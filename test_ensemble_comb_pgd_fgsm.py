import torch
import timm
from torchvision import transforms
from PIL import Image
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
import itertools


class LocalImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        with open(f"{data_dir}/dataset_info.json", "r") as f:
            self.info = json.load(f)
        
        self.images = self.info['images']
        self.labels = self.info['labels']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "images", self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, self.images[idx]


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.005, num_iter=10, random_start=True):
    """Классическая PGD атака"""
    perturbed = images.clone().detach()
    
    if random_start:
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    for _ in range(num_iter):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = perturbed.grad.sign()
            perturbed = perturbed + alpha * grad
            perturbed = torch.clamp(perturbed, images - epsilon, images + epsilon)
            perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    return perturbed.detach()


def fgsm_attack(model, images, labels, epsilon=0.03):
    """Классическая FGSM атака (один шаг)"""
    perturbed = images.clone().detach().requires_grad_(True)
    outputs = model(perturbed)
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        grad = perturbed.grad.sign()
        perturbed = perturbed + epsilon * grad
        perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    return perturbed.detach()


def ensemble_combined_attack(target_model, ensemble_models, images, labels, 
                            epsilon=0.03, num_iter=10, alpha=0.005):
    """
    Ансамблевая атака с поддержкой PGD и FGSM
    ensemble_models: список кортежей (model, attack_type, param)
    attack_type: 'PGD' или 'FGSM'
    """
    alpha_step = alpha
    perturbed = images.clone().detach()
    
    # Для PGD нужен random start
    use_random_start = any(attack_type == 'PGD' for _, attack_type, _ in ensemble_models)
    if use_random_start:
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    for iteration in range(num_iter):
        total_grad = torch.zeros_like(images)
        
        for idx, (model, attack_type, param) in enumerate(ensemble_models):
            model.zero_grad()
            
            if attack_type == 'PGD':
                perturbed_iter = perturbed.clone().detach().requires_grad_(True)
                outputs = model(perturbed_iter)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                
                if perturbed_iter.grad is not None:
                    total_grad += perturbed_iter.grad.sign()
            
            elif attack_type == 'FGSM':
                perturbed_iter = perturbed.clone().detach().requires_grad_(True)
                outputs = model(perturbed_iter)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                
                if perturbed_iter.grad is not None:
                    total_grad += perturbed_iter.grad.sign()
            
            perturbed_iter.grad = None
        
        # Усредняем градиенты
        avg_grad = total_grad / len(ensemble_models)
        
        with torch.no_grad():
            perturbed = perturbed + alpha_step * avg_grad
            perturbed = torch.clamp(perturbed, images - epsilon, images + epsilon)
            perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    return perturbed


def load_model(model_name, device):
    """Загрузка одной модели"""
    try:
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"    Ошибка загрузки {model_name}: {e}")
        raise


def compute_clean_accuracy(target_model, dataloader, device):
    """Вычисление чистой точности модели на датасете"""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="   Проверка чистой точности", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = target_model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total, correct, total


def test_ensemble_config(config_name, model_list, target_model_name, target_display_name, device, dataloader, dataset_size):
    """Тестирование конфигурации ансамбля с проверкой accuracy"""
    
    print(f"\n    {config_name}")
    
    # Загрузка целевой модели
    target_model = load_model(target_model_name, device)
    
    # === ПРОВЕРКА ЧИСТОЙ ТОЧНОСТИ ===
    print(f"   Проверка чистой точности...")
    clean_accuracy, clean_correct, clean_total = compute_clean_accuracy(target_model, dataloader, device)
    print(f"      Чистая точность: {clean_accuracy:.2f}% ({clean_correct}/{clean_total})")
    
    # Загрузка моделей ансамбля
    ensemble_models = []
    for model_name, attack_type in model_list:
        try:
            model = load_model(model_name, device)
            ensemble_models.append((model, attack_type, None))
        except Exception as e:
            print(f"      Пропускаем {model_name}: {e}")
            continue
    
    if len(ensemble_models) == 0:
        return None
    
    # Запуск атаки
    print(f"   Запуск атаки...")
    correct_before = []
    incorrect_after = []
    
    # Для подсчёта точности до атаки на атакуемых изображениях
    total_before_correct = 0
    total_before_count = 0
    
    for images, labels, img_names in tqdm(dataloader, desc=f"      Атака", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Проверка до атаки
        with torch.no_grad():
            outputs = target_model(images)
            _, preds = outputs.max(1)
        
        batch_correct = (preds == labels).sum().item()
        total_before_correct += batch_correct
        total_before_count += labels.size(0)
        
        for i, (is_correct, img_name) in enumerate(zip(preds == labels, img_names)):
            if is_correct:
                correct_before.append(img_name)
        
        # Атака
        try:
            perturbed = ensemble_combined_attack(
                target_model, ensemble_models, images, labels, 
                epsilon=16/255, num_iter=10, alpha=2/255
            )
        except Exception as e:
            print(f"       Ошибка атаки: {e}")
            continue
        
        # Проверка после атаки
        with torch.no_grad():
            outputs = target_model(perturbed)
            _, attack_preds = outputs.max(1)
        
        for i, (is_correct, img_name) in enumerate(zip(attack_preds == labels, img_names)):
            if not is_correct:
                incorrect_after.append(img_name)
    
    # Точность ДО атаки (на атакуемых изображениях)
    before_attack_accuracy = 100.0 * total_before_correct / total_before_count if total_before_count > 0 else 0.0
    
    # Результаты атаки
    correct_set = set(correct_before)
    incorrect_set = set(incorrect_after)
    successful = correct_set & incorrect_set
    
    correct_before_count = len(correct_set)
    successful_count = len(successful)
    asr = 100.0 * successful_count / correct_before_count if correct_before_count > 0 else 0.0
    after_attack_accuracy = 100.0 * (dataset_size - len(incorrect_set)) / dataset_size
    accuracy_drop = clean_accuracy - after_attack_accuracy
    
    print(f"\n      РЕЗУЛЬТАТЫ ДЛЯ {config_name}:")
    print(f"         Чистая точность:           {clean_accuracy:.2f}%")
    print(f"         Точность ДО атаки:         {before_attack_accuracy:.2f}%")
    print(f"         Правильно ДО атаки:        {correct_before_count}/{dataset_size}")
    print(f"         Успешно атаковано:         {successful_count}/{correct_before_count}")
    print(f"         ASR:                       {asr:.2f}%")
    print(f"         Точность ПОСЛЕ атаки:      {after_attack_accuracy:.2f}%")
    print(f"         Падение точности:          {accuracy_drop:.2f}%")
    
    del target_model
    for model, _, _ in ensemble_models:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'config_name': config_name, 
        'asr': asr, 
        'successful': successful_count, 
        'correct_before': correct_before_count,
        'clean_accuracy': clean_accuracy,
        'before_attack_accuracy': before_attack_accuracy,
        'after_attack_accuracy': after_attack_accuracy,
        'accuracy_drop': accuracy_drop
    }


def main():
    print("="*80)
    print("ЭКСПЕРИМЕНТ: Сравнение PGD vs FGSM в ансамблях")
    print("ЦЕЛЕВЫЕ МОДЕЛИ: ResNet50 и ViT-B/16")
    print("="*80)
    
    DATASET_PATH = "./imagenet_1000_random"
    if not os.path.exists(DATASET_PATH):
        print(f"\nДатасет не найден: {DATASET_PATH}")
        return
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LocalImageNetDataset(DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"\n Загружено {len(dataset)} изображений")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Устройство: {device}")
    
    # Суррогатные модели
    surrogate_models = ["efficientnet_b0", "mobilenetv3_large_100", "convnext_tiny"]
    
    # Конфигурации для тестирования
    configs = {
        "All_FGSM": [(m, 'FGSM') for m in surrogate_models],
        "All_PGD": [(m, 'PGD') for m in surrogate_models],
        "2FGSM_1PGD": [
            (surrogate_models[0], 'PGD'),
            (surrogate_models[1], 'PGD'),
            (surrogate_models[2], 'FGSM'),
        ],
        "1FGSM_2PGD": [
            (surrogate_models[0], 'PGD'),
            (surrogate_models[1], 'FGSM'),
            (surrogate_models[2], 'FGSM'),
        ]
    }
    
    # Целевые модели
    target_models = [
        {"name": "resnet50", "display": "ResNet50"},
        {"name": "vit_base_patch16_224", "display": "ViT-B/16"}
    ]
    
    all_results = {}
    
    for target in target_models:
        print(f"\n{'='*80}")
        print(f" ЦЕЛЕВАЯ МОДЕЛЬ: {target['display']}")
        print(f"{'='*80}")
        
        results = []
        for config_name, model_list in configs.items():
            result = test_ensemble_config(
                config_name, model_list, target["name"], target["display"], 
                device, dataloader, len(dataset)
            )
            if result:
                results.append(result)
        
        all_results[target["display"]] = results
    
    # Финальное сравнение
    print("\n" + "="*80)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*80)
    
    for target_name, results in all_results.items():
        if results:
            print(f"\n {target_name}:")
            print("-"*100)
            print(f"{'Конфигурация':<18} {'ASR %':<10} {'Успешно':<10} {'Clean Acc':<12} {'Acc после':<12} {'Падение':<12}")
            print("-"*100)
            
            results_sorted = sorted(results, key=lambda x: x['asr'], reverse=True)
            for r in results_sorted:
                print(f"{r['config_name']:<18} {r['asr']:>6.2f}%   {r['successful']:>3}/{r['correct_before']:<3}   {r['clean_accuracy']:>6.2f}%   {r['after_attack_accuracy']:>6.2f}%   {r['accuracy_drop']:>6.2f}%")
            
            best_asr = max(results, key=lambda x: x['asr'])
            best_clean = max(results, key=lambda x: x['clean_accuracy'])
            best_drop = max(results, key=lambda x: x['accuracy_drop'])
            
            print(f"\n    Лучший ASR: {best_asr['config_name']} ({best_asr['asr']:.2f}%, падение: {best_asr['accuracy_drop']:.2f}%)")
            print(f"    Лучшая чистая точность: {best_clean['config_name']} ({best_clean['clean_accuracy']:.2f}%)")
            print(f"    Макс. падение: {best_drop['config_name']} (падение: {best_drop['accuracy_drop']:.2f}%)")
    
    # Сравнение целевых моделей
    print("\n" + "="*80)
    print(" СРАВНЕНИЕ УЯЗВИМОСТИ ЦЕЛЕВЫХ МОДЕЛЕЙ")
    print("="*80)
    print(f"{'Целевая модель':<20} {'Clean Acc':<12} {'Лучший ASR':<12} {'Средний ASR':<12} {'Падение':<12}")
    print("-"*70)
    
    for target_name, results in all_results.items():
        if results:
            avg_clean = sum(r['clean_accuracy'] for r in results) / len(results)
            best_asr = max(r['asr'] for r in results)
            avg_asr = sum(r['asr'] for r in results) / len(results)
            avg_drop = sum(r['accuracy_drop'] for r in results) / len(results)
            
            print(f"{target_name:<20} {avg_clean:>10.2f}%    {best_asr:>10.2f}%    {avg_asr:>10.2f}%    {avg_drop:>10.2f}%")


if __name__ == "__main__":
    main()