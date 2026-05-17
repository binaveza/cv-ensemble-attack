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
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification


class LocalImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None, processor=None):
        self.data_dir = data_dir
        self.transform = transform
        self.processor = processor
        
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
        
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, label, self.images[idx]
        elif self.transform:
            image = self.transform(image)
            return image, label, self.images[idx]
        
        return image, label, self.images[idx]


class DINOv2Wrapper(torch.nn.Module):
    """Обёртка для DINOv2"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits


def get_model_output(model, x):
    """Универсальная функция для получения логгитов из модели"""
    outputs = model(x)
    
    # Обработка tuple (YOLO)
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    
    # Обработка случаев, когда outputs имеет атрибут logits (DINOv2 через transformers)
    if hasattr(outputs, 'logits'):
        outputs = outputs.logits
    
    return outputs


def load_yolo_model(model_path, device):
    """Загрузка YOLO как в вашем файле long_test2(yolo)_1000.py"""
    print("   Загрузка YOLO11n-cls...")
    yolo = YOLO(model_path, task='classify')
    torch_model = yolo.model
    torch_model = torch_model.to(device)
    torch_model.eval()
    return torch_model


def load_dinov2_model(device):
    """Загрузка DINOv2 как в вашем файле long_test3(foundation)_1000.py"""
    print("   Загрузка DINOv2...")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base-imagenet1k-1-layer')
    model = AutoModelForImageClassification.from_pretrained(
        'facebook/dinov2-base-imagenet1k-1-layer'
    )
    model.eval()
    target_model = DINOv2Wrapper(model).to(device)
    return target_model, processor


def load_convnextv2_model(device):
    """Загрузка ConvNeXt-V2-base"""
    print("   Загрузка ConvNeXt-V2-base...")
    model = timm.create_model('convnextv2_base', pretrained=True)
    model = model.to(device)
    model.eval()
    return model


def load_model_by_type(model_name, device, return_processor=False):
    """Умная загрузка модели по имени"""
    
    if model_name == "yolo11n-cls":
        model_path = "./yolo11n-cls.pt"
        if not os.path.exists(model_path):
            for file in os.listdir("."):
                if file.endswith(".pt") and "yolo" in file.lower():
                    model_path = file
                    break
        model = load_yolo_model(model_path, device)
        return model, None
    
    elif model_name == "dinov2_vitb14":
        model, processor = load_dinov2_model(device)
        if return_processor:
            return model, processor
        return model, None
    
    elif model_name == "convnextv2_base":
        model = load_convnextv2_model(device)
        return model, None
    
    else:
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        return model, None


def ensemble_combined_attack(target_model, ensemble_models, images, labels, 
                            epsilon=0.03, num_iter=10, alpha=0.005):
    """
    Ансамблевая атака с поддержкой PGD и FGSM
    """
    alpha_step = alpha
    perturbed = images.clone().detach()
    
    # Random start для PGD
    use_random_start = any(attack_type == 'PGD' for _, attack_type in ensemble_models)
    if use_random_start:
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    for iteration in range(num_iter):
        total_grad = torch.zeros_like(images)
        
        for model, attack_type in ensemble_models:
            model.zero_grad()
            
            perturbed_iter = perturbed.clone().detach().requires_grad_(True)
            outputs = get_model_output(model, perturbed_iter)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            if perturbed_iter.grad is not None:
                total_grad += perturbed_iter.grad.sign()
            
            perturbed_iter.grad = None
        
        avg_grad = total_grad / len(ensemble_models)
        
        with torch.no_grad():
            perturbed = perturbed + alpha_step * avg_grad
            perturbed = torch.clamp(perturbed, images - epsilon, images + epsilon)
            perturbed = torch.clamp(perturbed, -2.5, 2.5)
    
    return perturbed


def test_ensemble_config(config_name, model_list, target_model_name, target_display_name, 
                        device, dataloader, dataset_size, processor=None):
    """Тестирование конфигурации ансамбля"""
    
    print(f"\n    {config_name}")
    
    # Загрузка целевой модели
    try:
        target_model, target_processor = load_model_by_type(target_model_name, device, return_processor=True)
        if target_processor and processor is None:
            processor = target_processor
    except Exception as e:
        print(f"    Не удалось загрузить целевую модель {target_model_name}: {e}")
        return None
    
    # Загрузка моделей ансамбля (суррогатные модели из timm)
    ensemble_models = []
    for model_name, attack_type in model_list:
        try:
            model = timm.create_model(model_name, pretrained=True)
            model = model.to(device)
            model.eval()
            ensemble_models.append((model, attack_type))
            print(f"      Загружена {model_name} -> {attack_type}")
        except Exception as e:
            print(f"       Пропускаем {model_name}: {e}")
            continue
    
    if len(ensemble_models) == 0:
        print("    Нет моделей в ансамбле")
        return None
    
    # Запуск атаки
    correct_before = []
    incorrect_after = []
    
    for images, labels, img_names in tqdm(dataloader, desc=f"   Атака на {target_display_name}", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Проверка до атаки
        with torch.no_grad():
            outputs = get_model_output(target_model, images)
            _, preds = outputs.max(1)
        
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
            outputs = get_model_output(target_model, perturbed)
            _, attack_preds = outputs.max(1)
        
        for i, (is_correct, img_name) in enumerate(zip(attack_preds == labels, img_names)):
            if not is_correct:
                incorrect_after.append(img_name)
    
    correct_set = set(correct_before)
    incorrect_set = set(incorrect_after)
    successful = correct_set & incorrect_set
    
    correct_before_count = len(correct_set)
    successful_count = len(successful)
    asr = 100.0 * successful_count / correct_before_count if correct_before_count > 0 else 0.0
    adv_accuracy = 100.0 * (dataset_size - len(incorrect_set)) / dataset_size
    
    print(f"      ASR: {asr:.2f}% ({successful_count}/{correct_before_count})")
    print(f"      Точность после атаки: {adv_accuracy:.2f}%")
    
    # Очистка памяти
    del target_model
    for model, _ in ensemble_models:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'config_name': config_name, 
        'asr': asr, 
        'successful': successful_count, 
        'correct_before': correct_before_count,
        'adv_accuracy': adv_accuracy
    }


def main():
    print("="*80)
    print("ЭКСПЕРИМЕНТ: Сравнение PGD vs FGSM в ансамблях")
    print("ЦЕЛЕВЫЕ МОДЕЛИ: YOLO11n-cls, ConvNeXt-V2-base, DINOv2")
    print("Суррогатные модели: efficientnet_b0, mobilenetv3_large_100, convnext_tiny")
    print("="*80)
    
    DATASET_PATH = "./imagenet_1000_random"
    if not os.path.exists(DATASET_PATH):
        print(f"\n Датасет не найден: {DATASET_PATH}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Устройство: {device}")
    
    # Суррогатные модели
    surrogate_models = ["efficientnet_b0", "mobilenetv3_large_100", "convnext_tiny"]
    
    # Конфигурации для тестирования
    configs = {
        "All_FGSM": [(m, 'FGSM') for m in surrogate_models],
        "All_PGD": [(m, 'PGD') for m in surrogate_models],
        "2FGSM_1PGD": [
            (surrogate_models[0], 'FGSM'),
            (surrogate_models[1], 'FGSM'),
            (surrogate_models[2], 'PGD'),
        ],
        "1FGSM_2PGD": [
            (surrogate_models[0], 'FGSM'),
            (surrogate_models[1], 'PGD'),
            (surrogate_models[2], 'PGD'),
        ],
        "FGSM_PGD_Alt": [
            (surrogate_models[0], 'FGSM'),
            (surrogate_models[1], 'PGD'),
            (surrogate_models[2], 'FGSM'),
        ],
    }
    
    # Целевые модели
    target_models = [
        {"name": "yolo11n-cls", "display": "YOLO11n-cls"},
        {"name": "convnextv2_base", "display": "ConvNeXt-V2-base"},
        {"name": "dinov2_vitb14", "display": "DINOv2"},
    ]
    
    all_results = {}
    
    for target in target_models:
        print(f"\n{'='*80}")
        print(f" ЦЕЛЕВАЯ МОДЕЛЬ: {target['display']} ({target['name']})")
        print(f"{'='*80}")
        
        # Создаем даталоадер для каждой модели
        if target['name'] == "dinov2_vitb14":
            _, processor = load_model_by_type(target['name'], device, return_processor=True)
            transform = None
        else:
            processor = None
            # Для YOLO своя трансформация (как в вашем файле)
            if target['name'] == "yolo11n-cls":
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        
        dataset = LocalImageNetDataset(DATASET_PATH, transform=transform, processor=processor)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        print(f" Загружено {len(dataset)} изображений")
        
        # Проверяем чистую точность целевой модели
        print(f"\n   ПРОВЕРКА ЧИСТОЙ ТОЧНОСТИ:")
        try:
            target_model, _ = load_model_by_type(target['name'], device, return_processor=True)
            correct = 0
            total = 0
            for images, labels, _ in tqdm(dataloader, desc="   Testing", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = get_model_output(target_model, images)
                    _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            clean_accuracy = 100.0 * correct / total
            print(f"   Clean accuracy: {clean_accuracy:.2f}% ({correct}/{total})")
            del target_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Не удалось проверить чистую точность: {e}")
        
        # Запуск тестов для конфигураций
        results = []
        for config_name, model_list in configs.items():
            result = test_ensemble_config(
                config_name, model_list, target["name"], target["display"], 
                device, dataloader, len(dataset), processor
            )
            if result:
                results.append(result)
        
        all_results[target["display"]] = results
    
    # Финальное сравнение
    print("\n" + "="*80)
    print(" ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*80)
    
    for target_name, results in all_results.items():
        if results:
            print(f"\n {target_name}:")
            print("-"*70)
            print(f"{'Конфигурация':<20} {'ASR %':<12} {'Успешно':<12}")
            print("-"*70)
            
            results_sorted = sorted(results, key=lambda x: x['asr'], reverse=True)
            for r in results_sorted:
                print(f"{r['config_name']:<20} {r['asr']:>6.2f}%     {r['successful']:>3}/{r['correct_before']:<3}")
            
            best = results_sorted[0]
            print(f"\n    Лучшая: {best['config_name']} (ASR: {best['asr']:.2f}%)")


if __name__ == "__main__":
    main()