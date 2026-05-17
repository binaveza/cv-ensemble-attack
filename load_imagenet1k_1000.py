import torch
import timm
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import os
import json
from tqdm import tqdm
import random
from collections import defaultdict

def download_random_imagenet_subset(num_images=1000, save_dir="./imagenet_1000_random", random_seed=42):
    """
    Скачать num_images случайных изображений из ImageNet-1K
    """
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    
    print(f"Downloading {num_images} random images from ImageNet-1K")
    print(f"Save directory: {save_dir}")
    print(f"Random seed: {random_seed}")
    
    print("\nLoading ImageNet-1K dataset...")
    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        streaming=True  
    )
    
    print(f"\nSelecting {num_images} random indices...")
    total_dataset_size = 50000
    indices_to_download = random.sample(range(total_dataset_size), num_images)
    indices_to_download.sort()
    
    print(f"Will download {len(indices_to_download)} images")
    
    
    print("\nLoading class names...")
    class_names = None
    try:
        
        for example in dataset:
            if hasattr(example, 'features') and hasattr(example.features['label'], 'names'):
                class_names = example.features['label'].names
                break
    except Exception as e:
        print(f"  Warning: Could not load class names: {e}")
    
    if class_names:
        print(f"  Found {len(class_names)} classes")
    else:
        print("  Warning: Could not load class names, using numeric labels")
    
    indices_set = set(indices_to_download)
    
    print("\nDownloading images...")
    downloaded = 0
    dataset_info = {
        'num_images': num_images,
        'total_images': len(indices_to_download),
        'images': [],
        'labels': [],
        'indices': [],
        'class_info': {}
    }
    
    current_idx = 0
    for example in tqdm(dataset, total=total_dataset_size, desc="Downloading"):
        if current_idx in indices_set:
            label = example["label"]
            image = example["image"]
            
            if class_names:
                class_name = class_names[label]
                safe_class_name = "".join(c if c.isalnum() or c == '_' else '_' for c in class_name).replace(' ', '_')
                if len(safe_class_name) > 50:
                    safe_class_name = safe_class_name[:50]
            else:
                safe_class_name = f"class_{label}"
            
            image_filename = f"img_{current_idx:06d}_class_{label:04d}_{safe_class_name}.jpg"
            if len(image_filename) > 200:
                image_filename = f"img_{current_idx:06d}_class_{label:04d}.jpg"
            
            image_path = os.path.join(save_dir, "images", image_filename)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(image_path)
            
            dataset_info['images'].append(image_filename)
            dataset_info['labels'].append(label)
            dataset_info['indices'].append(current_idx)
            
            if label not in dataset_info['class_info']:
                dataset_info['class_info'][label] = {
                    'name': class_names[label] if class_names else f"class_{label}",
                    'image_count': 0,
                    'image_files': []
                }
            
            dataset_info['class_info'][label]['image_count'] += 1
            dataset_info['class_info'][label]['image_files'].append(image_filename)
            downloaded += 1
        
        current_idx += 1
        
        if downloaded == len(indices_to_download):
            break
    
    print("\nSaving metadata...")
    
    with open(os.path.join(save_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    with open(os.path.join(save_dir, "image_list.txt"), "w") as f:
        for img_file, label, idx in zip(dataset_info['images'], dataset_info['labels'], dataset_info['indices']):
            class_name = dataset_info['class_info'].get(label, {}).get('name', f'class_{label}')
            f.write(f"{img_file} {label} {class_name} {idx}\n")
    
    actual_classes = len(dataset_info['class_info'])
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully downloaded {downloaded} images")
    print(f"  Classes covered: {actual_classes} / 1000")
    print(f"  Images saved to: {save_dir}/images")
    
    return save_dir, dataset_info


def verify_downloaded_dataset(dataset_dir="./imagenet_1000_random"):
    """Проверка скачанного датасета"""
    
    print("\n" + "=" * 60)
    print("VERIFYING DOWNLOADED DATASET")
    print("=" * 60)
    
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        print(f"❌ Dataset info not found at {info_path}")
        return
    
    with open(info_path, "r") as f:
        info = json.load(f)
    
    images_dir = os.path.join(dataset_dir, "images")
    images_list = info.get('images', [])
    labels_list = info.get('labels', [])
    class_info = info.get('class_info', {})
    
    print(f"📁 Dataset path: {dataset_dir}")
    print(f"📊 Expected images: {len(images_list)}")
    
    missing = 0
    for img_file in images_list:
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path):
            missing += 1
    
    print(f"✅ Existing images: {len(images_list) - missing}")
    
    print(f"\n📋 Sample images (first 5):")
    for i in range(min(5, len(images_list))):
        img_file = images_list[i]
        label = labels_list[i]
        
        if label in class_info:
            class_name = class_info[label].get('name', f'class_{label}')
        else:
            class_name = f'class_{label}'
        
        img_path = os.path.join(images_dir, img_file)
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                print(f"  {img_file}")
                print(f"    -> class {label}: {class_name} ({img.size})")
        else:
            print(f"  {img_file} -> MISSING")
    
    print(f"\n📊 Statistics:")
    print(f"  Total images: {len(images_list)}")
    print(f"  Unique classes in dataset: {len(class_info)}")


if __name__ == "__main__":
    print("=" * 60)
    print("IMAGENET-1K RANDOM DATASET DOWNLOADER")
    print("=" * 60)
    
    dataset_path, dataset_info = download_random_imagenet_subset(
        num_images=1000,
        save_dir="./imagenet_1000_random",
        random_seed=42
    )
    
    verify_downloaded_dataset(dataset_path)
    
    print("\n" + "=" * 60)
    print("💡 HOW TO USE")
    print("=" * 60)
    print(f'\nDATASET_PATH = "{dataset_path}"')