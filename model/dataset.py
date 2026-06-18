"""
数据加载与预处理模块
"""

import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

CLASS_MAPPING = {
    "COVID": 0, "COVID-19": 0,
    "Normal": 1,
    "Lung_Opacity": 2, "Lung Opacity": 2,
    "Viral Pneumonia": 3, "Viral_Pneumonia": 3,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
CLASS_NAMES = ["新冠肺炎", "正常", "肺部阴影", "病毒性肺炎"]


class COVIDRadiographyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


def get_train_transforms(input_size=224):
    return A.Compose([
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.01, 0.05), p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(input_size=224):
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def scan_dataset(data_dir):
    image_paths, labels = [], []
    visited_dirs = set()
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    for class_name, class_idx in CLASS_MAPPING.items():
        candidates = [
            os.path.join(data_dir, class_name),
            os.path.join(data_dir, class_name.replace("_", " ")),
            os.path.join(data_dir, class_name.replace(" ", "_")),
        ]
        actual_dir = None
        for d in candidates:
            if os.path.isdir(d) and os.path.realpath(d) not in visited_dirs:
                actual_dir = d; break
        if actual_dir is None:
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and class_name.lower() in item.lower() and os.path.realpath(item_path) not in visited_dirs:
                    actual_dir = item_path; break
        if actual_dir is None:
            print(f"⚠️ 未找到类别目录: {class_name}")
            continue
        visited_dirs.add(os.path.realpath(actual_dir))
        # 兼容两种数据集结构：文件直接在目录下 或 在 images/ 子目录中
        search_dir = actual_dir
        images_subdir = os.path.join(actual_dir, "images")
        if os.path.isdir(images_subdir):
            search_dir = images_subdir
        files = [f for f in os.listdir(search_dir) if f.lower().endswith(valid_extensions)]
        for f in files:
            image_paths.append(os.path.join(search_dir, f))
            labels.append(class_idx)
        print(f"  ✓ {class_name}: {len(files)} 张图像")

    print(f"\n总计: {len(image_paths)} 张图像")
    return image_paths, labels


def split_dataset(image_paths, labels, val_ratio=0.1, test_ratio=0.1, random_state=42):
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_ratio, stratify=labels, random_state=random_state)
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_ratio_adjusted,
        stratify=train_val_labels, random_state=random_state)
    print(f"\n数据集划分: 训练 {len(train_paths)} | 验证 {len(val_paths)} | 测试 {len(test_paths)}")
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def create_dataloaders(data_dir, batch_size=32, input_size=224, num_workers=0):
    print("扫描数据集...")
    image_paths, labels = scan_dataset(data_dir)
    print("\n划分数据集...")
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(image_paths, labels)

    train_dataset = COVIDRadiographyDataset(train_paths, train_labels, transform=get_train_transforms(input_size))
    val_dataset = COVIDRadiographyDataset(val_paths, val_labels, transform=get_val_transforms(input_size))
    test_dataset = COVIDRadiographyDataset(test_paths, test_labels, transform=get_val_transforms(input_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader, CLASS_NAMES
