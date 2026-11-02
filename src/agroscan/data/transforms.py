"""Data transforms and augmentations for leaf image classification."""
from typing import Tuple
from torchvision import transforms


def get_agri_transforms(
    image_size: Tuple[int, int] = (224, 224), is_training: bool = True
) -> transforms.Compose:
    """Standard ImageNet preprocessing pipeline with leaf rotations for training."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
