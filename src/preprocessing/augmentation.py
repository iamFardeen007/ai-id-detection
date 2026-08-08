"""
Data Augmentation Pipeline
==========================
Defines torchvision transforms for training and evaluation.
Training uses augmentation (flips, rotations, color jitter) to prevent overfitting.
Evaluation uses only resize + normalize (no augmentation).

Used in: Training pipeline, DataLoader setup.
"""

from torchvision import transforms


# ImageNet normalization values (used for all pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default image size for pretrained models
IMAGE_SIZE = 224


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Get augmentation transforms for training.
    
    Augmentations applied:
    - Random resize crop (simulates different document scales)
    - Random horizontal flip (documents can be flipped)
    - Random rotation (±10°, simulates scanner misalignment)
    - Color jitter (simulates different scanning/camera conditions)
    - Random grayscale (5% chance - robustness to color variations)
    - Normalize to ImageNet statistics
    
    Args:
        image_size: Target size for the images (default 224)
    
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.1, 
            hue=0.05
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Get transforms for validation/testing (no augmentation).
    
    Only resizes and normalizes — we want consistent evaluation.
    
    Args:
        image_size: Target size (default 224)
    
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_ela_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Get transforms for ELA images (different normalization).
    
    ELA images have different intensity distributions than natural images,
    so we use simpler normalization.
    
    Args:
        image_size: Target size (default 224)
    
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Simple 0-1 normalization, no ImageNet stats
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse ImageNet normalization for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor clipped to [0, 1]
    """
    import torch
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


if __name__ == "__main__":
    print("Augmentation module loaded ✅")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Train transforms: {get_train_transforms()}")
    print(f"Val transforms: {get_val_transforms()}")
