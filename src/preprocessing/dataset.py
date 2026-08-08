"""
IDNet Dataset Module
====================
PyTorch Dataset class for loading IDNet identity document images.
Handles the IDNet folder structure: positive/ (authentic) vs fraud*/ (tampered).
Also supports generic image folder classification (real/ vs fake/).

Used in: Training, evaluation, data loading pipeline.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2


class IDNetDataset(Dataset):
    """
    PyTorch Dataset for IDNet identity document images.
    
    Loads images from IDNet folder structure:
      LOC_XX/positive/     → label 0 (authentic)
      LOC_XX/fraud5_*/     → label 1 (fraudulent)
      LOC_XX/fraud6_*/     → label 1 (fraudulent)
    
    Also supports simple folder structure:
      authentic/  → label 0
      fraudulent/ → label 1
    """
    
    # Class labels
    CLASSES = ['authentic', 'fraudulent']
    
    def __init__(self, root_dir: str, transform=None, max_per_class: Optional[int] = None,
                 mode: str = 'idnet'):
        """
        Args:
            root_dir: Root directory containing the dataset
            transform: torchvision transforms to apply
            max_per_class: Maximum images per class (for quick testing)
            mode: 'idnet' for IDNet structure, 'folder' for simple authentic/fraudulent folders
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        
        if mode == 'idnet':
            self._load_idnet(root_dir, max_per_class)
        elif mode == 'folder':
            self._load_folder(root_dir, max_per_class)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'idnet' or 'folder'.")
        
        print(f"📊 Dataset loaded: {len(self.samples)} images")
        print(f"   Authentic: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"   Fraudulent: {sum(1 for _, l in self.samples if l == 1)}")
    
    def _load_idnet(self, root_dir: str, max_per_class: Optional[int]):
        """Load images from IDNet folder structure."""
        authentic = []
        fraudulent = []
        
        root = Path(root_dir)
        
        # Walk through all location directories
        for loc_dir in sorted(root.iterdir()):
            if not loc_dir.is_dir():
                continue
            
            # Positive (authentic) images
            positive_dir = loc_dir / 'positive'
            if positive_dir.exists():
                for img_path in self._get_images(positive_dir):
                    authentic.append((str(img_path), 0))
            
            # Fraud directories (any folder starting with 'fraud')
            for fraud_dir in sorted(loc_dir.iterdir()):
                if fraud_dir.is_dir() and fraud_dir.name.startswith('fraud'):
                    for img_path in self._get_images(fraud_dir):
                        fraudulent.append((str(img_path), 1))
        
        # Apply max_per_class limit
        if max_per_class:
            authentic = authentic[:max_per_class]
            fraudulent = fraudulent[:max_per_class]
        
        self.samples = authentic + fraudulent
    
    def _load_folder(self, root_dir: str, max_per_class: Optional[int]):
        """Load images from simple folder structure (authentic/ + fraudulent/)."""
        root = Path(root_dir)
        
        for label, folder_name in enumerate(['authentic', 'fraudulent']):
            folder = root / folder_name
            if not folder.exists():
                # Try alternative names
                alt_names = {
                    'authentic': ['real', 'genuine', 'positive', 'original'],
                    'fraudulent': ['fake', 'tampered', 'fraud', 'negative']
                }
                for alt in alt_names.get(folder_name, []):
                    folder = root / alt
                    if folder.exists():
                        break
            
            if folder.exists():
                images = list(self._get_images(folder))
                if max_per_class:
                    images = images[:max_per_class]
                for img_path in images:
                    self.samples.append((str(img_path), label))
    
    def _get_images(self, directory: Path) -> List[Path]:
        """Get all valid image files from a directory (recursive)."""
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = []
        for f in sorted(directory.rglob('*')):
            if f.is_file() and f.suffix.lower() in valid_ext:
                images.append(f)
        return images
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize to 224x224 and convert to tensor
            image = image.resize((224, 224))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_path(self, idx: int) -> str:
        """Get the file path for a specific index."""
        return self.samples[idx][0]
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of images per class."""
        counts = {'authentic': 0, 'fraudulent': 0}
        for _, label in self.samples:
            counts[self.CLASSES[label]] += 1
        return counts


def create_data_splits(dataset: IDNetDataset, 
                       train_ratio: float = 0.70,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: The full dataset
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # Remainder goes to test
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {train_size} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {val_size} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {test_size} ({test_ratio*100:.0f}%)")
    
    return train_set, val_set, test_set


def create_dataloaders(train_set, val_set, test_set, 
                       batch_size: int = 32, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from dataset splits.
    
    Args:
        train_set, val_set, test_set: Dataset splits
        batch_size: Batch size (default 32)
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    print("IDNet Dataset module loaded successfully ✅")
    print(f"Classes: {IDNetDataset.CLASSES}")
