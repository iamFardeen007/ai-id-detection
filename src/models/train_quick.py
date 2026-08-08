"""
Quick Training Script — 5-epoch sanity check on MPS
Run: python src/models/train_quick.py
"""
import sys
sys.path.insert(0, '.')

import torch
from src.preprocessing.dataset import IDNetDataset
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
from src.models.classifier import (
    DocumentClassifier, train_model, compute_class_weights
)
from src.utils.helpers import get_device, set_seed
from torch.utils.data import DataLoader

if __name__ == '__main__':
    set_seed(42)
    device = get_device()

    # Load data (subset for quick test)
    print("\n📂 Loading datasets...")
    train_ds = IDNetDataset('data/processed/train', transform=get_train_transforms(),
                            mode='folder', max_per_class=500)
    val_ds = IDNetDataset('data/processed/val', transform=get_val_transforms(),
                          mode='folder', max_per_class=200)

    # num_workers=0 for macOS compatibility with stdin scripts
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_loader)

    # Create model
    model = DocumentClassifier('efficientnet_b0', pretrained=True, freeze_backbone=True)

    # Quick 5-epoch sanity check
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=5,
        learning_rate=1e-3,
        save_dir='models/saved',
        model_name='efficientnet_b0_quick',
        class_weights=class_weights
    )

    print(f"\n✅ Quick training complete!")
    print(f"Best val accuracy: {max(history['val_acc']):.4f}")
