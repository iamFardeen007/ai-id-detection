"""
Full Training Script — EfficientNet-B0 on IDNet GRC
====================================================
Two-phase training:
  Phase 1: Frozen backbone, train classifier head (10 epochs)
  Phase 2: Unfreeze backbone, fine-tune everything (15 epochs, lower LR)

Run: python src/models/train_full.py
"""
import sys
sys.path.insert(0, '.')

import torch
from src.preprocessing.dataset import IDNetDataset
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
from src.models.classifier import (
    DocumentClassifier, train_model, evaluate, compute_class_weights
)
from src.evaluation.metrics import (
    plot_confusion_matrix, plot_roc_curve, plot_training_history, print_full_report
)
from src.utils.helpers import get_device, set_seed, Timer
from torch.utils.data import DataLoader
import numpy as np


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    timer = Timer().start()

    # ═══════════════════════════════════════════
    # Load full dataset
    # ═══════════════════════════════════════════
    print("\n📂 Loading FULL datasets...")
    train_ds = IDNetDataset('data/processed/train', transform=get_train_transforms(), mode='folder')
    val_ds = IDNetDataset('data/processed/val', transform=get_val_transforms(), mode='folder')
    test_ds = IDNetDataset('data/processed/test', transform=get_val_transforms(), mode='folder')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_loader)

    # ═══════════════════════════════════════════
    # Phase 1: Frozen backbone (train head only)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📌 PHASE 1: Frozen backbone — training classifier head")
    print("=" * 60)

    model = DocumentClassifier('efficientnet_b0', pretrained=True, freeze_backbone=True)

    history1 = train_model(
        model, train_loader, val_loader, device,
        num_epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir='models/saved',
        model_name='efficientnet_b0_phase1',
        class_weights=class_weights
    )

    # ═══════════════════════════════════════════
    # Phase 2: Unfreeze backbone (fine-tune all)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📌 PHASE 2: Unfreezing backbone — fine-tuning")
    print("=" * 60)

    # Load best phase 1 model
    checkpoint = torch.load('models/saved/efficientnet_b0_phase1.pth', 
                           map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded Phase 1 best model (val acc: {checkpoint['val_accuracy']:.4f})")

    # Unfreeze last 3 backbone blocks
    model.unfreeze_backbone(unfreeze_from=-3)

    history2 = train_model(
        model, train_loader, val_loader, device,
        num_epochs=15,
        learning_rate=1e-4,  # Lower LR for fine-tuning
        weight_decay=1e-5,
        save_dir='models/saved',
        model_name='efficientnet_b0_best',
        class_weights=class_weights
    )

    # ═══════════════════════════════════════════
    # Final Evaluation on Test Set
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📌 FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load('models/saved/efficientnet_b0_best.pth',
                           map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    test_results = evaluate(model, test_loader, criterion, device, 'TEST')

    # Print detailed report
    print_full_report(test_results['labels'], test_results['predictions'], 
                      'EfficientNet-B0')

    # ═══════════════════════════════════════════
    # Generate plots
    # ═══════════════════════════════════════════
    print("\n📊 Generating evaluation plots...")

    # Combine histories
    full_history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'val_loss': history1['val_loss'] + history2['val_loss'],
        'train_acc': history1['train_acc'] + history2['train_acc'],
        'val_acc': history1['val_acc'] + history2['val_acc'],
    }

    plot_training_history(full_history, 'EfficientNet-B0 Training History',
                          save_path='reports/figures/training_history.png')

    plot_confusion_matrix(test_results['labels'], test_results['predictions'],
                          'EfficientNet-B0 — Test Set Confusion Matrix',
                          save_path='reports/figures/confusion_matrix_effnet.png')

    plot_roc_curve(test_results['labels'], test_results['probabilities'],
                   'EfficientNet-B0 — ROC Curve',
                   save_path='reports/figures/roc_curve_effnet.png')

    elapsed = timer.stop()
    print(f"\n⏱️ Total training time: {elapsed/60:.1f} minutes")
    print(f"🏆 Final test accuracy: {test_results['accuracy']:.4f}")
    print(f"\n🎉 Training complete!")
