"""
Evaluate the trained model on test set.
Run: python src/models/evaluate_model.py
"""
import sys
sys.path.insert(0, '.')
import os
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing.dataset import IDNetDataset
from src.preprocessing.augmentation import get_val_transforms
from src.models.classifier import DocumentClassifier, evaluate
from src.evaluation.metrics import (
    plot_confusion_matrix, plot_roc_curve, print_full_report
)
from src.utils.helpers import get_device, set_seed
from torch.utils.data import DataLoader

if __name__ == '__main__':
    set_seed(42)
    device = get_device()

    # Load test data
    print("\n📂 Loading test set...")
    test_ds = IDNetDataset('data/processed/test', transform=get_val_transforms(), mode='folder')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load best model
    print("\n📦 Loading best model...")
    model = DocumentClassifier('efficientnet_b0', pretrained=False, freeze_backbone=False)
    
    checkpoint = torch.load('models/saved/efficientnet_b0_best.pth',
                           map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"   Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"   Val accuracy at save: {checkpoint.get('val_accuracy', '?'):.4f}")

    # Evaluate on test set
    criterion = torch.nn.CrossEntropyLoss()
    print("\n📊 Evaluating on test set...")
    test_results = evaluate(model, test_loader, criterion, device, 'TEST')

    # Print full report
    print_full_report(test_results['labels'], test_results['predictions'], 
                      'EfficientNet-B0')

    # Generate plots (non-interactive)
    print("\n📊 Generating evaluation plots...")
    
    plot_confusion_matrix(test_results['labels'], test_results['predictions'],
                          'EfficientNet-B0 — Test Set Confusion Matrix',
                          save_path='reports/figures/confusion_matrix_effnet.png')
    plt.close()

    plot_roc_curve(test_results['labels'], test_results['probabilities'],
                   'EfficientNet-B0 — ROC Curve',
                   save_path='reports/figures/roc_curve_effnet.png')
    plt.close()

    # Also evaluate Phase 1 model for comparison
    print("\n\n📦 Also evaluating Phase 1 model...")
    checkpoint1 = torch.load('models/saved/efficientnet_b0_phase1.pth',
                            map_location=device, weights_only=True)
    model.load_state_dict(checkpoint1['model_state_dict'])
    model = model.to(device)
    
    p1_results = evaluate(model, test_loader, criterion, device, 'Phase1-TEST')
    
    print(f"\n\n{'='*60}")
    print(f"📊 FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Val Acc':>10} {'Test Acc':>10}")
    print(f"{'-'*47}")
    print(f"{'Phase 1 (frozen)':<25} {checkpoint1.get('val_accuracy', 0):.4f}     {p1_results['accuracy']:.4f}")
    print(f"{'Phase 2 (fine-tuned)':<25} {checkpoint.get('val_accuracy', 0):.4f}     {test_results['accuracy']:.4f}")
    print(f"\n🎉 Evaluation complete!")
