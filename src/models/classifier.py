"""
Transfer Learning Classifier
==============================
Fine-tunes pretrained models (EfficientNet-B0, ResNet50) for
binary classification: authentic vs fraudulent documents.

Architecture:
    Pretrained CNN backbone (frozen/unfrozen) → Global Avg Pool →
    Dropout → Linear(features, 256) → ReLU → Dropout → Linear(256, 2)

Usage:
    from src.models.classifier import DocumentClassifier, train_one_epoch, evaluate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
from pathlib import Path
from typing import Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️ timm not installed. Run: pip install timm")


class DocumentClassifier(nn.Module):
    """
    Transfer learning model for document fraud detection.
    
    Uses a pretrained backbone (EfficientNet-B0 or ResNet50) with
    a custom classification head for binary classification.
    """
    
    # Supported model architectures
    SUPPORTED_MODELS = {
        'efficientnet_b0': 'tf_efficientnet_b0',
        'resnet50': 'resnet50',
        'mobilenetv3': 'mobilenetv3_large_100',
    }
    
    def __init__(self, model_name: str = 'efficientnet_b0', 
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 pretrained: bool = True,
                 freeze_backbone: bool = True):
        """
        Args:
            model_name: One of 'efficientnet_b0', 'resnet50', 'mobilenetv3'
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate for regularization
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers initially
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Get timm model name
        timm_name = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        # Create backbone (remove original classification head)
        self.backbone = timm.create_model(
            timm_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'  # Keep global average pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        print(f"📦 Model: {model_name} (features={self.feature_dim})")
        print(f"   Pretrained: {pretrained} | Backbone frozen: {freeze_backbone}")
        print(f"   Classifier: {self.feature_dim} → 256 → {num_classes}")
    
    def freeze_backbone(self):
        """Freeze all backbone parameters (for initial training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("   🔒 Backbone frozen")
    
    def unfreeze_backbone(self, unfreeze_from: int = -3):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_from: Unfreeze last N layers (-3 = last 3 blocks)
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get all children modules
        children = list(self.backbone.children())
        
        # Unfreeze from the specified layer onwards
        for child in children[unfreeze_from:]:
            for param in child.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"   🔓 Backbone partially unfrozen: {trainable:,}/{total:,} params trainable")
    
    def forward(self, x):
        features = self.backbone(x)  # (batch, feature_dim)
        logits = self.classifier(features)  # (batch, num_classes)
        return logits
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def train_one_epoch(model, train_loader, criterion, optimizer, device, 
                    epoch: int = 0) -> dict:
    """
    Train model for one epoch.
    
    Returns:
        Dict with 'loss' and 'accuracy' for this epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress print every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.1f}%", end='\r')
    
    elapsed = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    print(f"  Epoch {epoch+1} Train | Loss: {epoch_loss:.4f} | "
          f"Acc: {epoch_acc:.4f} ({correct}/{total}) | Time: {elapsed:.1f}s")
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, 
             split_name: str = 'Val') -> dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        Dict with 'loss', 'accuracy', 'predictions', 'labels'
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    print(f"  {split_name:>5} | Loss: {epoch_loss:.4f} | "
          f"Acc: {epoch_acc:.4f} ({correct}/{total})")
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train_model(model, train_loader, val_loader, device,
                num_epochs: int = 20,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-4,
                save_dir: str = 'models/saved',
                model_name: str = 'best_model',
                class_weights: Optional[torch.Tensor] = None) -> dict:
    """
    Full training loop with early stopping and model saving.
    
    Args:
        model: The classifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Compute device
        num_epochs: Maximum epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        save_dir: Directory to save model checkpoints
        model_name: Name for saved model file
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    
    # Loss function (with optional class weights for imbalance)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer — only train parameters that require grad
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting training: {num_epochs} epochs")
    print(f"   LR: {learning_rate} | Weight Decay: {weight_decay}")
    print(f"   Trainable params: {model.get_trainable_params():,}")
    print(f"   Device: {device}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, 
                                        optimizer, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, 'Val')
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'model_name': model.model_name,
            }, save_path / f'{model_name}.pth')
            
            print(f"  💾 Best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break
        
        print()
    
    print("=" * 60)
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved: {save_path / f'{model_name}.pth'}")
    
    return history


def compute_class_weights(train_loader) -> torch.Tensor:
    """
    Compute class weights to handle class imbalance.
    
    With 2:1 fraud:authentic ratio, we upweight the minority class.
    """
    label_counts = [0, 0]
    for _, labels in train_loader:
        for l in labels:
            label_counts[l.item()] += 1
    
    total = sum(label_counts)
    weights = [total / (2 * count) for count in label_counts]
    
    print(f"📊 Class weights: authentic={weights[0]:.3f}, fraudulent={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    print("Classifier module loaded ✅")
    
    if TIMM_AVAILABLE:
        # Quick test — create model
        model = DocumentClassifier('efficientnet_b0', pretrained=False)
        print(f"\nTotal params: {model.get_total_params():,}")
        print(f"Trainable params: {model.get_trainable_params():,}")
        
        # Test forward pass
        dummy = torch.randn(2, 3, 224, 224)
        output = model(dummy)
        print(f"Output shape: {output.shape}")
    else:
        print("Install timm to test: pip install timm")
