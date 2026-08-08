"""
Traditional ML Baseline
========================
SVM and Random Forest classifiers using ELA + histogram features.
This provides a baseline accuracy to compare against deep learning.

Usage:
    python src/models/baseline.py
"""

import sys
import os
import random
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.ela import compute_ela_features


def extract_histogram_features(image_path: str) -> dict:
    """
    Extract color histogram features from an image.
    
    Returns statistics of the RGB histogram — captures overall
    color distribution differences between authentic and tampered docs.
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    features = {}
    for i, channel in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()  # Normalize
        
        features[f'hist_{channel}_mean'] = float(np.mean(hist))
        features[f'hist_{channel}_std'] = float(np.std(hist))
        features[f'hist_{channel}_skew'] = float(
            np.sum(((hist - np.mean(hist)) / (np.std(hist) + 1e-7)) ** 3) / len(hist)
        )
        features[f'hist_{channel}_entropy'] = float(
            -np.sum(hist * np.log2(hist + 1e-10))
        )
    
    return features


def extract_all_features(image_path: str) -> np.ndarray:
    """
    Extract combined ELA + histogram features for a single image.
    Returns a flat feature vector.
    """
    ela_feats = compute_ela_features(image_path)
    hist_feats = extract_histogram_features(image_path)
    
    # Combine all features into a single dict
    all_feats = {**ela_feats, **hist_feats}
    
    # Convert to ordered array
    return np.array(list(all_feats.values()))


def get_feature_names() -> list:
    """Get ordered list of feature names."""
    ela_names = [
        'ela_mean_b', 'ela_mean_g', 'ela_mean_r',
        'ela_std_b', 'ela_std_g', 'ela_std_r',
        'ela_max_b', 'ela_max_g', 'ela_max_r',
        'ela_gray_mean', 'ela_gray_std', 'ela_gray_max', 'ela_gray_median',
        'ela_hot_pixel_pct', 'ela_very_hot_pixel_pct'
    ]
    hist_names = []
    for ch in ['b', 'g', 'r']:
        hist_names.extend([
            f'hist_{ch}_mean', f'hist_{ch}_std',
            f'hist_{ch}_skew', f'hist_{ch}_entropy'
        ])
    return ela_names + hist_names


def load_dataset_features(data_dir: str, max_per_class: int = None) -> tuple:
    """
    Extract features from all images in a processed split directory.
    
    Args:
        data_dir: Path like 'data/processed/train'
        max_per_class: Limit images per class (for speed)
    
    Returns:
        (X, y) - feature matrix and labels
    """
    data_path = Path(data_dir)
    X_list = []
    y_list = []
    
    for label, class_name in enumerate(['authentic', 'fraudulent']):
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"  ⚠️ {class_dir} not found")
            continue
        
        files = sorted(list(class_dir.iterdir()))
        
        if max_per_class and len(files) > max_per_class:
            random.seed(42)
            files = random.sample(files, max_per_class)
        
        print(f"  Extracting features from {class_name}: {len(files)} images...")
        
        for f in tqdm(files, desc=f"  {class_name}"):
            try:
                # Resolve symlinks
                real_path = str(f.resolve())
                features = extract_all_features(real_path)
                X_list.append(features)
                y_list.append(label)
            except Exception as e:
                pass  # Skip corrupted images
    
    return np.array(X_list), np.array(y_list)


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train SVM and Random Forest, evaluate both.
    
    Returns dict of results.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # === SVM ===
    print("\n🔧 Training SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    svm_val_pred = svm.predict(X_val_scaled)
    svm_test_pred = svm.predict(X_test_scaled)
    
    results['svm'] = {
        'model': svm,
        'val_accuracy': accuracy_score(y_val, svm_val_pred),
        'test_accuracy': accuracy_score(y_test, svm_test_pred),
        'test_precision': precision_score(y_test, svm_test_pred),
        'test_recall': recall_score(y_test, svm_test_pred),
        'test_f1': f1_score(y_test, svm_test_pred),
        'test_predictions': svm_test_pred,
        'report': classification_report(y_test, svm_test_pred, 
                                        target_names=['Authentic', 'Fraudulent'])
    }
    
    print(f"   Val Accuracy: {results['svm']['val_accuracy']:.4f}")
    print(f"   Test Accuracy: {results['svm']['test_accuracy']:.4f}")
    
    # === Random Forest ===
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    rf_val_pred = rf.predict(X_val_scaled)
    rf_test_pred = rf.predict(X_test_scaled)
    
    results['rf'] = {
        'model': rf,
        'val_accuracy': accuracy_score(y_val, rf_val_pred),
        'test_accuracy': accuracy_score(y_test, rf_test_pred),
        'test_precision': precision_score(y_test, rf_test_pred),
        'test_recall': recall_score(y_test, rf_test_pred),
        'test_f1': f1_score(y_test, rf_test_pred),
        'test_predictions': rf_test_pred,
        'report': classification_report(y_test, rf_test_pred,
                                        target_names=['Authentic', 'Fraudulent']),
        'feature_importance': rf.feature_importances_
    }
    
    print(f"   Val Accuracy: {results['rf']['val_accuracy']:.4f}")
    print(f"   Test Accuracy: {results['rf']['test_accuracy']:.4f}")
    
    results['scaler'] = scaler
    
    return results


def save_models(results: dict, save_dir: str):
    """Save trained models and scaler."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_path / 'svm_baseline.pkl', 'wb') as f:
        pickle.dump(results['svm']['model'], f)
    
    with open(save_path / 'rf_baseline.pkl', 'wb') as f:
        pickle.dump(results['rf']['model'], f)
    
    with open(save_path / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(results['scaler'], f)
    
    print(f"\n💾 Models saved to {save_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    print("=" * 60)
    print("🔬 Traditional ML Baseline Training")
    print("=" * 60)
    
    # Extract features (use subset for speed)
    MAX_SAMPLES = 500  # Per class — adjust for full run
    
    print(f"\n📊 Extracting features (max {MAX_SAMPLES}/class)...")
    print("\nTrain set:")
    X_train, y_train = load_dataset_features(
        str(PROJECT_ROOT / 'data/processed/train'), max_per_class=MAX_SAMPLES
    )
    
    print("\nVal set:")
    X_val, y_val = load_dataset_features(
        str(PROJECT_ROOT / 'data/processed/val'), max_per_class=200
    )
    
    print("\nTest set:")
    X_test, y_test = load_dataset_features(
        str(PROJECT_ROOT / 'data/processed/test'), max_per_class=200
    )
    
    print(f"\n📊 Feature matrix shapes:")
    print(f"   Train: {X_train.shape} (labels: {np.bincount(y_train)})")
    print(f"   Val:   {X_val.shape} (labels: {np.bincount(y_val)})")
    print(f"   Test:  {X_test.shape} (labels: {np.bincount(y_test)})")
    
    # Train and evaluate
    results = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("📊 BASELINE RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Val Acc':>10} {'Test Acc':>10} {'F1':>10}")
    print("-" * 52)
    for name, display in [('svm', 'SVM (RBF)'), ('rf', 'Random Forest')]:
        r = results[name]
        print(f"{display:<20} {r['val_accuracy']:>10.4f} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f}")
    
    print(f"\n📋 SVM Classification Report:")
    print(results['svm']['report'])
    
    print(f"\n📋 Random Forest Classification Report:")
    print(results['rf']['report'])
    
    # Save models
    save_models(results, str(PROJECT_ROOT / 'models/saved'))
