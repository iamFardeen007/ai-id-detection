"""
Data Preparation Script
=======================
Creates train/val/test splits from the IDNet GRC dataset.
Uses symlinks to avoid duplicating ~3.6GB of images.

Usage:
    conda activate aiid
    cd ~/ai_id_detection
    python src/preprocessing/prepare_data.py
"""

import os
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Max images per class (None = use all)
# Set to a smaller number for quick testing
MAX_PER_CLASS = None  # e.g., 1000 for quick runs

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
IDNET_DIR = PROJECT_ROOT / "data" / "idnet" / "GRC"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Source folders → class mapping
AUTHENTIC_FOLDERS = ["positive"]
FRAUD_FOLDERS = ["fraud5_inpaint_and_rewrite", "fraud6_crop_and_replace"]

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def collect_images(base_dir: Path, folders: list) -> list:
    """Collect all image paths from specified folders."""
    images = []
    for folder_name in folders:
        folder = base_dir / folder_name
        if not folder.exists():
            print(f"  ⚠️ Folder not found: {folder}")
            continue
        
        count = 0
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                images.append(f)
                count += 1
        print(f"  📁 {folder_name}: {count} images")
    
    return images


def create_splits(images: list, train_r: float, val_r: float, test_r: float, 
                  seed: int) -> dict:
    """Split images into train/val/test sets."""
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_r)
    val_end = train_end + int(total * val_r)
    
    return {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
        'test': shuffled[val_end:]
    }


def create_split_folders(authentic_splits: dict, fraud_splits: dict, 
                         output_dir: Path, use_symlinks: bool = True):
    """Create the train/val/test folder structure with images."""
    
    for split_name in ['train', 'val', 'test']:
        for class_name, images in [('authentic', authentic_splits[split_name]), 
                                     ('fraudulent', fraud_splits[split_name])]:
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for src_path in images:
                dst_path = target_dir / src_path.name
                
                # Handle duplicate filenames (from different fraud folders)
                if dst_path.exists():
                    stem = src_path.stem
                    suffix = src_path.suffix
                    parent_name = src_path.parent.name[:6]  # e.g., "fraud5" or "fraud6"
                    dst_path = target_dir / f"{stem}_{parent_name}{suffix}"
                
                if use_symlinks:
                    # Symlink to save disk space
                    if dst_path.exists() or dst_path.is_symlink():
                        dst_path.unlink()
                    os.symlink(src_path.resolve(), dst_path)
                else:
                    # Copy file
                    shutil.copy2(src_path, dst_path)


def main():
    print("=" * 60)
    print("🔧 IDNet Data Preparation")
    print("=" * 60)
    
    # Check source directory
    if not IDNET_DIR.exists():
        print(f"❌ IDNet directory not found: {IDNET_DIR}")
        print("   Run the download + extraction first.")
        sys.exit(1)
    
    # Collect images
    print(f"\n📂 Collecting authentic images from: {IDNET_DIR}")
    authentic_images = collect_images(IDNET_DIR, AUTHENTIC_FOLDERS)
    
    print(f"\n📂 Collecting fraudulent images from: {IDNET_DIR}")
    fraud_images = collect_images(IDNET_DIR, FRAUD_FOLDERS)
    
    # Apply max limit
    if MAX_PER_CLASS:
        authentic_images = authentic_images[:MAX_PER_CLASS]
        fraud_images = fraud_images[:MAX_PER_CLASS]
        print(f"\n⚡ Limited to {MAX_PER_CLASS} per class for quick testing")
    
    print(f"\n📊 Total collected:")
    print(f"   Authentic:  {len(authentic_images)}")
    print(f"   Fraudulent: {len(fraud_images)}")
    print(f"   Total:      {len(authentic_images) + len(fraud_images)}")
    
    # Create splits
    print(f"\n✂️ Splitting data ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")
    authentic_splits = create_splits(authentic_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    fraud_splits = create_splits(fraud_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    
    # Print split summary
    print(f"\n📊 Split Summary:")
    print(f"{'Split':<8} {'Authentic':>10} {'Fraudulent':>12} {'Total':>8}")
    print("-" * 42)
    for split in ['train', 'val', 'test']:
        a = len(authentic_splits[split])
        f = len(fraud_splits[split])
        print(f"{split:<8} {a:>10} {f:>12} {a+f:>8}")
    
    total_a = sum(len(authentic_splits[s]) for s in ['train', 'val', 'test'])
    total_f = sum(len(fraud_splits[s]) for s in ['train', 'val', 'test'])
    print("-" * 42)
    print(f"{'TOTAL':<8} {total_a:>10} {total_f:>12} {total_a+total_f:>8}")
    
    # Create folder structure
    print(f"\n📁 Creating split folders in: {OUTPUT_DIR}")
    
    # Clean existing splits
    if OUTPUT_DIR.exists():
        print("   Cleaning existing splits...")
        shutil.rmtree(OUTPUT_DIR)
    
    create_split_folders(authentic_splits, fraud_splits, OUTPUT_DIR, use_symlinks=True)
    
    # Verify
    print(f"\n✅ Verification:")
    for split in ['train', 'val', 'test']:
        for cls in ['authentic', 'fraudulent']:
            d = OUTPUT_DIR / split / cls
            count = len(list(d.iterdir())) if d.exists() else 0
            print(f"   {split}/{cls}: {count} files")
    
    print(f"\n🎉 Data preparation complete!")
    print(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
