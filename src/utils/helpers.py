"""
Utility Functions
=================
Common helper functions used across the project.
"""

import os
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════
# Device & Reproducibility
# ═══════════════════════════════════════════════

def get_device() -> torch.device:
    """
    Get the best available compute device.
    Priority: CUDA > MPS (Apple Silicon/Intel) > CPU
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple MPS acceleration")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Random seed set to {seed}")


# ═══════════════════════════════════════════════
# File System Helpers
# ═══════════════════════════════════════════════

def count_images(directory: str) -> dict:
    """
    Count images in a directory and its subdirectories.
    
    Args:
        directory: Path to scan
    
    Returns:
        Dict with folder names as keys and counts as values
    """
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    counts = {}
    
    root = Path(directory)
    if not root.exists():
        return {'error': f'Directory not found: {directory}'}
    
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir():
            count = sum(1 for f in subdir.rglob('*') 
                       if f.is_file() and f.suffix.lower() in valid_ext)
            if count > 0:
                counts[subdir.name] = count
    
    # Also count files directly in root
    root_count = sum(1 for f in root.iterdir() 
                     if f.is_file() and f.suffix.lower() in valid_ext)
    if root_count > 0:
        counts['.'] = root_count
    
    return counts


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


# ═══════════════════════════════════════════════
# Visualization Helpers
# ═══════════════════════════════════════════════

def show_images(images: list, titles: list = None, cols: int = 4, 
                figsize: tuple = None, save_path: Optional[str] = None):
    """
    Display a grid of images.
    
    Args:
        images: List of numpy arrays (BGR or RGB) or PIL images
        titles: Optional list of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size (auto-calculated if None)
        save_path: If provided, save figure to this path
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    if figsize is None:
        figsize = (4 * cols, 4 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Check if BGR (OpenCV) → convert to RGB
                img = img[:, :, ::-1]  # BGR to RGB
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        else:
            ax.imshow(img)
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Figure saved: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════
# Training Helpers
# ═══════════════════════════════════════════════

class Timer:
    """Simple timer for tracking training time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __str__(self):
        mins = int(self.elapsed // 60)
        secs = int(self.elapsed % 60)
        return f"{mins}m {secs}s"


if __name__ == "__main__":
    print("Utils module loaded ✅")
    device = get_device()
    set_seed(42)
    print(f"Project root: {get_project_root()}")
