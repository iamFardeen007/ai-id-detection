"""
Error Level Analysis (ELA) Module
=================================
Detects image manipulation by analyzing JPEG compression artifacts.
Re-saves an image at a known quality level and compares the difference.
Tampered regions show higher error levels (brighter in ELA image).

Used in: Baseline features, Streamlit visualization, report figures.
"""

import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os


def compute_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Compute Error Level Analysis for a single image.
    
    How it works:
    1. Re-save the image as JPEG at a specific quality level
    2. Compare the re-saved version with the original
    3. Amplify the differences → tampered areas appear brighter
    
    Args:
        image_path: Path to the input image
        quality: JPEG re-compression quality (default 90)
        scale: Amplification factor for differences (default 15)
    
    Returns:
        ELA image as numpy array (BGR format, same size as input)
    """
    # Open original image
    original = Image.open(image_path).convert('RGB')
    
    # Re-save at specified JPEG quality into memory buffer
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    
    # Open the re-compressed version
    recompressed = Image.open(buffer)
    
    # Compute pixel-wise absolute difference
    ela_image = ImageChops.difference(original, recompressed)
    
    # Get extrema for scaling
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1  # Avoid division by zero
    
    # Scale the differences to make them visible
    # Higher scale = more amplification of subtle differences
    scale_factor = 255.0 / max_diff * scale
    ela_image = ela_image.point(lambda x: min(int(x * scale_factor), 255))
    
    # Convert PIL Image to numpy array (RGB → BGR for OpenCV)
    ela_array = np.array(ela_image)
    ela_array = cv2.cvtColor(ela_array, cv2.COLOR_RGB2BGR)
    
    return ela_array


def compute_ela_features(image_path: str, quality: int = 90) -> dict:
    """
    Extract statistical features from ELA image for ML classification.
    
    Returns a dictionary of features:
    - Mean, std, max of each channel
    - Overall brightness statistics
    - Percentage of "hot" pixels (high error areas)
    
    Args:
        image_path: Path to the input image
        quality: JPEG re-compression quality
    
    Returns:
        Dictionary of ELA feature values
    """
    ela_img = compute_ela(image_path, quality=quality, scale=1)
    
    # Convert to grayscale for overall stats
    gray = cv2.cvtColor(ela_img, cv2.COLOR_BGR2GRAY)
    
    features = {
        # Per-channel statistics
        'ela_mean_b': float(np.mean(ela_img[:, :, 0])),
        'ela_mean_g': float(np.mean(ela_img[:, :, 1])),
        'ela_mean_r': float(np.mean(ela_img[:, :, 2])),
        'ela_std_b': float(np.std(ela_img[:, :, 0])),
        'ela_std_g': float(np.std(ela_img[:, :, 1])),
        'ela_std_r': float(np.std(ela_img[:, :, 2])),
        'ela_max_b': float(np.max(ela_img[:, :, 0])),
        'ela_max_g': float(np.max(ela_img[:, :, 1])),
        'ela_max_r': float(np.max(ela_img[:, :, 2])),
        
        # Grayscale statistics
        'ela_gray_mean': float(np.mean(gray)),
        'ela_gray_std': float(np.std(gray)),
        'ela_gray_max': float(np.max(gray)),
        'ela_gray_median': float(np.median(gray)),
        
        # Hot pixel percentage (pixels with high error > threshold)
        'ela_hot_pixel_pct': float(np.sum(gray > 50) / gray.size),
        'ela_very_hot_pixel_pct': float(np.sum(gray > 100) / gray.size),
    }
    
    return features


def save_ela_image(image_path: str, output_path: str, quality: int = 90, 
                   scale: int = 15) -> str:
    """
    Compute ELA and save the result to disk.
    
    Args:
        image_path: Input image path
        output_path: Where to save the ELA image
        quality: JPEG quality for re-compression
        scale: Amplification scale
    
    Returns:
        Path to saved ELA image
    """
    ela_img = compute_ela(image_path, quality=quality, scale=scale)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, ela_img)
    return output_path


def batch_compute_ela(image_dir: str, output_dir: str, quality: int = 90,
                      scale: int = 15) -> int:
    """
    Compute ELA for all images in a directory.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save ELA images
        quality: JPEG quality
        scale: Amplification scale
    
    Returns:
        Number of images processed
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            input_path = os.path.join(image_dir, filename)
            output_path = os.path.join(output_dir, filename)
            try:
                save_ela_image(input_path, output_path, quality, scale)
                count += 1
            except Exception as e:
                print(f"  ⚠️ Failed to process {filename}: {e}")
    
    return count


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Computing ELA for: {path}")
        features = compute_ela_features(path)
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Usage: python ela.py <image_path>")
