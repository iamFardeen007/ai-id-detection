"""
Streamlit Web Application
==========================
Document Fraud Detection — Upload & Analyze

Features:
    - Upload an identity document image
    - Predict: Authentic or Fraudulent (with confidence %)
    - Show ELA (Error Level Analysis) visualization
    - Display model details and performance metrics

Run:
    cd ~/ai_id_detection
    streamlit run src/app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageChops
import io
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing.augmentation import get_val_transforms
from src.models.classifier import DocumentClassifier
from src.utils.helpers import get_device


# ═══════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="DocGuard AI — Document Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-authentic {
        background: linear-gradient(135deg, #0d9488 0%, #059669 100%);
        color: white;
    }
    .result-fraud {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
    }
    .result-card h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    .result-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Confidence bar */
    .confidence-container {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        transition: width 0.5s ease;
    }
    
    /* Info cards */
    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics row */
    .metric-box {
        background: #1e293b;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-box h3 {
        color: #60a5fa;
        margin: 0;
        font-size: 1.5rem;
    }
    .metric-box p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Model Loading (cached)
# ═══════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)."""
    device = torch.device('cpu')  # Use CPU for Streamlit (more stable)
    
    model = DocumentClassifier('efficientnet_b0', pretrained=False, freeze_backbone=False)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 
                              'models', 'saved', 'efficientnet_b0_best.pth')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.stop()
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


@st.cache_data
def get_transforms():
    """Get validation transforms (cached)."""
    return get_val_transforms()


# ═══════════════════════════════════════════════
# ELA Computation
# ═══════════════════════════════════════════════
def compute_ela_for_display(image: Image.Image, quality: int = 90, 
                             scale: int = 15) -> np.ndarray:
    """
    Compute ELA image for visualization.
    Returns RGB numpy array with amplified error levels.
    """
    # Save and re-load at specified quality
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    
    # Compute difference
    ela = ImageChops.difference(image, recompressed)
    
    # Amplify
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_factor = 255.0 / max_diff * scale
    
    ela = ela.point(lambda x: min(int(x * scale_factor), 255))
    
    return np.array(ela)


def compute_ela_stats(image: Image.Image) -> dict:
    """Compute ELA statistics for the uploaded image."""
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    
    ela = ImageChops.difference(image, recompressed)
    ela_array = np.array(ela)
    gray = np.mean(ela_array, axis=2) if len(ela_array.shape) == 3 else ela_array
    
    return {
        'mean_error': float(np.mean(gray)),
        'std_error': float(np.std(gray)),
        'max_error': float(np.max(gray)),
        'hot_pixels': float(np.sum(gray > 50) / gray.size * 100),
    }


# ═══════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════
def predict(image: Image.Image, model, device, transforms) -> dict:
    """
    Run prediction on a single image.
    
    Returns dict with prediction, confidence, and probabilities.
    """
    # Preprocess
    img_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    class_names = ['Authentic', 'Fraudulent']
    
    return {
        'prediction': class_names[pred_class],
        'confidence': confidence,
        'is_fraud': pred_class == 1,
        'prob_authentic': probs[0][0].item(),
        'prob_fraudulent': probs[0][1].item(),
    }


# ═══════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ DocGuard AI</h1>
        <p>AI-Powered Identity Document Fraud Detection</p>
        <p style="font-size: 0.85rem; color: #64748b;">
            Powered by EfficientNet-B0 + Error Level Analysis | Accuracy: 93%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()
        transforms = get_transforms()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        ela_quality = st.slider("ELA Quality", 50, 99, 90, 
                                help="JPEG quality for ELA computation")
        ela_scale = st.slider("ELA Amplification", 1, 30, 15,
                              help="How much to amplify ELA differences")
        show_ela = st.checkbox("Show ELA Analysis", value=True)
        show_stats = st.checkbox("Show ELA Statistics", value=True)
        
        st.markdown("---")
        st.markdown("## 📊 Model Info")
        st.markdown("""
        - **Model:** EfficientNet-B0
        - **Accuracy:** 93.0%
        - **F1 Score:** 0.95
        - **AUC:** 0.989
        - **Dataset:** IDNet (GRC)
        - **Training:** 2-phase transfer learning
        """)
        
        st.markdown("---")
        st.markdown("## 📖 About")
        st.markdown("""
        **DocGuard AI** detects tampered identity 
        documents using deep learning and forensic 
        image analysis (ELA).
        
        Built as part of B.Tech internship project 
        at SMVDU.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Upload an identity document image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        # Demo images
        st.markdown("---")
        st.markdown("**Or try a sample image:**")
        sample_col1, sample_col2 = st.columns(2)
        
        sample_dir = os.path.join(os.path.dirname(__file__), '..', '..', 
                                   'data', 'idnet', 'GRC')
        
        use_sample_auth = sample_col1.button("✅ Authentic Sample")
        use_sample_fraud = sample_col2.button("🚨 Fraud Sample")
        
        sample_image = None
        if use_sample_auth:
            auth_dir = os.path.join(sample_dir, 'positive')
            if os.path.exists(auth_dir):
                files = sorted(os.listdir(auth_dir))[:1]
                if files:
                    sample_image = Image.open(os.path.join(auth_dir, files[0])).convert('RGB')
                    st.info("Loaded authentic sample image")
        elif use_sample_fraud:
            fraud_dir = os.path.join(sample_dir, 'fraud5_inpaint_and_rewrite')
            if os.path.exists(fraud_dir):
                files = sorted(os.listdir(fraud_dir))[:1]
                if files:
                    sample_image = Image.open(os.path.join(fraud_dir, files[0])).convert('RGB')
                    st.info("Loaded fraudulent sample image")
    
    # Process image
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    elif sample_image is not None:
        image = sample_image
    
    if image is not None:
        with col1:
            st.image(image, caption="Uploaded Document", width='stretch')
        
        # Run prediction
        with st.spinner("🔍 Analyzing document..."):
            result = predict(image, model, device, transforms)
            ela_stats = compute_ela_stats(image)
        
        with col2:
            st.markdown("### 🎯 Analysis Result")
            
            # Result card
            if result['is_fraud']:
                st.markdown(f"""
                <div class="result-card result-fraud">
                    <h2>🚨 FRAUDULENT</h2>
                    <p>This document appears to be tampered</p>
                    <p style="font-size: 2rem; font-weight: bold; margin-top: 0.5rem;">
                        {result['confidence']*100:.1f}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-authentic">
                    <h2>✅ AUTHENTIC</h2>
                    <p>This document appears to be genuine</p>
                    <p style="font-size: 2rem; font-weight: bold; margin-top: 0.5rem;">
                        {result['confidence']*100:.1f}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence breakdown
            st.markdown("#### Confidence Breakdown")
            m1, m2 = st.columns(2)
            m1.metric("Authentic", f"{result['prob_authentic']*100:.1f}%")
            m2.metric("Fraudulent", f"{result['prob_fraudulent']*100:.1f}%")
            
            # Progress bars
            st.progress(result['prob_authentic'], text=f"Authentic: {result['prob_authentic']*100:.1f}%")
            st.progress(result['prob_fraudulent'], text=f"Fraudulent: {result['prob_fraudulent']*100:.1f}%")
            
            # ELA Statistics
            if show_stats:
                st.markdown("#### 📊 ELA Statistics")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Mean Error", f"{ela_stats['mean_error']:.2f}")
                s2.metric("Std Error", f"{ela_stats['std_error']:.2f}")
                s3.metric("Max Error", f"{ela_stats['max_error']:.0f}")
                s4.metric("Hot Pixels", f"{ela_stats['hot_pixels']:.2f}%")
        
        # ELA Visualization (full width below)
        if show_ela:
            st.markdown("---")
            st.markdown("### 🔬 Error Level Analysis (ELA)")
            st.markdown("""
            > **How to read ELA:** Brighter areas indicate higher compression error 
            > → potential tampering. Authentic documents have uniform ELA patterns, 
            > while tampered documents show bright spots where edits were made.
            """)
            
            ela_image = compute_ela_for_display(image, quality=ela_quality, 
                                                 scale=ela_scale)
            
            ela_col1, ela_col2 = st.columns(2)
            with ela_col1:
                st.image(image, caption="Original Document", 
                        width='stretch')
            with ela_col2:
                st.image(ela_image, caption="ELA Visualization (brighter = higher error)", 
                        width='stretch')
    else:
        with col2:
            st.markdown("### 👈 Upload a document to begin")
            st.markdown("""
            **How it works:**
            1. Upload a document image (passport, ID card, etc.)
            2. Our AI model analyzes the image
            3. Get an instant verdict: **Authentic** or **Fraudulent**
            4. View the ELA forensic analysis
            
            **What is ELA?**  
            Error Level Analysis is a forensic technique that reveals 
            image tampering by analyzing compression artifacts. Edited 
            regions show different error patterns than untouched areas.
            """)
    
    # Performance metrics
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    
    perf_cols = st.columns(5)
    metrics = [
        ("93.0%", "Test Accuracy"),
        ("0.95", "F1 Score"),
        ("0.989", "AUC Score"),
        ("97%", "Precision (Auth)"),
        ("99%", "Recall (Fraud)"),
    ]
    for col, (value, label) in zip(perf_cols, metrics):
        col.markdown(f"""
        <div class="metric-box">
            <h3>{value}</h3>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🛡️ DocGuard AI — Internship Project | SMVDU | 2026</p>
        <p>Built with EfficientNet-B0 + PyTorch + Streamlit | Dataset: IDNet (GRC)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
