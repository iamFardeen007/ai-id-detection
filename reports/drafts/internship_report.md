# Detection of AI-Generated and Tampered Identity Documents Using Deep Learning and Image Forensics

**Internship Project Report**  
Submitted in partial fulfillment of the requirements for B.Tech in Computer Science & Engineering

---

**Submitted by:** Fardeen Ahmad | Roll No.: 24BCM014  
**Department:** Computer Science & Engineering  
**Institution:** Shri Mata Vaishno Devi University (SMVDU), Katra, J&K  
**Internship Duration:** August 2026  
**Supervisor/Guide:** [Your Guide's Name]

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation

In the digital age, identity documents such as passports, Aadhaar cards, driving licences, and national identity cards form the backbone of authentication systems across government services, banking, border control, and healthcare. The proliferation of sophisticated image editing tools — including Adobe Photoshop, deep learning-based inpainting, and generative adversarial networks (GANs) — has made document forgery increasingly accessible.

Traditional document verification methods, such as UV light inspection, hologram verification, and physical watermark analysis, are inherently manual and cannot be scaled to meet the demands of modern digital identity verification systems. Furthermore, as AI-generated identity documents continue to improve in visual realism, even trained human examiners struggle to identify tampered documents.

This project addresses this critical gap by developing **DocGuard AI**, an automated system for detecting AI-generated and tampered identity documents. The system employs a hybrid approach combining classical digital image forensics techniques with state-of-the-art deep learning to deliver high-accuracy fraud detection.

## 1.2 Problem Statement

Given an identity document image (passport, national ID, etc.), the system must classify it as one of two categories:

- **Authentic (Label 0):** A genuine, unmodified identity document image.
- **Fraudulent (Label 1):** A document image that has been digitally tampered with using techniques such as text inpainting/rewriting, face image replacement (crop-and-replace), or AI-generated synthesis.

The system must achieve at least **85% classification accuracy** on unseen test data while providing explainable forensic visualisations.

## 1.3 Objectives

1. Implement an **Error Level Analysis (ELA)** engine for forensic feature extraction from JPEG-compressed document images.
2. Develop and evaluate **traditional machine learning baselines** (SVM, Random Forest) using ELA and colour histogram features.
3. Fine-tune a **pre-trained EfficientNet-B0** using two-phase transfer learning for binary document authentication.
4. Build a **real-time interactive web application** (DocGuard AI) using Streamlit.
5. Achieve a test accuracy of **≥85%** with comprehensive evaluation.

## 1.4 Report Organisation

This report is organised into seven chapters covering Literature Review (Ch. 2), System Design (Ch. 3), Implementation (Ch. 4), Results (Ch. 5), Web Application (Ch. 6), and Conclusion (Ch. 7).

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Document Forgery and Digital Fraud

The problem of document forgery has existed since the introduction of standardised identity credentials. Researchers have categorised digital document attacks into two major classes (Guo et al., 2023):

1. **Intra-document forgeries:** Modification of content within the same document template (e.g., changing the name, altering the date of birth).
2. **Document synthesis:** Generating entirely new fraudulent documents using generative models.

The IDNet dataset (Sultana et al., 2024) used in this project represents both attack types: text-level tampering via inpainting/rewriting, and region-level tampering via crop-and-replace of the photograph.

## 2.2 Classical Image Forensics

**Error Level Analysis (ELA)** is a well-established digital forensics technique introduced by Krawetz (2007). It exploits the lossy nature of JPEG compression: when a region of an image is edited and re-saved separately, it retains a higher compression error than surrounding unmodified regions, making it visually distinguishable.

Limitations of ELA:
- Sensitive to the format of the source image (PNG vs. JPG creates false positives — a key finding in this project).
- Cannot localise tampering when the entire image has been re-saved multiple times.

## 2.3 Deep Learning for Document Analysis

CNNs have demonstrated remarkable success in document image classification. Mansour et al. (2022) achieved 94.3% accuracy distinguishing authentic from forged documents using ResNet-50 fine-tuned on a private dataset. Their work demonstrated that ImageNet-pretrained backbone features generalise well to document textures and structural artifacts.

## 2.4 Transfer Learning

Transfer learning (Pan & Yang, 2010) involves adapting a model pre-trained on a large dataset to a target domain with limited labelled data. **EfficientNet** (Tan & Le, 2019) achieved state-of-the-art performance on ImageNet using compound scaling of network depth, width, and resolution. EfficientNet-B0 achieves 77.1% top-1 accuracy on ImageNet with only 5.3M parameters, making it an ideal backbone for fine-tuning on our dataset.

## 2.5 Gap Analysis

Existing work either relies on proprietary datasets or uses full-image classification without forensic analysis. This project bridges this gap by combining **normalised ELA forensics** with **two-phase EfficientNet-B0 fine-tuning** on the publicly available IDNet dataset, providing both high accuracy and forensic explainability.

---

# CHAPTER 3: SYSTEM DESIGN AND METHODOLOGY

## 3.1 System Architecture Overview

```
Stage 1: Input and Preprocessing
   Image Loading → Normalisation → ELA Computation → Augmentation

Stage 2: Dual-Path Classification
   Path A: ELA Features → SVM / Random Forest  [Baseline]
   Path B: Image Tensor → EfficientNet-B0       [Primary]

Stage 3: Output and Explanation
   Prediction + Confidence Score + ELA Visualisation
```

## 3.2 Dataset Description

**IDNet Dataset — GRC (Greece) Subset:**

| Category | Count | Format |
|----------|-------|--------|
| Authentic (genuine passports) | 5,979 | PNG |
| Fraud Type 5 — Inpaint & Rewrite | 5,979 | JPG |
| Fraud Type 6 — Crop & Replace | 5,978 | JPG |
| **Total** | **17,936** | |

**Stratified Train / Validation / Test Splits (70/15/15):**

| Split | Authentic | Fraudulent | Total |
|-------|-----------|------------|-------|
| Train | 4,185 | 8,369 | 12,554 |
| Validation | 896 | 1,793 | 2,689 |
| Test | 898 | 1,795 | 2,693 |

**Class Imbalance:** Dataset has 1:2 authentic-to-fraudulent ratio, addressed via inverse frequency class weights (1.5 for authentic, 0.75 for fraudulent) in CrossEntropyLoss.

## 3.3 Error Level Analysis (ELA)

**Algorithm:**

1. **Normalisation:** Source image is re-saved as JPEG at quality 90% (equalises PNG vs JPG format artifacts).
2. **Recompression:** Normalised JPEG is re-saved again at 90%.
3. **Difference Map:** Pixel-level absolute difference:  
   `E(x,y) = |I_normalised(x,y) − I_recompressed(x,y)|`
4. **Amplification:** Values scaled to [0, 255] range for visualisation.
5. **Feature Extraction:** 15 statistical features extracted: Mean, Standard Deviation, Maximum per channel (R,G,B) + Hot Pixel % (E>50) per channel.

**Key Research Finding — Format Leakage:**  
Raw ELA achieved 100% accuracy because authentic images were PNG (ELA error ≈ 0) and fraudulent were JPG (non-zero ELA errors). After normalisation, fraud images still showed **2× higher mean ELA error**, confirming genuine forensic signal beyond format differences.

## 3.4 Deep Learning Architecture

**DocumentClassifier (EfficientNet-B0):**

```
Input Image (224×224×3)
    ↓
EfficientNet-B0 Backbone [pretrained on ImageNet]
    ↓ Global Average Pooling
Feature Vector (1,280 dimensions)
    ↓
Custom Classification Head:
    Dropout(0.30) → Linear(1280→256) → ReLU → Dropout(0.15) → Linear(256→2)
    ↓
Softmax → [P(Authentic), P(Fraudulent)]
```

Total parameters: 4,335,998 | Trainable in Phase 1: 328,450

**Two-Phase Training Strategy:**

| Phase | Backbone | Epochs | LR | Optimiser | Best Val Acc |
|-------|----------|--------|----|-----------|-------------|
| Phase 1 — Head Only | Frozen | 10 (early stop @7) | 0.001 | AdamW | **87.76%** |
| Phase 2 — Fine-tuning | Top-3 blocks unfrozen | 15 (early stop @7) | 0.0001 | AdamW | **93.19%** |

**Data Augmentation (training only):** Random Resized Crop (224×224), Horizontal Flip (p=0.5), Random Rotation (±10°), Colour Jitter, Random Grayscale (p=0.05), ImageNet Normalisation.

---

# CHAPTER 4: IMPLEMENTATION

## 4.1 Development Environment

| Component | Details |
|-----------|---------|
| OS | macOS (Apple Silicon M-series) |
| Python | 3.11 (Conda env: `aiid`) |
| Deep Learning | PyTorch 2.2.2 + timm 1.0.28 |
| Hardware | Apple MPS (Metal Performance Shaders) |
| Classical ML | scikit-learn 1.9.0 |
| Image Processing | OpenCV 4.10.0, Pillow 12.3.0 |
| Web App | Streamlit 1.61.1 |
| Version Control | Git / GitHub (iamFardeen007/ai-id-detection) |

## 4.2 Project Structure

```
ai_id_detection/
├── data/idnet/GRC/             # Raw IDNet dataset
│   ├── positive/               # 5,979 authentic
│   ├── fraud5_inpaint_and_rewrite/  # 5,979 fraud
│   └── fraud6_crop_and_replace/     # 5,978 fraud
├── data/processed/             # Stratified splits (symlinks)
├── src/
│   ├── preprocessing/
│   │   ├── ela.py              # ELA engine
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── augmentation.py     # Transform pipelines
│   │   └── prepare_data.py     # Split script
│   ├── models/
│   │   ├── classifier.py       # DocumentClassifier
│   │   ├── baseline.py         # SVM + Random Forest
│   │   ├── train_full.py       # 2-phase training
│   │   └── evaluate_model.py   # Test evaluation
│   ├── evaluation/
│   │   └── metrics.py          # Confusion matrix, ROC, plots
│   ├── app/
│   │   └── app.py              # Streamlit web app
│   └── utils/helpers.py        # Device, seeding, timing
├── models/saved/               # Checkpoints (.pth, .pkl)
├── reports/figures/            # Generated plots
├── requirements.txt
└── README.md
```

## 4.3 Key Implementation Challenges & Solutions

| Challenge | Solution Applied |
|-----------|----------------|
| ELA format leakage (PNG vs JPG → 100% accuracy) | Normalise all images to JPEG before ELA |
| Class imbalance (1:2 ratio) | Inverse frequency class weights in CrossEntropyLoss |
| `plt.show()` blocking background training | `matplotlib.use('Agg')` non-interactive backend |
| DataLoader multiprocessing crash on macOS stdin scripts | Write training as Python modules; `num_workers=0` |
| Large notebooks (13MB+) failing GitHub push | Remove tutorial notebooks from git tracking |

---

# CHAPTER 5: RESULTS AND ANALYSIS

## 5.1 Baseline Model Performance

| Model | Accuracy (normalised ELA) | Note |
|-------|--------------------------|------|
| SVM (RBF Kernel) | ~65% | 27 hand-crafted features |
| Random Forest | ~67% | 200 trees, 27 features |

**Conclusion:** Traditional ML with hand-crafted ELA features alone is insufficient for robust document authentication, motivating deep learning.

## 5.2 Deep Learning — Final Test Set Results

Evaluated on **2,693 unseen test images**:

| Metric | Value | Target |
|--------|-------|--------|
| **Test Accuracy** | **92.98%** | 85–90% → ✅ EXCEEDED |
| **F1-Score (Weighted)** | **0.9494** | — |
| **ROC AUC** | **0.9887** | — |
| **Test Loss** | 0.2900 | — |

### Per-Class Metrics:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.97 | 0.81 | 0.89 | 898 |
| **Fraudulent** | 0.91 | **0.99** | **0.95** | 1,795 |
| Macro Average | 0.94 | 0.90 | 0.92 | 2,693 |

### Confusion Matrix:

| | Predicted: Authentic | Predicted: Fraudulent |
|---|---|---|
| **Actual Authentic** | ✅ 731 (True Negative) | ❌ 167 (False Positive) |
| **Actual Fraudulent** | ❌ 22 (False Negative) | ✅ 1,773 (True Positive) |

**Key Insight:** Only 22 fraud documents were missed (1.2% miss rate) — critical for a security application. The 167 false alarms (18.6% of authentic) represent a known trade-off when maximising fraud recall.

## 5.3 Model Comparison

| Model | Test Accuracy | AUC | F1 |
|-------|--------------|-----|----|
| SVM (ELA features) | ~65% | — | — |
| Random Forest | ~67% | — | — |
| EfficientNet-B0 Phase 1 | 86.9% | — | — |
| **EfficientNet-B0 Phase 2** | **92.98%** | **0.989** | **0.95** |

## 5.4 Training Curve Analysis

**Phase 1 (epochs 1–7):** Val accuracy jumped from 76.7% → 87.8% in just 2 epochs, demonstrating how quickly ImageNet features transfer to document authentication when fine-tuning the head only.

**Phase 2 (fine-tuning):** Val accuracy rose further to 93.2% by epoch 4 as the backbone adapted its lower-level feature detectors to document-specific forensic signals.

**Early stopping:** Both phases triggered early stopping (patience=5), preventing overfitting on the training data.

## 5.5 ROC Curve Analysis

The ROC curve achieves AUC = 0.9887, indicating near-perfect discrimination across all classification thresholds. The curve rises steeply at very low false positive rates, confirming the model's high sensitivity to fraudulent documents.

---

# CHAPTER 6: WEB APPLICATION — DocGuard AI

## 6.1 Overview

DocGuard AI is a Streamlit-based interactive web application providing real-time identity document authentication. It exposes the trained EfficientNet-B0 model through a user-friendly dark-themed interface.

- **Launch:** `streamlit run src/app/app.py`
- **URL:** http://localhost:8501
- **GitHub:** https://github.com/iamFardeen007/ai-id-detection

## 6.2 Core Features

1. **Document Upload:** Drag-and-drop image upload (JPG/PNG/BMP).
2. **Sample Testing:** One-click authentic and fraud sample loading for demonstration.
3. **Instant Verdict:** Colour-coded result banner — Green (✅ AUTHENTIC) or Red (🚨 FRAUDULENT) — with confidence percentage.
4. **Confidence Breakdown:** Side-by-side probability display for both classes with visual progress bars.
5. **ELA Forensic View:** Side-by-side original vs. ELA heatmap with adjustable amplification and quality sliders.
6. **ELA Statistics:** Mean Error, Std Error, Max Error, Hot Pixel % displayed as live metrics.
7. **Model Info Panel:** Model specifications and performance metrics in the sidebar.

## 6.3 Interface Design

- Dark theme with linear-gradient navy header.
- Colour-coded result cards: `#0d9488` (authentic) / `#dc2626` (fraudulent).
- Responsive two-column layout for image/result display.
- `@st.cache_resource` model caching for sub-second inference after initial load.

---

# CHAPTER 7: CONCLUSION AND FUTURE WORK

## 7.1 Conclusion

This project successfully developed DocGuard AI, achieving **92.98% test accuracy** and **0.9887 AUC** — substantially exceeding the 85–90% target. Three key contributions:

1. **Format-normalised ELA pipeline** — Discovered and resolved PNG vs. JPG format leakage in ELA-based document forensics. After normalisation, fraud documents still show 2× higher ELA error, confirming genuine forensic discriminability.

2. **Two-phase EfficientNet-B0 fine-tuning** — Phase 1 (frozen backbone, 87.8% val accuracy) rapidly adapts ImageNet features. Phase 2 (partial unfreeze, 93.2% val accuracy) further adapts backbone to document-specific forensic signals.

3. **DocGuard AI web application** — Full-stack ML deployment with real-time document verification, ELA forensic visualisation, and confidence reporting — ready for classroom demonstration.

The project was completed in **4 days of active development** against an 18-day planned schedule.

## 7.2 Limitations

1. **Dataset scope:** Tested only on Greek passport images (IDNet GRC). Generalisation to Aadhaar cards, PAN cards, or other ID types is untested.
2. **False positive rate:** 18.6% of authentic documents incorrectly flagged as fraudulent. Threshold tuning is needed for production deployment.
3. **Attack coverage:** Not evaluated on fully GAN-synthesised documents or physical photograph substitution.
4. **No text-level OCR verification:** The system treats documents as pixel grids; OCR-based field consistency checking is out of scope.

## 7.3 Future Work

1. **Multi-document generalisation:** Train on Aadhaar, PAN, driving licence datasets.
2. **Threshold optimisation:** Tune classification threshold using PR curves to improve the precision-recall trade-off.
3. **Grad-CAM explainability:** Add gradient-weighted class activation maps to highlight which document regions drove the fraud prediction.
4. **GAN detection:** Add frequency-domain (DCT/FFT) features to detect fully AI-synthesised documents.
5. **Mobile deployment:** Export to ONNX/TFLite for integration into mobile scanning applications.

---

# REFERENCES

1. Sultana, N., et al. (2024). *IDNet: A Novel Dataset for Identity Document Analysis*. Zenodo. DOI: 10.5281/zenodo.13854938.
2. Krawetz, N. (2007). *A Picture's Worth: Digital Image Analysis and Forensics*. Black Hat Briefings.
3. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019.
4. Pan, S.J., & Yang, Q. (2010). *A Survey on Transfer Learning*. IEEE TKDE, 22(10), 1345–1359.
5. Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition*. ICLR 2021.
6. Fridrich, J., et al. (2003). *Detection of Copy-Move Forgery in Digital Images*. DFRWS 2003.
7. Mansour, R.F., et al. (2022). *Deep Learning based Document Forgery Detection*. Computers and Electrical Engineering.
8. Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
9. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
10. PyTorch Team (2023). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. https://pytorch.org.

---

# APPENDIX A: Setup Instructions

```bash
# Clone and set up
git clone https://github.com/iamFardeen007/ai-id-detection.git
cd ai-id-detection
conda create -n aiid python=3.11
conda activate aiid
pip install -r requirements.txt

# Launch web app
streamlit run src/app/app.py

# Re-run evaluation
python src/models/evaluate_model.py
```

# APPENDIX B: Hardware Used

| Component | Details |
|-----------|---------|
| Processor | Apple M-series (MPS acceleration) |
| RAM | 16 GB |
| Storage | 30 GB (for dataset) |
| Training Time | ~4 hours (full 2-phase training on MPS) |
| Inference | < 1 second per image (CPU) |

# APPENDIX C: Reproducibility Note

All experiments use `random_seed = 42` across Python `random`, NumPy, and PyTorch. Reproduce full training with:

```bash
conda activate aiid && python src/models/train_full.py
```
