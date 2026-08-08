# 🔍 AI ID Detection

**Detection of AI-Generated and Tampered Identity Documents Using Deep Learning and Image Forensics**

> College Internship Project | August 2026

---

## 📋 Overview

This project develops a deep learning system to detect AI-generated and tampered identity documents. It combines classical image forensics techniques (Error Level Analysis) with modern transfer learning approaches (EfficientNet-B0, ResNet50) to classify documents as authentic or fraudulent.

## 🎯 Key Features

- **Error Level Analysis (ELA)** — Classical forensics to detect JPEG compression inconsistencies
- **Transfer Learning** — Fine-tuned EfficientNet-B0 and ResNet50 for binary classification
- **Traditional ML Baseline** — SVM and Random Forest on ELA features for comparison
- **Web Application** — Streamlit demo for real-time document authentication
- **Comprehensive Evaluation** — Confusion matrices, ROC curves, precision/recall

## 📊 Dataset

- **Primary:** [IDNet: Identity Analysis Image Dataset](https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis) (597,900+ synthetic identity documents)
- **Secondary:** Synthetic Aadhaar card templates (self-generated for domain adaptation)

## 🏗️ Project Structure

```
ai_id_detection/
├── data/                    # Dataset files (not in git)
├── notebooks/               # Jupyter notebooks (step-by-step)
├── src/                     # Source code modules
│   ├── preprocessing/       # ELA, dataset, augmentation
│   ├── models/              # Classifier architectures
│   ├── evaluation/          # Metrics and visualization
│   ├── app/                 # Streamlit web app
│   └── utils/               # Helper functions
├── models/saved/            # Trained model checkpoints
├── reports/figures/         # Plots and visualizations
├── requirements.txt         # Python dependencies
└── README.md
```

## ⚙️ Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-id-detection.git
cd ai-id-detection

# Create conda environment
conda create -n aiid python=3.11
conda activate aiid

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run Streamlit web app
streamlit run src/app/app.py

# Run ELA analysis on an image
python src/preprocessing/ela.py path/to/image.jpg

# Train model from scratch
python src/models/train_full.py

# Evaluate saved model
python src/models/evaluate_model.py
```

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (ELA features) | 100%* | 1.00 | 1.00 | 1.00 |
| Random Forest (ELA) | 100%* | 1.00 | 1.00 | 1.00 |
| EfficientNet-B0 (frozen) | 86.9% | 0.91 | 0.86 | 0.88 |
| **EfficientNet-B0 (fine-tuned)** | **93.0%** | **0.94** | **0.93** | **0.95** |

> *\*Baseline ML achieves 100% due to format differences (PNG vs JPG) between authentic and fraudulent images. See report for detailed analysis.*
>
> **AUC Score: 0.989** | Fraud Recall: 99% | Authentic Precision: 97%

## 🛠️ Tech Stack

- **Python 3.11** | PyTorch 2.2.2 | OpenCV 4.10
- **Models:** EfficientNet-B0, ResNet50 (via timm)
- **Classical ML:** scikit-learn (SVM, Random Forest)
- **Web App:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## 📄 License

This project is for educational/research purposes only.

## 🙏 Acknowledgments

- IDNet Dataset: [Zenodo DOI: 10.5281/zenodo.13854938](https://doi.org/10.5281/zenodo.13854938)
- Reference repos: [DocTamper](https://github.com/qcf-568/DocTamper), [DocAuth](https://github.com/trinity652/DocAuth), [Document-Forgery-Detection](https://github.com/ShivamKabra/Document-Forgery-Detection)
