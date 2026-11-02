# 🌿 AgroScan AI — Agricultural Disease Detection & Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/moise744/agroscan-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/moise744/agroscan-ai/actions)

> **Computer Vision system for automated plant disease classification with Grad-CAM saliency maps and interactive web-based diagnosis.**

---

## 🎯 Project Overview

AgroScan AI applies deep learning to agricultural computer vision — classifying plant diseases across 38 categories spanning 14 crop types. The system uses a ResNet-18 backbone fine-tuned on the PlantVillage dataset and produces **Grad-CAM visual explanations** that highlight which leaf regions influenced each prediction.

This project demonstrates:
- Transfer learning and fine-tuning with PyTorch
- Classification metrics: accuracy, F1, precision, recall, confusion matrix
- Interpretable AI through Gradient-weighted Class Activation Mapping
- Streamlit interactive web application
- Production-ready project structure with full test suite

---

## 🏗️ Architecture

```
Input Leaf Image (RGB)
        │
        ▼
  ┌─────────────────────────────┐
  │  Preprocessing Pipeline     │
  │  Resize → Normalize         │
  └────────────┬────────────────┘
               │
               ▼
  ┌─────────────────────────────┐
  │  ResNet-18 Feature Extractor│
  │  conv1 → layer1..4          │
  │  ↑ layer4 used for Grad-CAM │
  └────────────┬────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  ┌─────────┐    ┌──────────────┐
  │ Avg Pool│    │  Grad-CAM    │
  │ + Head  │    │  Heatmap     │
  └────┬────┘    └──────────────┘
       │
       ▼
  ┌─────────────────────────────┐
  │  38-Class Softmax Output    │
  │  + Top-K Predictions        │
  └─────────────────────────────┘
```

---

## 🌾 Supported Crops & Diseases

| Crop | Diseases Covered |
|------|-----------------|
| Tomato | 9 classes (Early blight, Late blight, Mosaic virus, Leaf mold, etc.) |
| Corn (Maize) | 4 classes (Common rust, Gray leaf spot, Northern blight, healthy) |
| Potato | 3 classes (Early blight, Late blight, healthy) |
| Apple | 4 classes (Apple scab, Black rot, Cedar rust, healthy) |
| Pepper, Bell | 2 classes (Bacterial spot, healthy) |
| + 9 more crops | 38 total disease/healthy categories |

---

## 📂 Project Structure

```
agroscan-ai/
├── src/agroscan/
│   ├── data/
│   │   ├── classes.py          # 38 PlantVillage class definitions
│   │   ├── dataset.py          # PlantVillageDataset loader
│   │   └── transforms.py       # Training/validation augmentation pipelines
│   ├── models/
│   │   └── classifier.py       # PlantDiseaseClassifier (ResNet-18)
│   ├── explainability/
│   │   └── gradcam.py          # Gradient-weighted Class Activation Mapping
│   ├── inference/
│   │   └── predictor.py        # PlantDiseasePredictor (+ DiagnosisResult)
│   └── visualization/
│       └── visualizer.py       # Leaf saliency overlay renderer
├── app/
│   └── streamlit_app.py        # Interactive diagnosis web app
├── scripts/
│   ├── train.py                # Training script
│   └── predict.py              # Single-image prediction script
├── configs/
│   └── classifier.yaml         # Training configuration
├── tests/
│   ├── test_dataset.py         # Dataset & transform tests
│   ├── test_model.py           # Classifier forward pass tests
│   ├── test_gradcam.py         # Grad-CAM output tests
│   └── test_inference.py       # End-to-end predictor tests
├── data/
│   └── README.md               # Dataset download instructions
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/moise744/agroscan-ai.git
cd agroscan-ai
pip install -e .
```

### 2. Download Dataset

Follow instructions in [data/README.md](data/README.md) to download the PlantVillage dataset from Kaggle.

Expected directory structure:
```
data/plantvillage/
├── Tomato___healthy/
├── Tomato___Early_blight/
├── Corn_(maize)___Common_rust_/
└── ... (38 class folders)
```

### 3. Train

```bash
python scripts/train.py \
    --data-dir data/plantvillage \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-4
```

### 4. Predict on a Leaf Image

```bash
python scripts/predict.py \
    --image path/to/leaf.jpg \
    --checkpoint outputs/models/best_model.pth \
    --top-k 3
```

### 5. Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Evaluation Metrics

The system reports the following metrics after training:

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Fraction of images where the top prediction is correct |
| **Top-5 Accuracy** | Fraction of images where the correct class appears in top 5 |
| **Macro F1-Score** | Mean F1 across all 38 classes (handles class imbalance) |
| **Per-class Precision / Recall** | Per-disease breakdown |
| **Confusion Matrix** | 38×38 class confusion visualization |
| **Inference Latency (ms)** | Time per image on CPU |

> **Note:** Severity estimation is not implemented. The PlantVillage dataset provides class labels only — not pixel-level severity ground truth. Confidence scores indicate class probability, not disease severity percentage.

---

## 🔍 Grad-CAM Visual Explanations

Every prediction includes a Grad-CAM saliency map overlaid on the original image. This shows which pixels activated the network's classification decision.

```python
from agroscan.inference.predictor import PlantDiseasePredictor
from agroscan.models.classifier import PlantDiseaseClassifier

model = PlantDiseaseClassifier(num_classes=38)
predictor = PlantDiseasePredictor(checkpoint_path="outputs/models/best_model.pth")

result = predictor.predict("leaf.jpg", top_k=3, generate_saliency=True)
print(result.top_prediction.class_name)     # e.g. "Tomato___Early_blight"
print(f"Confidence: {result.top_prediction.confidence:.1%}")

# Save saliency overlay
import cv2
cv2.imwrite("saliency.jpg", result.gradcam_overlay)
```

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v --tb=short
```

All tests run on CPU. No GPU required. No dataset download required.

---

## 📋 Dataset

**PlantVillage** — 54,305 images of healthy and diseased plant leaves across 38 classes.

- **License:** CC0 (Public Domain)
- **Source:** [Kaggle PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Paper:** Hughes, D.P. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.*

**Important dataset limitations:**
- Images are taken on plain backgrounds — real-field performance may differ
- No pixel-level disease segmentation masks provided
- Class distribution is slightly imbalanced

---

## 🛤️ Roadmap

- [x] ResNet-18 classifier with 38-class PlantVillage support
- [x] Grad-CAM explainability
- [x] Streamlit diagnostic web app
- [x] Full pytest test suite (CPU-only)
- [ ] ResNet-50 / EfficientNet-B0 comparison
- [ ] Field image domain adaptation
- [ ] ONNX export for edge deployment
- [ ] REST API with FastAPI

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 👤 Author

**Habimana Moise**
Computer Vision & Machine Learning Engineer
📧 habimanamoise77@gmail.com
🐙 [github.com/moise744](https://github.com/moise744)
