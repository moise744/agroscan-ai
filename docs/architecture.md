# AgroScan AI — System Architecture

## Overview

AgroScan AI is a plant disease classification and visual explanation system. It combines a fine-tuned ResNet-18 classifier with Gradient-weighted Class Activation Mapping (Grad-CAM) to produce both a disease prediction and a saliency map that highlights which leaf regions drove the decision.

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Input Layer                                  │
│           Leaf image (JPEG/PNG, any resolution)                       │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Preprocessing    │
                    │  Resize 224×224    │
                    │  Normalize         │
                    │  (ImageNet stats)  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   ResNet-18        │
                    │   Backbone         │
                    │   (fine-tuned or   │
                    │   feature extract) │
                    └─────────┬──────────┘
                              │
               ┌──────────────┼──────────────┐
               │                             │
    ┌──────────▼──────────┐     ┌────────────▼──────────┐
    │  Classification     │     │    Grad-CAM Module     │
    │  Head               │     │    Hooks on layer4     │
    │  Linear(512→38)     │     │    Gradient × Activation│
    │  → softmax          │     │    → saliency heatmap  │
    └──────────┬──────────┘     └────────────┬──────────┘
               │                             │
       ┌───────▼──────┐             ┌────────▼──────────┐
       │ disease name │             │  heatmap overlay  │
       │ confidence   │             │  on leaf image    │
       └───────┬──────┘             └────────┬──────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
     ┌─────────▼────┐  ┌──────▼──────┐ ┌────▼──────────┐
     │ Evaluation   │  │  REST API   │ │  Streamlit    │
     │ confusion    │  │  FastAPI    │ │  Dashboard    │
     │ matrix, etc. │  │  /diagnose  │ │  disease UI   │
     └──────────────┘  └─────────────┘ └───────────────┘
```

---

## Component Descriptions

| Component | Location | Responsibility |
|-----------|----------|---------------|
| `PlantVillageDataset` | `src/agroscan/data/dataset.py` | Loads PlantVillage directory structure; maps folder names to integer class IDs; supports train/val split; returns `{image, label, class_name}` |
| `get_train_transforms` | `src/agroscan/data/transforms.py` | Random resized crop, horizontal flip, colour jitter, normalization for training |
| `validate_dataset` | `src/agroscan/data/validation.py` | Verifies expected directory layout; counts samples per class; flags empty or unreadable images |
| `DiseaseClassifier` | `src/agroscan/models/classifier.py` | ResNet-18 with replaceable final FC layer sized to `num_classes`; supports `finetune` (all layers) and `feature_extract` (head only) modes |
| `GradCAM` | `src/agroscan/models/gradcam.py` | Registers forward and backward hooks on the target layer; computes `ReLU(Σ α_k · A_k)` where `α_k` = global-average-pooled gradients; returns upsampled heatmap |
| `DiseaseClassifierTrainer` | `src/agroscan/training/trainer.py` | Training loop with cross-entropy loss, LR scheduling, early stopping, and checkpoint saving; logs per-epoch accuracy and loss |
| `DiseaseEvaluator` | `src/agroscan/evaluation/evaluator.py` | Computes per-class precision, recall, F1; macro/weighted averages; confusion matrix; top-k accuracy |
| `DiseasePredictor` | `src/agroscan/inference/predictor.py` | Loads checkpoint; runs inference; returns predicted class, confidence, top-3 predictions, and Grad-CAM heatmap |
| `api.py` | `app/api.py` | FastAPI endpoints: `POST /diagnose` (image file → disease + confidence + heatmap), `GET /classes` (list all 38 classes), `GET /health` |
| `streamlit_app.py` | `app/streamlit_app.py` | Upload leaf image; display predicted disease; show Grad-CAM overlay; list top-3 predictions with confidence bars |

---

## Data Flow

```
PlantVillage dataset
     │
     ▼
PlantVillageDataset (loads image, returns {image, label, class_name})
     │
     ▼
DiseaseClassifierTrainer
  - cross-entropy loss
  - Adam optimizer
  - StepLR scheduler
  - early stopping (patience=5)
     │
     ▼
checkpoint.pth (saved best model by val accuracy)
     │
     ▼
DiseasePredictor.predict(image)
     │
     ├──► softmax(logits) → {class_name, confidence, top3}
     │
     └──► GradCAM(model, target_layer="layer4")
              → saliency heatmap [H×W] normalized [0,1]
                    │
                    ▼
           jet colormap overlay on original image
                    │
                    ▼
           FastAPI JSON response / Streamlit display
```

---

## Grad-CAM Computation

```
Forward pass:
  x → ResNet-18 layers → A_k (feature maps at layer4)
                       → logits → target class score y_c

Backward pass:
  ∂y_c/∂A_k  → gradients at layer4

Importance weights:
  α_k = (1/Z) Σ_{i,j} ∂y_c/∂A^k_{ij}   (global average pooling over spatial dims)

Class activation map:
  L_Grad-CAM = ReLU(Σ_k α_k · A_k)

Upsample to input resolution (bilinear) → normalize to [0,1]
```

---

## Key Design Decisions

### 1. ResNet-18 as backbone
ResNet-18 is a practical choice: it fits in CPU memory, trains quickly, and achieves strong accuracy on PlantVillage (>95% reported in literature). Larger backbones (ResNet-50, EfficientNet) can be substituted by changing the `backbone` parameter.

### 2. Grad-CAM for explainability
In agricultural and food-safety applications, predictions must be explainable. Grad-CAM highlights which leaf regions contributed most to the classification, allowing agronomists to validate the model's reasoning without deep ML expertise.

### 3. 38-class classification
PlantVillage covers 14 crop species × multiple disease states per species = 38 classes. The head is parameterised by `num_classes` so the same architecture works for subsets or future extended datasets.

### 4. Honest severity estimation
Severity is **not** estimated by default. PlantVillage labels identify disease type but do not quantify severity. Any severity output would require a separately annotated dataset. This constraint is documented in the README and respected in the API response schema.

---

## Performance Characteristics

| Model | Parameters | Inference time (CPU, 224×224) |
|-------|-----------|-------------------------------|
| ResNet-18 (38-class head) | ~11.2 M | ~30–80 ms |
| Grad-CAM overhead | — | ~5–15 ms |
| Full pipeline (classify + explain) | — | ~40–100 ms |

---

## Dataset

**PlantVillage** — 87,848 leaf images across 38 plant disease classes.
- 14 crop species, 26 disease conditions + healthy classes.
- License: CC0 (public domain).
- Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
- Paper: Hughes & Salathé (2015), arXiv:1511.08060.

See `data/README.md` for download and directory setup instructions.
