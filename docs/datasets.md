# AgroScan AI — Dataset Guide

## Primary Dataset: PlantVillage

**PlantVillage** is the standard benchmark dataset for plant disease classification.

| Property | Value |
|----------|-------|
| Source | Hughes & Salathé, Penn State University |
| License | CC0 (Public Domain) |
| Total images | ~87,848 |
| Classes | 38 (14 crops × disease conditions) |
| Image size | 256×256 px (varies) |
| Format | JPEG, directory-per-class |

### Disease Categories

The dataset covers:

| Crop | Conditions |
|------|-----------|
| Apple | Apple scab, Black rot, Cedar rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn | Gray leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca (Black measles), Leaf blight, Healthy |
| Orange | Citrus greening (HLB) |
| Peach | Bacterial spot, Healthy |
| Pepper bell | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early/Late blight, Leaf mold, Septoria, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |

### Download Instructions

**Option 1 — Kaggle (recommended):**
```bash
pip install kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/plantvillage
```

**Option 2 — Direct link:**
```
https://www.kaggle.com/datasets/emmarex/plantdisease
```

**Option 3 — Original GitHub:**
```
https://github.com/spMohanty/PlantVillage-Dataset
```

### Expected Directory Structure

```
data/
└── plantvillage/
    ├── Apple___Apple_scab/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    ├── Corn_(maize)___Cercospora_leaf_spot/
    └── ... (38 folders total)
```

### Verify the Download

```bash
python -c "
from agroscan.data.dataset import PlantVillageDataset
ds = PlantVillageDataset('data/plantvillage')
print(f'Classes: {ds.num_classes}')
print(f'Samples: {len(ds)}')
"
```

---

## Dataset Limitations

- **No severity labels.** Disease presence is labelled, but not severity (mild/moderate/severe).
- **Controlled conditions.** Images were taken under controlled lighting on a plain background — real field performance may be lower.
- **Class imbalance.** Some classes have 5,000+ images; others have ~150 (e.g., Potato healthy: 152).
- **No segmentation masks.** Only image-level classification labels are available.

---

## Alternative Datasets

### iPlant

- Field-collected leaf images with more variation.
- Download: https://github.com/fcakyon/plant-disease-detection

### IP102

- Insect pest classification (102 classes).
- Download: https://github.com/xpwu95/IP102

Do not commit dataset images to git. They are excluded by `.gitignore`.
