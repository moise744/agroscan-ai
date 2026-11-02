# AgroScan AI — PlantVillage Dataset Guidelines

## Overview

AgroScan AI is designed for the **PlantVillage** benchmark, encompassing 54,303 leaf images across 38 crop disease classes.

### Expected Directory Layout
```
data/plantvillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___healthy/
├── Corn_(maize)___Common_rust_/
├── Tomato___Early_blight/
└── ...
```

### Limitations & Domain Shift
The PlantVillage dataset was primarily captured in controlled laboratory settings with uniform neutral backgrounds. Models trained strictly on this distribution should be tested cautiously on field conditions (in-situ wild background clutter, direct sunlight specularities, and complex soil textures).
