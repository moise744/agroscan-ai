# AgroScan AI — API Reference

Base URL: `http://localhost:8003`

Interactive docs: `http://localhost:8003/docs` (Swagger UI)

---

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "agroscan-ai",
  "fastapi": "True",
  "opencv": "True",
  "torch": "True",
  "agroscan_modules": "True"
}
```

---

### `POST /diagnose`

Classify a leaf image and identify the plant disease.

**Request:** `multipart/form-data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | — | Leaf image (JPEG/PNG) |
| `top_k` | int | `3` | Number of top predictions to return |
| `return_gradcam` | bool | `false` | Include Grad-CAM heatmap in response |

**Response:**
```json
{
  "filename": "leaf.jpg",
  "predicted_class": "Tomato___Early_blight",
  "confidence": 0.912,
  "top_predictions": [
    {"class": "Tomato___Early_blight", "confidence": 0.912},
    {"class": "Tomato___Target_Spot", "confidence": 0.051},
    {"class": "Tomato___Septoria_leaf_spot", "confidence": 0.022}
  ],
  "severity_estimate": null,
  "severity_note": "Severity estimation requires separate ground-truth labels not present in PlantVillage.",
  "inference_time_ms": 67.4
}
```

> [!NOTE]
> `severity_estimate` is always `null`. PlantVillage labels identify disease type but not severity. Do not interpret absence of severity as "mild."

---

### `GET /classes`

List all 38 disease classes supported by the model.

**Response:**
```json
{
  "num_classes": 38,
  "classes": [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "..."
  ]
}
```

---

### `POST /diagnose/batch`

Diagnose multiple leaf images in a single request.

**Request:** `multipart/form-data`

| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | file[] | Multiple leaf images |
| `top_k` | int | Top predictions per image (default: `3`) |

**Response:**
```json
{
  "total": 3,
  "results": [
    {"filename": "leaf1.jpg", "predicted_class": "Tomato___Late_blight", "confidence": 0.878},
    {"filename": "leaf2.jpg", "predicted_class": "Apple___healthy", "confidence": 0.963},
    {"filename": "leaf3.jpg", "predicted_class": "Corn___Common_rust", "confidence": 0.831}
  ]
}
```

---

## Error Responses

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Invalid or unreadable image |
| `503` | AgroScan modules or model not available |

---

## Python Client Example

```python
import requests

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8003/diagnose",
        files={"file": f},
        params={"top_k": 5},
    )

data = response.json()
print(f"Disease: {data['predicted_class']}")
print(f"Confidence: {data['confidence']:.1%}")
for pred in data["top_predictions"]:
    print(f"  {pred['class']}: {pred['confidence']:.3f}")
```
