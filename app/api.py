"""
AgroScan AI — FastAPI REST Endpoint.

Production-ready plant disease diagnosis API with Grad-CAM saliency.

Endpoints:
    POST /diagnose         — Diagnose a leaf image
    POST /diagnose/batch   — Batch diagnose multiple images
    GET  /classes          — List all 38 supported disease classes
    GET  /health           — Health check

Launch:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import numpy as np
from PIL import Image

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AgroScan AI",
        description="Agricultural Plant Disease Diagnosis API with Grad-CAM Explainability",
        version="1.0.0",
        contact={"name": "Habimana Moise", "email": "habimanamoise77@gmail.com"},
    )

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    class TopPrediction(BaseModel):
        class_name: str
        confidence: float
        is_healthy: bool

    class DiagnosisResponse(BaseModel):
        top_prediction: TopPrediction
        top_k: List[TopPrediction]
        latency_ms: float
        has_saliency_map: bool
        saliency_map_b64: Optional[str] = None  # base64 PNG if requested

    class HealthResponse(BaseModel):
        status: str
        num_classes: int
        model: str

    _predictor = None

    def get_predictor():
        global _predictor
        if _predictor is None:
            from agroscan.inference.predictor import PlantDiseasePredictor
            from agroscan.models.classifier import PlantDiseaseClassifier
            model = PlantDiseaseClassifier(num_classes=38, pretrained=False)
            _predictor = PlantDiseasePredictor(model=model, image_size=(224, 224))
        return _predictor

    def load_image(upload: UploadFile) -> np.ndarray:
        try:
            content = upload.file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            return np.array(img)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        return HealthResponse(status="ok", num_classes=38, model="PlantDiseaseClassifier-ResNet18")

    @app.get("/classes", tags=["Info"])
    async def list_classes():
        """Return all 38 supported plant disease classes."""
        from agroscan.data.classes import PLANTVILLAGE_CLASSES, parse_class_name
        classes = []
        for i, name in enumerate(PLANTVILLAGE_CLASSES):
            crop, condition, is_healthy = parse_class_name(name)
            classes.append({"id": i, "name": name, "crop": crop,
                             "condition": condition, "is_healthy": is_healthy})
        return {"classes": classes, "total": len(classes)}

    @app.post("/diagnose", response_model=DiagnosisResponse, tags=["Diagnosis"])
    async def diagnose(
        file: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
        top_k: int = Query(5, ge=1, le=38, description="Number of top predictions to return"),
        include_saliency: bool = Query(False, description="Include Grad-CAM saliency map as base64"),
    ):
        """
        Diagnose a plant leaf image and return the top disease predictions
        with optional Grad-CAM visual explanation.
        """
        predictor = get_predictor()
        image = load_image(file)
        result = predictor.predict(image, top_k=top_k, generate_saliency=include_saliency)

        saliency_b64 = None
        if include_saliency and result.gradcam_overlay is not None:
            import cv2
            _, buf = cv2.imencode(".png", result.gradcam_overlay)
            saliency_b64 = base64.b64encode(buf.tobytes()).decode()

        top = result.top_prediction
        top_list = result.top_k

        return DiagnosisResponse(
            top_prediction=TopPrediction(
                class_name=top.class_name,
                confidence=round(top.confidence, 4),
                is_healthy=top.is_healthy,
            ),
            top_k=[
                TopPrediction(class_name=p.class_name,
                              confidence=round(p.confidence, 4),
                              is_healthy=p.is_healthy)
                for p in top_list
            ],
            latency_ms=round(result.latency_ms, 2),
            has_saliency_map=saliency_b64 is not None,
            saliency_map_b64=saliency_b64,
        )

    @app.post("/diagnose/batch", tags=["Diagnosis"])
    async def diagnose_batch(
        files: List[UploadFile] = File(...),
        top_k: int = Query(3, ge=1, le=38),
    ):
        """Batch diagnose multiple leaf images."""
        predictor = get_predictor()
        results = []
        for f in files:
            image = load_image(f)
            result = predictor.predict(image, top_k=top_k, generate_saliency=False)
            results.append({
                "filename": f.filename,
                "top_prediction": result.top_prediction.class_name,
                "confidence": round(result.top_prediction.confidence, 4),
                "is_healthy": result.top_prediction.is_healthy,
                "latency_ms": round(result.latency_ms, 2),
            })
        return {"predictions": results, "total": len(results)}

    @app.get("/", tags=["System"])
    async def root():
        return {"name": "AgroScan AI", "docs": "/docs", "health": "/health"}

else:
    class app:
        pass
