"""Unit tests for PlantDiseasePredictor inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agroscan.inference.predictor import PlantDiseasePredictor
from agroscan.models.classifier import PlantDiseaseClassifier


class TestPredictor:
    def test_predict_returns_valid_structure(self):
        model = PlantDiseaseClassifier(num_classes=38, pretrained=False)
        predictor = PlantDiseasePredictor(model=model, image_size=(128, 128))

        dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        diag = predictor.predict(dummy, top_k=3, generate_saliency=True)

        assert diag.top_prediction is not None
        assert len(diag.top_k) == 3
        assert diag.gradcam_heatmap.shape == (100, 100)
        assert diag.latency_ms > 0.0
        assert 0.0 <= diag.top_prediction.confidence <= 1.0
