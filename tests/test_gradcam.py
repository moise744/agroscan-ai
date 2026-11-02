"""Unit tests for Grad-CAM explainability module."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agroscan.explainability.gradcam import GradCAM
from agroscan.models.classifier import PlantDiseaseClassifier


class TestGradCAM:
    def test_gradcam_generation(self):
        model = PlantDiseaseClassifier(num_classes=10, pretrained=False)
        gradcam = GradCAM(model, model.layer4)

        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        heatmap = gradcam.generate(x, target_class=0)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all()

        gradcam.remove_hooks()
