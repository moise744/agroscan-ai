"""Unit tests for PlantDiseaseClassifier model."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agroscan.models.classifier import PlantDiseaseClassifier


class TestClassifier:
    def test_instantiation_and_forward(self):
        model = PlantDiseaseClassifier(num_classes=38, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        assert "logits" in out
        assert "probs" in out
        assert "features" in out

        assert out["logits"].shape == (2, 38)
        assert out["probs"].shape == (2, 38)
        assert torch.allclose(out["probs"].sum(dim=1), torch.ones(2), atol=1e-5)

    def test_cpu_device(self):
        model = PlantDiseaseClassifier(num_classes=38, pretrained=False).to(torch.device("cpu"))
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out["logits"].device.type == "cpu"
