"""Unit tests for dataset loading, class parsing, and transforms."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agroscan.data.classes import PLANTVILLAGE_CLASSES, parse_class_name
from agroscan.data.dataset import PlantVillageDataset
from agroscan.data.transforms import get_agri_transforms


class TestClassParsing:
    def test_parse_healthy(self):
        crop, cond, is_h = parse_class_name("Tomato___healthy")
        assert crop == "Tomato"
        assert cond == "healthy"
        assert is_h is True

    def test_parse_disease(self):
        crop, cond, is_h = parse_class_name("Corn_(maize)___Common_rust_")
        assert "Corn" in crop
        assert "Common rust" in cond
        assert is_h is False

    def test_class_count(self):
        assert len(PLANTVILLAGE_CLASSES) == 38


class TestTransforms:
    def test_transform_shape(self):
        tfm = get_agri_transforms(image_size=(224, 224), is_training=False)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = tfm(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32


class TestDataset:
    def test_empty_dataset(self, tmp_path):
        ds = PlantVillageDataset(root=tmp_path)
        assert len(ds) == 0

    def test_mock_dataset(self, tmp_path):
        cls_dir = tmp_path / "Tomato___Early_blight"
        cls_dir.mkdir(parents=True)
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img.save(cls_dir / "leaf_01.jpg")

        ds = PlantVillageDataset(root=tmp_path)
        assert len(ds) == 1
        item = ds[0]
        assert "image" in item
        assert "label" in item
        assert item["class_name"] == "Tomato___Early_blight"
