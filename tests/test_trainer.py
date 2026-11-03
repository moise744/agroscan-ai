"""Unit tests for DiseaseClassifierTrainer training module."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agroscan.models.classifier import PlantDiseaseClassifier
from agroscan.training.trainer import DiseaseClassifierTrainer, TrainingHistory


def _make_loader(n=16, num_classes=5, img_size=32):
    """Create a minimal DataLoader with random images and labels."""
    images = torch.randn(n, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (n,))
    # Mimic the dataset dict output
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls):
            self.imgs = imgs
            self.lbls = lbls
        def __len__(self): return len(self.imgs)
        def __getitem__(self, i): return {"image": self.imgs[i], "label": self.lbls[i]}
    ds = DictDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


class TestDiseaseClassifierTrainer:
    def _make_trainer(self, num_classes=5):
        model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return DiseaseClassifierTrainer(
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            output_dir="outputs/test_checkpoints",
        )

    def test_instantiation(self):
        trainer = self._make_trainer()
        assert trainer.device.type == "cpu"
        assert isinstance(trainer.history, TrainingHistory)

    def test_fit_runs_one_epoch(self):
        trainer = self._make_trainer(num_classes=5)
        loader = _make_loader(n=16, num_classes=5)
        history = trainer.fit(loader, loader, epochs=1, save_best=False)
        assert len(history.epochs) == 1
        assert history.epochs[0].epoch == 1
        assert history.epochs[0].train_loss >= 0.0
        assert 0.0 <= history.epochs[0].val_accuracy <= 1.0

    def test_fit_runs_two_epochs(self):
        trainer = self._make_trainer(num_classes=5)
        loader = _make_loader(n=16, num_classes=5)
        history = trainer.fit(loader, loader, epochs=2, save_best=False)
        assert len(history.epochs) == 2
        assert history.epochs[-1].epoch == 2

    def test_history_best_val_accuracy(self):
        trainer = self._make_trainer(num_classes=5)
        loader = _make_loader(n=16, num_classes=5)
        history = trainer.fit(loader, loader, epochs=2, save_best=False)
        assert history.best_val_accuracy >= 0.0
        assert history.best_epoch in {1, 2}

    def test_empty_history(self):
        trainer = self._make_trainer()
        assert trainer.history.best_val_accuracy == 0.0
        assert trainer.history.best_epoch == 0
