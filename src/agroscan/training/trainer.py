"""
AgroScan AI — Training Module.

Handles the training loop, validation, checkpointing, and metric logging
for PlantDiseaseClassifier.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainEpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    duration_sec: float


@dataclass
class TrainingHistory:
    epochs: List[TrainEpochResult] = field(default_factory=list)

    @property
    def best_val_accuracy(self) -> float:
        if not self.epochs:
            return 0.0
        return max(e.val_accuracy for e in self.epochs)

    @property
    def best_epoch(self) -> int:
        if not self.epochs:
            return 0
        return max(self.epochs, key=lambda e: e.val_accuracy).epoch


class DiseaseClassifierTrainer:
    """Training engine for PlantDiseaseClassifier.

    Args:
        model: PlantDiseaseClassifier instance.
        optimizer: PyTorch optimizer (e.g. Adam).
        criterion: Loss function (default: CrossEntropyLoss).
        device: Compute device.
        output_dir: Directory for checkpoints.
        scheduler: Optional LR scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs/models",
        scheduler: Optional[object] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.scheduler = scheduler
        self.history = TrainingHistory()

        self.model.to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(
        self, loader: DataLoader, is_training: bool
    ) -> Tuple[float, float]:
        """Run a single epoch. Returns (avg_loss, accuracy)."""
        self.model.train(is_training)
        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad() if is_training else torch.no_grad()
        with ctx:
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                out = self.model(images)
                logits = out["logits"]

                loss = self.criterion(logits, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_best: bool = True,
    ) -> TrainingHistory:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            epochs: Number of epochs to train.
            save_best: If True, checkpoint the model with best val accuracy.

        Returns:
            TrainingHistory with per-epoch results.
        """
        best_val_acc = 0.0
        logger.info("Starting training for %d epochs on %s", epochs, self.device)

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            train_loss, _ = self._run_epoch(train_loader, is_training=True)
            val_loss, val_acc = self._run_epoch(val_loader, is_training=False)

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            duration = time.perf_counter() - t0
            result = TrainEpochResult(
                epoch=epoch,
                train_loss=round(train_loss, 6),
                val_loss=round(val_loss, 6),
                val_accuracy=round(val_acc, 6),
                duration_sec=round(duration, 2),
            )
            self.history.epochs.append(result)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | %.1fs",
                epoch, epochs, train_loss, val_loss, val_acc, duration,
            )

            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)

        self._save_checkpoint(epochs, val_acc, is_best=False)
        logger.info("Training complete. Best val accuracy: %.4f (epoch %d)",
                    self.history.best_val_accuracy, self.history.best_epoch)
        return self.history

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool) -> None:
        """Save model checkpoint."""
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch:03d}.pth"
        checkpoint = {
            "epoch": epoch,
            "val_accuracy": val_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)
