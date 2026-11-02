"""
AgroScan AI — Training Script for Plant Disease Classification.

Usage:
    python scripts/train.py --config configs/classifier.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from agroscan.data.dataset import PlantVillageDataset
from agroscan.models.classifier import PlantDiseaseClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgroScan AI disease classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_root = Path(cfg["data"]["root"])
    if not data_root.exists():
        print(f"[WARN] Dataset directory {data_root} not found. Follow data/README.md.")
        return

    train_ds = PlantVillageDataset(root=data_root, is_training=True)
    val_ds = PlantVillageDataset(root=data_root, is_training=False)

    if len(train_ds) == 0:
        print("[WARN] No samples found.")
        return

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model = PlantDiseaseClassifier(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", False),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    print(f"[INFO] Starting training across {epochs} epochs...")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            preds = torch.argmax(out["logits"], dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

        acc = correct / max(1, total)
        print(f"Epoch {ep:02d}/{epochs:02d} — Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.2%}")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    print(f"[INFO] Model weights saved to {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
