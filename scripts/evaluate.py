"""
AgroScan AI — Model Evaluation Script.

Evaluates a trained PlantDiseaseClassifier checkpoint on a validation split
and prints a detailed per-class classification report.

Usage:
    python scripts/evaluate.py --checkpoint outputs/models/best_model.pth \\
                               --data-dir data/plantvillage --top-k 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from agroscan.data.classes import PLANTVILLAGE_CLASSES
from agroscan.data.dataset import PlantVillageDataset
from agroscan.evaluation.metrics import compute_classification_metrics
from agroscan.models.classifier import PlantDiseaseClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AgroScan AI classifier")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint")
    p.add_argument("--data-dir", type=Path, default=Path("data/plantvillage"))
    p.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation_report.json"))
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("[INFO]  Train a model first with: python scripts/train.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg.get("model", {}).get("num_classes", len(PLANTVILLAGE_CLASSES))

    # Dataset
    if not args.data_dir.exists():
        print(f"[WARN] Dataset not found at {args.data_dir}. See data/README.md.")
        sys.exit(0)

    val_ds = PlantVillageDataset(root=args.data_dir, is_training=False)
    if len(val_ds) == 0:
        print("[WARN] No samples found in validation split.")
        sys.exit(0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[INFO] Validation samples: {len(val_ds)}")

    # Model
    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # Inference
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            out = model(images)
            probs = out["probs"].cpu().numpy()
            preds = probs.argmax(axis=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    all_probs_arr = np.concatenate(all_probs, axis=0)

    # Metrics
    class_names = [PLANTVILLAGE_CLASSES[i] for i in range(num_classes)]
    report = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_pred_proba=all_probs_arr,
        class_names=class_names,
        top_k=args.top_k,
    )

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy  : {report.accuracy:.4f}  ({report.accuracy:.1%})")
    print(f"  Top-{args.top_k} Accuracy  : {report.top5_accuracy:.4f}  ({report.top5_accuracy:.1%})")
    print(f"  Macro F1 Score  : {report.macro_f1:.4f}")
    print(f"  Macro Precision : {report.macro_precision:.4f}")
    print(f"  Macro Recall    : {report.macro_recall:.4f}")
    print(f"  Num Samples     : {report.num_samples}")
    print(f"{'='*60}")

    print(f"\n  Per-Class F1 Scores (bottom 10):")
    sorted_f1 = sorted(report.per_class_f1.items(), key=lambda x: x[1])
    for cls, f1 in sorted_f1[:10]:
        print(f"    {cls:<45} F1={f1:.3f}")

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "accuracy": report.accuracy,
        f"top{args.top_k}_accuracy": report.top5_accuracy,
        "macro_f1": report.macro_f1,
        "macro_precision": report.macro_precision,
        "macro_recall": report.macro_recall,
        "num_samples": report.num_samples,
        "per_class_f1": report.per_class_f1,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
