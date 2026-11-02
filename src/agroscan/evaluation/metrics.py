"""Evaluation metrics for plant disease classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ClassificationReport:
    """Holds evaluation results for the disease classifier."""
    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    num_samples: int = 0


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> ClassificationReport:
    """Compute classification metrics from predictions.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices (argmax).
        y_pred_proba: Predicted probabilities array of shape (N, C).
        class_names: List of class name strings.
        top_k: k for top-k accuracy computation.

    Returns:
        ClassificationReport dataclass.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )
    except ImportError as exc:
        raise ImportError("scikit-learn is required for metric computation") from exc

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr)

    # Per-class scores
    per_f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    per_prec = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
    per_rec = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)

    num_classes = len(per_f1)
    names = class_names or [str(i) for i in range(num_classes)]

    # Top-k accuracy
    top5_acc = 0.0
    if y_pred_proba is not None:
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -top_k:]
        correct = sum(y_true_arr[i] in top_k_preds[i] for i in range(len(y_true_arr)))
        top5_acc = correct / len(y_true_arr)

    return ClassificationReport(
        accuracy=acc,
        top5_accuracy=top5_acc,
        macro_f1=macro_f1,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        per_class_f1={names[i]: float(per_f1[i]) for i in range(num_classes)},
        per_class_precision={names[i]: float(per_prec[i]) for i in range(num_classes)},
        per_class_recall={names[i]: float(per_rec[i]) for i in range(num_classes)},
        confusion_matrix=cm,
        num_samples=len(y_true),
    )
