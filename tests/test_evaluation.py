"""Unit tests for AgroScan evaluation metrics."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agroscan.evaluation.metrics import compute_classification_metrics, ClassificationReport


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        report = compute_classification_metrics(y_true, y_pred)
        assert report.accuracy == pytest.approx(1.0)
        assert report.macro_f1 == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        report = compute_classification_metrics(y_true, y_pred)
        assert report.accuracy == pytest.approx(0.0)

    def test_returns_classification_report(self):
        y_true = [0, 1, 2, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        report = compute_classification_metrics(y_true, y_pred)
        assert isinstance(report, ClassificationReport)

    def test_num_samples(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        report = compute_classification_metrics(y_true, y_pred)
        assert report.num_samples == 3

    def test_top5_accuracy_with_proba(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        proba = np.eye(38)[:3]  # 38-class identity matrix slice
        report = compute_classification_metrics(y_true, y_pred, y_pred_proba=proba, top_k=5)
        assert report.top5_accuracy == pytest.approx(1.0)

    def test_per_class_metrics_keys(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 1]
        class_names = ["healthy", "blight"]
        report = compute_classification_metrics(y_true, y_pred, class_names=class_names)
        assert "healthy" in report.per_class_f1
        assert "blight" in report.per_class_f1
