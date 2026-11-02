"""
AgroScan AI — Single Leaf Diagnosis & Grad-CAM CLI.

Usage:
    python scripts/predict.py --image path/to/leaf.jpg --output outputs/diag.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from agroscan.inference.predictor import PlantDiseasePredictor
from agroscan.visualization.visualizer import LeafVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose leaf pathology from an image")
    parser.add_argument("--image", type=Path, required=True, help="Input image")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Weights path")
    parser.add_argument("--output", type=Path, default=Path("outputs/explanation.png"))
    args = parser.parse_args()

    predictor = PlantDiseasePredictor(checkpoint_path=args.checkpoint)
    print(f"[INFO] Diagnosing image: {args.image}")

    diag = predictor.predict(args.image, top_k=5)
    top = diag.top_prediction

    print("\nDIAGNOSIS RESULTS:")
    print(f"  Crop:        {top.crop}")
    print(f"  Condition:   {top.condition}")
    print(f"  Healthy:     {top.is_healthy}")
    print(f"  Confidence:  {top.confidence:.2%}")
    print(f"  Latency:     {diag.latency_ms:.2f} ms")

    print("\nTop 5 Candidates:")
    for i, c in enumerate(diag.top_k, 1):
        print(f"  {i}. {c.crop} - {c.condition}: {c.confidence:.2%}")

    # Save Grad-CAM visualization
    raw_bgr = cv2.imread(str(args.image))
    if raw_bgr is not None:
        overlay = LeafVisualizer.overlay_gradcam(raw_bgr, diag.gradcam_heatmap, alpha=0.5)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), overlay)
        print(f"\n[INFO] Grad-CAM explanation saved to: {args.output}")


if __name__ == "__main__":
    main()
