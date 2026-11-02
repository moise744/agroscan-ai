"""
Plant Disease Inference & Diagnostics Predictor.

Executes classification forward pass, extracts top-k predictions,
and generates explainable Grad-CAM heatmaps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from agroscan.data.classes import PLANTVILLAGE_CLASSES, parse_class_name
from agroscan.explainability.gradcam import GradCAM
from agroscan.models.classifier import PlantDiseaseClassifier

logger = logging.getLogger(__name__)


@dataclass
class TopPrediction:
    class_id: int
    raw_class: str
    crop: str
    condition: str
    is_healthy: bool
    confidence: float


@dataclass
class DiagnosisResult:
    top_prediction: TopPrediction
    top_k: List[TopPrediction]
    gradcam_heatmap: np.ndarray  # (H, W) float32 [0.0, 1.0]
    latency_ms: float


class PlantDiseasePredictor:
    """Inference engine for leaf diagnosis."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str | Path] = None,
        device: str | torch.device = "auto",
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.image_size = image_size

        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = PlantDiseaseClassifier(pretrained=False)
        else:
            self.model = model

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                elif isinstance(state, dict):
                    self.model.load_state_dict(state)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Attach GradCAM to layer4
        self.gradcam = GradCAM(self.model, self.model.layer4)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
        generate_saliency: bool = True,
    ) -> DiagnosisResult:
        """Analyze a leaf image and return top-k predictions with Grad-CAM heatmap."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size

        # Preprocess
        resized = TF.resize(pil_img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized)
        norm_tensor = TF.normalize(tensor, mean=self.mean, std=self.std).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        # Compute forward pass with gradient tracking for GradCAM if needed
        norm_tensor.requires_grad_(True)
        out = self.model(norm_tensor)
        probs = out["probs"].detach().squeeze().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Rank predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions: List[TopPrediction] = []

        for idx in top_indices:
            raw_cls = PLANTVILLAGE_CLASSES[idx]
            crop, cond, is_healthy = parse_class_name(raw_cls)
            top_predictions.append(
                TopPrediction(
                    class_id=int(idx),
                    raw_class=raw_cls,
                    crop=crop,
                    condition=cond,
                    is_healthy=is_healthy,
                    confidence=float(probs[idx]),
                )
            )

        top_1 = top_predictions[0]

        # Generate Grad-CAM for top class
        if generate_saliency:
            cam_map = self.gradcam.generate(norm_tensor, target_class=top_1.class_id)
            # Upscale heatmap to match original image dimensions
            import cv2
            cam_resized = cv2.resize(cam_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_resized = np.zeros((orig_h, orig_w), dtype=np.float32)

        return DiagnosisResult(
            top_prediction=top_1,
            top_k=top_predictions,
            gradcam_heatmap=cam_resized,
            latency_ms=round(latency_ms, 2),
        )
