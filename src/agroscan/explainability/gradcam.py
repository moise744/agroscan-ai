"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Agricultural Vision.

Produces high-resolution heatmaps highlighting the exact leaf lesions, necrotic spots,
and rust pustules driving model classification decisions.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-Based Localization", ICCV 2017.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM saliency maps for a PyTorch convolutional network.

    Args:
        model: Convolutional classifier model.
        target_layer: The target convolutional layer (e.g. model.layer4).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a given input tensor and target class.

        Args:
            input_tensor: (1, 3, H, W) normalized tensor.
            target_class: Class index for which to explain prediction (defaults to top predicted class).

        Returns:
            Normalized 2D heatmap numpy array of shape (H, W) in [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        out = self.model(input_tensor)
        logits = out["logits"] if isinstance(out, dict) else out

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not invoked
            H, W = input_tensor.shape[2:]
            return np.zeros((H, W), dtype=np.float32)

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # (C,)

        # Weight activations by importance alpha
        activations = self.activations[0]  # (C, h, w)
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        # Channel-wise sum and ReLU
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0.0)

        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 1e-7:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        # Upsample to match original input image resolution
        H, W = input_tensor.shape[2:]
        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized.astype(np.float32)

    def remove_hooks(self) -> None:
        """Clean up PyTorch forward/backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
