"""
Leaf Inspection & Saliency Visualization Renderer.

Blends Grad-CAM heatmaps onto leaf images and formats agricultural diagnosis reports.
"""

from __future__ import annotations

import cv2
import numpy as np


class LeafVisualizer:
    """Overlays Grad-CAM saliency heatmaps on leaf photographs."""

    @staticmethod
    def overlay_gradcam(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Blend a 2D float [0, 1] Grad-CAM heatmap over an RGB/BGR image.

        Args:
            image: Original uint8 image (H, W, 3).
            heatmap: Normalized float32 array (H, W) in [0.0, 1.0].
            alpha: Transparency factor.
            colormap: OpenCV colormap enum.

        Returns:
            Blended uint8 image with warm colors over salient defect regions.
        """
        h, w = image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to 0-255 uint8 and apply colormap
        heat_uint8 = np.uint8(255 * heatmap)
        colored_heat = cv2.applyColorMap(heat_uint8, colormap)

        # Convert BGR colored heatmap to RGB if input is RGB
        blended = cv2.addWeighted(colored_heat, alpha, image, 1.0 - alpha, 0)
        return blended
