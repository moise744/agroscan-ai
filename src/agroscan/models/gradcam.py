"""
Compatibility shim: agroscan.models.gradcam → agroscan.explainability.gradcam

Allows notebooks and scripts to use:
    from agroscan.models.gradcam import GradCAM
"""
from agroscan.explainability.gradcam import GradCAM  # noqa: F401

__all__ = ["GradCAM"]
