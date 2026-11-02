"""AgroScan AI — Agricultural Computer Vision & Disease Analysis.

Deep learning plant disease classification with Grad-CAM visual explainability.
"""

__version__ = "0.1.0"
__author__ = "Habimana Moise"

from agroscan.models.classifier import PlantDiseaseClassifier
from agroscan.explainability.gradcam import GradCAM
from agroscan.inference.predictor import PlantDiseasePredictor

__all__ = [
    "PlantDiseaseClassifier",
    "GradCAM",
    "PlantDiseasePredictor",
]
