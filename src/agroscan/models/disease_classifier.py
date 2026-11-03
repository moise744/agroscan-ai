"""
Compatibility shim: agroscan.models.DiseaseClassifier → PlantDiseaseClassifier

Allows notebooks to use:
    from agroscan.models.classifier import DiseaseClassifier
"""
from agroscan.models.classifier import PlantDiseaseClassifier as DiseaseClassifier  # noqa: F401

__all__ = ["DiseaseClassifier", "PlantDiseaseClassifier"]
