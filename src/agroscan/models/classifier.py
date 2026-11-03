"""
Plant Disease Classifier using ResNet-18 Backbone.

Exposes the final convolutional layer (layer4) for Grad-CAM activation extraction.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from agroscan.data.classes import PLANTVILLAGE_CLASSES

logger = logging.getLogger(__name__)


class PlantDiseaseClassifier(nn.Module):
    """Deep residual network for agricultural disease categorization.

    Args:
        num_classes: Number of disease categories (38 for PlantVillage).
        pretrained: If True, loads ImageNet pretrained weights.
        dropout: Classification head dropout probability.
    """

    def __init__(
        self,
        num_classes: int = len(PLANTVILLAGE_CLASSES),
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Feature extractor components
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # Target layer for Grad-CAM

        self.avgpool = resnet.avgpool

        # Custom Head
        in_feats = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to layer4."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.extract_features(x)
        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        probs = torch.softmax(logits, dim=1)

        return {"logits": logits, "probs": probs, "features": features}


# Compatibility alias — allows notebooks to use DiseaseClassifier
DiseaseClassifier = PlantDiseaseClassifier
