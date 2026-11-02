"""
PlantVillage Dataset Loader.

Recursively indexes plant disease folders, parses botanical labels,
and yields normalized PyTorch tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from agroscan.data.classes import PLANTVILLAGE_CLASSES
from agroscan.data.transforms import get_agri_transforms

logger = logging.getLogger(__name__)


class PlantVillageDataset(Dataset):
    """Dataset loader for PlantVillage leaf images.

    Expects directory layout:
        <root>/<class_name>/<img_name>.jpg
    """

    def __init__(
        self,
        root: str | Path,
        classes: Optional[List[str]] = None,
        is_training: bool = False,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.root = Path(root)
        self.classes = classes or PLANTVILLAGE_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = get_agri_transforms(image_size=image_size, is_training=is_training)

        self.samples: List[Tuple[Path, int]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        if not self.root.exists():
            logger.warning("Dataset root does not exist: %s", self.root)
            return

        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue

            idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.glob("*.*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, idx))

        logger.info("PlantVillageDataset loaded %d images across %d classes", len(self.samples), len(self.classes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        img_path, target_idx = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)

        return {
            "image": tensor,
            "label": target_idx,
            "class_name": self.classes[target_idx],
            "path": str(img_path),
        }
