"""
Inference transform and preprocessing utilities for landscape classification.
Owner: progno-ml-vision agent

Every classifier checkpoint carries a :class:`PreprocessingContract`. Training
records it; inference, export, and scoring derive their transform from the
loaded checkpoint's contract instead of re-declaring normalization stats.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

try:
    from torchvision import transforms
except ImportError:
    raise ImportError(
        "torchvision required. Install with: uv sync"
    )


# ImageNet normalization statistics
# These are standard for ViT models pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# RESISC45-specific statistics (computed from training set)
RESISC45_MEAN = (0.368, 0.381, 0.344)
RESISC45_STD = (0.200, 0.186, 0.182)

NORMALIZATION_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "resisc45": (RESISC45_MEAN, RESISC45_STD),
}
DEFAULT_NORMALIZATION = "resisc45"
DEFAULT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class PreprocessingContract:
    """Classifier input contract persisted inside every classifier checkpoint.

    ``normalization`` names the statistics; ``mean``/``std`` are the exact
    values applied after ``ToTensor``; images are resized to
    ``image_size`` x ``image_size``; ``class_names`` is the logit order.
    """

    normalization: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    image_size: int
    class_names: tuple[str, ...]

    @classmethod
    def for_normalization(
        cls,
        normalization: str,
        *,
        class_names: Sequence[str],
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> "PreprocessingContract":
        if normalization not in NORMALIZATION_STATS:
            raise ValueError(
                f"unknown classifier normalization {normalization!r}; "
                f"expected one of {sorted(NORMALIZATION_STATS)}"
            )
        mean, std = NORMALIZATION_STATS[normalization]
        return cls.from_dict(
            {
                "normalization": normalization,
                "mean": list(mean),
                "std": list(std),
                "image_size": image_size,
                "class_names": list(class_names),
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> "PreprocessingContract":
        """Validate a persisted contract; stats must match the named normalization."""
        if not isinstance(value, Mapping):
            raise ValueError("classifier preprocessing contract must be a mapping")
        expected_keys = {"normalization", "mean", "std", "image_size", "class_names"}
        if set(value) != expected_keys:
            raise ValueError(
                "classifier preprocessing contract keys must be "
                f"{sorted(expected_keys)}, got {sorted(map(str, value))}"
            )
        normalization = value["normalization"]
        if normalization not in NORMALIZATION_STATS:
            raise ValueError(
                f"unknown classifier normalization {normalization!r}; "
                f"expected one of {sorted(NORMALIZATION_STATS)}"
            )
        stats = {}
        for key in ("mean", "std"):
            raw = value[key]
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 3
                or not all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(item)
                    for item in raw
                )
            ):
                raise ValueError(f"classifier preprocessing {key} must be 3 finite numbers")
            stats[key] = tuple(float(item) for item in raw)
        expected_mean, expected_std = NORMALIZATION_STATS[normalization]
        if stats["mean"] != tuple(expected_mean) or stats["std"] != tuple(expected_std):
            raise ValueError(
                f"classifier preprocessing stats do not match {normalization!r} normalization"
            )
        image_size = value["image_size"]
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
            raise ValueError("classifier preprocessing image_size must be a positive integer")
        class_names = value["class_names"]
        if (
            not isinstance(class_names, (list, tuple))
            or not class_names
            or not all(isinstance(name, str) and name for name in class_names)
            or len(set(class_names)) != len(class_names)
        ):
            raise ValueError(
                "classifier preprocessing class_names must be unique non-empty strings"
            )
        return cls(
            normalization=normalization,
            mean=stats["mean"],
            std=stats["std"],
            image_size=image_size,
            class_names=tuple(class_names),
        )

    def as_dict(self) -> dict[str, Any]:
        """Primitive form persisted in checkpoints (``weights_only`` safe)."""
        return {
            "normalization": self.normalization,
            "mean": list(self.mean),
            "std": list(self.std),
            "image_size": self.image_size,
            "class_names": list(self.class_names),
        }

    def require_normalization(self, expected: str | None) -> None:
        """Fail when an explicit override conflicts with the checkpoint contract."""
        if expected is not None and expected != self.normalization:
            raise ValueError(
                f"requested classifier normalization {expected!r} conflicts with the "
                f"checkpoint's {self.normalization!r}; preprocessing is defined by the "
                "checkpoint (omit the override or retrain)"
            )


def get_inference_transform(preprocessing: PreprocessingContract) -> transforms.Compose:
    """Inference pipeline defined by a checkpoint's preprocessing contract."""
    size = preprocessing.image_size
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocessing.mean, std=preprocessing.std),
    ])


def get_training_transform(preprocessing: PreprocessingContract) -> transforms.Compose:
    """
    Training pipeline with augmentation for the given preprocessing contract.

    Augmentations designed for satellite/aerial imagery:
    - Random horizontal/vertical flips (orientation invariance)
    - Random rotation (viewing angle invariance)
    - Random resized crop (scale invariance)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(preprocessing.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocessing.mean, std=preprocessing.std),
    ])
