# Landscape Classification Module
# Owner: progno-ml-vision agent
"""
Vision Transformer-based landscape classification for remote sensing imagery.

This module provides:
- LandscapeClassifier: ViT-B/16 model for 45-class terrain classification
- Image preprocessing and transform utilities
- Training pipeline for RESISC45 dataset
- Scenic weight mapping for terrain classes
"""

from .model import (
    LandscapeClassifier,
    TERRAIN_CLASSES,
    SCENIC_WEIGHTS,
    get_scenic_weight,
)
from .inference import (
    preprocess_image,
    get_inference_transform,
    get_training_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESISC45_MEAN,
    RESISC45_STD,
)

__all__ = [
    # Model
    "LandscapeClassifier",
    "TERRAIN_CLASSES",
    "SCENIC_WEIGHTS",
    "get_scenic_weight",
    # Inference
    "preprocess_image",
    "get_inference_transform",
    "get_training_transform",
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "RESISC45_MEAN",
    "RESISC45_STD",
]
