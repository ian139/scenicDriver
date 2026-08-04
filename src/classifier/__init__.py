# Landscape Classification Module
# Owner: progno-ml-vision agent
"""
Vision Transformer-based landscape classification for remote sensing imagery.

This module provides:
- LandscapeClassifier: ViT-B/16 model for 45-class terrain classification
- Inference utilities for single image and batch classification
- Training pipeline for RESISC45 dataset
- Scenic weight mapping for terrain classes

Example usage:
    >>> from src.classifier import LandscapeClassifier, classify_image, load_model
    >>> model = load_model("models/classifier/best_model.pt")
    >>> result = classify_image("data/tiles/sample.png", model)
    >>> print(f"Class: {result['class']} ({result['confidence']:.1%})")
    >>> print(f"Scenic weight: {result['scenic_weight']:.0%}")
"""

from .model import (
    LandscapeClassifier,
    TERRAIN_CLASSES,
    SCENIC_WEIGHTS,
    get_scenic_weight,
)
from .inference import (
    load_model,
    classify_image,
    batch_classify,
    classify_directory,
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
    "load_model",
    "classify_image",
    "batch_classify",
    "classify_directory",
    "preprocess_image",
    "get_inference_transform",
    "get_training_transform",
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "RESISC45_MEAN",
    "RESISC45_STD",
]
