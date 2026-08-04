"""
Inference utilities for landscape classification
Owner: progno-ml-vision agent

This module provides utilities for running inference with the
LandscapeClassifier model on single images or batches.
"""

from pathlib import Path
from typing import Union, List, Optional, Tuple
import torch
from torch import Tensor
from PIL import Image
import numpy as np

try:
    from torchvision import transforms
except ImportError:
    raise ImportError(
        "torchvision required. Install with: uv sync"
    )

from .model import LandscapeClassifier, TERRAIN_CLASSES, get_scenic_weight


# ImageNet normalization statistics
# These are standard for ViT models pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# RESISC45-specific statistics (computed from training set)
# Can be used for better performance when training from scratch
RESISC45_MEAN = (0.368, 0.381, 0.344)
RESISC45_STD = (0.200, 0.186, 0.182)


def get_inference_transform(
    image_size: int = 224,
    use_resisc45_stats: bool = False
) -> transforms.Compose:
    """
    Get the image transformation pipeline for inference.

    Args:
        image_size: Target image size (default: 224)
        use_resisc45_stats: If True, use RESISC45 dataset statistics
                           instead of ImageNet (default: False)

    Returns:
        torchvision transform composition
    """
    mean = RESISC45_MEAN if use_resisc45_stats else IMAGENET_MEAN
    std = RESISC45_STD if use_resisc45_stats else IMAGENET_STD

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_training_transform(
    image_size: int = 224,
    use_resisc45_stats: bool = False
) -> transforms.Compose:
    """
    Get the image transformation pipeline for training with augmentation.

    Augmentations designed for satellite/aerial imagery:
    - Random horizontal/vertical flips (orientation invariance)
    - Random rotation (viewing angle invariance)
    - Color jitter (lighting condition variance)
    - Random resized crop (scale invariance)

    Args:
        image_size: Target image size (default: 224)
        use_resisc45_stats: If True, use RESISC45 dataset statistics

    Returns:
        torchvision transform composition
    """
    mean = RESISC45_MEAN if use_resisc45_stats else IMAGENET_MEAN
    std = RESISC45_STD if use_resisc45_stats else IMAGENET_STD

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # NOTE: Avoid ColorJitter hue on some PIL/torchvision stacks where
        # negative hue factors can underflow uint8.
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_model(
    model_path: Optional[Path] = None,
    device: Optional[str] = None,
    pretrained: bool = True
) -> LandscapeClassifier:
    """
    Load a trained classifier model.

    Args:
        model_path: Path to trained weights. If None, loads pretrained ViT
        device: Device to load model on. Auto-detects if None
        pretrained: Whether to load ImageNet pretrained weights (when model_path is None)

    Returns:
        LandscapeClassifier model in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LandscapeClassifier(
        pretrained=pretrained,
        pretrained_path=model_path,
        device=device
    )
    model = model.to(device)
    model.eval()

    return model


def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    transform: Optional[transforms.Compose] = None
) -> Tensor:
    """
    Preprocess an image for classification.

    Args:
        image: Input image as:
            - str/Path: path to image file
            - PIL.Image: PIL image object
            - np.ndarray: numpy array (H, W, C) in RGB format
        transform: Optional custom transform. Uses default inference transform if None.

    Returns:
        Preprocessed tensor [1, 3, 224, 224] ready for model input
    """
    # Convert to PIL Image if necessary
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        # Assume numpy array is in RGB format (H, W, C)
        if image.dtype != np.uint8:
            # Scale float images to uint8
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Apply transform
    if transform is None:
        transform = get_inference_transform()

    tensor = transform(image)

    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor


def classify_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: LandscapeClassifier,
    transform: Optional[transforms.Compose] = None
) -> dict:
    """
    Classify a single image.

    Args:
        image: Image to classify (path, PIL Image, or numpy array)
        model: Loaded classifier model
        transform: Optional custom preprocessing transform

    Returns:
        Classification result dict with keys:
            - class: Predicted class name
            - class_idx: Predicted class index
            - confidence: Confidence score (0-1)
            - scenic_weight: Scenic beauty weight for the class
            - all_probs: Full probability distribution
            - top_5: List of (class_name, probability) for top 5

    Example:
        >>> model = load_model("models/classifier/best_model.pt")
        >>> result = classify_image("data/tiles/sample.png", model)
        >>> print(f"Class: {result['class']} ({result['confidence']:.2%})")
    """
    tensor = preprocess_image(image, transform)
    tensor = tensor.to(model.device)
    return model.predict(tensor)


def batch_classify(
    images: List[Union[str, Path, Image.Image, np.ndarray]],
    model: LandscapeClassifier,
    batch_size: int = 32,
    transform: Optional[transforms.Compose] = None,
    show_progress: bool = False
) -> List[dict]:
    """
    Classify multiple images in batches.

    Args:
        images: List of images to classify
        model: Loaded classifier model
        batch_size: Number of images per batch (default: 32)
        transform: Optional custom preprocessing transform
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        List of classification results, one dict per image

    Example:
        >>> model = load_model("models/classifier/best_model.pt")
        >>> image_paths = list(Path("data/tiles").glob("*.png"))
        >>> results = batch_classify(image_paths, model, batch_size=16)
        >>> for path, result in zip(image_paths, results):
        ...     print(f"{path.name}: {result['class']}")
    """
    results = []

    # Optional progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(images), batch_size), desc="Classifying")
        except ImportError:
            iterator = range(0, len(images), batch_size)
    else:
        iterator = range(0, len(images), batch_size)

    for i in iterator:
        batch_images = images[i:i + batch_size]

        # Preprocess batch
        tensors = torch.stack([
            preprocess_image(img, transform).squeeze(0)
            for img in batch_images
        ])
        tensors = tensors.to(model.device)

        # Batch prediction
        batch_results = model.predict_batch(tensors)
        results.extend(batch_results)

    return results


def classify_directory(
    directory: Union[str, Path],
    model: LandscapeClassifier,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    batch_size: int = 32,
    show_progress: bool = True
) -> dict:
    """
    Classify all images in a directory.

    Args:
        directory: Path to directory containing images
        model: Loaded classifier model
        extensions: Tuple of valid image extensions
        batch_size: Batch size for inference
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping filename to classification result

    Example:
        >>> model = load_model()
        >>> results = classify_directory("data/tiles/", model)
        >>> for filename, result in results.items():
        ...     print(f"{filename}: {result['class']} ({result['confidence']:.1%})")
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.glob(f"*{ext}"))
        image_paths.extend(directory.glob(f"*{ext.upper()}"))

    if not image_paths:
        print(f"No images found in {directory}")
        return {}

    # Sort for consistent ordering
    image_paths = sorted(set(image_paths))

    # Classify all images
    results = batch_classify(
        image_paths,
        model,
        batch_size=batch_size,
        show_progress=show_progress
    )

    # Map to filenames
    return {path.name: result for path, result in zip(image_paths, results)}




