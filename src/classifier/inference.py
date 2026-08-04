"""
Inference transform and preprocessing utilities for landscape classification.
Owner: progno-ml-vision agent
"""


from pathlib import Path
from typing import Union, Optional

from torch import Tensor
from PIL import Image
import numpy as np

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










