"""
Vision Transformer for Landscape Classification
Owner: progno-ml-vision agent

Trained on RESISC45 dataset (45 terrain categories)
Input: 224x224 RGB satellite/aerial imagery
Target: 94%+ classification accuracy
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

try:
    import timm
except ImportError:
    raise ImportError(
        "timm library required. Install with: uv sync"
    )


# RESISC45 terrain categories (alphabetically sorted as per dataset convention)
TERRAIN_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
    "island", "lake", "meadow", "medium_residential", "mobile_home_park",
    "mountain", "overpass", "palace", "parking_lot", "railway",
    "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
    "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
    "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
]

# Scenic value weights per terrain class (higher = more scenic)
# These weights are used to compute a scenic score from classification results
SCENIC_WEIGHTS = {
    # Natural landscapes - highest scenic value
    "mountain": 0.95,
    "beach": 0.9,
    "lake": 0.9,
    "island": 0.7,
    "forest": 0.85,
    "river": 0.85,
    "meadow": 0.8,
    "snowberg": 0.8,
    "wetland": 0.7,
    "sea_ice": 0.7,
    "desert": 0.45,
    "chaparral": 0.45,
    "cloud": 0.5,

    # Agricultural - moderate scenic value
    "circular_farmland": 0.45,
    "rectangular_farmland": 0.45,
    "terrace": 0.5,

    # Built environment with some character
    "bridge": 0.4,
    "harbor": 0.45,
    "palace": 0.35,
    "church": 0.45,
    "stadium": 0.3,
    "golf_course": 0.25,

    # Residential - lower scenic value
    "sparse_residential": 0.25,
    "medium_residential": 0.2,
    "dense_residential": 0.15,
    "mobile_home_park": 0.1,

    # Transportation infrastructure - low scenic value
    "freeway": 0.1,
    "railway": 0.15,
    "railway_station": 0.2,
    "intersection": 0.1,
    "roundabout": 0.1,
    "overpass": 0.1,
    "runway": 0.1,
    "airport": 0.1,

    # Industrial/Commercial - lowest scenic value
    "industrial_area": 0.05,
    "parking_lot": 0.05,
    "commercial_area": 0.1,
    "storage_tank": 0.05,
    "thermal_power_station": 0.05,

    # Sports/Recreation facilities
    "baseball_diamond": 0.25,
    "basketball_court": 0.15,
    "ground_track_field": 0.2,
    "tennis_court": 0.2,

    # Vehicles/Objects - context dependent
    "airplane": 0.2,
    "ship": 0.3,

    # Default for any unlisted classes
    "_default": 0.3
}


def get_scenic_weight(class_name: str) -> float:
    """Get the scenic weight for a terrain class."""
    return SCENIC_WEIGHTS.get(class_name, SCENIC_WEIGHTS["_default"])


class LandscapeClassifier(nn.Module):
    """
    ViT-based landscape classifier for remote sensing imagery.

    Architecture: Vision Transformer (ViT-B/16) pretrained on ImageNet,
    fine-tuned on RESISC45 dataset for 45-class terrain classification.

    The model uses transfer learning from ImageNet to leverage general
    visual features, then fine-tunes the classifier head for remote
    sensing specific categories.

    Attributes:
        num_classes: Number of output classes (45 for RESISC45)
        device: Computation device (cuda/cpu)
        classes: List of class names
        backbone: ViT backbone model from timm
    """

    def __init__(
        self,
        num_classes: int = 45,
        pretrained: bool = True,
        pretrained_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dropout: float = 0.1,
    ):
        """
        Initialize the LandscapeClassifier.

        Args:
            num_classes: Number of terrain classes (default: 45 for RESISC45)
            pretrained: Whether to use ImageNet pretrained weights
            pretrained_path: Path to custom pretrained weights (overrides pretrained)
            device: Device to run inference on
            dropout: Dropout rate for classifier head
        """
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.classes = TERRAIN_CLASSES

        # Initialize ViT backbone from timm
        # Using vit_base_patch16_224: 86M params, 224x224 input, 16x16 patches
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained and pretrained_path is None,
            num_classes=0,  # Remove default classifier head
            drop_rate=dropout,
        )

        # Get the feature dimension from backbone
        self.feature_dim = self.backbone.num_features  # 768 for ViT-B

        # Custom classifier head for RESISC45 classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )

        # Load custom weights if provided
        if pretrained_path is not None:
            self.load_weights(pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Args:
            x: Batch of images [B, 3, 224, 224] normalized to ImageNet stats

        Returns:
            Class logits [B, num_classes]
        """
        # Extract features from ViT backbone
        # backbone returns [B, feature_dim] when num_classes=0
        features = self.backbone(x)

        # Pass through classifier head
        logits = self.classifier(features)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without classification.

        Useful for visualization, clustering, or transfer learning.

        Args:
            x: Batch of images [B, 3, 224, 224]

        Returns:
            Feature vectors [B, feature_dim]
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def load_weights(self, path: Path) -> None:
        """
        Load pretrained weights from a file.

        Args:
            path: Path to the weights file (.pt or .pth)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        state_dict = torch.load(path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.load_state_dict(state_dict)
        print(f"Loaded weights from {path}")

    def save_weights(self, path: Path) -> None:
        """
        Save model weights to a file.

        Args:
            path: Path to save the weights file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Saved weights to {path}")

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture": "ViT-B/16",
            "input_size": (3, 224, 224),
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }
