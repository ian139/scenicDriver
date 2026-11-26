"""
Vision Transformer for Landscape Classification
Owner: progno-ml-vision agent

Trained on RESISC45 dataset (45 terrain categories)
Input: 224x224 RGB satellite/aerial imagery
Target: 94%+ classification accuracy
"""

from collections.abc import Mapping
from typing import Any, Optional
from pathlib import Path

import torch
import torch.nn as nn

from scenicdriver.scenic_scorer.regression import atomic_torch_save, load_checkpoint

from .inference import PreprocessingContract

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


CHECKPOINT_FORMAT = "scenicdriver.classifier"
CHECKPOINT_SCHEMA_VERSION = 1


def save_classifier_checkpoint(
    path: str | Path,
    *,
    model_state_dict: Mapping[str, torch.Tensor],
    preprocessing: PreprocessingContract,
    **training_state: Any,
) -> None:
    """Atomically write a classifier checkpoint carrying its preprocessing contract.

    ``training_state`` adds resume fields (optimizer, RNG, history, ...); every
    value must be ``weights_only``-loadable (tensors and primitives).
    """
    atomic_torch_save(
        path,
        {
            **training_state,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": dict(model_state_dict),
            "preprocessing": preprocessing.as_dict(),
        },
    )


def load_classifier_checkpoint(
    path: str | Path, map_location: Any = "cpu"
) -> tuple[dict[str, Any], PreprocessingContract]:
    """Safely load and validate a classifier checkpoint and its preprocessing contract.

    Raises:
        ValueError: For legacy bare state dicts or any other non-current format.
        pickle.UnpicklingError: For checkpoints containing non-allowlisted objects.
    """
    payload = load_checkpoint(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping) or (
        payload.get("checkpoint_format") != CHECKPOINT_FORMAT
        or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Classifier checkpoint {path} is not a {CHECKPOINT_FORMAT} "
            f"v{CHECKPOINT_SCHEMA_VERSION} checkpoint (legacy bare state dicts carry no "
            "preprocessing contract). Retrain with `python -m scenicdriver.classifier.train`, "
            "or migrate explicitly with `python -m scenicdriver.classifier.model migrate "
            "--input OLD --output NEW --normalization {imagenet,resisc45}` declaring the "
            "normalization the weights were trained with"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(f"Classifier checkpoint {path} model_state_dict must map names to tensors")
    preprocessing = PreprocessingContract.from_dict(payload.get("preprocessing"))
    return dict(payload), preprocessing


def migrate_legacy_checkpoint(
    source: str | Path, destination: str | Path, *, normalization: str
) -> None:
    """Wrap a legacy bare state dict with an operator-declared preprocessing contract."""
    state = load_checkpoint(Path(source), map_location="cpu")
    if not isinstance(state, Mapping) or "checkpoint_format" in state:
        raise ValueError(f"{source} is not a legacy bare classifier state dict")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"{source} is not a legacy bare classifier state dict")
    save_classifier_checkpoint(
        destination,
        model_state_dict=state,
        preprocessing=PreprocessingContract.for_normalization(
            normalization, class_names=TERRAIN_CLASSES
        ),
    )


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
        classes: List of class names (the checkpoint's class order once loaded)
        preprocessing: Preprocessing contract of the loaded checkpoint, or
            ``None`` for an untrained/pretrained-only model
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
        self.preprocessing: Optional[PreprocessingContract] = None

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
            x: Batch of images [B, 3, 224, 224] normalized per ``self.preprocessing``

        Returns:
            Class logits [B, num_classes]
        """
        return self.forward_with_features(x)[0]

    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ViT backbone once and return classifier logits and features."""
        features = self.backbone(x)
        return self.classifier(features), features


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
        Load a classifier checkpoint and adopt its preprocessing contract.

        Args:
            path: Checkpoint written by :func:`save_classifier_checkpoint`

        Raises:
            ValueError: For legacy/foreign checkpoints or a class-count mismatch
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

        payload, preprocessing = load_classifier_checkpoint(path, map_location=self.device)
        if len(preprocessing.class_names) != self.num_classes:
            raise ValueError(
                f"Classifier checkpoint {path} declares {len(preprocessing.class_names)} "
                f"classes; model has {self.num_classes}"
            )
        self.load_state_dict(payload["model_state_dict"])
        self.preprocessing = preprocessing
        self.classes = list(preprocessing.class_names)
        print(f"Loaded weights from {path}")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classifier checkpoint utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser(
        "migrate", help="Wrap a legacy bare state dict with a declared preprocessing contract"
    )
    migrate.add_argument("--input", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.add_argument("--normalization", choices=("imagenet", "resisc45"), required=True)
    args = parser.parse_args()
    migrate_legacy_checkpoint(args.input, args.output, normalization=args.normalization)
    print(f"Migrated {args.input} -> {args.output} ({args.normalization})")
