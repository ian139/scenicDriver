"""Canonical Terrain-RGB decoding and feature extraction."""

from .features import (
    TerrainFeatureResult,
    TerrainFeatures,
    compute_terrain_features,
    decode_terrain_rgb,
    repair_terrain_zero_seam,
)

__all__ = [
    "TerrainFeatureResult",
    "TerrainFeatures",
    "compute_terrain_features",
    "decode_terrain_rgb",
    "repair_terrain_zero_seam",
]
