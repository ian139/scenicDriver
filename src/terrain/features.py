"""Canonical Terrain-RGB decoding and feature extraction."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image


_SEAM_ZERO_ELEVATION = 0.0
_SEAM_ZERO_ATOL = 0.05
_SEAM_MAX_FRACTION = 0.02

TerrainImageInput = Image.Image | np.ndarray | PathLike[str] | str


@dataclass
class TerrainFeatures:
    """Features for scenic scoring."""

    slope_variation: float  # Normalized 0-1
    elevation_change: float  # Meters
    water_proximity: float  # Normalized 0-1 (1 = adjacent)
    vegetation_density: float  # NDVI-based, 0-1
    coastal: bool
    has_lake: bool
    has_river: bool

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array(
            [
                self.slope_variation,
                self.elevation_change / 1000,  # Normalize to km
                self.water_proximity,
                self.vegetation_density,
                float(self.coastal),
                float(self.has_lake or self.has_river),
            ]
        )


@dataclass(frozen=True)
class TerrainFeatureResult:
    """Terrain features plus scalar metrics used by heuristic scoring."""

    features: TerrainFeatures
    relief: float
    roughness: float
    slope_mean: float


def compute_terrain_features(
    terrain_img: TerrainImageInput,
    satellite_img: TerrainImageInput | None = None,
) -> TerrainFeatureResult:
    """Decode a Terrain-RGB image and compute all shared terrain metrics.

    ``terrain_img`` and ``satellite_img`` may be PIL images, RGB arrays, or
    paths to images. Terrain-RGB seam repair and the feature formulas live
    here so callers that operate on local files and callers that operate on
    downloaded images produce identical values.
    """
    elev = repair_terrain_zero_seam(decode_terrain_rgb(terrain_img))
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx**2 + gy**2)

    relief = float(elev.max() - elev.min())
    roughness = float(elev.std())
    slope_mean = float(slope.mean())
    slope_variation = float(min(slope.std() / 15.0, 1.0))

    low_elev = elev < np.percentile(elev, 10)
    flat = slope < np.percentile(slope, 10)
    water_proximity = float((low_elev & flat).mean())

    vegetation_density = 0.5
    if satellite_img is not None and (
        not isinstance(satellite_img, PathLike) or Path(satellite_img).exists()
    ):
        sat_arr = _as_rgb_array(satellite_img)
        r = sat_arr[..., 0]
        g = sat_arr[..., 1]
        b = sat_arr[..., 2]
        vegetation_density = float(_safe_div(g, r + g + b).mean())

    return TerrainFeatureResult(
        features=TerrainFeatures(
            slope_variation=slope_variation,
            elevation_change=relief,
            water_proximity=water_proximity,
            vegetation_density=vegetation_density,
            coastal=False,
            has_lake=False,
            has_river=False,
        ),
        relief=relief,
        roughness=roughness,
        slope_mean=slope_mean,
    )


def decode_terrain_rgb(terrain_img: TerrainImageInput) -> np.ndarray:
    """Decode Mapbox Terrain-RGB input into elevations in meters."""
    arr = _as_rgb_array(terrain_img)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1


def _as_rgb_array(image: TerrainImageInput) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    elif isinstance(image, (str, PathLike)):
        with Image.open(image) as opened:
            arr = np.asarray(opened.convert("RGB"))
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(
            "Terrain and satellite inputs must be PIL images, RGB arrays, or image paths."
        )

    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image array with shape (H, W, 3), got {arr.shape}.")
    return np.asarray(arr[..., :3], dtype=np.float32)


def repair_terrain_zero_seam(elev: np.ndarray) -> np.ndarray:
    """
    Repair thin zero-elevation seams that appear on tile borders.

    Some terrain-rgb tiles contain a 1-pixel border with elevation==0 due to
    source seam artifacts. This repair only applies when zero pixels are
    sparse and strictly on the border. That can inflate relief/slope metrics.
    """
    zero_mask = np.isclose(elev, _SEAM_ZERO_ELEVATION, atol=_SEAM_ZERO_ATOL)
    zero_fraction = float(zero_mask.mean())
    if zero_fraction <= 0.0 or zero_fraction > _SEAM_MAX_FRACTION:
        return elev

    border_mask = np.zeros_like(zero_mask, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    if not np.all((~zero_mask) | border_mask):
        return elev

    fixed = elev.copy()
    h, w = fixed.shape
    if w > 1 and np.all(zero_mask[:, 0]):
        fixed[:, 0] = fixed[:, 1]
    if w > 1 and np.all(zero_mask[:, -1]):
        fixed[:, -1] = fixed[:, -2]
    if h > 1 and np.all(zero_mask[0, :]):
        fixed[0, :] = fixed[1, :]
    if h > 1 and np.all(zero_mask[-1, :]):
        fixed[-1, :] = fixed[-2, :]

    return fixed


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.maximum(den, 1e-6)
