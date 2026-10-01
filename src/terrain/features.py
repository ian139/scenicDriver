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
    terrain_sea_level_fraction: float


def compute_terrain_features(
    terrain_img: TerrainImageInput,
    satellite_img: TerrainImageInput | None = None,
    *,
    valid_mask: np.ndarray,
) -> TerrainFeatureResult:
    """Decode a Terrain-RGB image and compute all shared terrain metrics.

    ``valid_mask`` is a boolean (H, W) land/DEM-validity mask from raster
    processing. No statistic or derivative includes outside-land fill.
    Empty masks return zero metrics and a neutral, all-zero feature vector;
    an isolated valid pixel has zero slope. Persisted canonical PNG readers
    can reconstruct the mask with ``compute_terrain_valid_mask``.
    """
    decoded_elevation = decode_terrain_rgb(terrain_img)
    valid = _validate_mask(valid_mask, decoded_elevation.shape)
    if not np.isfinite(decoded_elevation[valid]).all():
        raise ValueError("Valid terrain elevations must be finite")
    elev = repair_terrain_zero_seam(decoded_elevation, valid_mask=valid)
    values = elev[valid]
    slope = np.hypot(
        _masked_gradient(elev, valid, axis=0),
        _masked_gradient(elev, valid, axis=1),
    )[valid]

    relief = float(np.ptp(values)) if values.size else 0.0
    roughness = float(values.std()) if values.size else 0.0
    slope_mean = float(slope.mean()) if slope.size else 0.0
    slope_variation = float(min(slope.std() / 15.0, 1.0)) if slope.size else 0.0
    terrain_sea_level_fraction = (
        float(np.isclose(values, _SEAM_ZERO_ELEVATION, atol=_SEAM_ZERO_ATOL).mean())
        if values.size else 0.0
    )
    water_proximity = (
        float(((values < np.percentile(values, 10)) & (slope < np.percentile(slope, 10))).mean())
        if values.size else 0.0
    )

    vegetation_density = 0.5 if values.size else 0.0
    if satellite_img is not None and (
        not isinstance(satellite_img, (str, PathLike)) or Path(satellite_img).exists()
    ):
        sat_arr = _as_rgb_array(satellite_img)
        if sat_arr.shape[:2] != valid.shape:
            raise ValueError("Satellite shape must match the terrain validity mask")
        if not np.isfinite(sat_arr[valid]).all():
            raise ValueError("Valid satellite pixels must be finite")
        if values.size:
            r, g, b = sat_arr[valid].T
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
        terrain_sea_level_fraction=terrain_sea_level_fraction,
    )


def decode_terrain_rgb(terrain_img: TerrainImageInput) -> np.ndarray:
    """Decode Mapbox Terrain-RGB input into elevations in meters."""
    arr = _as_rgb_array(terrain_img)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1


def compute_terrain_valid_mask(terrain_img: TerrainImageInput) -> np.ndarray:
    """Recover land validity from persisted canonical Terrain-RGB PNGs.

    Black (-10000m) is reserved for outside-land fill by raster processing.
    New raster consumers must carry its returned mask instead of inferring it.
    This reconstruction is not valid for arbitrary third-party Terrain-RGB.
    """
    return np.any(_as_rgb_array(terrain_img) != 0, axis=-1)


def compute_terrain_sea_level_fraction(
    terrain_img: TerrainImageInput, *, valid_mask: np.ndarray
) -> float:
    """Fraction of valid pixels near zero elevation, excluding repaired seams.

    Elevation is an audit proxy, not a water classifier. Empty masks return 0.
    """
    elevation = decode_terrain_rgb(terrain_img)
    valid = _validate_mask(valid_mask, elevation.shape)
    if not np.isfinite(elevation[valid]).all():
        raise ValueError("Valid terrain elevations must be finite")
    values = repair_terrain_zero_seam(elevation, valid_mask=valid)[valid]
    return (
        float(np.isclose(values, _SEAM_ZERO_ELEVATION, atol=_SEAM_ZERO_ATOL).mean())
        if values.size else 0.0
    )


def _validate_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    valid = np.asarray(mask)
    if valid.dtype != np.bool_:
        raise TypeError("valid_mask must have boolean dtype")
    if valid.shape != shape:
        raise ValueError("valid_mask shape must match terrain dimensions")
    return valid


def _masked_gradient(elev: np.ndarray, valid: np.ndarray, *, axis: int) -> np.ndarray:
    """Centered differences where possible, otherwise valid one-sided edges."""
    values = np.moveaxis(elev, axis, 0)
    mask = np.moveaxis(valid, axis, 0)
    forward = np.zeros_like(values)
    backward = np.zeros_like(values)
    has_forward = np.zeros_like(mask)
    has_backward = np.zeros_like(mask)
    pairs = mask[:-1] & mask[1:]
    differences = np.subtract(values[1:], values[:-1], where=pairs, out=np.zeros_like(values[:-1]))
    forward[:-1] = differences
    backward[1:] = differences
    has_forward[:-1] = pairs
    has_backward[1:] = pairs
    count = has_forward.astype(np.uint8) + has_backward
    gradient = np.divide(
        forward + backward, count, out=np.zeros_like(values), where=count > 0
    )
    return np.moveaxis(gradient, 0, axis)


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
        raise ValueError(
            f"Expected an RGB image array with shape (H, W, 3), got {arr.shape}."
        )
    if not arr.shape[0] or not arr.shape[1]:
        raise ValueError("Terrain and satellite dimensions must be positive")
    return np.asarray(arr[..., :3], dtype=np.float32)


def repair_terrain_zero_seam(elev: np.ndarray, *, valid_mask: np.ndarray) -> np.ndarray:
    """
    Repair thin zero-elevation seams that appear on tile borders.

    Some terrain-rgb tiles contain a 1-pixel border with elevation==0 due to
    source seam artifacts. This repair only applies when zero pixels are
    sparse and strictly on the border. That can inflate relief/slope metrics.
    """
    valid = _validate_mask(valid_mask, elev.shape)
    zero_mask = valid & np.isclose(elev, _SEAM_ZERO_ELEVATION, atol=_SEAM_ZERO_ATOL)
    valid_count = int(valid.sum())
    zero_fraction = float(zero_mask.sum() / valid_count) if valid_count else 0.0
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
    if w > 1:
        left = zero_mask[:, 0] & valid[:, 1]
        right = zero_mask[:, -1] & valid[:, -2]
        fixed[left, 0] = fixed[left, 1]
        fixed[right, -1] = fixed[right, -2]
    if h > 1:
        top = zero_mask[0, :] & valid[1, :]
        bottom = zero_mask[-1, :] & valid[-2, :]
        fixed[0, top] = fixed[1, top]
        fixed[-1, bottom] = fixed[-2, bottom]

    return fixed


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.maximum(den, 1e-6)
