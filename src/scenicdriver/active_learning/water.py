"""Canonical water classification constants and metrics for active learning."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from typing import Any, Mapping

MAX_SELECTABLE_WATER_FRACTION: float = 0.50
UNUSABLE_REASON_EXCESSIVE_WATER: str = "excessive_water"

WATER_FILTER_STATUS_PASS: str = "pass"
WATER_FILTER_STATUS_EXCESSIVE: str = "excessive_water"
WATER_FILTER_STATUS_UNKNOWN: str = "unknown"

# Identity of the canonical satellite water metric. Weak-label provenance
# records it so labels from older (e.g. whole-image texture) metrics stay
# attributable. Bump ``version`` whenever compute_satellite_water_fraction or
# weak_score_water_components change semantics.
WATER_METRIC = {
    "version": 1,
    "function": "scenicdriver.active_learning.water.compute_satellite_water_fraction",
    "weak_score_components": "scenicdriver.active_learning.water.weak_score_water_components",
    "texture": "local-3x3-land-only",
    "land_mask": "scenicdriver.terrain.features.compute_terrain_valid_mask",
    "outside_land": "water",
}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def compute_satellite_water_fraction(
    image: Image.Image | np.ndarray, *, land_mask: np.ndarray | None = None
) -> float:
    """Estimate water coverage with local texture and optional land geometry.

    False pixels in the boolean (H, W) land mask are known outside-land sea,
    even when canonical imagery fills them black. They count as water in the
    full-tile denominator, not as RGB evidence in neighboring texture windows.
    Without geometry, only RGB evidence is used. Clouds/whitecaps can obscure
    individual pixels but cannot veto water elsewhere in the tile.
    """
    if isinstance(image, Image.Image):
        values = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 3:
        raise ValueError(f"Expected RGB image array, got shape {values.shape}")
    if not values.shape[0] or not values.shape[1]:
        raise ValueError("Water imagery must have positive dimensions")
    if land_mask is None:
        land = np.ones(values.shape[:2], dtype=bool)
    else:
        land = np.asarray(land_mask)
        if land.dtype != np.bool_:
            raise TypeError("land_mask must have boolean dtype")
        if land.shape != values.shape[:2]:
            raise ValueError("land_mask shape must match satellite dimensions")
    if not np.isfinite(values[land]).all():
        raise ValueError("Land satellite pixels must be finite")
    values = values[..., :3] / 255.0
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    brightness = (r + g + b) / 3.0
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    saturation = (maxc - minc) / np.maximum(maxc + 1e-6, 1e-6)
    luminance = 0.2989 * r + 0.5870 * g + 0.1140 * b
    weights = uniform_filter(land.astype(np.float32), size=3, mode="constant")
    mean = np.divide(
        uniform_filter(np.where(land, luminance, 0.0), size=3, mode="constant"),
        weights, out=np.zeros_like(luminance), where=weights > 0,
    )
    mean_square = np.divide(
        uniform_filter(np.where(land, luminance**2, 0.0), size=3, mode="constant"),
        weights, out=np.zeros_like(luminance), where=weights > 0,
    )
    texture = np.sqrt(np.maximum(mean_square - mean**2, 0.0))
    water_mask = (
        (b > r * 1.2)
        & (b > g * 1.15)
        & (brightness < 0.65)
        & (saturation > 0.18)
        & (texture < 0.12)
    )
    return float((~land | (land & water_mask)).mean())


def compute_effective_water_fraction(
    satellite_water_fraction: float | None,
    terrain_sea_level_fraction: float | None,
) -> float | None:
    """Return satellite-derived water coverage.

    Terrain-RGB only encodes elevation. Its sea-level fraction is retained for
    audit but must not independently classify land as water.
    """
    del terrain_sea_level_fraction
    return _finite_float(satellite_water_fraction)


def weak_score_water_components(
    class_score: float, *, water_fraction: float, water_proximity: float
) -> dict[str, float]:
    """Water-dependent heuristic weak-score terms shared by labeler and scorer.

    ``water_fraction`` must come from ``compute_satellite_water_fraction``
    with the terrain validity mask as ``land_mask``.
    """
    water_fraction = float(water_fraction)
    class_weight = 2.5 * max(0.3, 1.0 - 0.9 * water_fraction)
    return {
        "heuristic_class_component": class_weight * float(class_score),
        "heuristic_water_component": 1.5 * float(water_proximity),
        "heuristic_water_fraction_penalty": -1.2 * water_fraction,
    }


def evaluate_water_status(
    satellite_water_fraction: float | None,
    terrain_sea_level_fraction: float | None,
) -> tuple[float | None, str, str | None]:
    """Evaluate satellite water coverage while retaining terrain audit input."""
    effective = compute_effective_water_fraction(
        satellite_water_fraction, terrain_sea_level_fraction
    )
    if effective is not None:
        if effective >= MAX_SELECTABLE_WATER_FRACTION:
            return (
                effective,
                WATER_FILTER_STATUS_EXCESSIVE,
                UNUSABLE_REASON_EXCESSIVE_WATER,
            )
        return effective, WATER_FILTER_STATUS_PASS, None
    return None, WATER_FILTER_STATUS_UNKNOWN, None


def is_excessive_water(water_fraction: float | int | None) -> bool:
    """Return True if water_fraction is at or above MAX_SELECTABLE_WATER_FRACTION."""
    val = _finite_float(water_fraction)
    return val is not None and val >= MAX_SELECTABLE_WATER_FRACTION


def is_water_unusable(row: Mapping[str, Any]) -> bool:
    """Defensive check for candidate rows unusable due to excessive water."""
    reason = str(row.get("unusable_reason") or "").strip().lower()
    if reason == UNUSABLE_REASON_EXCESSIVE_WATER:
        return True

    status = str(row.get("water_filter_status") or "").strip().lower()
    if status == WATER_FILTER_STATUS_EXCESSIVE:
        return True

    effective = _finite_float(row.get("effective_water_fraction"))
    if effective is None:
        satellite = row.get("satellite_water_fraction", row.get("water_fraction"))
        effective = compute_effective_water_fraction(
            satellite,
            row.get("terrain_sea_level_fraction"),
        )
    return is_excessive_water(effective)
