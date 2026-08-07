"""Canonical water classification constants and metrics for active learning."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image
from typing import Any, Mapping

MAX_SELECTABLE_WATER_FRACTION: float = 0.50
UNUSABLE_REASON_EXCESSIVE_WATER: str = "excessive_water"

WATER_FILTER_STATUS_PASS: str = "pass"
WATER_FILTER_STATUS_EXCESSIVE: str = "excessive_water"
WATER_FILTER_STATUS_UNKNOWN: str = "unknown"


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def compute_satellite_water_fraction(image: Image.Image | np.ndarray) -> float:
    """Estimate open-water coverage from low-texture blue satellite pixels."""
    if isinstance(image, Image.Image):
        values = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 3:
        raise ValueError(f"Expected RGB image array, got shape {values.shape}")
    values = values[..., :3] / 255.0
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    brightness = (r + g + b) / 3.0
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    saturation = (maxc - minc) / np.maximum(maxc + 1e-6, 1e-6)
    texture = float((0.2989 * r + 0.5870 * g + 0.1140 * b).std())
    water_mask = (
        (b > r * 1.2)
        & (b > g * 1.15)
        & (brightness < 0.65)
        & (saturation > 0.18)
        & (texture < 0.12)
    )
    return float(water_mask.mean())


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
