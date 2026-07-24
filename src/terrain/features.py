"""Terrain-RGB feature extraction for scenic scoring."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image


_SEAM_ZERO_ELEVATION = 0.0
_SEAM_ZERO_ATOL = 0.05
_SEAM_MAX_FRACTION = 0.02


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
        return np.array([
            self.slope_variation,
            self.elevation_change / 1000,  # Normalize to km
            self.water_proximity,
            self.vegetation_density,
            float(self.coastal),
            float(self.has_lake or self.has_river)
        ])




def extract_terrain_features_from_tile(
    terrain_path: Path,
    satellite_path: Optional[Path] = None,
) -> TerrainFeatures:
    """
    Extract terrain features from a Mapbox terrain-rgb tile (and optional satellite tile).

    Args:
        terrain_path: Path to terrain-rgb PNG tile
        satellite_path: Optional satellite tile for vegetation proxy

    Returns:
        TerrainFeatures for the tile
    """
    terrain_img = Image.open(terrain_path).convert("RGB")
    elev = repair_terrain_zero_seam(_decode_terrain_rgb(terrain_img))

    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx ** 2 + gy ** 2)

    relief = float(elev.max() - elev.min())
    slope_variation = float(min(slope.std() / 15.0, 1.0))

    low_elev = elev < np.percentile(elev, 10)
    flat = slope < np.percentile(slope, 10)
    water_proximity = float((low_elev & flat).mean())

    vegetation_density = 0.5
    if satellite_path is not None and Path(satellite_path).exists():
        sat_img = Image.open(satellite_path).convert("RGB")
        sat_arr = np.array(sat_img).astype(np.float32)
        r = sat_arr[..., 0]
        g = sat_arr[..., 1]
        b = sat_arr[..., 2]
        vegetation_density = float(_safe_div(g, r + g + b).mean())

    return TerrainFeatures(
        slope_variation=slope_variation,
        elevation_change=relief,
        water_proximity=water_proximity,
        vegetation_density=vegetation_density,
        coastal=False,
        has_lake=False,
        has_river=False,
    )


def _decode_terrain_rgb(terrain_img: Image.Image) -> np.ndarray:
    arr = np.array(terrain_img).astype(np.float32)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1


def repair_terrain_zero_seam(elev: np.ndarray) -> np.ndarray:
    """
    Repair thin zero-elevation seams that appear on tile borders.

    Some terrain-rgb tiles contain a 1-pixel border with elevation==0 due to
    source seam artifacts. That can inflate relief/slope metrics. This repair
    only applies when zero pixels are sparse and strictly on the border.
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
    return num / (np.maximum(den, 1e-6))
