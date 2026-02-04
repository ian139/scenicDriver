"""
Terrain Feature Extraction
Owner: progno-geospatial agent

Extracts features needed for scenic score calculation.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .analyzer import TerrainAnalyzer, BoundingBox


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


def extract_terrain_features(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    analyzer: TerrainAnalyzer = None
) -> TerrainFeatures:
    """
    Extract terrain features for a location.

    Args:
        lat, lon: Center coordinates
        radius_km: Analysis radius in kilometers
        analyzer: TerrainAnalyzer instance (created if None)

    Returns:
        TerrainFeatures for the location
    """
    if analyzer is None:
        analyzer = TerrainAnalyzer()

    # Create bounding box from center + radius
    # Approximate: 1 degree lat ≈ 111 km
    lat_delta = radius_km / 111
    lon_delta = radius_km / (111 * np.cos(np.radians(lat)))

    bbox = BoundingBox(
        min_lat=lat - lat_delta,
        min_lon=lon - lon_delta,
        max_lat=lat + lat_delta,
        max_lon=lon + lon_delta
    )

    # Get terrain metrics
    metrics = analyzer.analyze_region(bbox)

    # Calculate slope variation (normalized)
    slope_variation = min(metrics.slope_std / 15, 1.0)  # Cap at 15 degrees std

    # TODO: Implement water detection
    water_proximity = detect_water_proximity(bbox)
    coastal, has_lake, has_river = detect_water_features(bbox)

    # TODO: Implement vegetation density from NDVI
    vegetation_density = calculate_vegetation_density(bbox)

    return TerrainFeatures(
        slope_variation=slope_variation,
        elevation_change=metrics.elevation_range,
        water_proximity=water_proximity,
        vegetation_density=vegetation_density,
        coastal=coastal,
        has_lake=has_lake,
        has_river=has_river
    )


def detect_water_proximity(bbox: BoundingBox) -> float:
    """
    Detect proximity to water features.

    Uses OpenStreetMap or similar data to find:
    - Lakes
    - Rivers
    - Coastline

    Returns:
        Distance score 0-1 (1 = water present in bbox)
    """
    # TODO: Implement water detection
    # Options:
    # 1. Query OSM via Overpass API
    # 2. Use pre-downloaded water body shapefiles
    # 3. Detect from satellite imagery (blue channel analysis)
    return 0.0


def detect_water_features(bbox: BoundingBox) -> Tuple[bool, bool, bool]:
    """
    Detect specific water feature types.

    Returns:
        Tuple of (is_coastal, has_lake, has_river)
    """
    # TODO: Implement feature detection
    return False, False, False


def calculate_vegetation_density(bbox: BoundingBox) -> float:
    """
    Calculate vegetation density using NDVI.

    NDVI = (NIR - Red) / (NIR + Red)

    Requires multispectral imagery (e.g., Landsat, Sentinel-2).

    Returns:
        Average NDVI normalized to 0-1
    """
    # TODO: Implement NDVI calculation
    # 1. Get Landsat/Sentinel imagery for bbox
    # 2. Extract NIR and Red bands
    # 3. Calculate NDVI
    # 4. Return mean value
    return 0.5  # Default middle value
