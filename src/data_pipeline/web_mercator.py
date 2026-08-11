"""
Web Mercator (EPSG:3857) grid definition and tile coordinate utilities.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Tuple

# Exact Web Mercator maximum latitude in degrees (~85.05112877980659)
WEB_MERCATOR_MAX_LAT: float = math.degrees(math.atan(math.sinh(math.pi)))

# WGS84 semi-major axis radius in meters used by Web Mercator (EPSG:3857)
EARTH_RADIUS: float = 6378137.0

# Half-circumference of the Earth in Web Mercator projection (~20037508.342789244 m)
HALF_CIRCUMFERENCE: float = math.pi * EARTH_RADIUS

# Standard CRS string
EPSG3857_CRS: str = "EPSG:3857"


def _validate_zoom(zoom: int) -> None:
    if not isinstance(zoom, int) or isinstance(zoom, bool) or not (0 <= zoom <= 22):
        raise ValueError(f"Zoom level must be an integer between 0 and 22, got {zoom}")


def _validate_tile_coords(x: int, y: int, zoom: int) -> None:
    _validate_zoom(zoom)
    n = 1 << zoom
    if not isinstance(x, int) or isinstance(x, bool) or not (0 <= x < n):
        raise ValueError(
            f"Tile x coordinate must be an integer in range [0, {n - 1}], got {x}"
        )
    if not isinstance(y, int) or isinstance(y, bool) or not (0 <= y < n):
        raise ValueError(
            f"Tile y coordinate must be an integer in range [0, {n - 1}], got {y}"
        )


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert lat/lon in degrees to tile coordinates (x, y) at given zoom level.

    Preserves clamped Mapbox semantics for lat/lon-to-tile.

    Args:
        lat: Latitude in degrees (-85.05112877980659 to 85.05112877980659)
        lon: Longitude in degrees (-180 to 180)
        zoom: Zoom level (0-22)

    Returns:
        Tuple of (x, y) tile coordinates
    """
    _validate_zoom(zoom)
    if (
        isinstance(lat, bool)
        or isinstance(lon, bool)
        or not isinstance(lat, (int, float))
        or not isinstance(lon, (int, float))
        or not (math.isfinite(lat) and math.isfinite(lon))
    ):
        raise ValueError(
            f"Latitude and longitude must be finite numbers, got lat={lat}, lon={lon}"
        )

    # Clamp latitude to valid range for Web Mercator
    clamped_lat = max(-WEB_MERCATOR_MAX_LAT, min(WEB_MERCATOR_MAX_LAT, float(lat)))
    lon_val = float(lon)

    lat_rad = math.radians(clamped_lat)
    n = 1 << zoom

    x = int((lon_val + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    # Clamp to valid tile range [0, n - 1]
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))

    return x, y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates to lat/lon (northwest corner of tile).

    Args:
        x: Tile x coordinate
        y: Tile y coordinate
        zoom: Zoom level (0-22)

    Returns:
        Tuple of (lat, lon) in degrees
    """
    _validate_tile_coords(x, y, zoom)
    n = 1 << zoom
    lon = (x / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def tile_to_lat_lon_center(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates to lat/lon (center of tile) using exact inverse Mercator.

    Args:
        x: Tile x coordinate
        y: Tile y coordinate
        zoom: Zoom level (0-22)

    Returns:
        Tuple of (lat, lon) for tile center (x + 0.5, y + 0.5)
    """
    _validate_tile_coords(x, y, zoom)
    n = 1 << zoom
    lon = ((x + 0.5) / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def tile_bounds_wgs84(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    Get WGS84 bounding box (min_lon, min_lat, max_lon, max_lat) in degrees for a tile.

    Args:
        x: Tile x coordinate
        y: Tile y coordinate
        zoom: Zoom level (0-22)

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) in degrees
    """
    _validate_tile_coords(x, y, zoom)
    n = 1 << zoom
    min_lon = (x / n) * 360.0 - 180.0
    max_lon = ((x + 1) / n) * 360.0 - 180.0
    nw_lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    se_lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
    max_lat = math.degrees(nw_lat_rad)
    min_lat = math.degrees(se_lat_rad)
    return min_lon, min_lat, max_lon, max_lat


def tile_bounds_web_mercator(
    x: int, y: int, zoom: int
) -> Tuple[float, float, float, float]:
    """
    Get exact EPSG:3857 bounding box (min_x, min_y, max_x, max_y) in meters for a tile.

    Args:
        x: Tile x coordinate
        y: Tile y coordinate
        zoom: Zoom level (0-22)

    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in EPSG:3857 meters
    """
    _validate_tile_coords(x, y, zoom)
    n = 1 << zoom
    tile_size = (2.0 * HALF_CIRCUMFERENCE) / n
    min_x = -HALF_CIRCUMFERENCE + x * tile_size
    max_x = -HALF_CIRCUMFERENCE + (x + 1) * tile_size
    max_y = HALF_CIRCUMFERENCE - y * tile_size
    min_y = HALF_CIRCUMFERENCE - (y + 1) * tile_size
    return min_x, min_y, max_x, max_y


def tile_transform_web_mercator(
    x: int, y: int, zoom: int, width: int = 256, height: int = 256
) -> Tuple[float, float, float, float, float, float]:
    """
    Get exact EPSG:3857 affine transform coefficients (a, b, c, d, e, f) for a tile image.

    Maps pixel (col, row) to EPSG:3857 coordinates:
        X = a * col + b * row + c
        Y = d * col + e * row + f

    Args:
        x: Tile x coordinate
        y: Tile y coordinate
        zoom: Zoom level (0-22)
        width: Tile pixel width (default 256)
        height: Tile pixel height (default 256)

    Returns:
        Tuple of (a, b, c, d, e, f) affine transform coefficients
    """
    _validate_tile_coords(x, y, zoom)
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width}")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height}")

    min_x, min_y, max_x, max_y = tile_bounds_web_mercator(x, y, zoom)
    a = (max_x - min_x) / width
    b = 0.0
    c = min_x
    d = 0.0
    e = -(max_y - min_y) / height
    f = max_y

    return a, b, c, d, e, f


@dataclass(frozen=True)
class TargetGrid:
    """
    Target grid definition for Web Mercator tile rasters.

    Stores z/x/y/width/height/crs/bounds/affine coefficients and provides
    deterministic serialization.
    """

    z: int
    x: int
    y: int
    width: int = 256
    height: int = 256
    crs: str = EPSG3857_CRS
    bounds: Tuple[float, float, float, float] = field(default_factory=tuple)
    affine: Tuple[float, float, float, float, float, float] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        _validate_tile_coords(self.x, self.y, self.z)
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
        ):
            raise ValueError(f"width must be a positive integer, got {self.width}")
        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
        ):
            raise ValueError(f"height must be a positive integer, got {self.height}")
        if not isinstance(self.crs, str) or self.crs != EPSG3857_CRS:
            raise ValueError(f"crs must be '{EPSG3857_CRS}', got {self.crs}")

        computed_bounds = tile_bounds_web_mercator(self.x, self.y, self.z)
        if not self.bounds:
            object.__setattr__(self, "bounds", computed_bounds)
        else:
            if isinstance(self.bounds, (list, tuple)):
                bounds_tuple = tuple(float(b) for b in self.bounds)
            else:
                raise ValueError(
                    f"Provided bounds must be a tuple or list of 4 floats, got {self.bounds}"
                )
            if len(bounds_tuple) != 4:
                raise ValueError(
                    f"Provided bounds must have 4 elements, got {self.bounds}"
                )
            if not all(
                math.isclose(b, c, rel_tol=1e-9, abs_tol=1e-9)
                for b, c in zip(bounds_tuple, computed_bounds)
            ):
                raise ValueError(
                    f"Provided bounds {self.bounds} do not match computed bounds {computed_bounds}"
                )
            object.__setattr__(self, "bounds", computed_bounds)

        computed_affine = tile_transform_web_mercator(
            self.x, self.y, self.z, self.width, self.height
        )
        if not self.affine:
            object.__setattr__(self, "affine", computed_affine)
        else:
            if isinstance(self.affine, (list, tuple)):
                affine_tuple = tuple(float(a) for a in self.affine)
            else:
                raise ValueError(
                    f"Provided affine must be a tuple or list of 6 floats, got {self.affine}"
                )
            if len(affine_tuple) != 6:
                raise ValueError(
                    f"Provided affine must have 6 elements, got {self.affine}"
                )
            if not all(
                math.isclose(a, c, rel_tol=1e-9, abs_tol=1e-9)
                for a, c in zip(affine_tuple, computed_affine)
            ):
                raise ValueError(
                    f"Provided affine {self.affine} does not match computed affine {computed_affine}"
                )
            object.__setattr__(self, "affine", computed_affine)

    @classmethod
    def from_tile(
        cls,
        z: int,
        x: int,
        y: int,
        width: int = 256,
        height: int = 256,
        crs: str = EPSG3857_CRS,
    ) -> TargetGrid:
        """Construct a TargetGrid for a tile coordinate."""
        return cls(z=z, x=x, y=y, width=width, height=height, crs=crs)

    @property
    def bounds_wgs84(self) -> Tuple[float, float, float, float]:
        """Get WGS84 bounding box (min_lon, min_lat, max_lon, max_lat)."""
        return tile_bounds_wgs84(self.x, self.y, self.z)

    @property
    def center_lat_lon(self) -> Tuple[float, float]:
        """Get tile center lat/lon (lat, lon)."""
        return tile_to_lat_lon_center(self.x, self.y, self.z)

    @property
    def nw_lat_lon(self) -> Tuple[float, float]:
        """Get tile northwest corner lat/lon (lat, lon)."""
        return tile_to_lat_lon(self.x, self.y, self.z)

    def to_dict(self) -> dict[str, Any]:
        """Convert TargetGrid to a canonical dictionary for serialization."""
        return {
            "z": self.z,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "bounds": list(self.bounds),
            "affine": list(self.affine),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetGrid:
        """Reconstruct TargetGrid from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict for TargetGrid.from_dict, got {type(data)}"
            )
        for k in ("z", "x", "y"):
            if k not in data:
                raise ValueError(f"TargetGrid dictionary missing required key '{k}'")
        raw_bounds = data.get("bounds")
        raw_affine = data.get("affine")
        bounds = (
            tuple(raw_bounds) if raw_bounds is not None and len(raw_bounds) == 4 else ()
        )
        affine = (
            tuple(raw_affine) if raw_affine is not None and len(raw_affine) == 6 else ()
        )
        return cls(
            z=int(data["z"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data.get("width", 256)),
            height=int(data.get("height", 256)),
            crs=str(data.get("crs", EPSG3857_CRS)),
            bounds=bounds,
            affine=affine,
        )

    def sha256(self) -> str:
        """Compute deterministic SHA256 hex digest from canonical JSON representation."""
        canonical_json = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
