"""
Raster processing module for Scenic Drive NAIP + USGS 3DEP transition.

Provides local-only raster processing against TargetGrid:
- Rejecting remote URI schemes and VSI virtual file prefixes
- Reprojecting and mosaicing imagery into uint8 RGB with declared resampling using rasterio
- Reprojecting float DEM and exact Terrain-RGB encoding/decoding with vertical datum verification
- Land geometry rasterization via Shapely and rasterio (no unverified fallbacks)
- Zero no-data validation on land pixels and outside-land filling after validation
- Deterministic QC and provenance stats calculation
- Deterministic atomic RGB PNG write and SHA-256 hash calculation
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
import urllib.parse
from typing import Any, Sequence

import numpy as np
from PIL import Image
import pyproj
import rasterio
import rasterio.features
from rasterio.enums import Resampling
import rasterio.transform
import rasterio.warp
import shapely.geometry
from shapely.ops import transform as shapely_transform

from src.data_pipeline.web_mercator import TargetGrid


class RemotePathError(ValueError):
    """Raised when a remote URI or GDAL virtual filesystem scheme is provided."""


class NoDataOnLandError(ValueError):
    """Raised when no-data pixels are detected on land."""


class TerrainRGBRangeError(ValueError):
    """Raised when float land elevation is non-finite or outside Terrain-RGB range."""


class VerticalDatumError(ValueError):
    """Raised when source and target vertical datums are incompatible or transformation is unimplemented."""


REMOTE_PREFIXES = (
    "http://",
    "https://",
    "s3://",
    "gs://",
    "az://",
    "ftp://",
    "ftps://",
    "/vsicurl",
    "/vsis3",
    "/vsigs",
    "/vsiang",
    "/vsiaz",
    "/vsioss",
    "/vsi",
)

WEB_MERCATOR_MAX_LAT: float = math.degrees(math.atan(math.sinh(math.pi)))


def validate_local_path(path: str | Path) -> Path:
    """
    Validates that a given file path is a local filesystem path.
    Raises RemotePathError (subclass of ValueError) if a remote scheme or VSI prefix is detected.
    """
    path_str = str(path).strip()
    path_lower = path_str.lower()

    for prefix in REMOTE_PREFIXES:
        if path_lower.startswith(prefix):
            raise RemotePathError(
                f"Remote path or virtual URI scheme rejected: {path_str}"
            )

    parsed = urllib.parse.urlparse(path_str)
    if parsed.scheme and parsed.scheme.lower() not in ("file", ""):
        raise RemotePathError(
            f"Unsupported remote URI scheme '{parsed.scheme}': {path_str}"
        )

    return Path(path_str)


def compute_file_sha256(path: str | Path) -> str:
    """Computes the SHA-256 hash of a local file in chunks."""
    local_p = validate_local_path(path)
    hasher = hashlib.sha256()
    with open(local_p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def encode_terrain_rgb(
    dem: np.ndarray, land_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Encodes float DEM (elevation in meters) to Terrain-RGB uint8 array of shape (3, H, W).
    Formula: q = floor((h + 10000) * 10 + 0.5) float64 math.
    Rejects non-finite or out-of-range land values if land_mask is provided.
    """
    dem_arr = np.asarray(dem, dtype=np.float64)

    land_bool = (
        np.ones(dem_arr.shape, dtype=bool)
        if land_mask is None
        else np.asarray(land_mask, dtype=bool)
    )
    if land_bool.shape != dem_arr.shape:
        raise ValueError("land_mask shape must match the DEM")
    land_values = dem_arr[land_bool]
    if not np.all(np.isfinite(land_values)):
        raise TerrainRGBRangeError(
            "Non-finite land elevation detected for Terrain-RGB encoding."
        )
    if np.any((land_values < -10000.0) | (land_values > 1667721.5)):
        raise TerrainRGBRangeError(
            "Land elevation is outside the Terrain-RGB representable range "
            "[-10000.0, 1667721.5]"
        )

    clean_dem = np.where(land_bool, dem_arr, -10000.0)

    q = np.floor((clean_dem + 10000.0) * 10.0 + 0.5).astype(np.int64)
    q = np.clip(q, 0, 16777215).astype(np.uint32)

    r = ((q // 65536) % 256).astype(np.uint8)
    g = ((q // 256) % 256).astype(np.uint8)
    b = (q % 256).astype(np.uint8)

    return np.stack([r, g, b], axis=0)


def decode_terrain_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Decodes Terrain-RGB uint8 array (shape (3, H, W) or (H, W, 3)) into float64 elevation array (shape (H, W)).
    Formula: h = -10000.0 + (R * 65536.0 + G * 256.0 + B) / 10.0
    Guaranteed accuracy: |decoded - h| <= 0.0500001 m.
    """
    rgb_arr = np.asarray(rgb, dtype=np.float64)

    if rgb_arr.ndim == 3 and rgb_arr.shape[0] == 3:
        r, g, b = rgb_arr[0], rgb_arr[1], rgb_arr[2]
    elif rgb_arr.ndim >= 3 and rgb_arr.shape[-1] == 3:
        r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    else:
        raise ValueError(f"Invalid shape for Terrain-RGB array: {rgb_arr.shape}")

    return -10000.0 + (r * 65536.0 + g * 256.0 + b) / 10.0


def write_atomic_png(
    rgb_array: np.ndarray, output_path: str | Path
) -> tuple[Path, str]:
    """
    Atomically writes uint8 RGB array (shape (3, H, W) or (H, W, 3)) to a PNG file.
    Returns (Path, sha256_hex).
    """
    path = validate_local_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(rgb_array, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Invalid RGB array shape: {arr.shape}")

    img = Image.fromarray(arr)

    # Atomic write to temporary file in same directory
    fd, tmp_file_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".png")
    os.close(fd)
    tmp_path = Path(tmp_file_path)

    try:
        img.save(tmp_path, format="PNG", optimize=False)
        sha256_hex = compute_file_sha256(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return path, sha256_hex


def rasterize_land_geometry(land_geometry: Any, target_grid: TargetGrid) -> np.ndarray:
    """
    Rasterizes land geometry onto target_grid returning a boolean land_mask array of shape (height, width).
    True indicates pixel center is on land.
    Requires Shapely and rasterio. Strict parsing without unverified fallbacks.
    """
    height = target_grid.height
    width = target_grid.width

    if land_geometry is None:
        raise ValueError("land_geometry is required")

    if isinstance(land_geometry, np.ndarray):
        if land_geometry.shape != (height, width):
            raise ValueError("land mask shape must match the target grid")
        return land_geometry.astype(bool, copy=True)

    if isinstance(land_geometry, dict):
        geom = shapely.geometry.shape(land_geometry)
    elif hasattr(land_geometry, "__geo_interface__"):
        geom = shapely.geometry.shape(land_geometry)
    elif isinstance(
        land_geometry,
        (
            shapely.geometry.base.BaseGeometry,
            shapely.geometry.Polygon,
            shapely.geometry.MultiPolygon,
        ),
    ):
        geom = land_geometry
    else:
        raise ValueError(
            f"Unsupported geometry format for rasterization: {type(land_geometry)}"
        )

    minx, miny, maxx, maxy = geom.bounds
    # If geometry coordinates appear to be WGS84 lat/lon degrees and target grid is EPSG:3857 (meters)
    target_crs = getattr(target_grid, "crs", "EPSG:3857")
    if (
        target_crs.upper() == "EPSG:3857"
        and -180.0 <= minx
        and maxx <= 180.0
        and -90.0 <= miny
        and maxy <= 90.0
    ):
        geom = shapely.clip_by_rect(
            geom, -180.0, -WEB_MERCATOR_MAX_LAT, 180.0, WEB_MERCATOR_MAX_LAT
        )
        proj = pyproj.Transformer.from_crs(
            "EPSG:4326", target_crs, always_xy=True
        ).transform
        geom = shapely_transform(proj, geom)

    transform = rasterio.transform.Affine(*target_grid.affine)
    mask = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def _load_and_reproject_raster(
    source_path: str | Path,
    target_grid: TargetGrid,
    bands_count: int,
    resampling: str,
    is_float: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads local raster file and reprojects it onto target_grid using rasterio.
    Returns (data_array, valid_mask_array).
    data_array shape: (bands_count, H, W)
    valid_mask_array shape: (H, W)
    """
    local_p = validate_local_path(source_path)
    height = target_grid.height
    width = target_grid.width
    dst_crs = target_grid.crs
    dst_transform = rasterio.transform.Affine(*target_grid.affine)

    try:
        resampl_enum = Resampling[resampling.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported raster resampling method: {resampling}") from exc

    with rasterio.open(local_p) as src:
        src_count = min(src.count, bands_count)

        if src.crs is None:
            raise ValueError(f"Source raster lacks CRS metadata: {local_p}")
        if src.transform == rasterio.transform.Affine.identity():
            raise ValueError(f"Source raster lacks a valid geotransform: {local_p}")
        src_crs = src.crs
        src_transform = src.transform

        if is_float:
            if src.count != 1 or not np.issubdtype(
                np.dtype(src.dtypes[0]), np.floating
            ):
                raise ValueError(
                    f"DEM source must be a single floating-point band: {local_p}"
                )
        elif src.count < bands_count or any(
            np.dtype(dtype) != np.dtype("uint8") for dtype in src.dtypes[:bands_count]
        ):
            raise ValueError(
                f"Imagery source must provide at least three uint8 bands: {local_p}"
            )
        src_valid = src.dataset_mask() > 0
        source_bands: list[np.ndarray] = []
        for b in range(1, src_count + 1):
            band_data = src.read(b).astype(np.float64)
            band_valid = np.isfinite(band_data)
            if src.nodata is not None:
                if np.isnan(src.nodata):
                    band_valid &= ~np.isnan(band_data)
                else:
                    band_valid &= band_data != src.nodata
            src_valid &= band_valid
            source_bands.append(band_data)

        reprojected = np.full((bands_count, height, width), np.nan, dtype=np.float64)
        for index, band_data in enumerate(source_bands):
            masked_source = band_data.copy()
            masked_source[~src_valid] = np.nan
            rasterio.warp.reproject(
                source=masked_source,
                destination=reprojected[index],
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                init_dest_nodata=True,
                resampling=resampl_enum,
            )

        for b in range(src_count, bands_count):
            reprojected[b] = reprojected[src_count - 1]
        valid_mask = np.all(np.isfinite(reprojected), axis=0)
        if is_float:
            out_data = reprojected
        else:
            out_data = np.rint(np.clip(reprojected, 0.0, 255.0))
            out_data[~np.isfinite(out_data)] = 0.0
            out_data = out_data.astype(np.uint8)
        return out_data, valid_mask


def _normalize_mosaic_order(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("mosaic_order must be 'first' or 'last'")
    normalized = value.strip().lower()
    if normalized not in {"first", "last"}:
        raise ValueError(
            f"Unsupported mosaic_order {value!r}; expected 'first' or 'last'"
        )
    return normalized


def process_imagery(
    source_paths: Sequence[str | Path],
    target_grid: TargetGrid,
    land_geometry: Any,
    resampling: str = "bilinear",
    fill_value: tuple[int, int, int] = (0, 0, 0),
    mosaic_order: str = "first",
    min_land_variance: float = 0.0,
    reject_all_black: bool = True,
    reject_all_white: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Reprojects and mosaics local imagery source files into uint8 RGB array (shape (3, H, W)).
    Validates zero no-data pixels on land; fills outside-land with fill_value after masking.
    Returns (rgb_array, qc_stats_dict).
    """
    validated_paths = [validate_local_path(p) for p in source_paths]
    if not validated_paths:
        raise ValueError(
            "At least one local source path is required for imagery processing."
        )
    normalized_mosaic_order = _normalize_mosaic_order(mosaic_order)

    height = target_grid.height
    width = target_grid.width

    target_data = np.zeros((3, height, width), dtype=np.uint8)
    target_valid = np.zeros((height, width), dtype=bool)
    source_contributions: list[dict[str, Any]] = []

    # Mosaic source rasters according to mosaic_order
    for p in validated_paths:
        sub_data, sub_valid = _load_and_reproject_raster(
            p, target_grid, 3, resampling, is_float=False
        )

        if normalized_mosaic_order == "first":
            update_mask = ~target_valid & sub_valid
        else:  # "last"
            update_mask = sub_valid

        target_data[:, update_mask] = sub_data[:, update_mask]
        target_valid |= sub_valid
        contributed_pixels = int(np.sum(update_mask))
        source_contributions.append(
            {
                "path": str(p),
                "pixel_count": contributed_pixels,
                "pixel_fraction": float(contributed_pixels / (height * width)),
            }
        )

    # Rasterize land geometry and validate zero no-data on land
    land_mask = rasterize_land_geometry(land_geometry, target_grid)
    nodata_land_count = int(np.sum(land_mask & ~target_valid))

    if nodata_land_count > 0:
        raise NoDataOnLandError(
            f"No-data validation failed: found {nodata_land_count} pixel(s) whose centers are on land with no valid data."
        )

    # Fill outside-land after land validation
    outside_land_mask = ~land_mask
    for c in range(3):
        target_data[c, outside_land_mask] = fill_value[c]
    land_pixel_count = int(np.sum(land_mask))

    land_data = target_data[:, land_mask] if land_pixel_count > 0 else target_data
    land_variance = float(np.var(land_data.astype(np.float64)))
    if reject_all_black and bool(np.all(land_data == 0)):
        raise ValueError("Satellite tile is all black over land")
    if reject_all_white and bool(np.all(land_data == 255)):
        raise ValueError("Satellite tile is all white over land")
    if not np.isfinite(min_land_variance) or min_land_variance < 0:
        raise ValueError("min_land_variance must be finite and non-negative")
    if land_variance < min_land_variance:
        raise ValueError(
            "Satellite tile land variance "
            f"{land_variance:.6f} is below {min_land_variance:.6f}"
        )
    # Compute deterministic QC and provenance stats
    total_pixel_count = height * width
    land_fraction = float(land_pixel_count / total_pixel_count)

    valid_pixel_count = int(np.sum(target_valid))
    source_hashes = {str(p): compute_file_sha256(p) for p in validated_paths}
    qc_stats = {
        "target_grid": target_grid.to_dict()
        if hasattr(target_grid, "to_dict")
        else str(target_grid),
        "source_files": [str(p) for p in validated_paths],
        "source_hashes": source_hashes,
        "total_pixels": total_pixel_count,
        "land_pixels": land_pixel_count,
        "outside_land_pixels": total_pixel_count - land_pixel_count,
        "land_fraction": land_fraction,
        "valid_pixels": valid_pixel_count,
        "valid_fraction": float(valid_pixel_count / total_pixel_count),
        "land_valid_fraction": 1.0,
        "source_contributions": source_contributions,
        "nodata_land_pixels": 0,
        "stats": {
            "r_mean": float(np.mean(land_data[0])) if land_pixel_count > 0 else 0.0,
            "g_mean": float(np.mean(land_data[1])) if land_pixel_count > 0 else 0.0,
            "b_mean": float(np.mean(land_data[2])) if land_pixel_count > 0 else 0.0,
            "land_variance": land_variance,
        },
        "parameters": {
            "resampling": resampling,
            "fill_value": list(fill_value),
            "mosaic_order": mosaic_order,
            "min_land_variance": min_land_variance,
            "reject_all_black": reject_all_black,
            "reject_all_white": reject_all_white,
        },
    }

    return target_data, qc_stats


def process_dem(
    source_paths: Sequence[str | Path],
    target_grid: TargetGrid,
    land_geometry: Any,
    resampling: str = "bilinear",
    outside_land_elevation: float = -10000.0,
    mosaic_order: str = "first",
    source_vertical_datum: str | Sequence[str] | None = "NAVD88",
    target_vertical_datum: str | None = "NAVD88",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Reprojects float DEM sources, validates land values, encodes to Terrain-RGB uint8 array (3, H, W).
    Rejects unimplemented vertical datum transformations.
    Fills outside-land with outside_land_elevation.
    Returns (terrain_rgb_array, float_dem_array, qc_stats_dict).
    """
    validated_paths = [validate_local_path(p) for p in source_paths]
    if not validated_paths:
        raise ValueError(
            "At least one local source path is required for DEM processing."
        )
    normalized_mosaic_order = _normalize_mosaic_order(mosaic_order)

    # Validate vertical datums
    tgt_vd = (target_vertical_datum or "NAVD88").strip().upper()
    if source_vertical_datum is not None:
        if isinstance(source_vertical_datum, str):
            src_vds = [source_vertical_datum]
        else:
            src_vds = list(source_vertical_datum)

        for src_vd in src_vds:
            if src_vd:
                normalized_src = src_vd.strip().upper()
                if normalized_src != tgt_vd:
                    raise VerticalDatumError(
                        f"Unsupported vertical datum transformation from '{src_vd}' to '{target_vertical_datum}'. Only matching vertical datums are supported."
                    )

    height = target_grid.height
    width = target_grid.width

    target_dem = np.full((height, width), np.nan, dtype=np.float64)
    target_valid = np.zeros((height, width), dtype=bool)
    source_contributions: list[dict[str, Any]] = []

    # Mosaic source float DEM rasters
    for p in validated_paths:
        sub_data, sub_valid = _load_and_reproject_raster(
            p, target_grid, 1, resampling, is_float=True
        )
        sub_dem = sub_data[0]

        if normalized_mosaic_order == "first":
            update_mask = ~target_valid & sub_valid
        else:  # "last"
            update_mask = sub_valid

        target_dem[update_mask] = sub_dem[update_mask]
        target_valid |= sub_valid
        contributed_pixels = int(np.sum(update_mask))
        source_contributions.append(
            {
                "path": str(p),
                "pixel_count": contributed_pixels,
                "pixel_fraction": float(contributed_pixels / (height * width)),
            }
        )

    # Rasterize land geometry and validate land pixel values
    land_mask = rasterize_land_geometry(land_geometry, target_grid)
    nodata_land_count = int(np.sum(land_mask & ~target_valid))

    if nodata_land_count > 0:
        raise NoDataOnLandError(
            f"No-data validation failed on land: found {nodata_land_count} pixel(s) whose centers are on land with no valid DEM data."
        )

    land_dem_values = target_dem[land_mask]
    if not np.all(np.isfinite(land_dem_values)):
        raise NoDataOnLandError(
            "Non-finite DEM elevation values (NaN/Inf) found on land."
        )

    if np.any((land_dem_values < -10000.0) | (land_dem_values > 1667721.5)):
        raise TerrainRGBRangeError(
            f"Land elevation out of Terrain-RGB encoding bounds [-10000.0, 1667721.5]. Min: {np.min(land_dem_values)}, Max: {np.max(land_dem_values)}"
        )

    # Fill outside-land after land validation
    outside_land_mask = ~land_mask
    target_dem[outside_land_mask] = outside_land_elevation

    # Encode float DEM into Terrain-RGB uint8 array
    terrain_rgb = encode_terrain_rgb(target_dem, land_mask=land_mask)

    # Compute deterministic QC and provenance stats
    land_pixel_count = int(np.sum(land_mask))
    total_pixel_count = height * width
    land_fraction = float(land_pixel_count / total_pixel_count)

    source_hashes = {str(p): compute_file_sha256(p) for p in validated_paths}

    qc_stats = {
        "target_grid": target_grid.to_dict()
        if hasattr(target_grid, "to_dict")
        else str(target_grid),
        "source_files": [str(p) for p in validated_paths],
        "source_hashes": source_hashes,
        "total_pixels": total_pixel_count,
        "land_pixels": land_pixel_count,
        "outside_land_pixels": total_pixel_count - land_pixel_count,
        "land_fraction": land_fraction,
        "valid_pixels": int(np.sum(target_valid)),
        "valid_fraction": float(np.sum(target_valid) / total_pixel_count),
        "land_valid_fraction": 1.0,
        "source_contributions": source_contributions,
        "nodata_land_pixels": 0,
        "stats": {
            "elevation_min": float(np.min(land_dem_values))
            if land_pixel_count > 0
            else 0.0,
            "elevation_max": float(np.max(land_dem_values))
            if land_pixel_count > 0
            else 0.0,
            "elevation_mean": float(np.mean(land_dem_values))
            if land_pixel_count > 0
            else 0.0,
        },
        "parameters": {
            "resampling": resampling,
            "outside_land_elevation": outside_land_elevation,
            "mosaic_order": mosaic_order,
            "vertical_datum": {
                "source": source_vertical_datum,
                "target": target_vertical_datum,
            },
        },
    }

    return terrain_rgb, target_dem, qc_stats
