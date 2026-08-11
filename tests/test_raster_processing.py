"""
Tests for src/data_pipeline/raster_processing.py.
Covers local-only path validation, remote URI and GDAL VSI rejection, exact Terrain-RGB
encoding/decoding, vertical datum rejection, tiny local raster reprojection, land geometry masking,
zero no-data on land validation, outside-land filling, mosaic order precedence, atomic PNG write/hash, and QC stats.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.data_pipeline.web_mercator import TargetGrid
from src.data_pipeline.raster_processing import (
    RemotePathError,
    NoDataOnLandError,
    TerrainRGBRangeError,
    VerticalDatumError,
    validate_local_path,
    encode_terrain_rgb,
    decode_terrain_rgb,
    write_atomic_png,
    rasterize_land_geometry,
    process_imagery,
    process_dem,
)


def _create_georeferenced_tif(
    path: Path,
    data: np.ndarray,
    grid: TargetGrid,
    is_float: bool = False,
    nodata: float | int | None = None,
) -> Path:
    """Helper to create a local GeoTIFF file properly georeferenced to grid."""
    height, width = data.shape[0], data.shape[1] if data.ndim == 2 else data.shape[1]
    count = 1 if data.ndim == 2 else data.shape[2]
    dtype = "float32" if is_float else "uint8"
    transform = from_bounds(*grid.bounds, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=grid.crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data.transpose(2, 0, 1))
    return path


def test_target_grid_contract():
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=64, height=64)
    assert grid.z == 10
    assert grid.x == 100
    assert grid.y == 200
    assert grid.width == 64
    assert grid.height == 64
    d = grid.to_dict()
    assert d["z"] == 10
    assert d["width"] == 64


def test_remote_path_rejection():
    remote_paths = [
        "http://example.com/naip.tif",
        "https://example.com/naip.tif",
        "s3://bucket/key.tif",
        "gs://bucket/key.tif",
        "az://container/key.tif",
        "/vsicurl/https://example.com/dem.tif",
        "/vsis3/bucket/key.tif",
        "/vsicurl_streaming/http://example.com/dem.tif",
    ]
    for p in remote_paths:
        with pytest.raises(RemotePathError):
            validate_local_path(p)

    # Local paths pass validation
    local_p = Path("/tmp/local_file.tif")
    assert validate_local_path(local_p) == local_p
    assert validate_local_path("relative/file.png") == Path("relative/file.png")


def test_vertical_datum_rejection():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)
    land_mask = np.ones((16, 16), dtype=bool)

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_arr = np.full((16, 16), 250.5, dtype=np.float32)
        dem_path = _create_georeferenced_tif(
            Path(tmpdir) / "dem.tif", dem_arr, grid, is_float=True
        )

        # Mismatched source vertical datum raises VerticalDatumError
        with pytest.raises(VerticalDatumError):
            process_dem(
                [dem_path],
                grid,
                land_mask,
                source_vertical_datum="NGVD29",
                target_vertical_datum="NAVD88",
            )

        # Matching vertical datum passes
        rgb_terrain, dem_out, stats = process_dem(
            [dem_path],
            grid,
            land_mask,
            source_vertical_datum="NAVD88",
            target_vertical_datum="NAVD88",
        )
        assert rgb_terrain.shape == (3, 16, 16)


def test_terrain_rgb_encode_decode_precision():
    test_elevations = np.array(
        [
            [-10000.0, -500.25, 0.0],
            [123.456, 4400.123, 8848.86],
            [10.0, 1000.0, 1667721.5],
        ],
        dtype=np.float64,
    )
    encoded = encode_terrain_rgb(test_elevations)
    assert encoded.shape == (3, 3, 3)
    assert encoded.dtype == np.uint8

    decoded = decode_terrain_rgb(encoded)
    assert decoded.shape == (3, 3)
    diff = np.abs(decoded - test_elevations)
    assert np.max(diff) <= 0.0500001, f"Max diff {np.max(diff)} exceeded 0.0500001 m"


def test_terrain_rgb_range_and_non_finite_rejection():
    land_mask = np.array([[True, False], [False, True]])

    # Non-finite on land
    dem_nan = np.array([[np.nan, 100.0], [50.0, 200.0]])
    with pytest.raises(TerrainRGBRangeError):
        encode_terrain_rgb(dem_nan, land_mask=land_mask)

    # Out of lower range on land (< -10000 m)
    dem_too_low = np.array([[-10005.0, 100.0], [50.0, 200.0]])
    with pytest.raises(TerrainRGBRangeError):
        encode_terrain_rgb(dem_too_low, land_mask=land_mask)

    # Out of upper range on land (> 1667721.5 m)
    dem_too_high = np.array([[100.0, 100.0], [50.0, 2000000.0]])
    with pytest.raises(TerrainRGBRangeError):
        encode_terrain_rgb(dem_too_high, land_mask=land_mask)

    # Non-finite outside land is safe if land is clean
    dem_outside_nan = np.array([[100.0, np.nan], [np.nan, 200.0]])
    encoded = encode_terrain_rgb(dem_outside_nan, land_mask=land_mask)
    assert encoded.shape == (3, 2, 2)


def test_write_atomic_png_and_hash():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "output" / "test_tile.png"
        arr = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)

        final_p, sha256_hex1 = write_atomic_png(arr, out_path)
        assert final_p.exists()
        assert len(sha256_hex1) == 64

        # Verify determinism: identical array produces identical SHA256 hash
        out_path2 = Path(tmpdir) / "output" / "test_tile2.png"
        _, sha256_hex2 = write_atomic_png(arr, out_path2)
        assert sha256_hex1 == sha256_hex2


def test_land_geometry_rasterization():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)

    # Polygon covering top half of WGS84 world (-180..180 lon, 0..85 lat)
    land_geojson = {
        "type": "Polygon",
        "coordinates": [
            [[-180.0, 0.0], [180.0, 0.0], [180.0, 85.0], [-180.0, 85.0], [-180.0, 0.0]]
        ],
    }
    land_mask = rasterize_land_geometry(land_geojson, grid)
    assert land_mask.shape == (16, 16)
    assert land_mask.dtype == bool
    # Top 8 rows out of 16 are True
    assert np.sum(land_mask) == 128


def test_process_imagery_and_no_data_validation():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a 16x16 uint8 image covering top half
        img_arr = np.full((16, 16, 3), 128, dtype=np.uint8)
        img_arr[0:8, :, 0] = 200  # distinct red channel in top half
        img_path = _create_georeferenced_tif(
            Path(tmpdir) / "source_img.tif", img_arr, grid
        )

        # Polygon covering top half
        land_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-180.0, 0.0],
                    [180.0, 0.0],
                    [180.0, 85.0],
                    [-180.0, 85.0],
                    [-180.0, 0.0],
                ]
            ],
        }

        rgb_out, stats = process_imagery(
            [img_path], grid, land_geojson, fill_value=(0, 0, 0)
        )
        assert rgb_out.shape == (3, 16, 16)
        assert stats["nodata_land_pixels"] == 0
        assert "source_hashes" in stats
        assert str(img_path) in stats["source_hashes"]

        # Outside land (bottom half) filled with fill_value (0, 0, 0)
        assert np.all(rgb_out[:, 8:16, :] == 0)

        # Reject remote paths in process_imagery
        with pytest.raises(RemotePathError):
            process_imagery(["s3://bucket/remote.tif"], grid, land_geojson)


def test_process_dem_and_qc_stats():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_arr = np.full((16, 16), 250.5, dtype=np.float32)
        dem_path = _create_georeferenced_tif(
            Path(tmpdir) / "dem.tif", dem_arr, grid, is_float=True
        )

        # Polygon covering top half of world
        land_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-180.0, 0.0],
                    [180.0, 0.0],
                    [180.0, 85.0],
                    [-180.0, 85.0],
                    [-180.0, 0.0],
                ]
            ],
        }

        rgb_terrain, dem_out, stats = process_dem(
            [dem_path], grid, land_geojson, outside_land_elevation=-10000.0
        )

        assert rgb_terrain.shape == (3, 16, 16)
        assert dem_out.shape == (16, 16)
        assert stats["nodata_land_pixels"] == 0
        assert stats["stats"]["elevation_min"] == 250.5

        # Outside land (bottom half) filled with -10000.0
        assert np.all(dem_out[8:16, :] == -10000.0)


def test_process_dem_nodata_on_land_failure():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_arr = np.full((16, 16), 250.5, dtype=np.float32)
        dem_arr[2, 2] = np.nan  # NaN on land
        dem_path = _create_georeferenced_tif(
            Path(tmpdir) / "dem_nan.tif", dem_arr, grid, is_float=True, nodata=-9999.0
        )

        land_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-180.0, 0.0],
                    [180.0, 0.0],
                    [180.0, 85.0],
                    [-180.0, 85.0],
                    [-180.0, 0.0],
                ]
            ],
        }

        with pytest.raises(NoDataOnLandError):
            process_dem([dem_path], grid, land_geojson)


def test_dem_resampling_does_not_bleed_nodata_into_valid_land() -> None:
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)
    with tempfile.TemporaryDirectory() as tmpdir:
        dem_arr = np.full((4, 4), 100.0, dtype=np.float32)
        dem_arr[:, 0] = -9999.0
        dem_path = _create_georeferenced_tif(
            Path(tmpdir) / "dem_nodata_edge.tif",
            dem_arr,
            grid,
            is_float=True,
            nodata=-9999.0,
        )
        land_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-90.0, -85.0],
                    [180.0, -85.0],
                    [180.0, 85.0],
                    [-90.0, 85.0],
                    [-90.0, -85.0],
                ]
            ],
        }
        _rgb, dem_out, stats = process_dem([dem_path], grid, land_geojson)
        assert stats["stats"]["elevation_min"] == pytest.approx(100.0)
        assert stats["stats"]["elevation_max"] == pytest.approx(100.0)
        land_mask = rasterize_land_geometry(land_geojson, grid)
        assert np.allclose(dem_out[land_mask], 100.0)


def test_mosaic_order_precedence():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)
    land_mask = np.ones((16, 16), dtype=bool)

    with tempfile.TemporaryDirectory() as tmpdir:
        img1_arr = np.full((16, 16, 3), 100, dtype=np.uint8)
        img2_arr = np.full((16, 16, 3), 200, dtype=np.uint8)

        path1 = _create_georeferenced_tif(Path(tmpdir) / "img1.tif", img1_arr, grid)
        path2 = _create_georeferenced_tif(Path(tmpdir) / "img2.tif", img2_arr, grid)

        # "first" order: img1 takes precedence
        rgb_first, _ = process_imagery(
            [path1, path2], grid, land_mask, mosaic_order="first"
        )
        assert np.all(rgb_first == 100)

        # "last" order: img2 takes precedence
        rgb_last, _ = process_imagery(
            [path1, path2], grid, land_mask, mosaic_order="last"
        )
        assert np.all(rgb_last == 200)

        with pytest.raises(ValueError, match="Unsupported mosaic_order"):
            process_imagery([path1, path2], grid, land_mask, mosaic_order="frist")
        with pytest.raises(ValueError, match="Unsupported mosaic_order"):
            process_dem([path1, path2], grid, land_mask, mosaic_order="newest")
