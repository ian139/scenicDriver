"""
Tests for src/scenicdriver/data_pipeline/raster_processing.py.
Covers local-only path validation, remote URI and GDAL VSI rejection, exact Terrain-RGB
encoding/decoding, vertical datum rejection, tiny local raster reprojection, land geometry masking,
zero no-data on land validation, outside-land filling, mosaic order precedence, atomic PNG write/hash, and QC stats.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pytest
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

from scenicdriver.data_pipeline.web_mercator import TargetGrid
from scenicdriver.data_pipeline.raster_processing import (
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
from scenicdriver.data_pipeline.tile_inventory import scan_tile_inventory, validate_png_image


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
        rgb_terrain, dem_out, returned_mask, stats = process_dem(
            [dem_path],
            grid,
            land_mask,
            source_vertical_datum="NAVD88",
            target_vertical_datum="NAVD88",
        )
        assert rgb_terrain.shape == (3, 16, 16)
        np.testing.assert_array_equal(returned_mask, land_mask)


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

        rgb_out, returned_mask, stats = process_imagery(
            [img_path], grid, land_geojson, fill_value=(0, 0, 0)
        )
        assert rgb_out.shape == (3, 16, 16)
        assert stats["nodata_land_pixels"] == 0
        assert "source_hashes" in stats
        assert str(img_path) in stats["source_hashes"]

        # Outside land (bottom half) filled with fill_value (0, 0, 0)
        assert np.all(rgb_out[:, 8:16, :] == 0)
        assert returned_mask.dtype == bool
        assert np.all(returned_mask[:8]) and not np.any(returned_mask[8:])

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

        rgb_terrain, dem_out, returned_mask, stats = process_dem(
            [dem_path], grid, land_geojson
        )

        assert rgb_terrain.shape == (3, 16, 16)
        assert dem_out.shape == (16, 16)
        assert stats["nodata_land_pixels"] == 0
        assert stats["stats"]["elevation_min"] == 250.5

        # Outside land (bottom half) filled with -10000.0
        assert np.all(dem_out[8:16, :] == -10000.0)
        assert returned_mask.dtype == bool
        assert np.all(returned_mask[:8]) and not np.any(returned_mask[8:])


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
        _rgb, dem_out, returned_mask, stats = process_dem([dem_path], grid, land_geojson)
        assert stats["stats"]["elevation_min"] == pytest.approx(100.0)
        assert stats["stats"]["elevation_max"] == pytest.approx(100.0)
        land_mask = rasterize_land_geometry(land_geojson, grid)
        assert np.allclose(dem_out[land_mask], 100.0)
        np.testing.assert_array_equal(returned_mask, land_mask)


def test_mosaic_order_precedence():
    grid = TargetGrid.from_tile(z=0, x=0, y=0, width=16, height=16)
    land_mask = np.ones((16, 16), dtype=bool)

    with tempfile.TemporaryDirectory() as tmpdir:
        img1_arr = np.full((16, 16, 3), 100, dtype=np.uint8)
        img2_arr = np.full((16, 16, 3), 200, dtype=np.uint8)

        path1 = _create_georeferenced_tif(Path(tmpdir) / "img1.tif", img1_arr, grid)
        path2 = _create_georeferenced_tif(Path(tmpdir) / "img2.tif", img2_arr, grid)

        # "first" order: img1 takes precedence
        rgb_first, _, _ = process_imagery(
            [path1, path2], grid, land_mask, mosaic_order="first"
        )
        assert np.all(rgb_first == 100)

        # "last" order: img2 takes precedence
        rgb_last, _, _ = process_imagery(
            [path1, path2], grid, land_mask, mosaic_order="last"
        )
        assert np.all(rgb_last == 200)

        with pytest.raises(ValueError, match="Unsupported mosaic_order"):
            process_imagery([path1, path2], grid, land_mask, mosaic_order="frist")
        with pytest.raises(ValueError, match="Unsupported mosaic_order"):
            process_dem([path1, path2], grid, land_mask, mosaic_order="newest")


def test_raster_coast_mask_carries_land_only_features(tmp_path):
    from scenicdriver.terrain.features import compute_terrain_features, compute_terrain_valid_mask

    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=20, height=20)
    land = np.zeros((20, 20), dtype=bool)
    land[:, 10:] = True
    dem_path = _create_georeferenced_tif(tmp_path / "dem.tif", np.zeros((20, 20)), grid, is_float=True)
    image_path = _create_georeferenced_tif(
        tmp_path / "satellite.tif", np.full((20, 20, 3), [10, 80, 10], dtype=np.uint8), grid
    )
    terrain, dem, dem_mask, _ = process_dem([dem_path], grid, land)
    satellite, image_mask, _ = process_imagery([image_path], grid, land)
    np.testing.assert_array_equal(dem_mask, land)
    np.testing.assert_array_equal(image_mask, land)
    np.testing.assert_array_equal(compute_terrain_valid_mask(terrain.transpose(1, 2, 0)), land)
    result = compute_terrain_features(
        terrain.transpose(1, 2, 0), satellite.transpose(1, 2, 0), valid_mask=dem_mask,
        pixel_spacing_m=1.0,
    )
    assert result.relief == result.roughness == result.slope_mean == 0.0
    assert result.features.vegetation_density == pytest.approx(0.8)
    assert np.all(dem[~land] == -10000.0)


def test_all_sea_rasters_have_empty_masks_and_finite_qc(tmp_path):
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=4, height=4)
    land = np.zeros((4, 4), dtype=bool)
    dem_path = _create_georeferenced_tif(tmp_path / "dem.tif", np.zeros((4, 4)), grid, is_float=True)
    image_path = _create_georeferenced_tif(tmp_path / "satellite.tif", np.zeros((4, 4, 3)), grid)
    terrain, dem, dem_mask, dem_qc = process_dem([dem_path], grid, land)
    satellite, image_mask, image_qc = process_imagery([image_path], grid, land)
    assert not dem_mask.any() and not image_mask.any()
    assert not terrain.any() and not satellite.any()
    assert np.all(dem == -10000.0)
    assert all(np.isfinite(value) for value in dem_qc["stats"].values())
    assert all(np.isfinite(value) for value in image_qc["stats"].values())


@pytest.mark.parametrize("mask,error", [
    (np.ones((4, 4), dtype=np.uint8), TypeError),
    (np.ones((4, 3), dtype=bool), ValueError),
])
def test_raster_land_mask_rejects_dtype_or_shape_mismatch(mask, error):
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=4, height=4)
    with pytest.raises(error):
        rasterize_land_geometry(mask, grid)
    with pytest.raises(error):
        encode_terrain_rgb(np.zeros((4, 4)), land_mask=mask)


def test_processed_land_cannot_collide_with_reserved_black_fill(tmp_path):
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=4, height=4)
    dem_path = _create_georeferenced_tif(
        tmp_path / "dem.tif", np.full((4, 4), -10000.0), grid, is_float=True
    )
    with pytest.raises(TerrainRGBRangeError):
        process_dem([dem_path], grid, np.ones((4, 4), dtype=bool))


def test_imagery_per_dataset_mask_overrides_numeric_zero_nodata(tmp_path):
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=4, height=4)
    path = tmp_path / "masked.tif"
    data = np.full((3, 4, 4), 100, dtype=np.uint8)
    data[0, 1, 1] = 0  # genuine black red channel, declared valid by the mask
    data[:, 3, 3] = 0
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[3, 3] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=3,
            dtype="uint8",
            crs=grid.crs,
            transform=from_bounds(*grid.bounds, 4, 4),
            nodata=0,
        ) as dst:
            dst.write(data)
            dst.write_mask(mask)
    land = np.ones((4, 4), dtype=bool)
    land[3, 3] = False

    rgb, _mask, qc = process_imagery([path], grid, land, resampling="nearest")

    assert rgb[:, 1, 1].tolist() == [0, 100, 100]
    assert qc["valid_pixels"] == 15
    with pytest.raises(NoDataOnLandError):
        process_imagery([path], grid, np.ones((4, 4), dtype=bool), resampling="nearest")


@pytest.mark.parametrize(
    "scale,offset",
    [(0.1, 0.0), (1.0, 5.0), (float("nan"), 0.0), (1.0, float("inf"))],
)
def test_dem_rejects_nonidentity_or_nonfinite_scale_offset(tmp_path, scale, offset):
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=4, height=4)
    path = _create_georeferenced_tif(
        tmp_path / "scaled.tif", np.full((4, 4), 250.0), grid, is_float=True
    )
    with rasterio.open(path, "r+") as dst:
        dst.scales = (scale,)
        dst.offsets = (offset,)

    with pytest.raises(ValueError, match="scale/offset"):
        process_dem([path], grid, np.ones((4, 4), dtype=bool))


def test_dem_accepts_explicit_identity_scale_offset(tmp_path):
    grid = TargetGrid.from_tile(z=10, x=100, y=200, width=4, height=4)
    path = _create_georeferenced_tif(
        tmp_path / "identity.tif", np.full((4, 4), 250.0), grid, is_float=True
    )
    with rasterio.open(path, "r+") as dst:
        dst.scales = (1.0,)
        dst.offsets = (0.0,)

    _rgb, dem, _mask, _qc = process_dem([path], grid, np.ones((4, 4), dtype=bool))

    assert np.allclose(dem, 250.0)


def _save_tile(path: Path, image: Image.Image, fmt: str = "PNG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format=fmt)
    return path


@pytest.mark.parametrize(
    "image,fmt,reason",
    [
        (Image.new("RGB", (8, 8), "green"), "JPEG", "not_png:JPEG"),
        (Image.new("L", (8, 8), 120), "PNG", "unexpected_mode:L"),
        (Image.new("RGBA", (8, 8), "green"), "PNG", "unexpected_mode:RGBA"),
        (Image.new("RGB", (4, 8), "green"), "PNG", "unexpected_dimensions:4x8"),
    ],
)
def test_png_validation_rejects_non_png_non_rgb_or_wrong_size(
    tmp_path, image, fmt, reason
):
    path = _save_tile(tmp_path / "tile.png", image, fmt)

    result = validate_png_image(path, (8, 8))

    assert result["valid"] is False
    assert result["reason"] == reason


def test_png_validation_accepts_contract_sized_rgb_png(tmp_path):
    path = _save_tile(tmp_path / "tile.png", Image.new("RGB", (8, 8), "green"))

    result = validate_png_image(path, (8, 8))

    assert result["valid"] is True
    assert (result["width"], result["height"]) == (8, 8)


def test_inventory_does_not_count_wrong_size_tiles_as_valid(tmp_path):
    row = {"region": "fixture", "z": 14, "x": 1, "y": 2}
    for style in ("satellite", "terrain"):
        _save_tile(
            tmp_path / style / "z14" / "fixture" / "1_2.png",
            Image.new("RGB", (4, 4), "green"),
        )

    rows, counts = scan_tile_inventory(
        [row], image_root=tmp_path, expected_dimensions=(8, 8)
    )

    assert counts["complete_pairs"] == 0
    assert counts["invalid_files"] == 2
    assert rows[0]["satellite_reason"] == "unexpected_dimensions:4x4"


def _full_read_reprojection_oracle(path, grid, bands_count, resampling, is_float):
    """Frozen pre-windowing behavior, including effective GDAL mask precedence."""
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        valid = src.dataset_mask() > 0
        bands = [src.read(b).astype(np.float64) for b in range(1, bands_count + 1)]
        for b, data in enumerate(bands, 1):
            valid &= (src.read_masks(b) > 0) & np.isfinite(data)
        result = np.full((bands_count, grid.height, grid.width), np.nan)
        for index, data in enumerate(bands):
            data[~valid] = np.nan
            rasterio.warp.reproject(
                source=data,
                destination=result[index],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=rasterio.Affine(*grid.affine),
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                init_dest_nodata=True,
                resampling=Resampling[resampling],
            )
    valid = np.all(np.isfinite(result), axis=0)
    if not is_float:
        result = np.rint(np.clip(result, 0.0, 255.0))
        result[~np.isfinite(result)] = 0.0
        result = result.astype(np.uint8)
    return result, valid


@pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic", "lanczos", "average"])
@pytest.mark.parametrize("scale", [0.65, 2.5])
@pytest.mark.parametrize("kind", ["dem_nodata", "dem_mask", "imagery_nodata", "imagery_mask", "imagery_alpha"])
def test_window_reprojection_matches_full_read_oracle(tmp_path, resampling, scale, kind):
    from scenicdriver.data_pipeline import raster_processing as rp
    from rasterio.enums import ColorInterp

    grid = TargetGrid.from_tile(z=14, x=4800, y=6000, width=16, height=16)
    transform = (
        rasterio.Affine(*grid.affine)
        * rasterio.Affine.translation(-20.25, -19.5)
        * rasterio.Affine.scale(1 / scale)
    )
    row, col = np.indices((128, 128))
    is_float = kind.startswith("dem")
    bands_count = 1 if is_float else 3
    if is_float:
        data = (100 + np.sin(col / 3) * 50 + np.cos(row / 4) * 20)[None].astype("float32")
        nodata = -9999.0
    else:
        data = np.stack([(row * 7 + col * 11 + b * 41) % 255 + 1 for b in range(3)]).astype("uint8")
        nodata = 0
    top, left = int(19.5 * scale), int(20.25 * scale)
    # Invalid values inside the tile and on both sides of its footprint border
    # exercise mask precedence and kernel support beyond the requested tile.
    data[0, top + 2:top + 4, left + 2:left + 4] = nodata
    if is_float:
        data[0, top + 5, left + 5] = np.nan
        data[0, top + 6, left + 5] = np.inf
    if kind.endswith("alpha"):
        alpha = np.full((128, 128), 255, dtype="uint8")
        alpha[top - 1:top + 2, left - 1:left + 5] = 0
        data = np.concatenate([data, alpha[None]])
        nodata = None
    path = tmp_path / f"{kind}.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=128, height=128,
        count=data.shape[0], dtype=data.dtype, crs=grid.crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
        if kind.endswith("mask"):
            mask = np.full((128, 128), 255, dtype="uint8")
            mask[top - 1:top + 2, left - 1:left + 5] = 0
            dst.write_mask(mask)
        if kind.endswith("alpha"):
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)

    expected, expected_valid = _full_read_reprojection_oracle(
        path, grid, bands_count, resampling, is_float
    )
    actual, actual_valid = rp._load_and_reproject_raster(
        path, grid, bands_count, resampling, is_float
    )
    np.testing.assert_array_equal(actual_valid, expected_valid)
    np.testing.assert_array_equal(actual, expected)


def test_windowed_reads_are_bounded_and_disjoint_sources_are_not_read(tmp_path, monkeypatch):
    from scenicdriver.data_pipeline import raster_processing as rp

    grid = TargetGrid.from_tile(z=14, x=4800, y=6000, width=16, height=16)
    path = tmp_path / "large_dem.tif"
    transform = rasterio.Affine(*grid.affine) * rasterio.Affine.translation(-200, -200)
    with rasterio.open(
        path, "w", driver="GTiff", width=512, height=512, count=1,
        dtype="float32", crs=grid.crs, transform=transform,
    ) as dst:
        dst.write(np.full((512, 512), 123.0, dtype="float32"), 1)
    original_open = rasterio.open
    calls = []

    class TrackedReader:
        def __init__(self, reader):
            self.reader = reader

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def __getattr__(self, name):
            attr = getattr(self.reader, name)
            if name not in {"read", "read_masks", "dataset_mask"}:
                return attr

            def tracked(*args, **kwargs):
                window = kwargs.get("window")
                assert window is not None, f"{name} attempted an unbounded read"
                assert window.width <= 32 and window.height <= 32
                calls.append((name, window))
                return attr(*args, **kwargs)
            return tracked

    monkeypatch.setattr(rp.rasterio, "open", lambda *a, **kw: TrackedReader(original_open(*a, **kw)))
    data, valid = rp._load_and_reproject_raster(path, grid, 1, "lanczos", True)
    assert valid.all()
    np.testing.assert_allclose(data, 123.0)
    assert {name for name, _ in calls} == {"read", "read_masks", "dataset_mask"}
    calls.clear()
    disjoint = TargetGrid.from_tile(z=14, x=4900, y=6100, width=16, height=16)
    data, valid = rp._load_and_reproject_raster(path, disjoint, 1, "bilinear", True)
    assert not valid.any()
    assert np.isnan(data).all()
    assert not calls


@pytest.mark.parametrize("processor", [process_imagery, process_dem])
def test_verified_hashes_bypass_source_rehash(tmp_path, monkeypatch, processor):
    from scenicdriver.data_pipeline import raster_processing as rp

    grid = TargetGrid.from_tile(z=14, x=4800, y=6000, width=4, height=4)
    is_float = processor is process_dem
    data = np.full((4, 4) if is_float else (4, 4, 3), 123, dtype="float32" if is_float else "uint8")
    path = _create_georeferenced_tif(tmp_path / "source.tif", data, grid, is_float)
    verified = {str(path): rp.compute_file_sha256(path)}

    def fail_rehash(*args):
        pytest.fail("verified source content was rehashed")

    monkeypatch.setattr(rp, "compute_file_sha256", fail_rehash)
    for _ in range(2):
        *_, qc = processor([path], grid, np.ones((4, 4), dtype=bool), verified_source_hashes=verified)
        assert qc["source_hashes"] == verified


@pytest.mark.parametrize("resampling", ["nearest", "bilinear", "cubic", "lanczos", "average"])
@pytest.mark.parametrize("source_crs", ["EPSG:32618", "EPSG:4326"])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("at_source_edge", [False, True])
@pytest.mark.parametrize("is_float", [False, True])
def test_projected_window_and_source_edges_match_oracle(
    tmp_path, resampling, source_crs, rotated, at_source_edge, is_float
):
    from scenicdriver.data_pipeline import raster_processing as rp

    grid = TargetGrid.from_tile(z=14, x=4800, y=6000, width=16, height=16)
    left, bottom, right, top = rasterio.warp.transform_bounds(
        grid.crs, source_crs, *grid.bounds, densify_pts=21
    )
    dx, dy = right - left, top - bottom
    if at_source_edge:
        # The dataset ends inside the requested tile: missing coverage, edge
        # kernels, and dataset-border mask pixels must match the original.
        right -= dx / 4
        bottom += dy / 4
    else:
        left -= dx
        right += dx
        bottom -= dy
        top += dy
    transform = from_bounds(left, bottom, right, top, 128, 128)
    if rotated:
        transform *= (
            rasterio.Affine.translation(64, 64)
            * rasterio.Affine.rotation(12)
            * rasterio.Affine.translation(-64, -64)
        )
    row, col = np.indices((128, 128))
    count = 1 if is_float else 3
    data = np.stack([
        (100 + np.sin(col / 3) * 40 + np.cos(row / 4) * 20 + b * 20)
        for b in range(count)
    ]).astype("float32" if is_float else "uint8")
    data[0, 62:66, 62:66] = np.nan if is_float else 0
    mask = np.full((128, 128), 255, dtype="uint8")
    mask[0:2] = 0
    mask[:, -2:] = 0
    mask[68:72, 70:74] = 0
    path = tmp_path / "projected.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=128, height=128, count=count,
        dtype=data.dtype, crs=source_crs, transform=transform,
        nodata=np.nan if is_float else 0,
    ) as dst:
        dst.write(data)
        dst.write_mask(mask)
    expected, expected_valid = _full_read_reprojection_oracle(
        path, grid, count, resampling, is_float
    )
    actual, actual_valid = rp._load_and_reproject_raster(
        path, grid, count, resampling, is_float
    )
    np.testing.assert_array_equal(actual_valid, expected_valid)
    np.testing.assert_array_equal(actual, expected)
