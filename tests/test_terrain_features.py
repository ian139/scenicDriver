from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenicdriver.data_pipeline.web_mercator import tile_pixel_ground_spacing_m
from scenicdriver.terrain.features import (
    TerrainFeatures,
    compute_terrain_features,
    compute_terrain_sea_level_fraction,
    compute_terrain_valid_mask,
    decode_terrain_rgb,
    repair_terrain_zero_seam,
)


def _terrain_rgb() -> np.ndarray:
    # Distinct channels and spatial values catch channel swaps and dispatch drift.
    return np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[90, 80, 70], [60, 50, 40], [30, 20, 10]],
        ],
        dtype=np.uint8,
    )


def test_decode_terrain_rgb_uses_mapbox_channel_order_and_formula():
    rgb = np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)

    expected = np.array(
        [
            [
                -10000.0 + (1 * 256 * 256 + 2 * 256 + 3) * 0.1,
                -10000.0 + (10 * 256 * 256 + 20 * 256 + 30) * 0.1,
            ]
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(decode_terrain_rgb(rgb), expected)


def test_compute_dispatch_is_identical_for_ndarray_pil_and_path(tmp_path: Path):
    rgb = _terrain_rgb()
    satellite = np.array(
        [
            [[10, 20, 30], [20, 40, 80], [5, 10, 5]],
            [[30, 60, 10], [40, 20, 10], [80, 10, 20]],
            [[10, 10, 80], [20, 30, 40], [50, 70, 10]],
        ],
        dtype=np.uint8,
    )
    terrain_path = tmp_path / "terrain.png"
    satellite_path = tmp_path / "satellite.png"
    Image.fromarray(rgb).save(terrain_path)
    Image.fromarray(satellite).save(satellite_path)

    valid = np.ones(rgb.shape[:2], dtype=bool)
    array_result = compute_terrain_features(rgb, satellite, valid_mask=valid, pixel_spacing_m=2.0)
    pil_result = compute_terrain_features(
        Image.open(terrain_path), Image.open(satellite_path), valid_mask=valid, pixel_spacing_m=2.0
    )
    path_result = compute_terrain_features(
        terrain_path, satellite_path, valid_mask=valid, pixel_spacing_m=2.0
    )

    for result in (pil_result, path_result):
        np.testing.assert_allclose(
            result.features.to_array(), array_result.features.to_array()
        )
        np.testing.assert_allclose(
            [result.relief, result.roughness, result.slope_mean],
            [array_result.relief, array_result.roughness, array_result.slope_mean],
        )


def test_zero_border_seam_is_repaired_without_changing_interior():
    elevations = np.full((100, 100), 123.0, dtype=np.float32)
    elevations[:, 0] = 0.0

    repaired = repair_terrain_zero_seam(elevations, valid_mask=np.ones_like(elevations, dtype=bool))

    assert np.all(repaired[:, 0] == 123.0)
    np.testing.assert_array_equal(repaired[:, 1:], elevations[:, 1:])


def test_satellite_vegetation_uses_green_channel_and_optional_default():
    terrain = np.full((2, 2, 3), [20, 30, 40], dtype=np.uint8)
    satellite = np.array(
        [
            [[10, 20, 70], [20, 40, 40]],
            [[50, 25, 25], [80, 10, 10]],
        ],
        dtype=np.uint8,
    )
    expected_density = np.mean([20 / 100, 40 / 100, 25 / 100, 10 / 100])

    valid = np.ones(terrain.shape[:2], dtype=bool)
    result = compute_terrain_features(terrain, satellite, valid_mask=valid, pixel_spacing_m=1.0)
    default_result = compute_terrain_features(terrain, valid_mask=valid, pixel_spacing_m=1.0)

    assert np.isclose(result.features.vegetation_density, expected_density)
    assert default_result.features.vegetation_density == 0.5


def test_feature_vector_matches_canonical_scalar_metrics():
    result = compute_terrain_features(
        _terrain_rgb(), valid_mask=np.ones((3, 3), dtype=bool), pixel_spacing_m=1.0
    )
    features = result.features

    expected_vector = np.array(
        [
            features.slope_variation,
            features.elevation_change / 1000,
            features.water_proximity,
            features.vegetation_density,
            float(features.coastal),
            float(features.has_lake or features.has_river),
        ]
    )

    np.testing.assert_allclose(features.to_array(), expected_vector)
    assert np.isclose(features.elevation_change, result.relief)
    assert np.isfinite(result.roughness)
    assert np.isfinite(result.slope_mean)


def test_terrain_features_to_array_preserves_boolean_water_contract():
    features = TerrainFeatures(0.25, 80.0, 0.5, 0.75, True, False, True)

    np.testing.assert_allclose(features.to_array(), [0.25, 0.08, 0.5, 0.75, 1.0, 1.0])


def test_compute_terrain_sea_level_fraction_zero_and_high_elevation():
    # Mapbox Terrain-RGB: elevation 0.0 corresponds to R=1, G=134, B=160
    sea_level_rgb = np.full((10, 10, 3), [1, 134, 160], dtype=np.uint8)
    assert compute_terrain_sea_level_fraction(sea_level_rgb, valid_mask=np.ones((10, 10), dtype=bool)) == 1.0

    # High land elevation
    land_rgb = np.full((10, 10, 3), [1, 138, 112], dtype=np.uint8)
    assert compute_terrain_sea_level_fraction(land_rgb, valid_mask=np.ones((10, 10), dtype=bool)) == 0.0


def test_compute_terrain_sea_level_fraction_border_seam_resistance():
    # Land elevation with 1-pixel zero-elevation border seam
    land_rgb = np.full((100, 100, 3), [1, 138, 112], dtype=np.uint8)
    land_rgb[:, 0] = [1, 134, 160]
    assert compute_terrain_sea_level_fraction(land_rgb, valid_mask=np.ones((100, 100), dtype=bool)) == 0.0


def test_compute_terrain_sea_level_fraction_mixed_tile():
    # Half sea level, half land
    mixed_rgb = np.full((10, 10, 3), [1, 138, 112], dtype=np.uint8)
    mixed_rgb[:5, :] = [1, 134, 160]
    assert compute_terrain_sea_level_fraction(mixed_rgb, valid_mask=np.ones((10, 10), dtype=bool)) == 0.5


def test_coastal_fill_does_not_create_relief_or_roughness():
    terrain = np.full((20, 20, 3), [1, 134, 160], dtype=np.uint8)
    terrain[:, :10] = 0
    result = compute_terrain_features(
        terrain, valid_mask=compute_terrain_valid_mask(terrain), pixel_spacing_m=1.0
    )
    assert result.relief == 0.0
    assert result.roughness == 0.0
    assert result.slope_mean == 0.0
    assert result.features.slope_variation == 0.0


@pytest.mark.parametrize("single_valid", [False, True])
def test_empty_or_isolated_land_has_finite_neutral_terrain(single_valid):
    terrain = np.zeros((1, 5, 3), dtype=np.uint8)
    valid = np.zeros((1, 5), dtype=bool)
    if single_valid:
        terrain[0, 2] = [1, 138, 112]
        valid[0, 2] = True
    result = compute_terrain_features(terrain, valid_mask=valid, pixel_spacing_m=1.0)
    assert result.relief == result.roughness == result.slope_mean == 0.0
    assert np.isfinite(result.features.to_array()).all()
    if not single_valid:
        np.testing.assert_array_equal(result.features.to_array(), np.zeros(6))
        assert compute_terrain_sea_level_fraction(terrain, valid_mask=valid) == 0.0


def test_invalid_neighborhoods_do_not_influence_valid_terrain_or_vegetation():
    from scenicdriver.data_pipeline.raster_processing import encode_terrain_rgb

    valid = np.zeros((20, 20), dtype=bool)
    valid[:, 5:15] = True
    elevation = np.tile(np.arange(20, dtype=float), (20, 1))
    terrain = encode_terrain_rgb(elevation).transpose(1, 2, 0)
    satellite = np.full((20, 20, 3), [10, 80, 10], dtype=np.uint8)
    reference = compute_terrain_features(
        terrain[:, 5:15], satellite[:, 5:15], valid_mask=valid[:, 5:15], pixel_spacing_m=1.0
    )
    terrain[~valid] = [255, 255, 255]
    satellite[~valid] = [255, 0, 0]
    masked = compute_terrain_features(terrain, satellite, valid_mask=valid, pixel_spacing_m=1.0)
    np.testing.assert_allclose(masked.features.to_array(), reference.features.to_array())
    assert masked.relief == pytest.approx(9.0)
    assert masked.roughness == pytest.approx(reference.roughness)
    # 1 m rise per 1 m pixel is a 45 degree slope.
    assert masked.slope_mean == pytest.approx(45.0)


def test_seam_repair_cannot_copy_elevation_from_invalid_neighbor():
    elevation = np.full((100, 100), 123.0)
    elevation[:, 0] = 0.0
    elevation[:, 1] = -10000.0
    valid = np.ones((100, 100), dtype=bool)
    valid[:, 1] = False
    repaired = repair_terrain_zero_seam(elevation, valid_mask=valid)
    np.testing.assert_array_equal(repaired[:, 0], np.zeros(100))


@pytest.mark.parametrize("mask,error", [
    (np.ones((2, 2), dtype=np.uint8), TypeError),
    (np.ones((2, 1), dtype=bool), ValueError),
])
def test_terrain_rejects_invalid_mask_contract(mask, error):
    terrain = np.full((2, 2, 3), [1, 134, 160], dtype=np.uint8)
    with pytest.raises(error):
        compute_terrain_features(terrain, valid_mask=mask, pixel_spacing_m=1.0)


def test_terrain_rejects_satellite_shape_mismatch():
    terrain = np.full((2, 2, 3), [1, 134, 160], dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_terrain_features(
            terrain, np.zeros((1, 2, 3)), valid_mask=np.ones((2, 2), dtype=bool), pixel_spacing_m=1.0
        )


def _float_terrain_rgb(elevation: np.ndarray) -> np.ndarray:
    """Unquantized Terrain-RGB carrier: decode() recovers ``elevation`` exactly."""
    rgb = np.zeros((*elevation.shape, 3), dtype=np.float64)
    rgb[..., 2] = (elevation + 10000.0) / 0.1
    return rgb


def _physical_surface(spacing: float, size: int = 32) -> np.ndarray:
    """A kinked ramp defined in ground meters, sampled at ``spacing`` m/pixel."""
    rows, cols = np.indices((size, size), dtype=float)
    east, north = cols * spacing, rows * spacing
    kink = (size // 2) * spacing
    east_rise = np.where(east < kink, 0.1 * east, 0.1 * kink + 0.6 * (east - kink))
    return 500.0 + east_rise + 0.25 * north


def test_pixel_ground_spacing_scales_with_zoom_and_latitude():
    equator = tile_pixel_ground_spacing_m(0, 0, 0)
    # Single z0 tile is centered on the equator: full circumference / 256 px.
    assert equator == pytest.approx(2 * np.pi * 6378137.0 / 256)
    # z1 tile (0, 0) is centered at atan(sinh(pi / 2)) north; ground pixel = cos(lat) * projected.
    center_lat = np.arctan(np.sinh(np.pi / 2))
    assert tile_pixel_ground_spacing_m(0, 0, 1, 512, 512) == pytest.approx(
        2 * np.pi * 6378137.0 / 1024 * np.cos(center_lat)
    )
    z14 = tile_pixel_ground_spacing_m(4945, 5975, 14)
    z16 = tile_pixel_ground_spacing_m(19780, 23900, 16)
    assert z14 > 3 * z16
    with pytest.raises(ValueError, match="square"):
        tile_pixel_ground_spacing_m(4945, 5975, 14, 256, 128)


def test_same_physical_terrain_yields_same_slope_features_across_zoom_and_latitude():
    spacings = [
        tile_pixel_ground_spacing_m(4945, 5975, 14),  # ~43N, ~7 m/px
        tile_pixel_ground_spacing_m(19073, 26793, 16),  # ~30N, ~2 m/px
    ]
    assert spacings[0] > 3 * spacings[1]
    valid = np.ones((32, 32), dtype=bool)
    results = [
        compute_terrain_features(
            _float_terrain_rgb(_physical_surface(spacing)), valid_mask=valid,
            pixel_spacing_m=spacing,
        )
        for spacing in spacings
    ]
    assert results[0].slope_mean == pytest.approx(results[1].slope_mean, rel=1e-4)
    assert results[0].features.slope_variation == pytest.approx(
        results[1].features.slope_variation, rel=1e-4
    )
    assert results[0].features.slope_variation > 0.0
    # Gentle half: hypot(0.1, 0.25) rise -> ~15.07 deg; steep half ~32.97 deg.
    assert np.degrees(np.arctan(np.hypot(0.1, 0.25))) < results[0].slope_mean
    assert results[0].slope_mean < np.degrees(np.arctan(np.hypot(0.6, 0.25)))
    # Unscaled (wrong-resolution) spacing would disagree.
    wrong = compute_terrain_features(
        _float_terrain_rgb(_physical_surface(spacings[1])), valid_mask=valid,
        pixel_spacing_m=spacings[0],
    )
    assert abs(wrong.slope_mean - results[1].slope_mean) > 5.0


@pytest.mark.parametrize("spacing,error", [
    (None, TypeError),
    (True, TypeError),
    ("2.0", TypeError),
    (0.0, ValueError),
    (-1.0, ValueError),
    (float("nan"), ValueError),
    (float("inf"), ValueError),
])
def test_terrain_requires_explicit_positive_pixel_spacing(spacing, error):
    terrain = np.full((2, 2, 3), [1, 134, 160], dtype=np.uint8)
    valid = np.ones((2, 2), dtype=bool)
    with pytest.raises(error):
        compute_terrain_features(terrain, valid_mask=valid, pixel_spacing_m=spacing)
    with pytest.raises(TypeError):
        compute_terrain_features(terrain, valid_mask=valid)
