from pathlib import Path

import numpy as np
from PIL import Image

from src.terrain.features import (
    TerrainFeatures,
    compute_terrain_features,
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
        [[-10000.0 + (1 * 256 * 256 + 2 * 256 + 3) * 0.1,
          -10000.0 + (10 * 256 * 256 + 20 * 256 + 30) * 0.1]],
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

    array_result = compute_terrain_features(rgb, satellite)
    pil_result = compute_terrain_features(Image.open(terrain_path), Image.open(satellite_path))
    path_result = compute_terrain_features(terrain_path, satellite_path)

    for result in (pil_result, path_result):
        np.testing.assert_allclose(result.features.to_array(), array_result.features.to_array())
        np.testing.assert_allclose(
            [result.relief, result.roughness, result.slope_mean],
            [array_result.relief, array_result.roughness, array_result.slope_mean],
        )


def test_zero_border_seam_is_repaired_without_changing_interior():
    elevations = np.full((100, 100), 123.0, dtype=np.float32)
    elevations[:, 0] = 0.0

    repaired = repair_terrain_zero_seam(elevations)

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
    expected_density = np.mean(
        [20 / 100, 40 / 100, 25 / 100, 10 / 100]
    )

    result = compute_terrain_features(terrain, satellite)
    default_result = compute_terrain_features(terrain)

    assert np.isclose(result.features.vegetation_density, expected_density)
    assert default_result.features.vegetation_density == 0.5


def test_feature_vector_matches_canonical_scalar_metrics():
    result = compute_terrain_features(_terrain_rgb())
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
