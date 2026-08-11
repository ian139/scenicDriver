"""
Tests for src/data_pipeline/web_mercator.py
"""

from __future__ import annotations

import math
import pytest

from src.data_pipeline.web_mercator import (
    EPSG3857_CRS,
    HALF_CIRCUMFERENCE,
    WEB_MERCATOR_MAX_LAT,
    TargetGrid,
    lat_lon_to_tile,
    tile_bounds_web_mercator,
    tile_bounds_wgs84,
    tile_to_lat_lon,
    tile_to_lat_lon_center,
    tile_transform_web_mercator,
)
from src.route_planner.graph import Edge, Node, RoadGraph
from src.route_planner.service import apply_tile_scores_to_graph


def test_lat_lon_to_tile_known_values():
    # Null Island (0, 0) at zoom 0 -> (0, 0)
    assert lat_lon_to_tile(0.0, 0.0, 0) == (0, 0)
    # Null Island at zoom 1 -> (1, 1)
    assert lat_lon_to_tile(0.0, 0.0, 1) == (1, 1)

    # Boston (42.3601, -71.0589) at zoom 14
    x14, y14 = lat_lon_to_tile(42.3601, -71.0589, 14)
    assert x14 == 4958
    assert y14 == 6059


def test_latitude_clamping():
    # Latitudes beyond max should be clamped to valid range [0, 2^z - 1]
    assert lat_lon_to_tile(90.0, 0.0, 4) == (8, 0)
    assert lat_lon_to_tile(-90.0, 0.0, 4) == (8, 15)
    assert lat_lon_to_tile(85.0511287798, 0.0, 4) == (8, 0)


def test_tile_to_lat_lon_corners():
    # Zoom 0 tile (0, 0) NW corner should be (MAX_LAT, -180)
    lat, lon = tile_to_lat_lon(0, 0, 0)
    assert math.isclose(lon, -180.0, abs_tol=1e-7)
    assert math.isclose(lat, WEB_MERCATOR_MAX_LAT, abs_tol=1e-7)

    # Zoom 1 tile (1, 1) NW corner is (0, 0)
    lat, lon = tile_to_lat_lon(1, 1, 1)
    assert math.isclose(lon, 0.0, abs_tol=1e-7)
    assert math.isclose(lat, 0.0, abs_tol=1e-7)


def test_tile_to_lat_lon_center_precision_vs_arithmetic_average():
    # Verify exact inverse Mercator center (x+0.5, y+0.5) behavior change
    # versus prior naive arithmetic average of NW and SE latitude corners
    x, y, z = 4958, 6059, 14
    center_lat, center_lon = tile_to_lat_lon_center(x, y, z)
    nw_lat, nw_lon = tile_to_lat_lon(x, y, z)
    se_lat, se_lon = tile_to_lat_lon(x + 1, y + 1, z)

    arithmetic_lat = (nw_lat + se_lat) / 2.0
    arithmetic_lon = (nw_lon + se_lon) / 2.0

    # Longitude is linear, so center longitude matches arithmetic average
    assert math.isclose(center_lon, arithmetic_lon, abs_tol=1e-12)

    # Latitude in Mercator is non-linear: exact center differs slightly from arithmetic mean
    assert not math.isclose(center_lat, arithmetic_lat, abs_tol=1e-12)
    assert (
        abs(center_lat - arithmetic_lat) < 1e-4
    )  # Delta is small but mathematically non-zero


def test_tile_bounds_wgs84():
    min_lon, min_lat, max_lon, max_lat = tile_bounds_wgs84(0, 0, 1)
    assert math.isclose(min_lon, -180.0, abs_tol=1e-7)
    assert math.isclose(max_lon, 0.0, abs_tol=1e-7)
    assert math.isclose(max_lat, WEB_MERCATOR_MAX_LAT, abs_tol=1e-7)
    assert math.isclose(min_lat, 0.0, abs_tol=1e-7)
    assert min_lon < max_lon
    assert min_lat < max_lat


def test_tile_bounds_web_mercator():
    # Zoom 0 full world tile
    min_x, min_y, max_x, max_y = tile_bounds_web_mercator(0, 0, 0)
    assert math.isclose(min_x, -HALF_CIRCUMFERENCE, abs_tol=1e-5)
    assert math.isclose(max_x, HALF_CIRCUMFERENCE, abs_tol=1e-5)
    assert math.isclose(min_y, -HALF_CIRCUMFERENCE, abs_tol=1e-5)
    assert math.isclose(max_y, HALF_CIRCUMFERENCE, abs_tol=1e-5)
    assert min_x < max_x
    assert min_y < max_y


def test_tile_transform_web_mercator_orientation():
    # Pixel (col=0, row=0) maps to (min_x, max_y)
    a, b, c, d, e, f = tile_transform_web_mercator(0, 0, 1, width=256, height=256)
    min_x, min_y, max_x, max_y = tile_bounds_web_mercator(0, 0, 1)

    assert math.isclose(c, min_x, abs_tol=1e-5)
    assert math.isclose(f, max_y, abs_tol=1e-5)
    assert a > 0.0  # Pixel size X positive
    assert e < 0.0  # Pixel size Y negative (image row 0 is top)
    assert b == 0.0
    assert d == 0.0

    # Top-left pixel center maps correctly
    px_x = a * 0 + c
    px_y = e * 0 + f
    assert math.isclose(px_x, min_x)
    assert math.isclose(px_y, max_y)

    # Bottom-right pixel maps correctly
    br_x = a * 256 + c
    br_y = e * 256 + f
    assert math.isclose(br_x, max_x, abs_tol=1e-5)
    assert math.isclose(br_y, min_y, abs_tol=1e-5)


def test_target_grid_creation_and_serialization():
    grid = TargetGrid.from_tile(z=14, x=4958, y=6059, width=256, height=256)
    assert grid.z == 14
    assert grid.x == 4958
    assert grid.y == 6059
    assert grid.crs == EPSG3857_CRS
    assert len(grid.bounds) == 4
    assert len(grid.affine) == 6

    # Test serialization round-trip
    grid_dict = grid.to_dict()
    reconstructed = TargetGrid.from_dict(grid_dict)
    assert grid == reconstructed
    assert grid.sha256() == reconstructed.sha256()
    assert isinstance(grid.sha256(), str)
    assert len(grid.sha256()) == 64


def test_validation_errors():
    with pytest.raises(
        ValueError, match="Zoom level must be an integer between 0 and 22"
    ):
        lat_lon_to_tile(0.0, 0.0, -1)

    with pytest.raises(
        ValueError, match="Zoom level must be an integer between 0 and 22"
    ):
        lat_lon_to_tile(0.0, 0.0, 23)

    with pytest.raises(ValueError, match="Latitude and longitude must be finite"):
        lat_lon_to_tile(float("nan"), 0.0, 10)

    with pytest.raises(ValueError, match="Latitude and longitude must be finite"):
        lat_lon_to_tile(True, 0.0, 10)

    with pytest.raises(ValueError, match="Tile x coordinate must be an integer"):
        tile_to_lat_lon(16, 0, 4)

    with pytest.raises(ValueError, match="Tile y coordinate must be an integer"):
        tile_to_lat_lon(0, -1, 4)

    with pytest.raises(ValueError, match="width must be a positive integer"):
        tile_transform_web_mercator(0, 0, 4, width=0, height=256)


def test_target_grid_rejection():
    # Invalid CRS
    with pytest.raises(ValueError, match="crs must be"):
        TargetGrid(z=14, x=4958, y=6059, crs="EPSG:4326")

    # Mismatched bounds
    with pytest.raises(ValueError, match="do not match computed bounds"):
        TargetGrid(z=14, x=4958, y=6059, bounds=(0.0, 0.0, 1.0, 1.0))

    # Mismatched affine
    with pytest.raises(ValueError, match="does not match computed affine"):
        TargetGrid(z=14, x=4958, y=6059, affine=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    # Missing dict keys
    with pytest.raises(ValueError, match="missing required key"):
        TargetGrid.from_dict({"z": 14, "x": 4958})


def test_route_tile_matching_integration():
    # Build simple RoadGraph and test apply_tile_scores_to_graph with neutral lat_lon_to_tile
    graph = RoadGraph()
    graph.add_node(Node(id="n1", lat=42.3601, lon=-71.0589))
    graph.add_node(Node(id="n2", lat=42.3610, lon=-71.0580))
    graph.add_node(Node(id="n3", lat=0.0, lon=0.0))
    graph.add_node(Node(id="n4", lat=0.001, lon=0.001))

    graph.add_edge(
        Edge(
            id="e1",
            start_node_id="n1",
            end_node_id="n2",
            distance_km=0.2,
            scenic_score=5.0,
        )
    )
    graph.add_edge(
        Edge(
            id="e2",
            start_node_id="n3",
            end_node_id="n4",
            distance_km=0.2,
            scenic_score=3.0,
        )
    )

    mid_lat = 0.5 * (42.3601 + 42.3610)
    mid_lon = 0.5 * (-71.0589 + -71.0580)
    tx, ty = lat_lon_to_tile(mid_lat, mid_lon, 14)

    score_map = {(14, tx, ty): 9.5}

    # Out of coverage without fallback leaves e2 unchanged
    matched, total = apply_tile_scores_to_graph(
        graph, score_map, zoom=14, fallback=None
    )
    assert matched == 1
    assert total == 2
    assert graph.edges["e1"].scenic_score == 9.5
    assert graph.edges["e2"].scenic_score == 3.0

    # Out of coverage with fallback applies fallback score to e2
    matched, total = apply_tile_scores_to_graph(graph, score_map, zoom=14, fallback=1.5)
    assert matched == 1
    assert total == 2
    assert graph.edges["e1"].scenic_score == 9.5
    assert graph.edges["e2"].scenic_score == 1.5


def test_coordinate_roundtrips():
    # Tile center lat/lon converted back via lat_lon_to_tile maps to same tile
    for zoom in [0, 5, 14, 22]:
        n = 1 << zoom
        test_tiles = [(0, 0), (n // 2, n // 2), (n - 1, n - 1)]
        for x, y in test_tiles:
            center_lat, center_lon = tile_to_lat_lon_center(x, y, zoom)
            rx, ry = lat_lon_to_tile(center_lat, center_lon, zoom)
            assert (rx, ry) == (x, y)
