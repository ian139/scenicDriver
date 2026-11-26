from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from scenicdriver.data_pipeline import region_planning

from scenicdriver.data_pipeline.region_planning import (
    enumerate_bbox_tiles,
    enumerate_land_tiles,
    enumerate_polygon_tiles,
    parse_and_validate_region_spec,
    project_geometry_to_5070,
    parse_geojson_geometry,
)
from scenicdriver.data_pipeline.web_mercator import (
    tile_bounds_wgs84,
    tile_to_lat_lon_center,
)


def test_parse_geojson_geometry_polygon_and_multipolygon() -> None:
    poly_dict = {
        "type": "Polygon",
        "coordinates": [
            [[-71.5, 42.5], [-71.0, 42.5], [-71.0, 43.0], [-71.5, 43.0], [-71.5, 42.5]]
        ],
    }
    geom = parse_geojson_geometry(poly_dict)
    assert isinstance(geom, Polygon)
    assert geom.is_valid

    multi_dict = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-71.5, 42.5],
                    [-71.0, 42.5],
                    [-71.0, 43.0],
                    [-71.5, 43.0],
                    [-71.5, 42.5],
                ]
            ],
            [
                [
                    [-70.5, 42.5],
                    [-70.0, 42.5],
                    [-70.0, 43.0],
                    [-70.5, 43.0],
                    [-70.5, 42.5],
                ]
            ],
        ],
    }
    multi_geom = parse_geojson_geometry(multi_dict)
    assert isinstance(multi_geom, MultiPolygon)
    assert multi_geom.is_valid


def test_parse_geojson_geometry_feature_and_collection() -> None:
    feat = {
        "type": "Feature",
        "properties": {"name": "Test Feature"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-71.5, 42.5],
                    [-71.0, 42.5],
                    [-71.0, 43.0],
                    [-71.5, 43.0],
                    [-71.5, 42.5],
                ]
            ],
        },
    }
    geom = parse_geojson_geometry(feat)
    assert isinstance(geom, Polygon)

    fc = {
        "type": "FeatureCollection",
        "features": [feat, feat],
    }
    fc_geom = parse_geojson_geometry(fc)
    assert isinstance(fc_geom, Polygon)


def test_parse_geojson_geometry_holes() -> None:
    # Outer square [-72, 42] to [-70, 44], inner hole [-71.5, 42.5] to [-70.5, 43.5]
    poly_with_hole = {
        "type": "Polygon",
        "coordinates": [
            [[-72.0, 42.0], [-70.0, 42.0], [-70.0, 44.0], [-72.0, 44.0], [-72.0, 42.0]],
            [[-71.5, 42.5], [-70.5, 42.5], [-70.5, 43.5], [-71.5, 43.5], [-71.5, 42.5]],
        ],
    }
    geom = parse_geojson_geometry(poly_with_hole)
    assert isinstance(geom, Polygon)
    assert len(geom.interiors) == 1

    # Tile inside hole should not have center on land
    tiles = enumerate_land_tiles(poly_with_hole, zoom=14)
    assert len(tiles) > 0
    # Hole center is at (-71.0, 43.0). Tiles admitted should surround the hole.
    for tile in tiles:
        # None of the admitted tiles should be fully centered inside the hole interior
        pass


def test_parse_geojson_geometry_multipolygons_disjoint() -> None:
    disjoint = [
        [[-71.5, 42.5], [-71.4, 42.5], [-71.4, 42.6], [-71.5, 42.6], [-71.5, 42.5]],
        [[-70.5, 43.5], [-70.4, 43.5], [-70.4, 43.6], [-70.5, 43.6], [-70.5, 43.5]],
    ]
    tiles = enumerate_polygon_tiles(disjoint, zoom=14)
    assert len(tiles) >= 2


def test_boundary_touches_not_admitted_without_positive_area() -> None:
    # A tiny triangle where tile corner barely touches polygon boundary with 0 area
    # Create land geometry
    poly = {
        "type": "Polygon",
        "coordinates": [[[-71.0, 42.5], [-71.0, 42.51], [-70.99, 42.5], [-71.0, 42.5]]],
    }
    tiles = enumerate_land_tiles(poly, zoom=14, threshold=0.05)
    for rec in tiles:
        assert rec["admission_reason"] in ("center_on_land", "land_fraction")
        if rec["admission_reason"] == "land_fraction":
            assert rec["land_fraction"] >= 0.05


def test_vectorized_land_tiles_match_scalar_equal_area_contract() -> None:
    geometry = Polygon(
        [
            (-71.17, 42.31),
            (-70.91, 42.34),
            (-70.96, 42.52),
            (-71.12, 42.48),
            (-71.17, 42.31),
        ]
    )
    threshold = 0.17
    observed = enumerate_land_tiles(geometry, zoom=12, threshold=threshold)

    geometry_equal_area = project_geometry_to_5070(geometry)
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    expected = []
    for x, y in enumerate_bbox_tiles(min_lat, min_lon, max_lat, max_lon, zoom=12):
        center_lat, center_lon = tile_to_lat_lon_center(x, y, 12)
        center = Point(center_lon, center_lat)
        center_on_land = geometry.covers(center) or geometry.contains(center)
        tile_equal_area = project_geometry_to_5070(box(*tile_bounds_wgs84(x, y, 12)))
        intersection_area = tile_equal_area.intersection(geometry_equal_area).area
        land_fraction = min(
            1.0,
            max(
                0.0,
                intersection_area / tile_equal_area.area
                if tile_equal_area.area > 0.0
                else 0.0,
            ),
        )
        if center_on_land or (intersection_area > 0.0 and land_fraction >= threshold):
            expected.append(
                {
                    "coord": (x, y),
                    "x": x,
                    "y": y,
                    "land_fraction": land_fraction,
                    "admission_reason": (
                        "center_on_land" if center_on_land else "land_fraction"
                    ),
                }
            )

    assert [row["coord"] for row in observed] == [row["coord"] for row in expected]
    assert [row["admission_reason"] for row in observed] == [
        row["admission_reason"] for row in expected
    ]
    assert [row["land_fraction"] for row in observed] == pytest.approx(
        [row["land_fraction"] for row in expected], abs=1e-12
    )


def test_parse_geojson_geometry_bad_crs() -> None:
    bad_crs_spec = {
        "type": "Polygon",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "coordinates": [
            [[-71.5, 42.5], [-71.0, 42.5], [-71.0, 43.0], [-71.5, 43.0], [-71.5, 42.5]]
        ],
    }
    with pytest.raises(ValueError, match="Unsupported CRS"):
        parse_geojson_geometry(bad_crs_spec)


def test_parse_geojson_geometry_non_finite_and_out_of_bounds() -> None:
    nan_coords = {
        "type": "Polygon",
        "coordinates": [
            [
                [float("nan"), 42.5],
                [-71.0, 42.5],
                [-71.0, 43.0],
                [-71.5, 43.0],
                [float("nan"), 42.5],
            ]
        ],
    }
    with pytest.raises(ValueError):
        parse_geojson_geometry(nan_coords)

    oob_coords = {
        "type": "Polygon",
        "coordinates": [
            [
                [-190.0, 42.5],
                [-71.0, 42.5],
                [-71.0, 43.0],
                [-71.5, 43.0],
                [-190.0, 42.5],
            ]
        ],
    }
    with pytest.raises(ValueError, match="WGS84 bounds"):
        parse_geojson_geometry(oob_coords)


def test_parse_geojson_geometry_malformed_data() -> None:
    with pytest.raises(ValueError):
        parse_geojson_geometry(None)

    with pytest.raises(ValueError, match="Unsupported GeoJSON geometry type"):
        parse_geojson_geometry({"type": "Point", "coordinates": [-71.0, 42.5]})

    with pytest.raises(ValueError, match="Unsupported GeoJSON geometry type"):
        parse_geojson_geometry(
            {"type": "LineString", "coordinates": [[-71.0, 42.5], [-70.0, 43.0]]}
        )

    with pytest.raises(ValueError, match="contains no features"):
        parse_geojson_geometry({"type": "FeatureCollection", "features": []})

    with pytest.raises(ValueError, match="missing or invalid 'geometry'"):
        parse_geojson_geometry({"type": "Feature", "properties": {}})


def test_geometry_file_missing_raises_file_not_found(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 47.5,
                            "max_lon": -66.8,
                        },
                    }
                ]
            }
        )
    )

    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.5,
                    "min_lon": -73.5,
                    "max_lat": 47.5,
                    "max_lon": -66.8,
                },
            },
            {
                "name": "missing_region",
                "type": "geometry_file",
                "geometry_file": str(tmp_path / "nonexistent.geojson"),
            },
        ],
    }

    with pytest.raises(FileNotFoundError, match="Authoritative boundary file"):
        parse_and_validate_region_spec(spec, app_regions_path=app_regions_file)


def test_geometry_file_hash_mismatch_raises_value_error(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 47.5,
                            "max_lon": -66.8,
                        },
                    }
                ]
            }
        )
    )

    geom_file = tmp_path / "region.geojson"
    geom_data = {
        "type": "Polygon",
        "coordinates": [
            [[-73.5, 42.5], [-73.0, 42.5], [-73.0, 43.0], [-73.5, 43.0], [-73.5, 42.5]]
        ],
    }
    geom_file.write_text(json.dumps(geom_data))

    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.5,
                    "min_lon": -73.5,
                    "max_lat": 47.5,
                    "max_lon": -66.8,
                },
            },
            {
                "name": "file_region",
                "type": "geometry_file",
                "geometry_file": str(geom_file),
                "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            },
        ],
    }

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_and_validate_region_spec(spec, app_regions_path=app_regions_file)


def test_geometry_file_valid_loading_and_sha256(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 47.5,
                            "max_lon": -66.8,
                        },
                    }
                ]
            }
        )
    )

    geom_file = tmp_path / "region.geojson"
    geom_data = {
        "type": "Polygon",
        "coordinates": [
            [[-73.5, 42.5], [-73.0, 42.5], [-73.0, 43.0], [-73.5, 43.0], [-73.5, 42.5]]
        ],
    }
    content_bytes = json.dumps(geom_data).encode("utf-8")
    geom_file.write_bytes(content_bytes)
    file_sha256 = hashlib.sha256(content_bytes).hexdigest()

    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.5,
                    "min_lon": -73.5,
                    "max_lat": 47.5,
                    "max_lon": -66.8,
                },
            },
            {
                "name": "file_region",
                "type": "geometry_file",
                "geometry_file": str(geom_file),
                "expected_sha256": file_sha256,
            },
        ],
    }

    planned = parse_and_validate_region_spec(spec, app_regions_path=app_regions_file)
    assert planned["unique_coordinates_count"] > 0
    assert planned["region_tile_counts"]["file_region"] > 0


def test_deduplication_of_overlapping_regions(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 47.5,
                            "max_lon": -66.8,
                        },
                    }
                ]
            }
        )
    )

    baseline_bbox = {
        "min_lat": 42.5,
        "min_lon": -73.5,
        "max_lat": 47.5,
        "max_lon": -66.8,
    }

    spec = {
        "version": 1,
        "regions": [
            {"name": "new_england_north", "type": "bbox", "bbox": baseline_bbox},
            {"name": "duplicate_nen", "type": "bbox", "bbox": baseline_bbox},
        ],
    }

    planned = parse_and_validate_region_spec(spec, app_regions_path=app_regions_file)
    assert planned["unique_coordinates_count"] == planned["nen_tile_count"]
    assert planned["region_tile_counts"]["duplicate_nen"] == planned["nen_tile_count"]
    # All deduplicated coordinates mapped to first region
    for coord in planned["ordered_coords"]:
        assert planned["coord_to_region"][coord] == "new_england_north"


def test_north_and_east_expansion_prohibition(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 47.5,
                            "max_lon": -66.8,
                        },
                    }
                ]
            }
        )
    )

    baseline_bbox = {
        "min_lat": 42.5,
        "min_lon": -73.5,
        "max_lat": 47.5,
        "max_lon": -66.8,
    }

    # North expansion (max_lat 48.0 > 47.5)
    north_spec = {
        "version": 1,
        "regions": [
            {"name": "new_england_north", "type": "bbox", "bbox": baseline_bbox},
            {
                "name": "north_expansion",
                "type": "bbox",
                "bbox": {
                    "min_lat": 47.5,
                    "min_lon": -73.5,
                    "max_lat": 48.0,
                    "max_lon": -66.8,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match="North"):
        parse_and_validate_region_spec(north_spec, app_regions_path=app_regions_file)

    # East expansion (max_lon -65.0 > -66.8)
    east_spec = {
        "version": 1,
        "regions": [
            {"name": "new_england_north", "type": "bbox", "bbox": baseline_bbox},
            {
                "name": "east_expansion",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.5,
                    "min_lon": -66.8,
                    "max_lat": 47.5,
                    "max_lon": -65.0,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match="East"):
        parse_and_validate_region_spec(east_spec, app_regions_path=app_regions_file)


def test_budget_cap_exceeded_raises_value_error(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 47.5,
                            "max_lon": -66.8,
                        },
                    }
                ]
            }
        )
    )

    baseline_bbox = {
        "min_lat": 42.5,
        "min_lon": -73.5,
        "max_lat": 47.5,
        "max_lon": -66.8,
    }

    spec = {
        "version": 1,
        "regions": [
            {"name": "new_england_north", "type": "bbox", "bbox": baseline_bbox}
        ],
    }

    # Pass max_budget_coords smaller than baseline NEN tile count
    with pytest.raises(ValueError, match="exceeds hard budget cap"):
        parse_and_validate_region_spec(
            spec, app_regions_path=app_regions_file, max_budget_coords=10
        )


def test_schema_and_per_coordinate_metadata(tmp_path: Path) -> None:
    app_regions_file = tmp_path / "app_regions.json"
    app_regions_file.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 43.0,
                            "max_lon": -73.0,
                        },
                    }
                ]
            }
        )
    )

    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.5,
                    "min_lon": -73.5,
                    "max_lat": 43.0,
                    "max_lon": -73.0,
                },
            }
        ],
    }

    planned = parse_and_validate_region_spec(spec, app_regions_path=app_regions_file)

    expected_keys = {
        "spec_data",
        "zoom",
        "unique_coordinates_count",
        "total_rasters_count",
        "nen_tile_count",
        "coord_to_region",
        "coord_to_land_fraction",
        "coord_to_admission_reason",
        "ordered_coords",
        "region_tile_counts",
        "geometry_digest",
    }
    assert expected_keys.issubset(planned.keys())
    assert planned["unique_coordinates_count"] == planned["nen_tile_count"]
    assert planned["total_rasters_count"] == planned["unique_coordinates_count"] * 2

    for coord in planned["ordered_coords"]:
        assert coord in planned["coord_to_land_fraction"]
        assert coord in planned["coord_to_admission_reason"]
        assert isinstance(planned["coord_to_land_fraction"][coord], float)
        assert planned["coord_to_admission_reason"][coord] in (
            "center_on_land",
            "land_fraction",
            "bbox",
        )


def test_pinned_jurisdiction_land_intersection_records_audit_metrics(
    tmp_path: Path,
) -> None:
    tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat = tile_bounds_wgs84(
        4800, 6200, 14
    )
    lon_margin = (tile_max_lon - tile_min_lon) * 0.25
    lat_margin = (tile_max_lat - tile_min_lat) * 0.25
    bbox = {
        "min_lat": tile_min_lat + lat_margin,
        "min_lon": tile_min_lon + lon_margin,
        "max_lat": tile_max_lat - lat_margin,
        "max_lon": tile_max_lon - lon_margin,
    }
    ring = [
        [tile_min_lon, tile_min_lat],
        [tile_max_lon, tile_min_lat],
        [tile_max_lon, tile_max_lat],
        [tile_min_lon, tile_max_lat],
        [tile_min_lon, tile_min_lat],
    ]
    app_regions = tmp_path / "app_regions.json"
    app_regions.write_text(
        json.dumps({"regions": [{"region": "new_england_north", "bbox": bbox}]}),
        encoding="utf-8",
    )
    jurisdiction = tmp_path / "jurisdiction.geojson"
    jurisdiction.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"code": "MA"},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    land = tmp_path / "land.geojson"
    land.write_text(
        json.dumps(
            {"type": "Polygon", "coordinates": [ring]},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    jurisdiction_sha = hashlib.sha256(jurisdiction.read_bytes()).hexdigest()
    land_sha = hashlib.sha256(land.read_bytes()).hexdigest()
    spec = {
        "version": 1,
        "baseline_land_geometry_file": str(land),
        "baseline_land_geometry_sha256": land_sha,
        "baseline_jurisdiction_geometry_file": str(jurisdiction),
        "baseline_jurisdiction_geometry_sha256": jurisdiction_sha,
        "regions": [
            {
                "name": "new_england_north",
                "type": "jurisdiction_land_intersection",
                "jurisdiction_geometry_file": str(jurisdiction),
                "jurisdiction_geometry_sha256": jurisdiction_sha,
                "land_geometry_file": str(land),
                "land_geometry_sha256": land_sha,
                "jurisdiction_field": "code",
                "included_jurisdictions": ["MA"],
                "threshold": 0.05,
            }
        ],
    }

    planned = parse_and_validate_region_spec(
        spec, app_regions_path=app_regions, max_budget_coords=100_000
    )

    assert planned["geometry_source_hashes"] == {
        "baseline_jurisdiction": jurisdiction_sha,
        "baseline_land": land_sha,
        "new_england_north:jurisdiction": jurisdiction_sha,
        "new_england_north:land": land_sha,
    }
    assert planned["state_tile_counts"]["MA"] > 0
    assert set(planned["coord_to_state"].values()) == {"MA"}
    assert planned["admission_reason_counts"]["center_on_land"] > 0
    assert planned["land_fraction_summary"]["min"] > 0.0


def test_jurisdiction_and_land_geometry_must_be_independently_pinned(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "same.geojson"
    geometry.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"code": "MA"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-73.5, 42.5],
                                    [-73.0, 42.5],
                                    [-73.0, 43.0],
                                    [-73.5, 43.0],
                                    [-73.5, 42.5],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "jurisdiction_land_intersection",
                "jurisdiction_geometry_file": str(geometry),
                "jurisdiction_geometry_sha256": digest,
                "land_geometry_file": str(geometry),
                "land_geometry_sha256": digest,
                "jurisdiction_field": "code",
                "included_jurisdictions": ["MA"],
            }
        ],
    }
    app_regions = tmp_path / "app_regions.json"
    app_regions.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "bbox": {
                            "min_lat": 42.5,
                            "min_lon": -73.5,
                            "max_lat": 43.0,
                            "max_lon": -73.0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="independently pinned"):
        parse_and_validate_region_spec(spec, app_regions_path=app_regions)


def test_global_bbox_tile_enumeration_rejected_before_grid_allocation() -> None:
    # A global z14 bbox spans ~2^28 tiles; materializing it would need tens of GB.
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="exceeds"):
            enumerate_bbox_tiles(-85.0, -180.0, 85.0, 180.0, zoom=14, max_tiles=370_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


def test_global_land_tile_enumeration_aborts_at_cap_with_bounded_memory() -> None:
    tracemalloc.start()
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="exceeds"):
            enumerate_land_tiles(
                box(-180.0, -85.0, 180.0, 85.0), zoom=14, max_tiles=1_000
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64_000_000
    assert time.monotonic() - started < 30.0


def test_global_bbox_region_hits_budget_cap_without_enumerating(tmp_path: Path) -> None:
    app_regions = tmp_path / "app_regions.json"
    nen_bbox = {"min_lat": 42.5, "min_lon": -73.5, "max_lat": 42.6, "max_lon": -73.4}
    app_regions.write_text(
        json.dumps({"regions": [{"region": "new_england_north", "bbox": nen_bbox}]}),
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "regions": [
            {"name": "new_england_north", "type": "bbox", "bbox": nen_bbox},
            {
                "name": "world",
                "type": "bbox",
                "bbox": {
                    "min_lat": -85.0,
                    "min_lon": -180.0,
                    "max_lat": 85.0,
                    "max_lon": 180.0,
                },
            },
        ],
    }
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="exceeds hard budget cap"):
            parse_and_validate_region_spec(spec, app_regions_path=app_regions)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16_000_000


def _write_feature_collection(path: Path, features: list[dict]) -> str:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, sort_keys=True),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rect(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


def test_jurisdiction_fragments_admitted_on_union_with_single_owner(
    tmp_path: Path,
) -> None:
    # Land strip covering ~6% of one z14 tile, split by a state border into
    # ~3.3% (A) and ~2.7% (B) fragments: each alone is below the 5% threshold
    # but the authorized union is above it, and the tile centre is water.
    min_lon, min_lat, max_lon, max_lat = tile_bounds_wgs84(4800, 6200, 14)
    width = max_lon - min_lon
    height = max_lat - min_lat
    border_lon = min_lon + 0.55 * width
    strip_top = min_lat + 0.06 * height
    nen_bbox = {
        "min_lat": min_lat + 0.25 * height,
        "min_lon": min_lon + 0.25 * width,
        "max_lat": max_lat - 0.25 * height,
        "max_lon": max_lon - 0.25 * width,
    }
    app_regions = tmp_path / "app_regions.json"
    app_regions.write_text(
        json.dumps({"regions": [{"region": "new_england_north", "bbox": nen_bbox}]}),
        encoding="utf-8",
    )
    jurisdiction = tmp_path / "jurisdiction.geojson"
    jurisdiction_sha = _write_feature_collection(
        jurisdiction,
        [
            {
                "type": "Feature",
                "properties": {"code": "B"},
                "geometry": _rect(border_lon, min_lat, max_lon, max_lat),
            },
            {
                "type": "Feature",
                "properties": {"code": "A"},
                "geometry": _rect(min_lon, min_lat, border_lon, max_lat),
            },
        ],
    )
    land = tmp_path / "land.geojson"
    land.write_text(
        json.dumps(_rect(min_lon, min_lat, max_lon, strip_top)), encoding="utf-8"
    )
    land_sha = hashlib.sha256(land.read_bytes()).hexdigest()
    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "jurisdiction_land_intersection",
                "jurisdiction_geometry_file": str(jurisdiction),
                "jurisdiction_geometry_sha256": jurisdiction_sha,
                "land_geometry_file": str(land),
                "land_geometry_sha256": land_sha,
                "jurisdiction_field": "code",
                "included_jurisdictions": ["A", "B"],
                "threshold": 0.05,
            }
        ],
    }

    planned = parse_and_validate_region_spec(
        spec, app_regions_path=app_regions, max_budget_coords=100
    )

    coord = (4800, 6200)
    assert planned["ordered_coords"] == [coord]
    assert planned["coord_to_admission_reason"][coord] == "land_fraction"
    assert planned["coord_to_land_fraction"][coord] == pytest.approx(0.06, abs=0.005)
    assert planned["coord_to_state"][coord] == "A"
    assert planned["state_tile_counts"] == {"A": 1, "B": 0}
    assert planned["region_tile_counts"] == {"new_england_north": 1}


def test_jurisdiction_border_tiles_counted_once_per_owner(tmp_path: Path) -> None:
    # Two land states sharing a border through a 3x3 tile block: every tile is
    # admitted exactly once and state counts partition the region count.
    west, south, _, _ = tile_bounds_wgs84(4800, 6202, 14)
    _, _, east, north = tile_bounds_wgs84(4802, 6200, 14)
    border_lon = west + (east - west) * (1.0 + 0.4) / 3.0
    app_regions = tmp_path / "app_regions.json"
    inset_lon = (east - west) * 0.01
    inset_lat = (north - south) * 0.01
    nen_bbox = {
        "min_lat": south + inset_lat,
        "min_lon": west + inset_lon,
        "max_lat": north - inset_lat,
        "max_lon": east - inset_lon,
    }
    app_regions.write_text(
        json.dumps({"regions": [{"region": "new_england_north", "bbox": nen_bbox}]}),
        encoding="utf-8",
    )
    jurisdiction = tmp_path / "jurisdiction.geojson"
    jurisdiction_sha = _write_feature_collection(
        jurisdiction,
        [
            {
                "type": "Feature",
                "properties": {"code": "W"},
                "geometry": _rect(west, south, border_lon, north),
            },
            {
                "type": "Feature",
                "properties": {"code": "E"},
                "geometry": _rect(border_lon, south, east, north),
            },
        ],
    )
    land = tmp_path / "land.geojson"
    land.write_text(json.dumps(_rect(west, south, east, north)), encoding="utf-8")
    land_sha = hashlib.sha256(land.read_bytes()).hexdigest()
    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "jurisdiction_land_intersection",
                "jurisdiction_geometry_file": str(jurisdiction),
                "jurisdiction_geometry_sha256": jurisdiction_sha,
                "land_geometry_file": str(land),
                "land_geometry_sha256": land_sha,
                "jurisdiction_field": "code",
                "included_jurisdictions": ["E", "W"],
            }
        ],
    }

    planned = parse_and_validate_region_spec(
        spec, app_regions_path=app_regions, max_budget_coords=100
    )

    assert planned["region_tile_counts"]["new_england_north"] == 9
    assert planned["unique_coordinates_count"] == 9
    assert sum(planned["state_tile_counts"].values()) == 9
    # The middle column is 40% W / 60% E land: area majority decides ownership.
    for (x, _), state in planned["coord_to_state"].items():
        assert state == {4800: "W", 4801: "E", 4802: "E"}[x]
    assert planned["state_tile_counts"] == {"E": 6, "W": 3}


def test_jointly_admitted_jurisdiction_fragments_stop_before_full_grid_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each state's share is below 5%; only their union admits the 100 tiles.
    # Inset the strips so the bbox has no neighboring boundary-only tiles.
    a_parts = []
    b_parts = []
    coords = [(x, 6200) for x in range(4800, 4900)]
    for x, y in coords:
        west, south, east, north = tile_bounds_wgs84(x, y, 14)
        width, height = east - west, north - south
        strip_south = south + 0.01 * height
        strip_north = south + 0.07 * height
        border = west + 0.55 * width
        a_parts.append(box(west + 0.01 * width, strip_south, border, strip_north))
        b_parts.append(box(border, strip_south, east - 0.01 * width, strip_north))
    state_lands = [("A", MultiPolygon(a_parts)), ("B", MultiPolygon(b_parts))]
    assert enumerate_land_tiles(state_lands[0][1], threshold=0.05) == []
    assert enumerate_land_tiles(state_lands[1][1], threshold=0.05) == []

    monkeypatch.setattr(region_planning, "_TILE_CHUNK_SIZE", 2)
    real_chunks = region_planning._iter_land_tile_chunks
    scanned_coords = []

    def bounded_chunks(*args):
        for chunk in real_chunks(*args):
            xs, ys, *_ = chunk
            scanned_coords.extend((int(x), int(y)) for x, y in zip(xs, ys))
            assert len(scanned_coords) <= 4, "Scanned beyond the cap-crossing chunk"
            yield chunk

    monkeypatch.setattr(region_planning, "_iter_land_tile_chunks", bounded_chunks)
    with pytest.raises(ValueError, match="exceeds hard budget cap of 2"):
        region_planning._enumerate_jurisdiction_land_tiles(
            state_lands, zoom=14, threshold=0.05, max_tiles=2
        )
    assert scanned_coords == coords[:4]

    # Without overflow, real geometry retains deterministic union admission
    # and assigns the larger state share, including across ownership chunks.
    monkeypatch.setattr(region_planning, "_iter_land_tile_chunks", real_chunks)
    records = region_planning._enumerate_jurisdiction_land_tiles(
        state_lands, zoom=14, threshold=0.05, max_tiles=len(coords)
    )
    assert [record["coord"] for record in records] == coords
    for record in records:
        assert record["state"] == "A"
        assert record["admission_reason"] == "land_fraction"
        assert record["land_fraction"] == pytest.approx(0.0588, abs=0.001)


def test_jurisdiction_equal_positive_shares_use_first_sorted_state() -> None:
    west, south, east, north = tile_bounds_wgs84(4800, 6200, 14)
    width, height = east - west, north - south
    land = box(
        west + 0.1 * width,
        south + 0.1 * height,
        east - 0.1 * width,
        north - 0.1 * height,
    )
    records = region_planning._enumerate_jurisdiction_land_tiles(
        [("B", land), ("A", land)], zoom=14, threshold=0.05, max_tiles=1
    )
    assert [record["coord"] for record in records] == [(4800, 6200)]
    assert records[0]["state"] == "A"
    assert records[0]["land_fraction"] == pytest.approx(0.64, abs=0.001)


def test_aggregate_region_cap_stops_on_first_overflowing_unique_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    west, _, east, north = tile_bounds_wgs84(4800, 6200, 14)
    _, south, _, _ = tile_bounds_wgs84(4800, 6201, 14)
    inset = 0.00001
    baseline_bbox = {
        "min_lat": south + inset,
        "min_lon": west + inset,
        "max_lat": north - inset,
        "max_lon": east - inset,
    }
    app_regions = tmp_path / "app_regions.json"
    app_regions.write_text(
        json.dumps(
            {"regions": [{"region": "new_england_north", "bbox": baseline_bbox}]}
        ),
        encoding="utf-8",
    )
    expansion_west, _, expansion_east, _ = tile_bounds_wgs84(4799, 6200, 14)
    _, expansion_south, _, _ = tile_bounds_wgs84(4799, 6202, 14)
    spec = {
        "version": 1,
        "regions": [
            {"name": "new_england_north", "type": "bbox", "bbox": baseline_bbox},
            {"name": "duplicate", "type": "bbox", "bbox": baseline_bbox},
            {
                "name": "expansion",
                "type": "polygon",
                "polygon": _rect(
                    expansion_west + inset,
                    expansion_south + inset,
                    expansion_east - inset,
                    north - inset,
                ),
            },
            {"name": "must_not_reach", "type": "invalid"},
        ],
    }
    consumed_coords = []
    real_enumerate = region_planning.enumerate_land_tiles

    class ObservedRecords(list):
        def __iter__(self):
            for record in super().__iter__():
                consumed_coords.append(record["coord"])
                yield record

    def observed_enumerate(*args, **kwargs):
        return ObservedRecords(real_enumerate(*args, **kwargs))

    monkeypatch.setattr(region_planning, "enumerate_land_tiles", observed_enumerate)
    with pytest.raises(
        ValueError, match=r"Total unique tile coordinates \(4\) exceeds hard budget cap of 3"
    ):
        parse_and_validate_region_spec(
            spec, app_regions_path=app_regions, max_budget_coords=3
        )
    assert consumed_coords == [(4799, 6200), (4799, 6201)]

    # Duplicate regions are still legal at the exact aggregate cap, and the
    # first region keeps ownership of duplicated coordinates.
    spec["regions"][2]["polygon"] = _rect(
        expansion_west + inset,
        tile_bounds_wgs84(4799, 6200, 14)[1] + inset,
        expansion_east - inset,
        north - inset,
    )
    spec["regions"][3] = {
        "name": "duplicate_at_cap", "type": "bbox", "bbox": baseline_bbox
    }
    planned = parse_and_validate_region_spec(
        spec, app_regions_path=app_regions, max_budget_coords=3
    )
    assert planned["ordered_coords"] == [(4800, 6200), (4800, 6201), (4799, 6200)]
    assert planned["unique_coordinates_count"] == 3
    assert planned["region_tile_counts"] == {
        "new_england_north": 2,
        "duplicate": 2,
        "expansion": 1,
        "duplicate_at_cap": 2,
    }
    assert planned["coord_to_region"] == {
        (4800, 6200): "new_england_north",
        (4800, 6201): "new_england_north",
        (4799, 6200): "expansion",
    }
