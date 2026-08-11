"""Region planning and specification validation for active learning data acquisition.

Enforces geographic boundaries, New England North coverage requirements,
north/east expansion prohibitions, tile deduplication, and budget caps.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Polygon,
    box,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

from src.data_pipeline.web_mercator import (
    lat_lon_to_tile,
    tile_bounds_wgs84,
)


def load_app_region_bounds(
    config_path: Path | str = "config/app_regions.json",
) -> dict[str, Any]:
    """Load region configurations from app_regions.json."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"App regions config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = {}
    for r in data.get("regions", []):
        r_name = r.get("region")
        if r_name and "bbox" in r:
            regions[r_name] = r["bbox"]
    return regions


def enumerate_bbox_tiles(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int = 14
) -> list[tuple[int, int]]:
    """Enumerate all (x, y) tile coordinates covering the given bounding box at specified zoom."""
    if zoom < 0 or zoom > 22:
        raise ValueError(f"Unsupported zoom: {zoom}; expected 0..22.")
    if not all(math.isfinite(float(v)) for v in (min_lat, min_lon, max_lat, max_lon)):
        raise ValueError("Bounding box coordinates must be finite numbers.")
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("Bounding box must have min values strictly below max values.")
    if not (
        -90 <= min_lat <= 90
        and -90 <= max_lat <= 90
        and -180 <= min_lon <= 180
        and -180 <= max_lon <= 180
    ):
        raise ValueError(
            "Bounding box coordinates are outside valid latitude/longitude ranges."
        )

    nw_x, nw_y = lat_lon_to_tile(max_lat, min_lon, zoom)
    se_x, se_y = lat_lon_to_tile(min_lat, max_lon, zoom)
    min_x, max_x = min(nw_x, se_x), max(nw_x, se_x)
    min_y, max_y = min(nw_y, se_y), max(nw_y, se_y)

    tiles = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tiles.append((x, y))
    return tiles


_TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


def _validate_wgs84_coordinates(geometry: BaseGeometry) -> None:
    """Validate that all coordinates in geometry are finite and within WGS84 bounds."""
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            _validate_wgs84_coordinates(part)
        return

    coords: list[tuple[float, ...]] = []
    if isinstance(geometry, Polygon):
        coords.extend(geometry.exterior.coords)
        for hole in geometry.interiors:
            coords.extend(hole.coords)
    elif hasattr(geometry, "coords"):
        coords.extend(geometry.coords)

    for pt in coords:
        if len(pt) < 2:
            raise ValueError("Coordinate tuple must have at least 2 dimensions.")
        x, y = float(pt[0]), float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Non-finite coordinate encountered: ({x}, {y})")
        if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
            raise ValueError(
                f"Coordinate out of WGS84 bounds: lon={x}, lat={y}; expected lon in [-180, 180], lat in [-90, 90]"
            )


def parse_geojson_geometry(geometry_data: Any) -> BaseGeometry:
    """Parse Polygon, MultiPolygon, Feature, FeatureCollection, or raw coordinates fail-closed."""
    if geometry_data is None:
        raise ValueError("Geometry data is None.")

    if isinstance(geometry_data, BaseGeometry):
        geometry = geometry_data
    elif isinstance(geometry_data, dict):
        if "crs" in geometry_data:
            crs_val = geometry_data.get("crs")
            if crs_val is not None:
                crs_name = ""
                if isinstance(crs_val, dict):
                    props = crs_val.get("properties", {})
                    crs_name = str(props.get("name", ""))
                elif isinstance(crs_val, str):
                    crs_name = crs_val
                crs_upper = crs_name.upper()
                if crs_name and not any(
                    valid in crs_upper
                    for valid in ("4326", "CRS84", "WGS84", "OGC:1.3:CRS84")
                ):
                    raise ValueError(
                        f"Unsupported CRS '{crs_name}': expected WGS84 / EPSG:4326"
                    )

        geometry_type = geometry_data.get("type")
        if geometry_type == "Feature":
            feat_geom = geometry_data.get("geometry")
            if feat_geom is None or not isinstance(feat_geom, (dict, BaseGeometry)):
                raise ValueError("GeoJSON Feature missing or invalid 'geometry' field.")
            return parse_geojson_geometry(feat_geom)
        elif geometry_type == "FeatureCollection":
            features = geometry_data.get("features")
            if not isinstance(features, list) or not features:
                raise ValueError("FeatureCollection contains no features.")
            parsed_geoms = [parse_geojson_geometry(feature) for feature in features]
            geometry = unary_union(parsed_geoms)
        elif geometry_type in ("Polygon", "MultiPolygon", "GeometryCollection"):
            try:
                geometry = shape(geometry_data)
            except Exception as exc:
                raise ValueError(
                    f"Invalid GeoJSON geometry for {geometry_type}: {exc}"
                ) from exc
        else:
            raise ValueError(f"Unsupported GeoJSON geometry type: {geometry_type}")
    elif isinstance(geometry_data, (list, tuple)):
        if not geometry_data:
            raise ValueError("Empty coordinate list.")

        def _depth(item: Any) -> int:
            return (
                1 + _depth(item[0]) if isinstance(item, (list, tuple)) and item else 0
            )

        depth = _depth(geometry_data)
        try:
            if depth == 2:
                geometry = Polygon(geometry_data)
            elif depth == 3:
                geometry = Polygon(geometry_data[0], holes=geometry_data[1:] or None)
            elif depth == 4:
                geometry = MultiPolygon(
                    [
                        Polygon(coords[0], holes=coords[1:] or None)
                        for coords in geometry_data
                    ]
                )
            else:
                raise ValueError(f"Invalid polygon coordinate nesting depth: {depth}")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid polygon coordinate structure: {exc}") from exc
    else:
        raise ValueError(f"Unsupported geometry value: {type(geometry_data).__name__}")

    if geometry.is_empty:
        raise ValueError("Parsed geometry is empty.")

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    # Filter to polygonal components only
    if isinstance(geometry, GeometryCollection) or not isinstance(
        geometry, (Polygon, MultiPolygon)
    ):
        polys: list[Polygon] = []
        if hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                if isinstance(part, Polygon):
                    polys.append(part)
                elif isinstance(part, MultiPolygon):
                    polys.extend(part.geoms)
        if not polys:
            raise ValueError(f"Expected polygonal geometry, got {geometry.geom_type}")
        geometry = unary_union(polys)

    if geometry.is_empty or not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError("Parsed geometry contains no non-empty polygons.")

    _validate_wgs84_coordinates(geometry)

    return geometry


def project_geometry_to_5070(geometry: BaseGeometry) -> BaseGeometry:
    """Project WGS84 geometry to EPSG:5070 for equal-area calculations."""
    return transform(_TO_EQUAL_AREA.transform, geometry)


# Number of tiles processed per vectorized chunk; bounds per-chunk memory
# (one chunk of quad geometries at a time) regardless of total tile count.
_TILE_CHUNK_SIZE = 8192


def _tile_centers_xy(
    xs: np.ndarray, ys: np.ndarray, zoom: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized tile-center (lon, lat), bit-identical to tile_to_lat_lon_center.

    Mirrors the scalar formula and its exact operation order so every element
    equals ``tile_to_lat_lon_center(int(x), int(y), zoom)`` bit for bit
    (numpy and the math module dispatch to the same libm functions for
    ``sinh``/``atan`` and the same ``radians(180/pi)`` scaling).
    """
    n = 1 << zoom
    lons = (xs + 0.5) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (ys + 0.5) / n)))
    lats = np.rad2deg(lat_rad)
    return lons, lats


def _tile_box_corners_xy(
    xs: np.ndarray, ys: np.ndarray, zoom: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized WGS84 tile bounds, bit-identical to tile_bounds_wgs84.

    Returns (min_lon, min_lat, max_lon, max_lat) arrays reproducing the scalar
    helper's arithmetic and evaluation order exactly.
    """
    n = 1 << zoom
    min_lon = xs / n * 360.0 - 180.0
    max_lon = (xs + 1.0) / n * 360.0 - 180.0
    max_lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ys / n))))
    min_lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (ys + 1.0) / n))))
    return min_lon, min_lat, max_lon, max_lat


def _project_tile_quads_5070(
    min_lon: np.ndarray,
    min_lat: np.ndarray,
    max_lon: np.ndarray,
    max_lat: np.ndarray,
) -> np.ndarray:
    """Project WGS84 tile boxes to EPSG:5070 polygons in one vectorized pass.

    Ring order matches ``shapely.geometry.box`` (so each quad equals the
    ``project_geometry_to_5070(box(*tile_bounds_wgs84(...)))`` polygon the
    scalar loop built): (max_lon, min_lat), (max_lon, max_lat),
    (min_lon, max_lat), (min_lon, min_lat), closed. pyproj's array transform
    runs the same per-point ``proj_trans`` math as the scalar transform path,
    so projected vertices are bit-identical.
    """
    corner_lons = np.column_stack((max_lon, max_lon, min_lon, min_lon)).reshape(-1)
    corner_lats = np.column_stack((min_lat, max_lat, max_lat, min_lat)).reshape(-1)
    x_ea, y_ea = _TO_EQUAL_AREA.transform(corner_lons, corner_lats)
    x_ea = x_ea.reshape(-1, 4)
    y_ea = y_ea.reshape(-1, 4)
    ring_x = np.column_stack((x_ea, x_ea[:, :1]))
    ring_y = np.column_stack((y_ea, y_ea[:, :1]))
    return shapely.polygons(np.stack((ring_x, ring_y), axis=-1))


def enumerate_land_tiles(
    land_geometry: Any,
    zoom: int = 14,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Admit tiles whose center is on land or whose land fraction reaches the threshold in EPSG:5070.

    Deterministic, chunked, vectorized Shapely 2 implementation that returns
    the same ordered records as the scalar per-tile loop:

    * tile centers and WGS84 box corners come from the exact scalar formulas
      (bit-identical inputs);
    * every chunk's tile boxes are projected to EPSG:5070 in one pyproj array
      call and intersected with the equal-area land geometry using the same
      GEOS operations the scalar path used;
    * only one chunk of tile geometries is materialized at a time;
    * ``land_fraction`` is clamped to [0.0, 1.0] to neutralize EPSG:5070
      floating-point overflow (canonical preflight observed 1.0000000000000286);
      the threshold test uses the clamped value.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Land fraction threshold must be between 0 and 1.")
    geometry_wgs84 = parse_geojson_geometry(land_geometry)
    geometry_equal_area = project_geometry_to_5070(geometry_wgs84)
    # Prepare both geometries once: the center predicates and the disjoint
    # pre-filter below then run against STRtree-indexed prepared geometries.
    shapely.prepare(geometry_wgs84)
    shapely.prepare(geometry_equal_area)

    min_lon, min_lat, max_lon, max_lat = geometry_wgs84.bounds
    tiles = enumerate_bbox_tiles(min_lat, min_lon, max_lat, max_lon, zoom=zoom)
    admitted: list[dict[str, Any]] = []

    for start in range(0, len(tiles), _TILE_CHUNK_SIZE):
        chunk = tiles[start : start + _TILE_CHUNK_SIZE]
        xs = np.asarray([x for x, _ in chunk], dtype=np.float64)
        ys = np.asarray([y for _, y in chunk], dtype=np.float64)

        center_lon, center_lat = _tile_centers_xy(xs, ys, zoom)
        # For a point against a (Multi)Polygon, covers == intersects: both hold
        # exactly for interior and boundary points. intersects_xy avoids
        # constructing 370k Point objects and uses the prepared geometry.
        center_on_land = shapely.intersects_xy(geometry_wgs84, center_lon, center_lat)

        min_lon_c, min_lat_c, max_lon_c, max_lat_c = _tile_box_corners_xy(xs, ys, zoom)
        tile_quads_5070 = _project_tile_quads_5070(
            min_lon_c, min_lat_c, max_lon_c, max_lat_c
        )
        tile_area = shapely.area(tile_quads_5070)

        # Exact disjoint pre-filter: a quad with empty intersection with the
        # land geometry contributes intersection_area == 0.0 (the scalar loop
        # would produce exactly 0.0 from its empty overlay result), so it can
        # skip the full GEOS overlay. The predicate and the overlay operate on
        # the same bit-identical inputs with the same exact-arithmetic core.
        intersects_land = shapely.intersects(geometry_equal_area, tile_quads_5070)
        intersection_area = np.zeros(len(chunk), dtype=np.float64)
        if np.any(intersects_land):
            intersection_area[intersects_land] = shapely.area(
                shapely.intersection(
                    tile_quads_5070[intersects_land], geometry_equal_area
                )
            )

        land_fraction = np.divide(
            intersection_area,
            tile_area,
            out=np.zeros_like(intersection_area),
            where=tile_area > 0.0,
        )
        np.clip(land_fraction, 0.0, 1.0, out=land_fraction)

        admitted_mask = center_on_land | (
            (intersection_area > 0.0) & (land_fraction >= threshold)
        )
        for idx in np.nonzero(admitted_mask)[0]:
            x_i, y_i = int(xs[idx]), int(ys[idx])
            admitted.append(
                {
                    "coord": (x_i, y_i),
                    "x": x_i,
                    "y": y_i,
                    "land_fraction": float(land_fraction[idx]),
                    "admission_reason": (
                        "center_on_land" if center_on_land[idx] else "land_fraction"
                    ),
                }
            )
    return admitted


def enumerate_polygon_tiles(
    polygon: Any, zoom: int = 14, threshold: float = 0.05
) -> list[tuple[int, int]]:
    """Enumerate tiles admitted by equal-area land intersection."""
    return [
        record["coord"]
        for record in enumerate_land_tiles(polygon, zoom=zoom, threshold=threshold)
    ]


def get_builtin_region_spec() -> dict[str, Any]:
    """Return the canonical near-cap west/south inland expansion region specification."""
    return {
        "version": 1,
        "description": (
            "Preserve New England North and add contiguous land coverage "
            "south to 38N and west to 83.55W using authoritative geometry"
        ),
        "geographic_source": "authoritative land and state boundary geometry file",
        "included_jurisdictions": [
            "Connecticut",
            "Delaware",
            "District of Columbia",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "New Hampshire",
            "New Jersey",
            "New York",
            "Ohio",
            "Pennsylvania",
            "Rhode Island",
            "Vermont",
            "Virginia",
            "West Virginia",
        ],
        "excluded_jurisdictions": [
            "US states wholly south of 38N",
            "US areas wholly west of 83.55W",
        ],
        "known_non_target_coverage": [
            "southern Ontario and Quebec",
            "Great Lakes water",
            "water already inside the preserved New England North bbox",
        ],
        "limitations": [
            "Requires authoritative land and state boundary geometry file for polygon filtering; "
            "preflight fails closed if boundary file is absent or checksum mismatched."
        ],
        "regions": [
            {
                "name": "new_england_north",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.488301979602255,
                    "min_lon": -73.5205078125,
                    "max_lat": 47.50235895196859,
                    "max_lon": -66.796875,
                },
            },
            {
                "name": "west_south_inland",
                "type": "geometry_file",
                "geometry_file": "data/raw/boundaries/us_states_2025.geojson",
                "expected_sha256": None,
            },
        ],
    }


def compute_geometry_digest(spec_data: Any) -> str:
    """Compute deterministic SHA-256 digest of region specification geometry."""
    if isinstance(spec_data, (dict, list)):
        canonical_json = json.dumps(spec_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if isinstance(spec_data, bytes):
        return hashlib.sha256(spec_data).hexdigest()
    if isinstance(spec_data, str):
        return hashlib.sha256(spec_data.encode("utf-8")).hexdigest()
    raise ValueError(
        f"Cannot compute geometry digest for type {type(spec_data).__name__}"
    )


def _load_pinned_geojson(
    path_value: Any, expected_sha256: Any, *, label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(
            f"Authoritative boundary file for {label} not found: {path}"
        )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label} expected_sha256 must be a 64-character hash")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is not valid UTF-8 GeoJSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} GeoJSON must be an object")
    return data, actual_sha256


def parse_and_validate_region_spec(
    spec_data: dict[str, Any],
    app_regions_path: Path | str = "config/app_regions.json",
    max_budget_coords: int = 370000,
    zoom: int = 14,
) -> dict[str, Any]:
    """Parse region specification and enforce all geographic/budget invariants.

    Invariants:
    1. Schema version must be 1.
    2. Must cover all New England North tiles defined in app_regions.json.
    3. Added geometry must NOT expand North or East of New England North bounds.
    4. Unique tile coordinate count must not exceed max_budget_coords.
    """
    version = spec_data.get("version")
    if version != 1:
        raise ValueError(f"Unsupported region spec version: {version}. Expected 1.")

    regions_list = spec_data.get("regions")
    if not isinstance(regions_list, list) or not regions_list:
        raise ValueError(
            "Region specification must contain a non-empty 'regions' list."
        )

    # Load baseline New England North bounds
    app_bounds = load_app_region_bounds(app_regions_path)
    if "new_england_north" not in app_bounds:
        raise ValueError(
            f"Baseline region 'new_england_north' not found in {app_regions_path}."
        )

    nen_bbox = app_bounds["new_england_north"]
    nen_bbox_tiles = set(
        enumerate_bbox_tiles(
            nen_bbox["min_lat"],
            nen_bbox["min_lon"],
            nen_bbox["max_lat"],
            nen_bbox["max_lon"],
            zoom=zoom,
        )
    )
    geometry_source_hashes: dict[str, str] = {}
    baseline_land_path = spec_data.get("baseline_land_geometry_file")
    baseline_land_sha = spec_data.get("baseline_land_geometry_sha256")
    if baseline_land_path is not None or baseline_land_sha is not None:
        baseline_land_data, computed_baseline_land_sha = _load_pinned_geojson(
            baseline_land_path,
            baseline_land_sha,
            label="New England North baseline land geometry",
        )
        geometry_source_hashes["baseline_land"] = computed_baseline_land_sha
        baseline_land = parse_geojson_geometry(baseline_land_data)
        baseline_jurisdiction_path = spec_data.get(
            "baseline_jurisdiction_geometry_file"
        )
        baseline_jurisdiction_sha = spec_data.get(
            "baseline_jurisdiction_geometry_sha256"
        )
        if (
            baseline_jurisdiction_path is not None
            or baseline_jurisdiction_sha is not None
        ):
            (
                baseline_jurisdiction_data,
                computed_baseline_jurisdiction_sha,
            ) = _load_pinned_geojson(
                baseline_jurisdiction_path,
                baseline_jurisdiction_sha,
                label="New England North baseline jurisdiction geometry",
            )
            geometry_source_hashes["baseline_jurisdiction"] = (
                computed_baseline_jurisdiction_sha
            )
            baseline_land = baseline_land.intersection(
                parse_geojson_geometry(baseline_jurisdiction_data)
            )
        nen_min_x = min(x for x, _ in nen_bbox_tiles)
        nen_max_x = max(x for x, _ in nen_bbox_tiles)
        nen_min_y = min(y for _, y in nen_bbox_tiles)
        nen_max_y = max(y for _, y in nen_bbox_tiles)
        nen_west, nen_south, _, _ = tile_bounds_wgs84(nen_min_x, nen_max_y, zoom)
        _, _, nen_east, nen_north = tile_bounds_wgs84(nen_max_x, nen_min_y, zoom)
        nen_land = baseline_land.intersection(
            box(nen_west, nen_south, nen_east, nen_north)
        )
        nen_threshold = float(spec_data.get("land_fraction_threshold", 0.05))
        nen_tiles_set = {
            record["coord"]
            for record in enumerate_land_tiles(
                nen_land, zoom=zoom, threshold=nen_threshold
            )
        }
        if not nen_tiles_set:
            raise ValueError("New England North baseline land geometry admits no tiles")
    else:
        nen_tiles_set = nen_bbox_tiles

    nen_max_x = max(x for x, _ in nen_bbox_tiles)
    nen_min_y = min(y for _, y in nen_bbox_tiles)

    # Process regions in order
    coord_to_region: dict[tuple[int, int], str] = {}
    coord_to_land_fraction: dict[tuple[int, int], float] = {}
    coord_to_admission_reason: dict[tuple[int, int], str] = {}
    region_tile_counts: dict[str, int] = {}
    coord_to_state: dict[tuple[int, int], str | None] = {}
    state_tile_counts: dict[str, int] = {}
    excluded_water_count = 0
    admission_reason_counts = {"center_on_land": 0, "land_fraction": 0, "bbox": 0}
    admitted_land_fractions: list[float] = []
    ordered_coords: list[tuple[int, int]] = []
    seen_coords: set[tuple[int, int]] = set()

    for r_entry in regions_list:
        r_name = r_entry.get("name")
        if not r_name:
            raise ValueError(f"Region entry missing 'name': {r_entry}")

        r_type = r_entry.get("type")
        geom_file_key = next(
            (
                k
                for k in ("geometry_file", "geometry_path", "path", "file")
                if k in r_entry
            ),
            None,
        )
        is_intersection = r_type == "jurisdiction_land_intersection"
        is_geom_file = (
            r_type in ("geometry_file", "polygon_file", "geojson")
            or geom_file_key is not None
        )
        is_bbox = r_type == "bbox" or ("bbox" in r_entry and not is_geom_file)
        is_polygon = r_type == "polygon" or ("polygon" in r_entry and not is_geom_file)
        records: list[dict[str, Any]] = []
        if is_intersection:
            jurisdiction_data, jurisdiction_sha = _load_pinned_geojson(
                r_entry.get("jurisdiction_geometry_file"),
                r_entry.get("jurisdiction_geometry_sha256"),
                label=f"{r_name} jurisdiction geometry",
            )
            land_data, land_sha = _load_pinned_geojson(
                r_entry.get("land_geometry_file"),
                r_entry.get("land_geometry_sha256"),
                label=f"{r_name} land geometry",
            )
            if jurisdiction_sha == land_sha:
                raise ValueError(
                    f"{r_name} jurisdiction and land geometry must be independently pinned"
                )
            geometry_source_hashes[f"{r_name}:jurisdiction"] = jurisdiction_sha
            geometry_source_hashes[f"{r_name}:land"] = land_sha
            land_geometry = parse_geojson_geometry(land_data)
            features = jurisdiction_data.get("features")
            if not isinstance(features, list) or not features:
                raise ValueError(
                    f"{r_name} jurisdiction geometry must be a non-empty FeatureCollection"
                )
            field = r_entry.get("jurisdiction_field")
            included = r_entry.get("included_jurisdictions")
            if not isinstance(field, str) or not field:
                raise ValueError(f"{r_name} jurisdiction_field is required")
            if not isinstance(included, list) or not included:
                raise ValueError(f"{r_name} included_jurisdictions must be non-empty")
            included_values = {str(value) for value in included}
            found_values: set[str] = set()
            threshold = float(r_entry.get("threshold", 0.05))
            for feature in features:
                if not isinstance(feature, dict):
                    raise ValueError(f"{r_name} contains a non-object feature")
                properties = feature.get("properties")
                if not isinstance(properties, dict) or field not in properties:
                    raise ValueError(f"{r_name} feature is missing property {field!r}")
                state = str(properties[field])
                if state not in included_values:
                    continue
                found_values.add(state)
                jurisdiction = parse_geojson_geometry(feature)
                state_land = jurisdiction.intersection(land_geometry)
                if state_land.is_empty:
                    raise ValueError(
                        f"{r_name} has no verified land for jurisdiction {state}"
                    )
                state_records = enumerate_land_tiles(
                    state_land, zoom=zoom, threshold=threshold
                )
                for record in state_records:
                    record["state"] = state
                records.extend(state_records)
                state_tile_counts[state] = len(state_records)
                bounds = state_land.bounds
                bbox_count = len(
                    enumerate_bbox_tiles(
                        bounds[1], bounds[0], bounds[3], bounds[2], zoom=zoom
                    )
                )
                excluded_water_count += bbox_count - len(state_records)
            missing = included_values - found_values
            if missing:
                raise ValueError(
                    f"{r_name} is missing jurisdictions: {sorted(missing)}"
                )

        elif is_geom_file:
            path_str = r_entry.get(geom_file_key) if geom_file_key else None
            json_data, computed_sha = _load_pinned_geojson(
                path_str,
                r_entry.get("expected_sha256") or r_entry.get("sha256"),
                label=f"{r_name} geometry",
            )
            parsed_geom = parse_geojson_geometry(json_data)
            clip_bbox = r_entry.get("clip_bbox")
            if clip_bbox is not None:
                if not isinstance(clip_bbox, dict) or not all(
                    key in clip_bbox
                    for key in ("min_lat", "min_lon", "max_lat", "max_lon")
                ):
                    raise ValueError(
                        f"{r_name} clip_bbox requires min_lat, min_lon, max_lat, and max_lon"
                    )
                parsed_geom = parsed_geom.intersection(
                    box(
                        float(clip_bbox["min_lon"]),
                        float(clip_bbox["min_lat"]),
                        float(clip_bbox["max_lon"]),
                        float(clip_bbox["max_lat"]),
                    )
                )
            threshold = float(r_entry.get("threshold", 0.05))
            records = enumerate_land_tiles(parsed_geom, zoom=zoom, threshold=threshold)

        elif is_bbox:
            bbox = r_entry.get("bbox")
            if not isinstance(bbox, dict) or not all(
                key in bbox for key in ("min_lat", "min_lon", "max_lat", "max_lon")
            ):
                raise ValueError(
                    f"Bbox region '{r_name}' requires min_lat, min_lon, max_lat, and max_lon."
                )
            try:
                b_tiles = enumerate_bbox_tiles(
                    bbox["min_lat"],
                    bbox["min_lon"],
                    bbox["max_lat"],
                    bbox["max_lon"],
                    zoom=zoom,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid bbox for region '{r_name}': {exc}") from exc
            records = [
                {
                    "coord": (x, y),
                    "x": x,
                    "y": y,
                    "land_fraction": 1.0,
                    "admission_reason": "bbox",
                }
                for x, y in b_tiles
            ]

        elif is_polygon:
            poly = r_entry.get("polygon")
            if not poly:
                raise ValueError(
                    f"Polygon region '{r_name}' missing 'polygon' definition."
                )
            parsed_geom = parse_geojson_geometry(poly)
            threshold = float(r_entry.get("threshold", 0.05))
            records = enumerate_land_tiles(parsed_geom, zoom=zoom, threshold=threshold)

        else:
            raise ValueError(f"Unknown region type '{r_type}' for region '{r_name}'.")

        region_tile_counts[r_name] = len(records)
        for rec in records:
            coord = rec["coord"]
            if coord not in seen_coords:
                seen_coords.add(coord)
                ordered_coords.append(coord)
                coord_to_region[coord] = r_name
                coord_to_land_fraction[coord] = rec["land_fraction"]
                coord_to_admission_reason[coord] = rec["admission_reason"]
                coord_to_state[coord] = rec.get("state")
                admission_reason_counts[rec["admission_reason"]] += 1
                admitted_land_fractions.append(float(rec["land_fraction"]))

    # 1. Check New England North Coverage
    missing_nen = nen_tiles_set - seen_coords
    if missing_nen:
        raise ValueError(
            f"Region specification fails to cover {len(missing_nen)} tiles of "
            f"required New England North coverage."
        )

    # 2. Check North / East Expansion Prohibition
    added_tiles = seen_coords - nen_tiles_set
    for x, y in added_tiles:
        if y < nen_min_y:
            raise ValueError(
                f"Region specification violates bounds: added tile ({x}, {y}) is North "
                f"of New England North boundary (y < {nen_min_y})."
            )
        if x > nen_max_x:
            raise ValueError(
                f"Region specification violates bounds: added tile ({x}, {y}) is East "
                f"of New England North boundary (x > {nen_max_x})."
            )

    # 3. Check Budget Cap
    unique_count = len(seen_coords)
    total_rasters = unique_count * 2
    if unique_count > max_budget_coords:
        raise ValueError(
            f"Total unique tile coordinates ({unique_count:,}) exceeds hard budget cap "
            f"of {max_budget_coords:,} ({total_rasters:,} rasters > {max_budget_coords * 2:,})."
        )

    geometry_digest = compute_geometry_digest(
        {
            "region_spec": spec_data,
            "geometry_source_hashes": geometry_source_hashes,
        }
    )

    return {
        "spec_data": spec_data,
        "zoom": zoom,
        "unique_coordinates_count": unique_count,
        "total_rasters_count": total_rasters,
        "nen_tile_count": len(nen_tiles_set),
        "nen_bbox_tile_count": len(nen_bbox_tiles),
        "coord_to_region": coord_to_region,
        "coord_to_land_fraction": coord_to_land_fraction,
        "coord_to_admission_reason": coord_to_admission_reason,
        "ordered_coords": ordered_coords,
        "region_tile_counts": region_tile_counts,
        "geometry_digest": geometry_digest,
        "coord_to_state": coord_to_state,
        "state_tile_counts": state_tile_counts,
        "excluded_water_count": excluded_water_count,
        "admission_reason_counts": admission_reason_counts,
        "land_fraction_summary": {
            "count": len(admitted_land_fractions),
            "min": min(admitted_land_fractions) if admitted_land_fractions else None,
            "max": max(admitted_land_fractions) if admitted_land_fractions else None,
            "mean": (
                sum(admitted_land_fractions) / len(admitted_land_fractions)
                if admitted_land_fractions
                else None
            ),
        },
        "geometry_source_hashes": geometry_source_hashes,
    }
