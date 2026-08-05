"""Region planning and specification validation for active learning data acquisition.

Enforces geographic boundaries, New England North coverage requirements,
north/east expansion prohibitions, tile deduplication, and budget caps.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.data_pipeline.mapbox import lat_lon_to_tile, tile_to_lat_lon_center


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


def point_in_polygon(
    lon: float, lat: float, polygon: Sequence[Sequence[float]]
) -> bool:
    """Ray-casting point-in-polygon test. Polygon points expected as [lon, lat]."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def enumerate_polygon_tiles(
    polygon: Sequence[Sequence[float]], zoom: int = 14
) -> list[tuple[int, int]]:
    """Enumerate tile coordinates whose centers fall within the given polygon [[lon, lat], ...]."""
    if len(polygon) < 3:
        raise ValueError("Polygon must contain at least three points.")
    if any(
        len(point) != 2 or not all(math.isfinite(float(value)) for value in point)
        for point in polygon
    ):
        raise ValueError("Polygon points must be finite [longitude, latitude] pairs.")
    if any(
        not (-180 <= float(point[0]) <= 180 and -90 <= float(point[1]) <= 90)
        for point in polygon
    ):
        raise ValueError(
            "Polygon coordinates are outside valid latitude/longitude ranges."
        )
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    candidate_tiles = enumerate_bbox_tiles(
        min_lat, min_lon, max_lat, max_lon, zoom=zoom
    )
    valid_tiles = []
    for x, y in candidate_tiles:
        clat, clon = tile_to_lat_lon_center(x, y, zoom)
        if point_in_polygon(clon, clat, polygon):
            valid_tiles.append((x, y))
    return valid_tiles


def get_builtin_region_spec() -> dict[str, Any]:
    """Return the canonical near-cap west/south inland expansion."""
    return {
        "version": 1,
        "description": (
            "Preserve New England North and add one contiguous inland band "
            "south to 38N and west to 83.55W while avoiding Atlantic expansion"
        ),
        "geographic_source": "deterministic collection of two adjoining zoom-14 bounding boxes",
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
            "No compatible land/state polygon dataset is available in the repository; "
            "tile centers are filtered by adjoining bounding boxes, so non-US land and "
            "inland water remain and must be measured downstream."
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
                "type": "bbox",
                "bbox": {
                    "min_lat": 38.0,
                    "min_lon": -83.55,
                    "max_lat": 47.50235895196859,
                    "max_lon": -73.5205078125,
                },
            },
        ],
    }


def compute_geometry_digest(spec_data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of region specification geometry."""
    canonical_json = json.dumps(spec_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


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
    nen_tiles_list = enumerate_bbox_tiles(
        nen_bbox["min_lat"],
        nen_bbox["min_lon"],
        nen_bbox["max_lat"],
        nen_bbox["max_lon"],
        zoom=zoom,
    )
    nen_tiles_set = set(nen_tiles_list)

    nen_max_x = max(x for x, _ in nen_tiles_set)
    nen_min_y = min(y for _, y in nen_tiles_set)

    # Process regions in order
    coord_to_region: dict[tuple[int, int], str] = {}
    region_tile_counts: dict[str, int] = {}
    ordered_coords: list[tuple[int, int]] = []
    seen_coords: set[tuple[int, int]] = set()

    for r_entry in regions_list:
        r_name = r_entry.get("name")
        r_type = r_entry.get("type", "bbox")
        if not r_name:
            raise ValueError(f"Region entry missing 'name': {r_entry}")

        if r_type == "bbox":
            bbox = r_entry.get("bbox")
            if not isinstance(bbox, dict) or not all(
                key in bbox for key in ("min_lat", "min_lon", "max_lat", "max_lon")
            ):
                raise ValueError(
                    f"Bbox region '{r_name}' requires min_lat, min_lon, max_lat, and max_lon."
                )
            try:
                r_tiles = enumerate_bbox_tiles(
                    bbox["min_lat"],
                    bbox["min_lon"],
                    bbox["max_lat"],
                    bbox["max_lon"],
                    zoom=zoom,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid bbox for region '{r_name}': {exc}") from exc
        elif r_type == "polygon":
            poly = r_entry.get("polygon")
            if not poly:
                raise ValueError(f"Polygon region '{r_name}' missing 'polygon' list.")
            r_tiles = enumerate_polygon_tiles(poly, zoom=zoom)
        else:
            raise ValueError(f"Unknown region type '{r_type}' for region '{r_name}'.")

        added_count = 0
        for coord in r_tiles:
            if coord not in seen_coords:
                seen_coords.add(coord)
                ordered_coords.append(coord)
                coord_to_region[coord] = r_name
                added_count += 1
        region_tile_counts[r_name] = len(r_tiles)

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

    geometry_digest = compute_geometry_digest(spec_data)

    return {
        "spec_data": spec_data,
        "zoom": zoom,
        "unique_coordinates_count": unique_count,
        "total_rasters_count": total_rasters,
        "nen_tile_count": len(nen_tiles_set),
        "coord_to_region": coord_to_region,
        "ordered_coords": ordered_coords,
        "region_tile_counts": region_tile_counts,
        "geometry_digest": geometry_digest,
    }
