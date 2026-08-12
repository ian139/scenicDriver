"""
Build a RoadGraph from local OSM PBF extracts.

The repeatable local-PBF inputs are merged, filtered, and converted to XML
before OSMnx loads the resulting graph.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import re
import subprocess
import sys
import gc
from typing import AbstractSet, Any, Iterable, Mapping

try:
    from shapely.geometry import LineString, box as shapely_box, shape as shapely_shape
except ImportError:  # pragma: no cover - dependency is declared by the project
    LineString = None
    shapely_box = None
    shapely_shape = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import (  # noqa: E402
    CompactRoadGraph,
    EdgeProjectionIndex,
    RoadGraph,
    _graph_from_osmnx,
    _iter_osmnx_graph_rows,
    _write_sqlite_graph,
    write_compact_graph,
)


PBF_HASH_CHUNK_SIZE = 8 * 1024 * 1024
DRIVE_EXCLUDED_HIGHWAYS = frozenset(
    {
        "abandoned",
        "bridleway",
        "bus_guideway",
        "construction",
        "corridor",
        "cycleway",
        "elevator",
        "escalator",
        "footway",
        "no",
        "path",
        "pedestrian",
        "planned",
        "platform",
        "proposed",
        "raceway",
        "razed",
        "rest_area",
        "service",
        "services",
        "steps",
        "track",
    }
)
DRIVE_EXCLUDED_SERVICES = frozenset(
    {
        "alley",
        "driveway",
        "emergency_access",
        "parking",
        "parking_aisle",
        "private",
    }
)
DRIVE_FILTER_VERSION = "osmnx-2.1-drive-v2"
PBF_DERIVED_CACHE_VERSION = "complete-ways-highway-filter-v1"
TILE_EXTRACTION_VERSION = "tile-footprint-v1"
DEFAULT_TILE_BUFFER = 1
TILE_MAX_ZOOM = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build road graph from local OSM PBF extracts")
    parser.add_argument("--min-lat", type=float, default=None)
    parser.add_argument("--min-lon", type=float, default=None)
    parser.add_argument("--max-lat", type=float, default=None)
    parser.add_argument("--max-lon", type=float, default=None)
    parser.add_argument(
        "--tile-report",
        type=Path,
        default=None,
        help=(
            "Heuristic run report JSON whose tiles define the extraction footprint. "
            "Mutually exclusive with the --min/max bbox flags."
        ),
    )
    parser.add_argument(
        "--tile-buffer",
        type=int,
        default=1,
        help="Integer tile-count connectivity buffer around the report tiles (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit graph output path. If omitted, a deterministic run folder is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/road_graphs"),
        help="Root folder for deterministic graph-cache runs when --output is omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name. If omitted, a deterministic name is generated.",
    )
    parser.add_argument("--network", type=str, default="drive")
    parser.add_argument(
        "--source-pbf",
        type=Path,
        action="append",
        required=True,
        help="Local PBF extract. Repeat for every source in the merged build.",
    )
    parser.add_argument("--require-source-checksums", action="store_true")
    parser.add_argument(
        "--graph-format",
        choices=("json", "sqlite3", "compact"),
        default="json",
        help=(
            "Artifact layout: json (legacy), sqlite3 (SQLite audit only), or "
            "compact (SQLite audit plus the canonical mmap runtime manifest)."
        ),
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=None,
        help="OSMnx/cache folder. Local conversion intermediates are stored below it.",
    )
    parser.add_argument(
        "--coverage-probe",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "LAT", "LON"),
        help="Named coordinate used to validate graph coverage. Repeatable.",
    )
    args = parser.parse_args()
    _validate_mode_flags(parser, args)
    return args


def _validate_mode_flags(
    parser: argparse.ArgumentParser | None,
    args: argparse.Namespace,
) -> None:
    bbox_flags = [
        name
        for name, value in (
            ("--min-lat", getattr(args, "min_lat", None)),
            ("--min-lon", getattr(args, "min_lon", None)),
            ("--max-lat", getattr(args, "max_lat", None)),
            ("--max-lon", getattr(args, "max_lon", None)),
        )
        if value is not None
    ]
    tile_report = getattr(args, "tile_report", None)
    if tile_report is not None and bbox_flags:
        message = (
            "--tile-report is mutually exclusive with the bbox flags "
            f"({', '.join(bbox_flags)})"
        )
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    if tile_report is None and len(bbox_flags) != 4:
        message = "bbox mode requires --min-lat, --min-lon, --max-lat and --max-lon"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    if getattr(args, "tile_buffer", DEFAULT_TILE_BUFFER) < 0:
        message = "--tile-buffer must be >= 0"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)




def _slug_float(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    tile_report = getattr(args, "tile_report", None)
    if tile_report is not None:
        return (
            f"osm_{args.network}_tiles_"
            f"{Path(tile_report).stem}_buf{getattr(args, 'tile_buffer', DEFAULT_TILE_BUFFER)}"
        )
    return (
        f"osm_{args.network}_"
        f"{_slug_float(args.min_lat)}_{_slug_float(args.min_lon)}_"
        f"{_slug_float(args.max_lat)}_{_slug_float(args.max_lon)}"
    )


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    suffix = (
        ".sqlite3"
        if args.graph_format in ("sqlite3", "compact")
        else ".json"
    )
    if args.output is not None:
        run_name = args.run_name or args.output.stem
        return args.output, args.output.parent, run_name

    run_name = args.run_name or _default_run_name(args)
    run_dir = args.output_root / run_name
    return run_dir / f"road_graph{suffix}", run_dir, run_name


def _bbox(args: argparse.Namespace) -> dict[str, float]:
    min_lat = getattr(args, "min_lat", None)
    min_lon = getattr(args, "min_lon", None)
    max_lat = getattr(args, "max_lat", None)
    max_lon = getattr(args, "max_lon", None)
    if (
        min_lat is None
        or min_lon is None
        or max_lat is None
        or max_lon is None
        or min_lat >= max_lat
        or min_lon >= max_lon
    ):
        raise ValueError("Invalid bbox: --min-lat < --max-lat and --min-lon < --max-lon required")
    return {
        "min_lat": float(min_lat),
        "min_lon": float(min_lon),
        "max_lat": float(max_lat),
        "max_lon": float(max_lon),
    }


def _parse_probes(values: Iterable[Iterable[str]]) -> list[dict[str, float | str]]:
    probes: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for raw in values:
        name, raw_lat, raw_lon = raw
        if name in seen:
            raise ValueError(f"Duplicate coverage probe: {name}")
        lat = float(raw_lat)
        lon = float(raw_lon)
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not -90.0 <= lat <= 90.0
            or not -180.0 <= lon <= 180.0
        ):
            raise ValueError(f"Invalid coverage probe coordinates: {name}")
        seen.add(name)
        probes.append({"name": name, "lat": lat, "lon": lon})
    return probes


def _parse_tile_report(report_path: Path) -> tuple[int, list[tuple[int, int]]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Tile report JSON is missing: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Tile report JSON is not readable: {report_path}") from exc
    tiles = payload.get("tiles") if isinstance(payload, dict) else None
    if not isinstance(tiles, list) or not tiles:
        raise ValueError(f"Tile report has no tiles entries: {report_path}")

    zoom: int | None = None
    seen: set[tuple[int, int]] = set()
    coords: list[tuple[int, int]] = []
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"Tile report tile {index} is not an object")
        raw_z = tile.get("z")
        raw_x = tile.get("x")
        raw_y = tile.get("y")
        if not isinstance(raw_z, int) or not isinstance(raw_x, int) or not isinstance(raw_y, int):
            raise ValueError(f"Tile report tile {index} has non-integer z/x/y")
        if not 0 <= raw_z <= TILE_MAX_ZOOM:
            raise ValueError(f"Tile report tile {index} has out-of-range zoom {raw_z}")
        limit = 2**raw_z
        if not 0 <= raw_x < limit or not 0 <= raw_y < limit:
            raise ValueError(f"Tile report tile {index} is outside the z{raw_z} grid")
        if zoom is None:
            zoom = raw_z
        elif raw_z != zoom:
            raise ValueError(f"Tile report mixes zoom levels: {zoom} and {raw_z}")
        if (raw_x, raw_y) not in seen:
            seen.add((raw_x, raw_y))
            coords.append((raw_x, raw_y))
    if zoom is None:
        raise ValueError(f"Tile report has no tiles entries: {report_path}")
    return zoom, coords


def _tile_lonlat_bounds(
    x: int,
    y: int,
    zoom: int,
) -> tuple[float, float, float, float]:
    n = 2**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def _compact_tile_rectangles(
    zoom: int,
    coords: list[tuple[int, int]],
    buffer: int,
) -> list[tuple[int, int, int, int]]:
    if buffer < 0:
        raise ValueError("tile buffer must be >= 0")
    buffered: set[tuple[int, int]] = set()
    for x, y in coords:
        for dx in range(-buffer, buffer + 1):
            for dy in range(-buffer, buffer + 1):
                buffered.add((x + dx, y + dy))

    runs: dict[int, list[tuple[int, int]]] = {}
    for y in sorted({y for _x, y in buffered}):
        row = sorted(x for x, _y in buffered if _y == y)
        row_runs: list[tuple[int, int]] = []
        run_start = row[0]
        run_end = row[0]
        for x in row[1:]:
            if x == run_end + 1:
                run_end = x
            else:
                row_runs.append((run_start, run_end))
                run_start = x
                run_end = x
        row_runs.append((run_start, run_end))
        runs[y] = row_runs

    rectangles: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    previous_y: int | None = None
    for y in sorted(runs):
        if previous_y is not None and y != previous_y + 1:
            rectangles.extend(active.values())
            active.clear()
        current_spans = set(runs[y])
        for span in sorted(set(active) - current_spans):
            rectangles.append(active.pop(span))
        next_active: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for start, end in runs[y]:
            prior = active.get((start, end))
            start_y = prior[2] if prior is not None else y
            next_active[(start, end)] = (start, end, start_y, y)
        active = next_active
        previous_y = y
    rectangles.extend(active.values())
    rectangles.sort(key=lambda rect: (rect[2], rect[0], rect[1], rect[3]))
    return rectangles


def _rectangles_to_polygons(
    zoom: int,
    rectangles: list[tuple[int, int, int, int]],
) -> list[list[list[list[float]]]]:
    polygons: list[list[list[list[float]]]] = []
    for x_start, x_end, y_start, y_end in rectangles:
        west, _south, _east, north = _tile_lonlat_bounds(x_start, y_start, zoom)
        _west, south, east, _north = _tile_lonlat_bounds(x_end, y_end, zoom)
        polygons.append(
            [
                [
                    [west, north],
                    [east, north],
                    [east, south],
                    [west, south],
                    [west, north],
                ]
            ]
        )
    return polygons


def _footprint_geojson(
    zoom: int,
    rectangles: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    polygons = _rectangles_to_polygons(zoom, rectangles)
    geometry: dict[str, Any]
    if len(polygons) == 1:
        geometry = {"type": "Polygon", "coordinates": polygons[0]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": polygons}
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "zoom": zoom,
                    "rectangle_count": len(rectangles),
                    "tile_footprint_version": TILE_EXTRACTION_VERSION,
                },
                "geometry": geometry,
            }
        ],
    }


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_tile_footprint(
    report_path: Path,
    buffer: int,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    zoom, coords = _parse_tile_report(report_path)
    rectangles = _compact_tile_rectangles(zoom, coords, buffer)
    polygons = _rectangles_to_polygons(zoom, rectangles)
    geojson = _footprint_geojson(zoom, rectangles)
    report_digest = _digest_file(report_path, "sha256")
    footprint_digest = _digest_json(
        {
            "version": TILE_EXTRACTION_VERSION,
            "report_sha256": report_digest,
            "buffer_tiles": buffer,
            "zoom": zoom,
            "tile_count": len(coords),
            "rectangles": rectangles,
        }
    )
    ring_points = [
        point for polygon in polygons for ring in polygon for point in ring
    ]
    west = min(point[0] for point in ring_points)
    south = min(point[1] for point in ring_points)
    east = max(point[0] for point in ring_points)
    north = max(point[1] for point in ring_points)
    return {
        "report_path": report_path,
        "report_sha256": report_digest,
        "zoom": zoom,
        "tile_count": len(coords),
        "buffer": buffer,
        "rectangles": rectangles,
        "polygons": polygons,
        "geojson": geojson,
        "footprint_digest": footprint_digest,
        "bbox": {
            "min_lon": west,
            "min_lat": south,
            "max_lon": east,
            "max_lat": north,
        },
    }


def _settings_record(ox: Any) -> dict[str, Any]:
    return {
        "version": getattr(ox, "__version__", "unknown"),
        "cache_folder": getattr(ox.settings, "cache_folder", None),
        "use_cache": getattr(ox.settings, "use_cache", None),
    }


def _configure_osmnx(ox: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_folder is not None:
        args.cache_folder.mkdir(parents=True, exist_ok=True)
        ox.settings.cache_folder = str(args.cache_folder)
    ox.settings.use_cache = True
    return _settings_record(ox)


def _write_build_state(
    path: Path,
    *,
    run_name: str,
    bbox: Mapping[str, float] | None,
    network: str,
    graph_format: str,
    source_pbf: list[str],
    settings: Mapping[str, Any],
    stage: str,
    status: str,
    last_error: str | None,
    extra: Mapping[str, Any] | None = None,
    footprint: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_name": run_name,
        "bbox": dict(bbox) if bbox is not None else None,
        "network": network,
        "graph_format": graph_format,
        "source_pbf": list(source_pbf),
        "settings": dict(settings),
        "stage": stage,
        "status": status,
        "last_error": last_error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if footprint is not None:
        payload["tile_footprint"] = _footprint_metadata(footprint)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(PBF_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expected_md5(path: Path) -> str:
    sidecar = Path(f"{path}.md5")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Source checksum sidecar is missing: {sidecar}")
    match = re.search(r"\b([0-9a-fA-F]{32})\b", sidecar.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Source checksum sidecar is malformed: {sidecar}")
    return match.group(1).lower()


def _source_manifest(source_paths: list[Path], require_checksums: bool) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for source in source_paths:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source PBF is missing: {source}")
        expected_md5 = _expected_md5(source) if require_checksums else None
        actual_md5 = _digest_file(source, "md5")
        if expected_md5 is not None and actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 checksum mismatch for {source}: expected {expected_md5}, got {actual_md5}"
            )
        manifest.append(
            {
                "path": str(source),
                "size_bytes": int(source.stat().st_size),
                "md5": actual_md5,
                "verified_md5": expected_md5 or actual_md5,
                "sha256": _digest_file(source, "sha256"),
            }
        )
    return manifest


def _replication_timestamp(path: Path) -> str:
    result = _run_command(["osmium", "fileinfo", "--extended", str(path)])
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(
        r"osmosis_replication_timestamp\s*[=:]\s*([^\s]+)",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"No osmosis_replication_timestamp in osmium fileinfo: {path}")
    return match.group(1).strip().rstrip(",")


def _source_digests(manifest: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in manifest:
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _footprint_cache_digest(footprint: Mapping[str, Any]) -> str:
    return _digest_json(
        {
            "version": TILE_EXTRACTION_VERSION,
            "report_sha256": footprint["report_sha256"],
            "zoom": footprint["zoom"],
            "tile_count": footprint["tile_count"],
            "buffer_tiles": footprint["buffer"],
            "rectangles": footprint["rectangles"],
        }
    )


def _derived_cache_digest(
    manifest: list[dict[str, Any]],
    bbox: Mapping[str, float],
    footprint: Mapping[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(_source_digests(manifest).encode("ascii"))
    if footprint is not None:
        extraction: dict[str, Any] = {
            "mode": "tiles",
            "footprint_digest": footprint["footprint_digest"],
            "tile_report": str(footprint["report_path"]),
            "tile_report_sha256": footprint["report_sha256"],
            "zoom": footprint["zoom"],
            "tile_count": footprint["tile_count"],
            "buffer_tiles": footprint["buffer"],
        }
    else:
        extraction = {
            "mode": "bbox",
            "bbox": {
                key: float(bbox[key])
                for key in ("min_lat", "min_lon", "max_lat", "max_lon")
            },
            "buffer_m": 500.0,
        }
    cache_descriptor = {
        "cache_version": PBF_DERIVED_CACHE_VERSION,
        "extraction": extraction,
        "filter_version": DRIVE_FILTER_VERSION,
        "excluded_highways": sorted(DRIVE_EXCLUDED_HIGHWAYS),
        "excluded_services": sorted(DRIVE_EXCLUDED_SERVICES),
        "extract_strategy": "complete_ways",
        "tag_filter": "w/highway",
    }
    digest.update(
        json.dumps(
            cache_descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()[:24]


def _buffered_bbox(bbox: Mapping[str, float], buffer_m: float = 500.0) -> tuple[float, float, float, float]:
    center_lat = (float(bbox["min_lat"]) + float(bbox["max_lat"])) / 2.0
    lat_delta = buffer_m / 111_320.0
    lon_scale = max(math.cos(math.radians(center_lat)), 0.1)
    lon_delta = buffer_m / (111_320.0 * lon_scale)
    return (
        float(bbox["min_lon"]) - lon_delta,
        float(bbox["min_lat"]) - lat_delta,
        float(bbox["max_lon"]) + lon_delta,
        float(bbox["max_lat"]) + lat_delta,
    )


def _cached_path(
    cache_root: Path,
    digest: str,
    name: str,
    suffix: str | None = None,
) -> Path:
    folder = cache_root / "pbf" / digest
    folder.mkdir(parents=True, exist_ok=True)
    if suffix is not None:
        name = f"{name}{suffix}"
    return folder / name


def _footprint_metadata(footprint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "tiles",
        "report_path": str(footprint["report_path"]),
        "report_sha256": footprint["report_sha256"],
        "zoom": footprint["zoom"],
        "tile_count": footprint["tile_count"],
        "buffer_tiles": footprint["buffer"],
        "rectangle_count": len(footprint["rectangles"]),
        "footprint_digest": footprint["footprint_digest"],
    }


def _persist_footprint_geometry(
    footprint: Mapping[str, Any],
    cache_root: Path,
) -> Path:
    geometry_path = _cached_path(
        cache_root,
        _footprint_cache_digest(footprint),
        "extraction.geojson",
    )
    geometry_path.write_text(
        json.dumps(footprint["geojson"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return geometry_path


def _merge_and_filter_pbf(
    source_paths: list[Path],
    manifest: list[dict[str, Any]],
    bbox: Mapping[str, float] | None,
    cache_root: Path,
    footprint: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, str]]:
    source_paths = sorted(
        (Path(path).expanduser().resolve() for path in source_paths),
        key=lambda path: str(path),
    )
    source_digest = _source_digests(manifest)
    derived_digest = _derived_cache_digest(manifest, bbox or {}, footprint)
    merged = _cached_path(cache_root, source_digest, "merged.osm.pbf")
    buffered = _cached_path(cache_root, derived_digest, "buffered.osm.pbf")
    filtered = _cached_path(cache_root, derived_digest, "filtered.osm.pbf")
    xml_bz2 = _cached_path(cache_root, derived_digest, "filtered.osm.bz2")
    timestamps = {str(path): _replication_timestamp(path) for path in source_paths}
    if len(set(timestamps.values())) != 1:
        raise ValueError(f"Source PBF replication timestamps differ: {timestamps}")

    if not merged.exists():
        _run_command(
            [
                "osmium",
                "merge",
                "--overwrite",
                "-o",
                str(merged),
                *(str(path) for path in source_paths),
            ]
        )
    if not buffered.exists():
        extract_command: list[str] = [
            "osmium",
            "extract",
            "--overwrite",
            "--strategy",
            "complete_ways",
        ]
        if footprint is not None:
            geometry_path = _persist_footprint_geometry(footprint, cache_root)
            extract_command.extend(
                [
                    "--polygon",
                    str(geometry_path),
                ]
            )
        else:
            west, south, east, north = _buffered_bbox(bbox or {})
            extract_command.extend(
                [
                    "--bbox",
                    f"{west},{south},{east},{north}",
                ]
            )
        extract_command.extend(
            [
                "-o",
                str(buffered),
                str(merged),
            ]
        )
        _run_command(extract_command)
    if not filtered.exists():
        _run_command(
            [
                "osmium",
                "tags-filter",
                "--overwrite",
                str(buffered),
                "w/highway",
                "-o",
                str(filtered),
            ]
        )
    _run_command(["osmium", "check-refs", str(filtered)])
    if not xml_bz2.exists():
        _run_command(
            [
                "osmium",
                "cat",
                "--overwrite",
                str(filtered),
                "-o",
                str(xml_bz2),
            ]
        )
    return xml_bz2, timestamps


def _tag_values(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value}
    if value is None:
        return set()
    return {str(value).strip().lower()}


def _tag_contains(value: Any, needles: AbstractSet[str]) -> bool:
    return any(needle in item for item in _tag_values(value) for needle in needles)


def _drive_edge_allowed(data: Mapping[str, Any]) -> bool:
    highway_values = _tag_values(data.get("highway"))
    if not highway_values or _tag_contains(data.get("highway"), DRIVE_EXCLUDED_HIGHWAYS):
        return False
    if _tag_contains(data.get("area"), {"yes"}):
        return False
    if _tag_contains(data.get("access"), {"private"}):
        return False
    if _tag_contains(data.get("motor_vehicle"), {"no"}):
        return False
    if _tag_contains(data.get("motorcar"), {"no"}):
        return False
    if _tag_contains(data.get("service"), DRIVE_EXCLUDED_SERVICES):
        return False
    return True


def _truncate_graph_to_bbox(
    G: Any,
    bbox: tuple[float, float, float, float],
) -> Any:
    west, south, east, north = bbox
    remove_nodes = [
        node_id
        for node_id, data in G.nodes(data=True)
        if not (
            south <= float(data.get("y")) <= north
            and west <= float(data.get("x")) <= east
        )
    ]
    if remove_nodes:
        G.remove_nodes_from(remove_nodes)
    return G


def _largest_weak_component(G: Any) -> Any:
    if len(G.nodes) == 0:
        return G
    import networkx as nx

    component = max(nx.weakly_connected_components(G), key=len)
    keep = set(component)
    remove_nodes = [node_id for node_id in G.nodes if node_id not in keep]
    if remove_nodes:
        G.remove_nodes_from(remove_nodes)
    return G


def _truncate_graph_to_footprint(G: Any, footprint: Mapping[str, Any]) -> Any:
    if LineString is None or shapely_box is None or shapely_shape is None:  # pragma: no cover
        raise RuntimeError("shapely is required for tile footprint graph filtering")
    footprint_geometry = shapely_shape(footprint["geojson"]["features"][0]["geometry"])
    remove_edges: list[tuple[Any, Any, Any]] = []
    for u, v, key, data in G.edges(keys=True, data=True):
        geometry = data.get("geometry")
        line: Any = None
        if isinstance(geometry, dict):
            try:
                line = shapely_shape(geometry)
            except Exception:
                line = None
        if line is None or line.is_empty:
            u_data = G.nodes.get(u)
            v_data = G.nodes.get(v)
            if u_data is None or v_data is None:
                remove_edges.append((u, v, key))
                continue
            line = LineString(
                [
                    (float(u_data.get("x")), float(u_data.get("y"))),
                    (float(v_data.get("x")), float(v_data.get("y"))),
                ]
            )
        if footprint_geometry.intersects(line):
            continue
        remove_edges.append((u, v, key))
    if remove_edges:
        G.remove_edges_from(remove_edges)

    incident_nodes = {
        node_id
        for u, v in G.edges()
        for node_id in (u, v)
    }
    remove_nodes = [
        node_id
        for node_id in G.nodes
        if node_id not in incident_nodes
        and not footprint_geometry.covers(
            shapely_box(
                float(G.nodes[node_id].get("x")),
                float(G.nodes[node_id].get("y")),
                float(G.nodes[node_id].get("x")),
                float(G.nodes[node_id].get("y")),
            )
        )
    ]
    if remove_nodes:
        G.remove_nodes_from(remove_nodes)
    return G




def _load_local_osm_graph(
    ox: Any,
    xml_path: Path,
    bbox: tuple[float, float, float, float] | None,
    footprint: Mapping[str, Any] | None = None,
) -> Any:
    useful_tags = list(getattr(ox.settings, "useful_tags_way", []))
    for tag in ("motor_vehicle", "motorcar"):
        if tag not in useful_tags:
            useful_tags.append(tag)
    ox.settings.useful_tags_way = useful_tags
    graph = ox.graph_from_xml(
        xml_path,
        bidirectional=False,
        simplify=False,
        retain_all=True,
    )
    if footprint is not None:
        graph = _filter_drive_graph(
            graph,
            (
                footprint["bbox"]["min_lon"],
                footprint["bbox"]["min_lat"],
                footprint["bbox"]["max_lon"],
                footprint["bbox"]["max_lat"],
            ),
        )
        graph = _truncate_graph_to_footprint(graph, footprint)
        graph = ox.simplify_graph(graph)
        return graph
    if bbox is None:
        raise ValueError("bbox is required when tile footprint mode is disabled")
    buffered_bbox = _buffered_bbox(
        {
            "min_lat": bbox[1],
            "min_lon": bbox[0],
            "max_lat": bbox[3],
            "max_lon": bbox[2],
        }
    )
    graph = _filter_drive_graph(graph, buffered_bbox)
    graph = _largest_weak_component(graph)
    graph = ox.simplify_graph(graph)
    graph = _truncate_graph_to_bbox(graph, bbox)
    graph = _largest_weak_component(graph)
    return graph
def _filter_drive_graph(G: Any, bbox: tuple[float, float, float, float]) -> Any:
    west, south, east, north = bbox
    remove_edges: list[tuple[Any, Any, Any]] = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if not _drive_edge_allowed(data):
            remove_edges.append((u, v, key))
    if remove_edges:
        G.remove_edges_from(remove_edges)

    remove_nodes = [
        node_id
        for node_id, data in G.nodes(data=True)
        if not (
            south <= float(data.get("y")) <= north
            and west <= float(data.get("x")) <= east
        )
    ]
    if remove_nodes:
        G.remove_nodes_from(remove_nodes)
    isolates = [node_id for node_id in G.nodes if G.degree(node_id) == 0]
    if isolates:
        G.remove_nodes_from(isolates)
    return G




def _coverage_metadata(graph: RoadGraph, probes: list[dict[str, float | str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for probe in probes:
        name = str(probe["name"])
        lat = float(probe["lat"])
        lon = float(probe["lon"])
        projections, distance_km = graph.find_nearest_edge_positions_with_distance(lat, lon)
        if not projections:
            raise ValueError(f"Coverage probe has no nearest edge: {name}")
        projection = projections[0]
        coordinate = [float(projection.lat), float(projection.lon)]
        result[name] = {
            "lat": lat,
            "lon": lon,
            "distance_km": float(distance_km),
            "nearest_edge_coordinate": coordinate,
            "nearest_edge": coordinate,
        }
    return result


def _update_sqlite_metadata(path: Path, updates: Mapping[str, Any]) -> None:
    with __import__("sqlite3").connect(path) as connection:
        for key, value in updates.items():
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(key), json.dumps(value, sort_keys=True, separators=(",", ":"))),
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity!r}")
        connection.commit()


def _publish_sqlite_graph(
    output_path: Path,
    run_dir: Path,
    G: Any,
    scenic_scores: Mapping[str, float],
    metadata: Mapping[str, Any],
    probes: list[dict[str, float | str]],
) -> tuple[int, int, dict[str, Any]]:
    candidate = run_dir / f".{output_path.stem}.candidate.sqlite3"
    candidate_sidecar = EdgeProjectionIndex.sidecar_path(candidate)
    candidate.unlink(missing_ok=True)
    candidate_sidecar.unlink(missing_ok=True)
    try:
        counts = _write_sqlite_graph(
            candidate,
            _iter_osmnx_graph_rows(G, scenic_scores),
            metadata=metadata,
        )
        clear_graph = getattr(G, "clear", None)
        if callable(clear_graph):
            clear_graph()
        del G
        gc.collect()
        loaded = RoadGraph.load(candidate)
        probe_metadata = _coverage_metadata(loaded, probes)
        _update_sqlite_metadata(candidate, {"coverage_probes": probe_metadata})
        loaded.persist_edge_projection_index(candidate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_sidecar = EdgeProjectionIndex.sidecar_path(output_path)
        os.replace(candidate, output_path)
        os.replace(candidate_sidecar, output_sidecar)
        return counts[0], counts[1], probe_metadata
    finally:
        candidate.unlink(missing_ok=True)
        candidate_sidecar.unlink(missing_ok=True)


def _publish_compact_graph(
    sqlite_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Publish the canonical compact runtime artifacts from a published SQLite.

    The SQLite file is the authoritative audit source; conversion streams its
    rows in deterministic rowid order and never rebuilds Node/Edge objects.
    """
    manifest_path = sqlite_path.with_name(f"{sqlite_path.stem}.compact.json")
    return write_compact_graph(sqlite_path, manifest_path)




def _build(args: argparse.Namespace) -> dict[str, Any]:
    _validate_mode_flags(None, args)
    source_values = getattr(args, "source_pbf", None) or []
    if not source_values:
        raise ValueError("At least one --source-pbf is required")
    footprint: dict[str, Any] | None = None
    if getattr(args, "tile_report", None) is not None:
        footprint = _load_tile_footprint(
            args.tile_report,
            getattr(args, "tile_buffer", DEFAULT_TILE_BUFFER),
        )
        bbox: dict[str, float] | None = footprint["bbox"]
    else:
        bbox = _bbox(args)
    probes = _parse_probes(args.coverage_probe)
    if args.network != "drive":
        raise ValueError("PBF ingestion currently supports --network drive only")
    if shutil.which("osmium") is None:
        raise FileNotFoundError(
            "PBF ingestion requires the 'osmium' executable on PATH"
        )

    output_path, run_dir, run_name = _resolve_output_paths(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    build_state_path = run_dir / "build_state.json"
    cache_root = args.cache_folder or Path("cache/osmnx") / run_name
    cache_root.mkdir(parents=True, exist_ok=True)
    source_names = [str(Path(path).expanduser()) for path in source_values]
    settings: dict[str, Any] = {
        "cache_folder": str(cache_root),
        "use_cache": True,
    }
    _write_build_state(
        build_state_path,
        run_name=run_name,
        bbox=bbox,
        network=args.network,
        graph_format=args.graph_format,
        source_pbf=source_names,
        settings=settings,
        stage="acquisition",
        status="running",
        last_error=None,
        footprint=footprint,
    )

    source_manifest: list[dict[str, Any]] = []
    replication_timestamps: dict[str, str] = {}
    try:
        import osmnx as ox

        osm_settings = _configure_osmnx(ox, args)
        settings.update(osm_settings)
        source_paths = sorted(
            (Path(path).expanduser().resolve() for path in source_values),
            key=lambda path: str(path),
        )
        source_manifest = _source_manifest(
            source_paths,
            args.require_source_checksums,
        )
        xml_path, replication_timestamps = _merge_and_filter_pbf(
            source_paths,
            source_manifest,
            bbox,
            cache_root,
            footprint=footprint,
        )
        for row in source_manifest:
            row["replication_timestamp"] = replication_timestamps[row["path"]]
        _write_build_state(
            build_state_path,
            run_name=run_name,
            bbox=bbox,
            network=args.network,
            graph_format=args.graph_format,
            source_pbf=source_names,
            settings=settings,
            stage="acquisition",
            status="running",
            last_error=None,
            extra={
                "source_manifest": source_manifest,
                "replication_timestamps": replication_timestamps,
            },
            footprint=footprint,
        )
        if footprint is not None:
            load_bbox: tuple[float, float, float, float] | None = None
        else:
            if bbox is None:
                raise AssertionError("bbox mode requires resolved bounds")
            load_bbox = (
                bbox["min_lon"],
                bbox["min_lat"],
                bbox["max_lon"],
                bbox["max_lat"],
            )
        G = _load_local_osm_graph(
            ox,
            xml_path,
            load_bbox,
            footprint=footprint,
        )

        _write_build_state(
            build_state_path,
            run_name=run_name,
            bbox=bbox,
            network=args.network,
            graph_format=args.graph_format,
            source_pbf=source_names,
            settings=settings,
            stage="conversion",
            status="running",
            last_error=None,
            extra={
                "source_manifest": source_manifest,
                "replication_timestamps": replication_timestamps,
            },
            footprint=footprint,
        )
        base_metadata: dict[str, Any] = {
            "bbox": bbox,
            "network": args.network,
            "source_manifest": source_manifest,
            "source_digests": {
                row["path"]: row["sha256"] for row in source_manifest
            },
            "replication_timestamps": replication_timestamps,
            "osmnx": settings,
            "filter": {
                "version": DRIVE_FILTER_VERSION,
                "network": "drive",
                "excluded_highways": sorted(DRIVE_EXCLUDED_HIGHWAYS),
                "excluded_services": sorted(DRIVE_EXCLUDED_SERVICES),
            },
            "coverage_probe_requests": probes,
        }
        if footprint is not None:
            base_metadata["tile_footprint"] = _footprint_metadata(footprint)
            extraction_geometry = _persist_footprint_geometry(footprint, cache_root)
            base_metadata["extraction_geometry"] = {
                "path": str(extraction_geometry),
                "sha256": _digest_file(extraction_geometry, "sha256"),
            }
        if args.graph_format in ("sqlite3", "compact"):
            node_count, edge_count, probe_metadata = _publish_sqlite_graph(
                output_path,
                run_dir,
                G,
                {},
                base_metadata,
                probes,
            )
        else:
            graph = _graph_from_osmnx(G, scenic_scores={})
            del G
            gc.collect()
            probe_metadata = _coverage_metadata(graph, probes)
            graph.save(output_path)
            node_count, edge_count = len(graph.nodes), len(graph.edges)

        compact_record: dict[str, Any] | None = None
        if args.graph_format == "compact":
            compact_record = _publish_compact_graph(output_path, run_dir)
            if compact_record["node_count"] != node_count:
                raise ValueError(
                    "Compact node count does not match the published SQLite "
                    f"({compact_record['node_count']} != {node_count})"
                )
            if compact_record["edge_count"] != edge_count:
                raise ValueError(
                    "Compact edge count does not match the published SQLite "
                    f"({compact_record['edge_count']} != {edge_count})"
                )

        _write_build_state(
            build_state_path,
            run_name=run_name,
            bbox=bbox,
            network=args.network,
            graph_format=args.graph_format,
            source_pbf=source_names,
            settings=settings,
            stage="publication",
            status="running",
            last_error=None,
            extra={
                "source_manifest": source_manifest,
                "replication_timestamps": replication_timestamps,
                "coverage_probes": probe_metadata,
                "node_count": node_count,
                "edge_count": edge_count,
            },
            footprint=footprint,
        )
        graph_stat = output_path.stat()
        run_record = {
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_graph": str(output_path),
            "output_dir": str(run_dir),
            "bbox": bbox,
            "network": args.network,
            "graph_format": args.graph_format,
            "settings": settings,
            "source_manifest": source_manifest,
            "replication_timestamps": replication_timestamps,
            "coverage_probes": probe_metadata,
            "counts": {"nodes": node_count, "edges": edge_count},
            "graph_sha256": _digest_file(output_path, "sha256"),
            "graph_size_bytes": int(graph_stat.st_size),
        }
        if footprint is not None:
            run_record["tile_footprint"] = _footprint_metadata(footprint)
            run_record["extraction_geometry"] = base_metadata["extraction_geometry"]
        if compact_record is not None:
            run_record["compact_graph"] = compact_record["manifest_path"]
            run_record["compact_manifest_sha256"] = compact_record[
                "manifest_sha256"
            ]
            run_record["compact_bin_sha256"] = compact_record["bin_sha256"]
            run_record["compact_size_bytes"] = compact_record["bin_size_bytes"]
            run_record["compact_traversal_count"] = compact_record[
                "traversal_count"
            ]
            run_record["runtime_graph"] = compact_record["manifest_path"]
        (run_dir / "run.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_build_state(
            build_state_path,
            run_name=run_name,
            bbox=bbox,
            network=args.network,
            graph_format=args.graph_format,
            source_pbf=source_names,
            settings=settings,
            stage="publication",
            status="complete",
            last_error=None,
            extra={
                "source_manifest": source_manifest,
                "replication_timestamps": replication_timestamps,
                "coverage_probes": probe_metadata,
                "node_count": node_count,
                "edge_count": edge_count,
                "graph_sha256": run_record["graph_sha256"],
                "graph_size_bytes": run_record["graph_size_bytes"],
                "compact_graph": run_record.get("compact_graph"),
                "compact_manifest_sha256": run_record.get(
                    "compact_manifest_sha256"
                ),
                "compact_bin_sha256": run_record.get("compact_bin_sha256"),
                "compact_size_bytes": run_record.get("compact_size_bytes"),
                "compact_traversal_count": run_record.get(
                    "compact_traversal_count"
                ),
            },
            footprint=footprint,
        )
        return run_record
    except Exception as exc:
        _write_build_state(
            build_state_path,
            run_name=run_name,
            bbox=bbox,
            network=args.network,
            graph_format=args.graph_format,
            source_pbf=source_names,
            settings=settings,
            stage="failed",
            status="failed",
            last_error=f"{type(exc).__name__}: {exc}",
            footprint=footprint,
        )
        raise


def main() -> None:
    args = parse_args()
    record = _build(args)
    output_path = record["output_graph"]
    print(f"Wrote {output_path}")
    if record.get("compact_graph"):
        print(f"Wrote {record['compact_graph']}")
    print(f"Wrote {Path(output_path).parent / 'run.json'}")
    print(f"Nodes: {record['counts']['nodes']}")
    print(f"Edges: {record['counts']['edges']}")


if __name__ == "__main__":
    main()
