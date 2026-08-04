"""
Build a RoadGraph from an OSM bbox or local PBF extracts.

The local-PBF mode is the reproducible path for large regions. The legacy
Overpass mode remains suitable for small builds.
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
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import (  # noqa: E402
    EdgeProjectionIndex,
    RoadGraph,
    _graph_from_osmnx,
    _iter_osmnx_graph_rows,
    _write_sqlite_graph,
)


DEFAULT_MAX_QUERY_AREA = 50_000_000_000.0
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_ATTEMPTS = 3
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build road graph from OSM bbox")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
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
    parser.add_argument("--max-query-area", type=float, default=None)
    parser.add_argument("--overpass-url", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--source-pbf",
        type=Path,
        action="append",
        default=[],
        help="Local PBF extract. Repeat for every source in the merged build.",
    )
    parser.add_argument("--require-source-checksums", action="store_true")
    parser.add_argument(
        "--graph-format",
        choices=("json", "sqlite3"),
        default="json",
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=None,
        help="OSMnx/cache folder. Local conversion intermediates are stored below it.",
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument(
        "--coverage-probe",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "LAT", "LON"),
        help="Named coordinate used to validate graph coverage. Repeatable.",
    )
    return parser.parse_args()




def _slug_float(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    return (
        f"osm_{args.network}_"
        f"{_slug_float(args.min_lat)}_{_slug_float(args.min_lon)}_"
        f"{_slug_float(args.max_lat)}_{_slug_float(args.max_lon)}"
    )


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    suffix = ".sqlite3" if args.graph_format == "sqlite3" else ".json"
    if args.output is not None:
        run_name = args.run_name or args.output.stem
        return args.output, args.output.parent, run_name

    run_name = args.run_name or _default_run_name(args)
    run_dir = args.output_root / run_name
    return run_dir / f"road_graph{suffix}", run_dir, run_name


def _bbox(args: argparse.Namespace) -> dict[str, float]:
    if args.min_lat >= args.max_lat:
        raise ValueError("Invalid bbox: --min-lat must be < --max-lat")
    if args.min_lon >= args.max_lon:
        raise ValueError("Invalid bbox: --min-lon must be < --max-lon")
    return {
        "min_lat": float(args.min_lat),
        "min_lon": float(args.min_lon),
        "max_lat": float(args.max_lat),
        "max_lon": float(args.max_lon),
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


def _settings_record(ox: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": getattr(ox, "__version__", "unknown"),
        "requests_timeout": getattr(ox.settings, "requests_timeout", None),
        "cache_folder": getattr(ox.settings, "cache_folder", None),
        "use_cache": getattr(ox.settings, "use_cache", None),
        "max_query_area_size": getattr(ox.settings, "max_query_area_size", None),
        "overpass_url": getattr(ox.settings, "overpass_url", None),
        "overpass_rate_limit": getattr(ox.settings, "overpass_rate_limit", None),
    }


def _configure_osmnx(ox: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_folder is not None:
        args.cache_folder.mkdir(parents=True, exist_ok=True)
        ox.settings.cache_folder = str(args.cache_folder)
    ox.settings.use_cache = True
    ox.settings.overpass_rate_limit = True
    ox.settings.requests_timeout = int(args.timeout)
    if args.max_query_area is not None:
        ox.settings.max_query_area_size = float(args.max_query_area)
    if args.overpass_url is not None:
        ox.settings.overpass_url = args.overpass_url
    return _settings_record(ox, args)


def _write_build_state(
    path: Path,
    *,
    run_name: str,
    bbox: Mapping[str, float],
    network: str,
    graph_format: str,
    source_pbf: list[str],
    settings: Mapping[str, Any],
    stage: str,
    status: str,
    last_error: str | None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_name": run_name,
        "bbox": dict(bbox),
        "network": network,
        "graph_format": graph_format,
        "source_pbf": list(source_pbf),
        "settings": dict(settings),
        "stage": stage,
        "status": status,
        "last_error": last_error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
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


def _derived_cache_digest(
    manifest: list[dict[str, Any]],
    bbox: Mapping[str, float],
) -> str:
    digest = hashlib.sha256()
    digest.update(_source_digests(manifest).encode("ascii"))
    cache_descriptor = {
        "cache_version": PBF_DERIVED_CACHE_VERSION,
        "bbox": {
            key: float(bbox[key])
            for key in ("min_lat", "min_lon", "max_lat", "max_lon")
        },
        "buffer_m": 500.0,
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


def _cached_path(cache_root: Path, digest: str, name: str) -> Path:
    folder = cache_root / "pbf" / digest
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _merge_and_filter_pbf(
    source_paths: list[Path],
    manifest: list[dict[str, Any]],
    bbox: Mapping[str, float],
    cache_root: Path,
) -> tuple[Path, dict[str, str]]:
    source_paths = sorted(
        (Path(path).expanduser().resolve() for path in source_paths),
        key=lambda path: str(path),
    )
    source_digest = _source_digests(manifest)
    derived_digest = _derived_cache_digest(manifest, bbox)
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
    west, south, east, north = _buffered_bbox(bbox)
    if not buffered.exists():
        _run_command(
            [
                "osmium",
                "extract",
                "--overwrite",
                "--strategy",
                "complete_ways",
                "--bbox",
                f"{west},{south},{east},{north}",
                "-o",
                str(buffered),
                str(merged),
            ]
        )
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


def _tag_contains(value: Any, needles: set[str]) -> bool:
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




def _load_local_osm_graph(
    ox: Any,
    xml_path: Path,
    bbox: tuple[float, float, float, float],
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


def _load_overpass_graph(ox: Any, args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    for attempt in range(1, args.attempts + 1):
        try:
            return (
                ox.graph_from_bbox(
                    bbox=(args.min_lon, args.min_lat, args.max_lon, args.max_lat),
                    network_type=args.network,
                ),
                errors,
            )
        except Exception as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    raise RuntimeError(f"OSMnx graph download failed after {args.attempts} attempts: {errors}")


def _build(args: argparse.Namespace) -> dict[str, Any]:
    bbox = _bbox(args)
    if args.attempts < 1:
        raise ValueError("--attempts must be positive")
    probes = _parse_probes(args.coverage_probe)
    if args.require_source_checksums and not args.source_pbf:
        raise ValueError("--require-source-checksums requires --source-pbf")
    if args.source_pbf and args.network != "drive":
        raise ValueError("Local PBF mode currently supports --network drive only")
    if args.source_pbf and shutil.which("osmium") is None:
        raise FileNotFoundError(
            "Local PBF mode requires the 'osmium' executable on PATH"
        )

    output_path, run_dir, run_name = _resolve_output_paths(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    build_state_path = run_dir / "build_state.json"
    cache_root = args.cache_folder or Path("cache/osmnx") / run_name
    cache_root.mkdir(parents=True, exist_ok=True)
    source_names = [str(Path(path).expanduser()) for path in args.source_pbf]
    settings: dict[str, Any] = {
        "timeout": args.timeout,
        "cache_folder": str(cache_root),
        "use_cache": True,
        "max_query_area_size": args.max_query_area,
        "overpass_url": args.overpass_url,
        "overpass_rate_limit": True,
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
    )

    source_manifest: list[dict[str, Any]] = []
    replication_timestamps: dict[str, str] = {}
    overpass_attempts: list[dict[str, Any]] = []
    try:
        import osmnx as ox

        osm_settings = _configure_osmnx(ox, args)
        settings.update(osm_settings)
        if args.source_pbf:
            source_paths = sorted(
                (Path(path).expanduser().resolve() for path in args.source_pbf),
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
            )
            G = _load_local_osm_graph(
                ox,
                xml_path,
                (
                    bbox["min_lon"],
                    bbox["min_lat"],
                    bbox["max_lon"],
                    bbox["max_lat"],
                ),
            )
        else:
            G, overpass_attempts = _load_overpass_graph(ox, args)

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
                "overpass_attempts": overpass_attempts,
            },
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
        if args.graph_format == "sqlite3":
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
            "overpass_attempts": overpass_attempts,
            "coverage_probes": probe_metadata,
            "counts": {"nodes": node_count, "edges": edge_count},
            "graph_sha256": _digest_file(output_path, "sha256"),
            "graph_size_bytes": int(graph_stat.st_size),
        }
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
            },
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
        )
        raise


def main() -> None:
    args = parse_args()
    record = _build(args)
    output_path = record["output_graph"]
    print(f"Wrote {output_path}")
    print(f"Wrote {Path(output_path).parent / 'run.json'}")
    print(f"Nodes: {record['counts']['nodes']}")
    print(f"Edges: {record['counts']['edges']}")


if __name__ == "__main__":
    main()
