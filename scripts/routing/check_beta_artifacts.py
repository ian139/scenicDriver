#!/usr/bin/env python3
"""Admit read-only beta graphs, coverage, usable reports, and learned model identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


from scenicdriver.route_planner._edge_projection import EdgeProjectionIndex, _SidecarError
from scenicdriver.heuristics.report import recover_report_run_info
from scenicdriver.route_planner.graph import (
    _COMPACT_DTYPE_ITEMSIZE,
    _COMPACT_FORMAT,
    _COMPACT_SCENIC_BYWAY_ROAD_TYPE,
    _COMPACT_SCHEMA_VERSION,
    _COMPACT_SECTION_NAMES,
    _COMPACT_SECTION_SPECS,
    CompactRoadGraph,
    RoadGraph,
    _compact_highway_road_types,
)
from scenicdriver.route_planner.service import _load_graph, load_tile_scores
CONFIG_PATH = Path("config/app_regions.json")
REGISTRY_PATH = Path("data/processed/regression/model_registry.json")

_SQLITE_GRAPH_FORMAT = "scenic-roadgraph-sqlite"
_SQLITE_SCHEMA_VERSION = 1

_REQUIRED_SQLITE_COLUMNS = {
    "metadata": ("key", "value"),
    "nodes": ("id", "lat", "lon"),
    "edges": (
        "id",
        "start_node_id",
        "end_node_id",
        "distance_km",
        "scenic_score",
        "road_name",
        "road_type",
        "speed_limit_kmh",
        "one_way",
    ),
}

_REQUIRED_SQLITE_TYPES = {
    "metadata": ("TEXT", "TEXT"),
    "nodes": ("TEXT", "REAL", "REAL"),
    "edges": (
        "TEXT",
        "TEXT",
        "TEXT",
        "REAL",
        "REAL",
        "TEXT",
        "TEXT",
        "REAL",
        "INTEGER",
    ),
}
_REQUIRED_SQLITE_DECLARATIONS = {
    # Each tuple is (notnull, primary-key ordinal), matching PRAGMA table_info.
    "metadata": ((0, 1), (1, 0)),
    "nodes": ((0, 1), (1, 0), (1, 0)),
    "edges": (
        (0, 1),
        (1, 0),
        (1, 0),
        (1, 0),
        (1, 0),
        (0, 0),
        (1, 0),
        (0, 0),
        (1, 0),
    ),
}


def _project_path(root: Path, value: str | Path) -> Path:
    """Resolve a configured path while preserving absolute container paths."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_active_checkpoint(
    root: Path, registry_path: Path, value: str
) -> tuple[Path, list[str]]:
    """Resolve a stored active checkpoint, preferring an existing file.

    Preserves absolute container paths; otherwise tries the registry-relative
    location (checkpoints/<sha>.pt) before the project-root-relative location
    (models/...). Returns the first existing candidate; when none exists,
    returns the registry-relative candidate and reports every tried path.
    """
    path = Path(value)
    if path.is_absolute():
        return path, []
    candidates = [
        registry_path.parent / path,
        root / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, []
    return candidates[0], [
        f"invalid: active checkpoint missing: {value} "
        f"(tried: {candidates[0]}, {candidates[1]})"
    ]


def _configured_route_regions(
    root: Path, cli_graph: Path | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve the default region and every route-enabled region without skipping defects.

    Returns a list of dicts, each describing a route region:
        - "region": region name (str)
        - "graph_path": Path
        - "run_name": run_name (str)
        - "region_dict": raw region configuration dict
        - "is_default": bool
    and a list of issue strings.
    """
    config_path = root / CONFIG_PATH
    issues: list[str] = []

    if not config_path.is_file():
        return [], [f"missing: {CONFIG_PATH} (canonical region config is required)"]

    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [f"invalid: {CONFIG_PATH} ({exc})"]

    if not isinstance(payload, dict):
        return [], [f"invalid: {CONFIG_PATH} (config payload is not a dict)"]

    default_region_name = payload.get("default_region")
    if not isinstance(default_region_name, str) or not default_region_name.strip():
        return [], [f"invalid: {CONFIG_PATH} (default_region is missing)"]

    regions_raw = payload.get("regions")
    if not isinstance(regions_raw, list):
        return [], [f"invalid: {CONFIG_PATH} (regions array is missing)"]

    route_regions: list[dict[str, Any]] = []
    found_default = False

    for item in regions_raw:
        if not isinstance(item, dict):
            issues.append(f"invalid: {CONFIG_PATH} (region entry is not a dict)")
            continue
        region_name = item.get("region")
        if not isinstance(region_name, str) or not region_name:
            issues.append(f"invalid: {CONFIG_PATH} (region name is missing)")
            continue

        is_default = region_name.strip().lower() == default_region_name.strip().lower()
        if is_default:
            found_default = True
            if item.get("route_planning") is False:
                issues.append(f"invalid: {CONFIG_PATH} ({region_name} default route planning is disabled)")

        if item.get("route_planning") is False and not is_default:
            continue

        raw_graph = item.get("graph_path") or item.get("graph")
        if not isinstance(raw_graph, str) or not raw_graph.strip():
            issues.append(f"invalid: {CONFIG_PATH} ({region_name} graph is missing)")
            continue

        if is_default and cli_graph is not None:
            graph_path = _project_path(root, cli_graph)
        else:
            graph_path = _project_path(root, raw_graph)

        run_name = item.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            issues.append(f"invalid: {CONFIG_PATH} ({region_name} run_name is missing)")
            continue

        route_regions.append(
            {
                "region": region_name,
                "graph_path": graph_path,
                "run_name": run_name,
                "region_dict": item,
                "is_default": is_default,
            }
        )

    if not found_default:
        issues.append(
            f"invalid: {CONFIG_PATH} ({default_region_name} is not configured)"
        )

    return route_regions, issues


def _metadata_bbox_matches(metadata_bbox: Any, configured_bbox: Any) -> bool:
    if not isinstance(metadata_bbox, Mapping) or not isinstance(
        configured_bbox, Mapping
    ):
        return False
    for key in ("min_lat", "min_lon", "max_lat", "max_lon"):
        actual = metadata_bbox.get(key)
        expected = configured_bbox.get(key)
        if isinstance(actual, bool) or isinstance(expected, bool):
            return False
        try:
            if not math.isfinite(float(actual)) or float(actual) != float(expected):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _validate_sqlite_graph(
    graph_path: Path, region: Mapping[str, Any] | None
) -> list[str]:
    """Validate SQLite schema, metadata, row values, and endpoint references read-only."""

    label = str(graph_path)
    if region is None:
        return [f"invalid: {label} (configured region is unavailable)"]
    uri = f"{graph_path.resolve().as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        expected_tables = set(_REQUIRED_SQLITE_COLUMNS)
        if tables != expected_tables:
            missing = sorted(expected_tables - tables)
            extra = sorted(tables - expected_tables)
            details: list[str] = []
            if missing:
                details.append(f"missing tables {', '.join(missing)}")
            if extra:
                details.append(f"unexpected tables {', '.join(extra)}")
            return [f"invalid: {label} ({'; '.join(details)})"]

        for table, expected_columns in _REQUIRED_SQLITE_COLUMNS.items():
            schema_rows = tuple(connection.execute(f"PRAGMA table_info({table})"))
            columns = tuple(str(row[1]) for row in schema_rows)
            types = tuple(str(row[2]).upper() for row in schema_rows)
            declarations = tuple((int(row[3]), int(row[5])) for row in schema_rows)
            if (
                columns != expected_columns
                or types != _REQUIRED_SQLITE_TYPES[table]
                or declarations != _REQUIRED_SQLITE_DECLARATIONS[table]
            ):
                return [
                    f"invalid: {label} ({table} schema does not match SQLite graph format)"
                ]

        edges_sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='edges'"
        ).fetchone()
        edges_sql = str(edges_sql_row[0]) if edges_sql_row else ""
        if not re.search(
            r"CHECK\s*\(\s*one_way\s+IN\s*\(\s*0\s*,\s*1\s*\)\s*\)",
            edges_sql,
            flags=re.IGNORECASE,
        ):
            return [
                f"invalid: {label} (edges schema is missing CHECK(one_way IN (0, 1)))"
            ]

        metadata: dict[str, Any] = {}
        for key, value in connection.execute("SELECT key, value FROM metadata"):
            if not isinstance(key, str) or key in metadata:
                return [f"invalid: {label} (duplicate or invalid metadata key)"]
            try:
                metadata[key] = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return [f"invalid: {label} (invalid metadata value: {exc})"]

        if metadata.get("graph_format") != _SQLITE_GRAPH_FORMAT:
            return [
                f"invalid: {label} (unsupported graph_format "
                f"{metadata.get('graph_format')!r})"
            ]
        schema_version = metadata.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != _SQLITE_SCHEMA_VERSION:
            return [f"invalid: {label} (unsupported schema_version {schema_version!r})"]
        configured_graph_bbox = region.get("graph_bbox", region.get("bbox"))
        if not _metadata_bbox_matches(metadata.get("bbox"), configured_graph_bbox):
            return [f"invalid: {label} (bbox does not match configured region)"]

        for name, table in (("node_count", "nodes"), ("edge_count", "edges")):
            value = metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return [f"invalid: {label} ({name} metadata count is not positive)"]
            actual = int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            if actual != value:
                return [
                    f"invalid: {label} ({name} metadata count {value} "
                    f"does not match SQLite COUNT(*) {actual})"
                ]

        numeric = "('integer', 'real')"
        invalid_nodes = connection.execute(f"""
            SELECT 1 FROM nodes WHERE id IS NULL OR typeof(id) != 'text' OR trim(id) = ''
                OR typeof(lat) NOT IN {numeric} OR NOT (lat BETWEEN -90 AND 90)
                OR typeof(lon) NOT IN {numeric} OR NOT (lon BETWEEN -180 AND 180)
            LIMIT 1
        """).fetchone()
        invalid_edges = connection.execute(f"""
            SELECT 1 FROM edges e
            LEFT JOIN nodes s ON s.id=e.start_node_id
            LEFT JOIN nodes t ON t.id=e.end_node_id
            WHERE e.id IS NULL OR typeof(e.id) != 'text' OR trim(e.id) = ''
                OR s.id IS NULL OR t.id IS NULL
                OR typeof(e.distance_km) NOT IN {numeric}
                OR NOT (e.distance_km BETWEEN 0 AND 1.7976931348623157e308)
                OR typeof(e.scenic_score) NOT IN {numeric}
                OR NOT (e.scenic_score BETWEEN 0 AND 10)
                OR typeof(e.road_type) != 'text' OR trim(e.road_type) = ''
                OR (e.road_name IS NOT NULL AND typeof(e.road_name) != 'text')
                OR (e.speed_limit_kmh IS NOT NULL AND
                    (typeof(e.speed_limit_kmh) NOT IN {numeric}
                     OR NOT (e.speed_limit_kmh > 0 AND e.speed_limit_kmh <= 1.7976931348623157e308)))
                OR typeof(e.one_way) != 'integer' OR e.one_way NOT IN (0, 1)
            LIMIT 1
        """).fetchone()
        if invalid_nodes or invalid_edges:
            return [f"invalid: {label} (invalid graph rows or node references)"]

        max_snap = region.get("max_route_snap_km")
        if isinstance(max_snap, bool):
            return [f"invalid: {label} (max_route_snap_km is invalid)"]
        try:
            max_snap_value = float(max_snap)
        except (TypeError, ValueError):
            return [f"invalid: {label} (max_route_snap_km is invalid)"]
        if not math.isfinite(max_snap_value) or max_snap_value < 0:
            return [f"invalid: {label} (max_route_snap_km is invalid)"]
        probes = metadata.get("coverage_probes")
        if not isinstance(probes, Mapping):
            return [f"invalid: {label} (coverage_probes metadata is missing)"]
        configured_probes = region.get("coverage_probes")
        if not isinstance(configured_probes, Mapping) or not configured_probes:
            return [f"invalid: {label} (configured coverage_probes are invalid)"]
        expected_probes = configured_probes

        for probe_name, expected_probe in expected_probes.items():
            if not isinstance(probe_name, str) or not isinstance(
                expected_probe, Mapping
            ):
                return [f"invalid: {label} (configured coverage_probes are invalid)"]
            probe = probes.get(probe_name)
            if not isinstance(probe, Mapping):
                return [f"invalid: {label} (coverage probe {probe_name} is missing)"]

            expected_lat = expected_probe.get("lat")
            expected_lon = expected_probe.get("lon")
            latitude = probe.get("lat")
            longitude = probe.get("lon")
            if (
                isinstance(latitude, bool)
                or isinstance(longitude, bool)
                or isinstance(expected_lat, bool)
                or isinstance(expected_lon, bool)
                or not isinstance(latitude, (int, float))
                or not isinstance(longitude, (int, float))
                or not isinstance(expected_lat, (int, float))
                or not isinstance(expected_lon, (int, float))
                or not math.isfinite(float(latitude))
                or not math.isfinite(float(longitude))
                or not math.isfinite(float(expected_lat))
                or not math.isfinite(float(expected_lon))
            ):
                return [
                    f"invalid: {label} (coverage probe {probe_name} "
                    "coordinates are invalid)"
                ]
            if (
                float(latitude) != float(expected_lat)
                or float(longitude) != float(expected_lon)
            ):
                return [
                    f"invalid: {label} (coverage probe {probe_name} "
                    "coordinates do not match configured values)"
                ]

            distance = probe.get("distance_km")
            if isinstance(distance, bool):
                return [
                    f"invalid: {label} (coverage probe {probe_name} "
                    "distance is invalid)"
                ]
            try:
                distance_value = float(distance)
            except (TypeError, ValueError, OverflowError):
                return [
                    f"invalid: {label} (coverage probe {probe_name} distance is invalid)"
                ]
            if (
                not math.isfinite(distance_value)
                or distance_value < 0
                or distance_value > max_snap_value
            ):
                return [
                    f"invalid: {label} (coverage probe {probe_name} exceeds "
                    f"max_route_snap_km)"
                ]

        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            return [f"invalid: {label} (PRAGMA integrity_check is not ok)"]
    except (OSError, sqlite3.Error, TypeError, ValueError, OverflowError) as exc:
        return [f"invalid: {label} ({exc})"]
    finally:
        if connection is not None:
            connection.close()
    return []
def _sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.digest()


def _validate_compact_graph(
    manifest_path: Path, region: Mapping[str, Any] | None
) -> list[str]:
    """Validate dependencies resolved relative to the manifest, including generations."""
    label = str(manifest_path)
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"invalid: {label} ({exc})"]

    if not isinstance(manifest, dict):
        return [f"invalid: {label} (manifest is not a dict)"]
    if manifest.get("format") != _COMPACT_FORMAT:
        return [
            f"invalid: {label} (unsupported compact format {manifest.get('format')!r})"
        ]

    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _COMPACT_SCHEMA_VERSION:
        return [
            f"invalid: {label} (unsupported compact schema_version {schema_version!r})"
        ]

    graph_counts = manifest.get("graph")
    if not isinstance(graph_counts, Mapping):
        return [f"invalid: {label} (missing graph counts mapping)"]

    node_count_val = graph_counts.get("node_count")
    edge_count_val = graph_counts.get("edge_count")
    traversal_count_val = graph_counts.get("traversal_count")
    if (
        isinstance(node_count_val, bool)
        or not isinstance(node_count_val, int)
        or node_count_val < 0
        or isinstance(edge_count_val, bool)
        or not isinstance(edge_count_val, int)
        or edge_count_val < 0
        or isinstance(traversal_count_val, bool)
        or not isinstance(traversal_count_val, int)
        or traversal_count_val < 0
    ):
        return [f"invalid: {label} (invalid compact graph counts)"]

    node_count = node_count_val
    edge_count = edge_count_val

    bin_name = manifest.get("bin_path")
    if not isinstance(bin_name, str) or not bin_name:
        return [f"invalid: {label} (missing bin_path)"]
    bin_path = (manifest_path.parent / bin_name).resolve()
    if not bin_path.is_file():
        return [f"invalid: {bin_path} (compact binary payload is missing)"]

    bin_size_expected = manifest.get("bin_size_bytes")
    if (
        isinstance(bin_size_expected, bool)
        or not isinstance(bin_size_expected, int)
        or bin_size_expected <= 0
    ):
        return [f"invalid: {label} (invalid bin_size_bytes)"]
    if bin_path.stat().st_size != bin_size_expected:
        return [f"invalid: {bin_path} (compact binary size mismatch)"]

    expected_bin_sha = manifest.get("bin_sha256")
    if (
        not isinstance(expected_bin_sha, str)
        or not expected_bin_sha
        or len(expected_bin_sha) != 64
    ):
        return [f"invalid: {label} (invalid or missing bin_sha256)"]
    actual_bin_sha = _sha256_file(bin_path).hex()
    if actual_bin_sha != expected_bin_sha:
        return [f"invalid: {bin_path} (compact binary SHA-256 mismatch)"]

    source_info = manifest.get("source")
    if not isinstance(source_info, Mapping):
        return [f"invalid: {label} (missing source metadata)"]
    source_name = source_info.get("path")
    if not isinstance(source_name, str) or not source_name:
        return [f"invalid: {label} (missing source path)"]
    source_path = (manifest_path.parent / source_name).resolve()
    if not source_path.is_file():
        return [f"invalid: {source_path} (compact source graph is missing)"]

    source_schema = source_info.get("schema_version")
    if (
        isinstance(source_schema, bool)
        or source_schema != _SQLITE_SCHEMA_VERSION
    ):
        return [
            f"invalid: {label} (unsupported compact source schema_version {source_schema!r})"
        ]

    src_node_count = source_info.get("node_count")
    src_edge_count = source_info.get("edge_count")
    if isinstance(src_node_count, bool) or src_node_count != node_count:
        return [f"invalid: {label} (compact source node count mismatch)"]
    if isinstance(src_edge_count, bool) or src_edge_count != edge_count:
        return [f"invalid: {label} (compact source edge count mismatch)"]

    source_size_expected = source_info.get("size_bytes")
    if (
        isinstance(source_size_expected, bool)
        or not isinstance(source_size_expected, int)
        or source_size_expected <= 0
    ):
        return [f"invalid: {label} (invalid source size_bytes)"]
    if source_path.stat().st_size != source_size_expected:
        return [f"invalid: {source_path} (compact source graph size mismatch)"]

    expected_source_sha = source_info.get("sha256")
    if (
        not isinstance(expected_source_sha, str)
        or not expected_source_sha
        or len(expected_source_sha) != 64
    ):
        return [f"invalid: {label} (invalid or missing source sha256)"]
    actual_source_sha = _sha256_file(source_path).hex()
    if actual_source_sha != expected_source_sha:
        return [f"invalid: {source_path} (compact source graph SHA-256 mismatch)"]

    sqlite_issues = _validate_sqlite_graph(source_path, region)
    if sqlite_issues:
        return sqlite_issues

    expected_sections = manifest.get("sections")
    if not isinstance(expected_sections, Mapping):
        return [f"invalid: {label} (missing sections mapping)"]
    missing_sections = _COMPACT_SECTION_NAMES.difference(expected_sections)
    if missing_sections:
        return [
            f"invalid: {label} (compact manifest missing sections: {sorted(missing_sections)})"
        ]
    unknown_sections = set(expected_sections.keys()).difference(_COMPACT_SECTION_NAMES)
    if unknown_sections:
        return [
            f"invalid: {label} (compact manifest has unknown section: {sorted(unknown_sections)})"
        ]

    for name, descriptor in expected_sections.items():
        if not isinstance(descriptor, Mapping):
            return [f"invalid: {label} (compact section {name} has no descriptor)"]
        dtype = descriptor.get("dtype")
        if not isinstance(dtype, str) or dtype not in _COMPACT_DTYPE_ITEMSIZE:
            return [f"invalid: {label} (compact section {name} has unknown dtype)"]
        if dtype != dict(_COMPACT_SECTION_SPECS)[name]:
            return [f"invalid: {label} (compact section {name} has incorrect dtype)"]
        offset_val = descriptor.get("offset")
        length_val = descriptor.get("length")
        count_val = descriptor.get("count")
        if (
            isinstance(offset_val, bool)
            or not isinstance(offset_val, int)
            or offset_val < 0
            or isinstance(length_val, bool)
            or not isinstance(length_val, int)
            or length_val < 0
            or isinstance(count_val, bool)
            or not isinstance(count_val, int)
            or count_val < 0
        ):
            return [f"invalid: {label} (compact section {name} has invalid bounds)"]

        offset = offset_val
        length = length_val
        count = count_val

        if dtype == "raw":
            if length != count:
                return [f"invalid: {label} (compact section {name} size/count mismatch)"]
        elif length != count * _COMPACT_DTYPE_ITEMSIZE[dtype]:
            return [f"invalid: {label} (compact section {name} size/count mismatch)"]
        if offset + length > bin_size_expected:
            return [f"invalid: {label} (compact section {name} extends past payload)"]

    previous_end = 0
    for name, _dtype in _COMPACT_SECTION_SPECS:
        descriptor = expected_sections[name]
        offset = descriptor["offset"]
        length = descriptor["length"]
        if offset != previous_end:
            return [
                f"invalid: {label} (compact section {name} offset is not contiguous)"
            ]
        previous_end = offset + length
    if previous_end != bin_size_expected:
        return [f"invalid: {label} (compact section payload has trailing bytes)"]

    if str(manifest.get("scenic_byway_road_type")) != _COMPACT_SCENIC_BYWAY_ROAD_TYPE:
        return [f"invalid: {label} (compact manifest scenic byway marker mismatch)"]

    expected_hw = ",".join(sorted(_compact_highway_road_types()))
    if str(manifest.get("highway_road_types", "")) != expected_hw:
        return [
            f"invalid: {label} (compact manifest highway road-type mask is stale; rebuild it)"
        ]

    proj_info = manifest.get("projection_index")
    if not isinstance(proj_info, Mapping):
        return [f"invalid: {label} (missing projection_index metadata)"]

    proj_name = proj_info.get("path")
    if not isinstance(proj_name, str) or not proj_name:
        return [f"invalid: {label} (missing projection sidecar path)"]
    sidecar_path = (manifest_path.parent / proj_name).resolve()
    if not sidecar_path.is_file():
        return [f"invalid: {sidecar_path} (projection sidecar is missing)"]

    proj_size_expected = proj_info.get("size_bytes")
    if (
        isinstance(proj_size_expected, bool)
        or not isinstance(proj_size_expected, int)
        or proj_size_expected <= 0
    ):
        return [f"invalid: {sidecar_path} (invalid projection sidecar size_bytes)"]
    if sidecar_path.stat().st_size != proj_size_expected:
        return [f"invalid: {sidecar_path} (projection sidecar size mismatch)"]

    expected_proj_sha = proj_info.get("sha256")
    if (
        not isinstance(expected_proj_sha, str)
        or not expected_proj_sha
        or len(expected_proj_sha) != 64
    ):
        return [f"invalid: {sidecar_path} (invalid or missing projection sidecar sha256)"]
    actual_proj_sha = _sha256_file(sidecar_path).hex()
    if actual_proj_sha != expected_proj_sha:
        return [f"invalid: {sidecar_path} (projection sidecar SHA-256 mismatch)"]

    try:
        index = EdgeProjectionIndex.load(sidecar_path, source_path, verify=True)
        index.close()
    except (_SidecarError, ValueError, OSError) as exc:
        return [f"invalid: {sidecar_path} ({exc})"]

    return _validate_admitted_graph(manifest_path, region)

def _active_checkpoint(root: Path) -> tuple[Path | None, list[str]]:
    """Resolve the checkpoint named by the active model-registry record."""

    registry_path = root / REGISTRY_PATH
    if not registry_path.is_file():
        return None, [f"missing: {REGISTRY_PATH} (model registry is required)"]

    try:
        payload: Any = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [f"invalid: {REGISTRY_PATH} ({exc})"]

    active = payload.get("active") if isinstance(payload, dict) else None
    configured_checkpoint = (
        active.get("checkpoint") if isinstance(active, dict) else None
    )
    if not isinstance(configured_checkpoint, str) or not configured_checkpoint.strip():
        return None, [f"invalid: {REGISTRY_PATH} (active checkpoint is missing)"]
    return _resolve_active_checkpoint(root, registry_path, configured_checkpoint)


def _score_report_path(root: Path, reg: Mapping[str, Any]) -> Path:
    """Match API startup score-report override precedence."""
    item = reg["region_dict"]
    configured = next(
        (item.get(key) for key in ("tile_scores_json", "tile_scores_path", "report", "report_json")
         if item.get(key)), None,
    )
    if configured is not None:
        return _project_path(root, configured)
    return root / "data/processed/heuristic_runs" / reg["run_name"] / "report/report.json"


def _validate_report(path: Path, checkpoint: Path, *, heatmap: bool = True) -> list[str]:
    """Validate consumed scores and canonical learned provenance/model association."""
    from fastapi import HTTPException
    from scenicdriver.app_api.main import (
        _parse_heatmap_artifact,
        _readiness_file_identity,
    )

    try:
        if heatmap:
            payload = _parse_heatmap_artifact(str(path), _readiness_file_identity(path))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("tiles"), list):
                raise ValueError("invalid score report")
            load_tile_scores(path)
        identity = recover_report_run_info(
            payload, source=str(path), guidance="Regenerate the admitted learned report.",
        )
        if identity["scoring_mode"] != "learned":
            raise ValueError("beta report must use learned scoring")
        if checkpoint.is_file() and identity["regression_checkpoint_sha256"] != _sha256_file(checkpoint).hex():
            raise ValueError("report regression checkpoint identity does not match configured model")
    except HTTPException as exc:
        return [f"invalid: {path} ({exc.detail})"]
    except (OSError, UnicodeError, TypeError, ValueError, KeyError, AttributeError) as exc:
        return [f"invalid: {path} ({exc})"]
    return []


def _validate_saved_routes(report_dir: Path) -> list[str]:
    """Validate the saved geometry and finite metrics consumed by validated-route."""
    try:
        geojson = json.loads((report_dir / "route.geojson").read_text(encoding="utf-8"))
        metrics = json.loads((report_dir / "route_metrics.json").read_text(encoding="utf-8"))
        if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
            raise ValueError("route must be a FeatureCollection")
        features = geojson.get("features")
        if not isinstance(features, list) or len(features) != 2:
            raise ValueError("route requires scenic and baseline features")

        def finite(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

        def coordinate(value: Any, *, latitude_first: bool = False) -> None:
            if not isinstance(value, list) or len(value) != 2 or not all(map(finite, value)):
                raise ValueError("invalid route coordinate")
            lat, lon = value if latitude_first else value[::-1]
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("route coordinate out of range")

        kinds = set()
        for feature in features:
            kinds.add(feature["properties"]["route_kind"])
            geometry = feature["geometry"]
            if geometry.get("type") != "LineString" or len(geometry.get("coordinates", [])) < 2:
                raise ValueError("route requires complete LineString geometry")
            for point in geometry["coordinates"]:
                coordinate(point)
        if kinds != {"scenic", "baseline"} or not isinstance(metrics["score_mapping"], dict):
            raise ValueError("invalid route kinds or score mapping")
        for kind in kinds:
            route = metrics[kind]
            if route["route_kind"] != kind or not all(
                finite(route.get(key)) for key in ("total_distance_km", "estimated_duration_minutes",
                                                   "raw_scenic_score", "normalized_scenic_score")
            ):
                raise ValueError("missing finite route metrics")
        request = metrics["request"]
        for endpoint in ("start", "end"):
            coordinate(request.get(endpoint), latitude_first=True)
        if not finite(request.get("max_detour_factor")):
            raise ValueError("missing finite route request")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"invalid: {report_dir} (saved route artifacts: {exc})"]
    return []


def _validate_json_graph_rows(path: Path) -> None:
    """Reject missing identities and boolean numerics before runtime normalization."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("graph payload must be a mapping")
    if path.suffix.lower() == ".geojson":
        if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
            raise ValueError("graph must be a FeatureCollection")
        for feature in payload["features"]:
            geometry = feature["geometry"]
            if geometry.get("type") != "LineString" or len(geometry.get("coordinates", [])) < 2:
                raise ValueError("road geometry must be a complete LineString")
            for point in geometry["coordinates"]:
                if not isinstance(point, list) or len(point) != 2 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) for value in point
                ):
                    raise ValueError("invalid road coordinates")
                if not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90:
                    raise ValueError("road coordinates out of range")
        return
    for table, required, numeric in (
        ("nodes", ("id", "lat", "lon"), ("lat", "lon")),
        ("edges", ("id", "start", "end", "distance_km"),
         ("distance_km", "scenic_score", "speed_limit_kmh")),
    ):
        rows = payload.get(table)
        if not isinstance(rows, list):
            raise ValueError(f"missing graph {table}")
        for row in rows:
            if not isinstance(row, dict) or any(row.get(key) is None for key in required):
                raise ValueError(f"missing required {table} fields")
            for key in ("id", "start", "end"):
                if key in row and (isinstance(row[key], bool) or not str(row[key]).strip()):
                    raise ValueError(f"invalid {table} identity")
            if any(isinstance(row.get(key), bool) for key in numeric):
                raise ValueError(f"invalid {table} numeric fields")


def _validate_admitted_graph(path: Path, region: Mapping[str, Any] | None) -> list[str]:
    """Probe admitted geometry, never trusting metadata or a source-graph sidecar.

    Compact payloads reuse the runtime's type/cardinality/topology validator.
    Coverage uses bounded projection chunks from the admitted endpoint arrays.
    """
    graph: RoadGraph | None = None
    try:
        if region is None:
            raise ValueError("configured region is unavailable")
        if path.suffix.lower() in {".json", ".geojson"} and not path.name.endswith(".compact.json"):
            _validate_json_graph_rows(path)
        graph = _load_graph(path)
        if not graph.nodes or not graph.edges:
            raise ValueError("graph must contain nodes and edges")
        if isinstance(graph, CompactRoadGraph):
            sections = graph._sections
            for name, bound in (("node_lat", 90), ("node_lon", 180)):
                if np.any(np.abs(sections[name]) > bound):
                    raise ValueError("node coordinates out of range")
            if np.any((sections["edge_scenic_score"] < 0) | (sections["edge_scenic_score"] > 10)):
                raise ValueError("scenic scores out of range")
        else:
            for node in graph.nodes.values():
                if not node.id.strip() or not math.isfinite(node.lat) or not -90 <= node.lat <= 90 or not math.isfinite(node.lon) or not -180 <= node.lon <= 180:
                    raise ValueError("invalid node identity or coordinates")
            for edge in graph.edges.values():
                if not edge.id.strip() or not edge.road_type.strip() or edge.start_node_id not in graph.nodes or edge.end_node_id not in graph.nodes:
                    raise ValueError("invalid edge identity or node references")
                if not math.isfinite(edge.distance_km) or edge.distance_km < 0 or not math.isfinite(edge.scenic_score) or not 0 <= edge.scenic_score <= 10:
                    raise ValueError("invalid edge numeric values")

        probes = region.get("coverage_probes")
        max_snap = region.get("max_route_snap_km")
        if isinstance(max_snap, bool) or not isinstance(max_snap, (int, float)) or not math.isfinite(max_snap) or max_snap < 0:
            raise ValueError("invalid max_route_snap_km")
        if not isinstance(probes, Mapping) or not probes:
            raise ValueError("invalid coverage_probes")
        queries = []
        for name, probe in probes.items():
            lat, lon = probe["lat"], probe["lon"]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (lat, lon)) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError(f"invalid coverage probe {name}")
            queries.append((name, lat, lon))
        nearest = [math.inf] * len(queries)

        def project(start_lat: np.ndarray, start_lon: np.ndarray,
                    end_lat: np.ndarray, end_lon: np.ndarray) -> None:
            for index, (_, lat, lon) in enumerate(queries):
                distances = EdgeProjectionIndex._project_edge_chunk(
                    lat, lon, math.cos(math.radians(lat)), start_lat, start_lon,
                    end_lat, end_lon, 0, len(start_lat),
                )[3]
                nearest[index] = min(nearest[index], float(np.min(distances)))

        chunk_size = 65_536
        if isinstance(graph, CompactRoadGraph):
            for start in range(0, graph.edge_count, chunk_size):
                stop = start + chunk_size
                start_ranks = sections["edge_start_rank"][start:stop]
                end_ranks = sections["edge_end_rank"][start:stop]
                project(sections["node_lat"][start_ranks], sections["node_lon"][start_ranks],
                        sections["node_lat"][end_ranks], sections["node_lon"][end_ranks])
        else:
            coordinates = []
            for edge in graph.edges.values():
                start_node = graph.nodes[edge.start_node_id]
                end_node = graph.nodes[edge.end_node_id]
                coordinates.append((start_node.lat, start_node.lon, end_node.lat, end_node.lon))
                if len(coordinates) == chunk_size:
                    project(*np.asarray(coordinates).T)
                    coordinates.clear()
            if coordinates:
                project(*np.asarray(coordinates).T)
        for (name, _, _), distance in zip(queries, nearest):
            if not math.isfinite(distance) or distance > max_snap:
                raise ValueError(f"coverage probe {name} on admitted graph exceeds max_route_snap_km ({distance} km)")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        return [f"invalid: {path} ({exc})"]
    finally:
        if graph is not None:
            graph.close()
    return []

def _validate_readiness_gate(root: Path, graph_path: Path) -> list[str]:
    """Validate the builder's ``run.json`` publication gate for one admitted graph.

    Mirrors the runtime gate in ``scenicdriver.app_api.main._graph_readiness``: the
    record must carry the shipped graph identity, project-root-relative paths,
    and (when the admitted graph is the compact manifest) the compact identity.
    A missing or stale gate means the API refuses to route the region, so
    admission must fail the same deployment the runtime would treat as
    not-ready.
    """
    gate = graph_path.parent / "run.json"
    label = str(gate)
    try:
        record = json.loads(gate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"invalid: {label} (readiness gate: {exc})"]
    if not isinstance(record, dict):
        return [f"invalid: {label} (readiness gate is not a dict)"]

    output = record.get("output_graph")
    digest = record.get("graph_sha256")
    size = record.get("graph_size_bytes")
    if (
        not isinstance(output, str) or not output
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size) is not int or size < 0
    ):
        return [
            f"invalid: {label} (readiness gate has invalid graph identity fields)"
        ]
    source = _project_path(root, output)
    if not source.is_file():
        return [
            f"invalid: {label} (readiness gate output_graph is missing: {source})"
        ]
    if source.stat().st_size != size or _sha256_file(source).hex() != digest:
        return [
            f"invalid: {label} (readiness gate graph identity does not match {source})"
        ]

    if source.resolve() != graph_path.resolve():
        compact = record.get("runtime_graph") or record.get("compact_graph")
        compact_digest = record.get("compact_manifest_sha256")
        if not isinstance(compact, str) or not compact:
            return [
                f"invalid: {label} (readiness gate does not name the admitted "
                f"compact manifest {graph_path})"
            ]
        compact_path = _project_path(root, compact)
        if compact_path.resolve() != graph_path.resolve():
            return [
                f"invalid: {label} (readiness gate runtime_graph resolves to "
                f"{compact_path}, not {graph_path})"
            ]
        if (
            not isinstance(compact_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", compact_digest) is None
            or _sha256_file(graph_path).hex() != compact_digest
        ):
            return [
                f"invalid: {label} (readiness gate compact manifest identity "
                f"does not match {graph_path})"
            ]
    return []


def check_artifacts(project_root: Path, graph: Path | None = None) -> int:
    root = project_root.resolve()
    regions, issues = _configured_route_regions(root, cli_graph=graph)
    checkpoint_path, registry_issues = _active_checkpoint(root)
    issues.extend(registry_issues)

    required_paths: list[Path] = [root / REGISTRY_PATH]
    if checkpoint_path is not None:
        required_paths.append(checkpoint_path)

    for reg in regions:
        graph_path = reg["graph_path"]
        region_dict = reg["region_dict"]
        model_path = _project_path(root, region_dict["model_checkpoint"]) if region_dict.get("model_checkpoint") else checkpoint_path
        reg["model_path"] = model_path
        if model_path is not None:
            required_paths.append(model_path)
        report_path = _score_report_path(root, reg)
        reg["report_path"] = report_path
        required_paths.append(report_path)
        run_name = reg["run_name"]
        if graph_path and graph_path not in required_paths:
            required_paths.append(graph_path)
        if graph_path:
            gate_path = graph_path.parent / "run.json"
            if gate_path not in required_paths:
                required_paths.append(gate_path)
        if run_name:
            report_dir = root / "data/processed/heuristic_runs" / run_name / "report"
            for name in ("report.json", "route.geojson", "route_metrics.json"):
                p = report_dir / name
                if p not in required_paths:
                    required_paths.append(p)

    missing: list[Path] = []
    for path in required_paths:
        if not path.is_file() and path not in missing:
            missing.append(path)

    for reg in regions:
        graph_path = reg["graph_path"]
        region_dict = reg["region_dict"]
        if not graph_path or graph_path in missing:
            continue

        if graph_path.suffix.lower() == ".sqlite3":
            graph_issues = _validate_sqlite_graph(graph_path, region_dict)
            issues.extend(graph_issues)
            if not graph_issues:
                issues.extend(_validate_admitted_graph(graph_path, region_dict))
            compact_manifest = graph_path.with_name(f"{graph_path.stem}.compact.json")
            if compact_manifest.is_file():
                issues.extend(_validate_compact_graph(compact_manifest, region_dict))
            sidecar_path = EdgeProjectionIndex.sidecar_path(graph_path)
            if sidecar_path.is_file() and not compact_manifest.is_file():
                try:
                    index = EdgeProjectionIndex.load(sidecar_path, graph_path, verify=True)
                    index.close()
                except (_SidecarError, ValueError, OSError) as exc:
                    issues.append(f"invalid: {sidecar_path} ({exc})")
        elif graph_path.name.endswith(".compact.json"):
            issues.extend(_validate_compact_graph(graph_path, region_dict))
        elif graph_path.suffix.lower() in {".json", ".geojson"}:
            issues.extend(_validate_admitted_graph(graph_path, region_dict))
        else:
            issues.append(f"invalid: {graph_path} (unsupported graph format)")
        issues.extend(_validate_readiness_gate(root, graph_path))

    for reg in regions:
        report_path = reg["report_path"]
        canonical_report = root / "data/processed/heuristic_runs" / reg["run_name"] / "report/report.json"
        model_path = reg["model_path"]
        if report_path != canonical_report and report_path not in missing and model_path is not None:
            issues.extend(_validate_report(report_path, model_path, heatmap=False))
        if reg["run_name"]:
            report_dir = root / "data/processed/heuristic_runs" / reg["run_name"] / "report"
            canonical_report = report_dir / "report.json"
            if canonical_report not in missing and model_path is not None:
                issues.extend(_validate_report(canonical_report, model_path))
            if all((report_dir / name) not in missing for name in ("route.geojson", "route_metrics.json")):
                issues.extend(_validate_saved_routes(report_dir))

    for path in missing:
        try:
            display_path = path.relative_to(root)
        except ValueError:
            display_path = path
        print(f"missing: {display_path}")
    for issue in issues:
        print(issue)
    return 1 if missing or issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Candidate graph overriding only the default region; all required regions remain admitted",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing config, processed artifacts, and models",
    )
    args = parser.parse_args()
    if args.graph is None:
        return check_artifacts(args.project_root)
    return check_artifacts(args.project_root, graph=args.graph)


if __name__ == "__main__":
    sys.exit(main())
