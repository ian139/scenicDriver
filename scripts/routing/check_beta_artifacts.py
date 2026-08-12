#!/usr/bin/env python3
"""Check the read-only artifacts required by the New England beta API."""

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.route_planner._edge_projection import EdgeProjectionIndex, _SidecarError
from src.route_planner.graph import (
    _COMPACT_DTYPE_ITEMSIZE,
    _COMPACT_FORMAT,
    _COMPACT_SCENIC_BYWAY_ROAD_TYPE,
    _COMPACT_SCHEMA_VERSION,
    _COMPACT_SECTION_NAMES,
    _COMPACT_SECTION_SPECS,
    _compact_highway_road_types,
)
CONFIG_PATH = Path("config/app_regions.json")
REGISTRY_PATH = Path("data/processed/regression/model_registry.json")
DEFAULT_GRAPH_PATH = Path(
    "data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3"
)
DEFAULT_RUN_NAME = "prompt_two_candidate_exp02_fresh_test20_20260810"
DEFAULT_CHECKPOINT_PATH = Path(
    "data/processed/regression/checkpoints/0a165e429c8ac050524c7da409dd533d2de8849600c5dd8605aa1a1024d823e9.pt"
)

_SQLITE_GRAPH_FORMAT = "scenic-roadgraph-sqlite"
_SQLITE_SCHEMA_VERSION = 1
_REQUIRED_PROBES = (
    "rutland_usps",
    "lisbon_police",
    "burlington",
    "bangor",
)
_CANONICAL_PROBE_COORDINATES = {
    "rutland_usps": (43.60784414, -72.98226538),
    "lisbon_police": (44.02516775, -70.10003245),
    "burlington": (44.475884, -73.214003),
    "bangor": (44.801616, -68.771305),
}

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
    """Resolve all configured route-enabled regions with graph paths.

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
        default_graph = (
            _project_path(root, cli_graph)
            if cli_graph is not None
            else root / DEFAULT_GRAPH_PATH
        )
        return [
            {
                "region": "new_england_north",
                "graph_path": default_graph,
                "run_name": DEFAULT_RUN_NAME,
                "region_dict": None,
                "is_default": True,
            }
        ], []

    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        default_graph = (
            _project_path(root, cli_graph)
            if cli_graph is not None
            else root / DEFAULT_GRAPH_PATH
        )
        return [
            {
                "region": "new_england_north",
                "graph_path": default_graph,
                "run_name": DEFAULT_RUN_NAME,
                "region_dict": None,
                "is_default": True,
            }
        ], [f"invalid: {CONFIG_PATH} ({exc})"]

    if not isinstance(payload, dict):
        default_graph = (
            _project_path(root, cli_graph)
            if cli_graph is not None
            else root / DEFAULT_GRAPH_PATH
        )
        return [
            {
                "region": "new_england_north",
                "graph_path": default_graph,
                "run_name": DEFAULT_RUN_NAME,
                "region_dict": None,
                "is_default": True,
            }
        ], [f"invalid: {CONFIG_PATH} (config payload is not a dict)"]

    default_region_name = payload.get("default_region")
    if not isinstance(default_region_name, str) or not default_region_name:
        default_region_name = "new_england_north"

    regions_raw = payload.get("regions")
    if not isinstance(regions_raw, list):
        default_graph = (
            _project_path(root, cli_graph)
            if cli_graph is not None
            else root / DEFAULT_GRAPH_PATH
        )
        return [
            {
                "region": default_region_name,
                "graph_path": default_graph,
                "run_name": DEFAULT_RUN_NAME,
                "region_dict": None,
                "is_default": True,
            }
        ], [f"invalid: {CONFIG_PATH} (regions array is missing)"]

    route_regions: list[dict[str, Any]] = []
    found_default = False

    for item in regions_raw:
        if not isinstance(item, dict):
            continue
        region_name = item.get("region")
        if not isinstance(region_name, str) or not region_name:
            continue

        is_default = region_name == default_region_name
        if is_default:
            found_default = True

        if item.get("route_planning") is False:
            continue

        raw_graph = item.get("graph")
        if raw_graph is None or not str(raw_graph).strip():
            if is_default:
                issues.append(
                    f"invalid: {CONFIG_PATH} ({region_name} graph is missing)"
                )
            continue

        if is_default and cli_graph is not None:
            graph_path = _project_path(root, cli_graph)
        else:
            graph_path = _project_path(root, raw_graph)

        run_name = item.get("run_name")
        if not isinstance(run_name, str) or not run_name:
            issues.append(f"invalid: {CONFIG_PATH} ({region_name} run_name is missing)")
            run_name = DEFAULT_RUN_NAME if is_default else ""

        route_regions.append(
            {
                "region": region_name,
                "graph_path": graph_path,
                "run_name": run_name,
                "region_dict": item,
                "is_default": is_default,
            }
        )

    if not route_regions:
        default_graph = (
            _project_path(root, cli_graph)
            if cli_graph is not None
            else root / DEFAULT_GRAPH_PATH
        )
        route_regions.append(
            {
                "region": default_region_name,
                "graph_path": default_graph,
                "run_name": DEFAULT_RUN_NAME,
                "region_dict": None,
                "is_default": True,
            }
        )
        if not found_default:
            issues.append(
                f"invalid: {CONFIG_PATH} ({default_region_name} is not configured)"
            )

    return route_regions, issues


def _configured_region(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the default configured region without validating artifacts."""
    regions, issues = _configured_route_regions(root)
    for reg in regions:
        if reg["is_default"]:
            return reg["region_dict"], issues
    return (regions[0]["region_dict"] if regions else None), issues


def _configured_new_england(root: Path) -> tuple[Path, str, list[str]]:
    """Read the default graph/run settings, with canonical fallbacks."""
    regions, issues = _configured_route_regions(root)
    for reg in regions:
        if reg["is_default"]:
            return reg["graph_path"], reg["run_name"], issues
    if regions:
        return regions[0]["graph_path"], regions[0]["run_name"], issues
    return root / DEFAULT_GRAPH_PATH, DEFAULT_RUN_NAME, issues


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
    """Validate SQLite graph metadata and aggregate invariants without loading rows."""

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
        if configured_probes is None:
            expected_probes: Mapping[str, Any] = {
                name: {"lat": coordinates[0], "lon": coordinates[1]}
                for name, coordinates in _CANONICAL_PROBE_COORDINATES.items()
            }
        elif isinstance(configured_probes, Mapping) and configured_probes:
            expected_probes = configured_probes
        else:
            return [f"invalid: {label} (configured coverage_probes are invalid)"]

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
    """Validate compact graph manifest, binary payload, source, and projection sidecar."""
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
        if dtype not in _COMPACT_DTYPE_ITEMSIZE:
            return [f"invalid: {label} (compact section {name} has unknown dtype)"]
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
        EdgeProjectionIndex.load(sidecar_path, source_path, verify=True)
    except (_SidecarError, ValueError, OSError) as exc:
        return [f"invalid: {sidecar_path} ({exc})"]

    return []

def _active_checkpoint(root: Path) -> tuple[Path, list[str]]:
    """Resolve the checkpoint named by the active model-registry record."""

    registry_path = root / REGISTRY_PATH
    checkpoint_path = root / DEFAULT_CHECKPOINT_PATH
    issues: list[str] = []
    if not registry_path.is_file():
        return checkpoint_path, issues

    try:
        payload: Any = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(f"invalid: {REGISTRY_PATH} ({exc})")
        return checkpoint_path, issues

    active = payload.get("active") if isinstance(payload, dict) else None
    configured_checkpoint = (
        active.get("checkpoint") if isinstance(active, dict) else None
    )
    if isinstance(configured_checkpoint, str) and configured_checkpoint:
        checkpoint_path, resolve_issues = _resolve_active_checkpoint(
            root, registry_path, configured_checkpoint
        )
        issues.extend(resolve_issues)
    else:
        issues.append(f"invalid: {REGISTRY_PATH} (active checkpoint is missing)")
    return checkpoint_path, issues

def check_artifacts(project_root: Path, graph: Path | None = None) -> int:
    root = project_root.resolve()
    regions, issues = _configured_route_regions(root, cli_graph=graph)
    checkpoint_path, registry_issues = _active_checkpoint(root)
    issues.extend(registry_issues)

    required_paths: list[Path] = [
        root / REGISTRY_PATH,
        checkpoint_path,
    ]

    for reg in regions:
        graph_path = reg["graph_path"]
        run_name = reg["run_name"]
        if graph_path and graph_path not in required_paths:
            required_paths.append(graph_path)
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
            issues.extend(_validate_sqlite_graph(graph_path, region_dict))
            compact_manifest = graph_path.with_name(f"{graph_path.stem}.compact.json")
            if compact_manifest.is_file():
                issues.extend(_validate_compact_graph(compact_manifest, region_dict))
            sidecar_path = EdgeProjectionIndex.sidecar_path(graph_path)
            if sidecar_path.is_file() and not compact_manifest.is_file():
                try:
                    EdgeProjectionIndex.load(sidecar_path, graph_path, verify=True)
                except (_SidecarError, ValueError, OSError) as exc:
                    issues.append(f"invalid: {sidecar_path} ({exc})")
        elif (
            graph_path.name.endswith(".compact.json")
            or graph_path.suffix.lower() == ".json"
        ) and graph_path not in missing:
            issues.extend(_validate_compact_graph(graph_path, region_dict))

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
        help="Optional candidate graph path overriding the configured graph",
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
