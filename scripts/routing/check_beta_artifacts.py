#!/usr/bin/env python3
"""Check the read-only artifacts required by the New England beta API."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping


CONFIG_PATH = Path("config/app_regions.json")
REGISTRY_PATH = Path("data/processed/regression/model_registry.json")
DEFAULT_GRAPH_PATH = Path(
    "data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3"
)
DEFAULT_RUN_NAME = "new_england_north_z14_v6_learned"
DEFAULT_CHECKPOINT_PATH = Path(
    "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt"
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


def _configured_region(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the configured New England region without validating artifacts."""

    config_path = root / CONFIG_PATH
    if not config_path.is_file():
        return None, []
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [f"invalid: {CONFIG_PATH} ({exc})"]
    regions = payload.get("regions") if isinstance(payload, dict) else None
    region = next(
        (
            item
            for item in regions or []
            if isinstance(item, dict) and item.get("region") == "new_england_north"
        ),
        None,
    )
    if not isinstance(region, dict):
        return None, [
            f"invalid: {CONFIG_PATH} (new_england_north is not configured)"
        ]
    return region, []


def _configured_new_england(root: Path) -> tuple[Path, str, list[str]]:
    """Read the New England graph/run settings, with canonical fallbacks."""

    graph_path = root / DEFAULT_GRAPH_PATH
    run_name = DEFAULT_RUN_NAME
    region, issues = _configured_region(root)
    if region is None:
        return graph_path, run_name, issues
    configured_graph = region.get("graph")
    configured_run = region.get("run_name")
    if isinstance(configured_graph, str) and configured_graph:
        graph_path = _project_path(root, configured_graph)
    else:
        issues.append(f"invalid: {CONFIG_PATH} (new_england_north graph is missing)")
    if isinstance(configured_run, str) and configured_run:
        run_name = configured_run
    else:
        issues.append(
            f"invalid: {CONFIG_PATH} (new_england_north run_name is missing)"
        )
    return graph_path, run_name, issues


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
            schema_rows = tuple(
                connection.execute(f"PRAGMA table_info({table})")
            )
            columns = tuple(str(row[1]) for row in schema_rows)
            types = tuple(str(row[2]).upper() for row in schema_rows)
            declarations = tuple(
                (int(row[3]), int(row[5])) for row in schema_rows
            )
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
                f"invalid: {label} (edges schema is missing "
                "CHECK(one_way IN (0, 1)))"
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
            return [
                f"invalid: {label} (unsupported schema_version "
                f"{schema_version!r})"
            ]
        if not _metadata_bbox_matches(metadata.get("bbox"), region.get("bbox")):
            return [f"invalid: {label} (bbox does not match configured region)"]

        for name, table in (("node_count", "nodes"), ("edge_count", "edges")):
            value = metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return [f"invalid: {label} ({name} metadata count is not positive)"]
            actual = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
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
        for probe_name in _REQUIRED_PROBES:
            probe = probes.get(probe_name)
            if not isinstance(probe, Mapping):
                return [f"invalid: {label} (coverage probe {probe_name} is missing)"]

            expected_lat, expected_lon = _CANONICAL_PROBE_COORDINATES[probe_name]
            latitude = probe.get("lat")
            longitude = probe.get("lon")
            if (
                isinstance(latitude, bool)
                or isinstance(longitude, bool)
                or not isinstance(latitude, (int, float))
                or not isinstance(longitude, (int, float))
                or not math.isfinite(float(latitude))
                or not math.isfinite(float(longitude))
            ):
                return [
                    f"invalid: {label} (coverage probe {probe_name} "
                    "coordinates are invalid)"
                ]
            if float(latitude) != expected_lat or float(longitude) != expected_lon:
                return [
                    f"invalid: {label} (coverage probe {probe_name} "
                    "coordinates do not match canonical values)"
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
    configured_checkpoint = active.get("checkpoint") if isinstance(active, dict) else None
    if isinstance(configured_checkpoint, str) and configured_checkpoint:
        checkpoint_path = _project_path(root, configured_checkpoint)
    else:
        issues.append(f"invalid: {REGISTRY_PATH} (active checkpoint is missing)")
    return checkpoint_path, issues


def check_artifacts(project_root: Path, graph: Path | None = None) -> int:
    root = project_root.resolve()
    configured_graph, run_name, issues = _configured_new_england(root)
    region, _ = _configured_region(root)
    graph_path = _project_path(root, graph) if graph is not None else configured_graph
    checkpoint_path, registry_issues = _active_checkpoint(root)
    issues.extend(registry_issues)

    report_dir = root / "data/processed/heuristic_runs" / run_name / "report"
    required_paths = [
        graph_path,
        report_dir / "report.json",
        report_dir / "route.geojson",
        report_dir / "route_metrics.json",
        root / REGISTRY_PATH,
        checkpoint_path,
    ]

    missing: list[Path] = []
    for path in required_paths:
        if not path.is_file() and path not in missing:
            missing.append(path)

    if graph_path.suffix.lower() == ".sqlite3" and graph_path not in missing:
        issues.extend(_validate_sqlite_graph(graph_path, region))

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
