from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.routing import check_beta_artifacts as checker
from src.route_planner.graph import RoadGraph, write_compact_graph

BBOX = {
    "min_lat": 42.488301979602255,
    "min_lon": -73.5205078125,
    "max_lat": 47.50235895196859,
    "max_lon": -66.796875,
}
PROBES = {
    name: {
        "lat": coordinates[0],
        "lon": coordinates[1],
        "distance_km": 0.25,
    }
    for name, coordinates in checker._CANONICAL_PROBE_COORDINATES.items()
}


def _write_project(root: Path, *, graph: str) -> None:
    (root / "config").mkdir(parents=True)
    (root / "config/app_regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region": "new_england_north",
                        "graph": graph,
                        "run_name": checker.DEFAULT_RUN_NAME,
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = (
        root / "data/processed/heuristic_runs" / checker.DEFAULT_RUN_NAME / "report"
    )
    report.mkdir(parents=True)
    for name in ("report.json", "route.geojson", "route_metrics.json"):
        (report / name).write_text("{}", encoding="utf-8")
    registry = root / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(checker.DEFAULT_CHECKPOINT_PATH)}}),
        encoding="utf-8",
    )
    checkpoint = root / checker.DEFAULT_CHECKPOINT_PATH
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")


def _write_sqlite_graph(
    path: Path,
    schema_overrides: dict[str, str] | None = None,
    **metadata_updates: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "graph_format": "scenic-roadgraph-sqlite",
        "schema_version": 1,
        "bbox": BBOX,
        "node_count": 2,
        "edge_count": 1,
        "coverage_probes": PROBES,
    }
    metadata.update(metadata_updates)
    schema_overrides = schema_overrides or {}
    with sqlite3.connect(path) as connection:
        connection.executescript(
            f"""
            CREATE TABLE metadata({
                schema_overrides.get(
                    "metadata", "key TEXT PRIMARY KEY, value TEXT NOT NULL"
                )
            });
            CREATE TABLE nodes({
                schema_overrides.get(
                    "nodes",
                    "id TEXT PRIMARY KEY, lat REAL NOT NULL, lon REAL NOT NULL",
                )
            });
            CREATE TABLE edges({
                schema_overrides.get(
                    "edges",
                    "id TEXT PRIMARY KEY, start_node_id TEXT NOT NULL, "
                    "end_node_id TEXT NOT NULL, distance_km REAL NOT NULL, "
                    "scenic_score REAL NOT NULL, road_name TEXT, "
                    "road_type TEXT NOT NULL, speed_limit_kmh REAL, "
                    "one_way INTEGER NOT NULL CHECK(one_way IN (0, 1))",
                )
            });
            """
        )
        connection.executemany(
            "INSERT INTO nodes(id, lat, lon) VALUES (?, ?, ?)",
            [("a", 44.0, -70.0), ("b", 44.1, -70.1)],
        )
        connection.execute(
            """INSERT INTO edges(
                id, start_node_id, end_node_id, distance_km, scenic_score,
                road_name, road_type, speed_limit_kmh, one_way
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("edge", "a", "b", 1.0, 0.5, "Road", "primary", 50.0, 0),
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                (key, json.dumps(value, allow_nan=False))
                for key, value in metadata.items()
            ],
        )


def test_valid_configured_sqlite_graph_passes(tmp_path: Path) -> None:
    relative = "data/processed/road_graphs/candidate/road_graph.sqlite3"
    _write_project(tmp_path, graph=relative)
    _write_sqlite_graph(tmp_path / relative)

    assert checker.check_artifacts(tmp_path) == 0


@pytest.mark.parametrize(
    ("schema_overrides", "expected"),
    [
        (
            {"metadata": "key TEXT, value TEXT NOT NULL"},
            "metadata schema",
        ),
        (
            {
                "nodes": "id TEXT PRIMARY KEY, lat REAL, lon REAL NOT NULL",
            },
            "nodes schema",
        ),
        (
            {
                "edges": (
                    "id TEXT PRIMARY KEY, start_node_id TEXT NOT NULL, "
                    "end_node_id TEXT NOT NULL, distance_km REAL NOT NULL, "
                    "scenic_score REAL NOT NULL, road_name TEXT, "
                    "road_type TEXT NOT NULL, speed_limit_kmh REAL, "
                    "one_way INTEGER NOT NULL CHECK(one_way IN (0, 1, 2))"
                )
            },
            "edges schema is missing",
        ),
    ],
)
def test_sqlite_schema_constraints_are_rejected(
    tmp_path: Path,
    schema_overrides: dict[str, str],
    expected: str,
    capsys,
) -> None:
    relative = "candidate.sqlite3"
    _write_project(tmp_path, graph=relative)
    _write_sqlite_graph(tmp_path / relative, schema_overrides=schema_overrides)

    assert checker.check_artifacts(tmp_path) == 1
    assert expected in capsys.readouterr().out


def test_wrong_canonical_probe_coordinates_are_rejected(tmp_path: Path, capsys) -> None:
    relative = "candidate.sqlite3"
    _write_project(tmp_path, graph=relative)
    probes = {
        **PROBES,
        "rutland_usps": {**PROBES["rutland_usps"], "lon": -72.0},
    }
    _write_sqlite_graph(tmp_path / relative, coverage_probes=probes)

    assert checker.check_artifacts(tmp_path) == 1
    output = capsys.readouterr().out
    assert "rutland_usps" in output
    assert "canonical" in output


def test_candidate_override_validates_before_cutover(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        graph="data/processed/road_graphs/old/road_graph.json",
    )
    old_graph = tmp_path / "data/processed/road_graphs/old/road_graph.json"
    old_graph.parent.mkdir(parents=True)
    old_graph.write_text("legacy", encoding="utf-8")
    candidate = tmp_path / "candidate.sqlite3"
    _write_sqlite_graph(candidate)

    assert checker.check_artifacts(tmp_path, graph=Path("candidate.sqlite3")) == 0


def test_graph_bbox_can_differ_from_display_bbox(tmp_path: Path) -> None:
    relative = "candidate.sqlite3"
    _write_project(tmp_path, graph=relative)
    config_path = tmp_path / "config/app_regions.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["regions"][0]["bbox"] = {**BBOX, "max_lon": -66.0}
    config["regions"][0]["graph_bbox"] = BBOX
    config["regions"][0]["coverage_probes"] = {
        "burlington": {
            "lat": PROBES["burlington"]["lat"],
            "lon": PROBES["burlington"]["lon"],
        }
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _write_sqlite_graph(tmp_path / relative)

    assert checker.check_artifacts(tmp_path) == 0


@pytest.mark.parametrize(
    ("metadata_updates", "expected"),
    [
        ({"bbox": {**BBOX, "max_lon": -66.0}}, "bbox"),
        ({"graph_format": "wrong-format"}, "graph_format"),
        ({"schema_version": 2}, "schema_version"),
        ({"node_count": 3}, "node_count"),
        (
            {"coverage_probes": {**PROBES, "bangor": None}},
            "coverage probe bangor is missing",
        ),
        (
            {
                "coverage_probes": {
                    **PROBES,
                    "rutland_usps": {
                        **PROBES["rutland_usps"],
                        "distance_km": 1.01,
                    },
                }
            },
            "coverage probe rutland_usps exceeds",
        ),
    ],
)
def test_invalid_sqlite_metadata_is_rejected(
    tmp_path: Path, metadata_updates: dict[str, Any], expected: str, capsys
) -> None:
    relative = "candidate.sqlite3"
    _write_project(tmp_path, graph=relative)
    candidate = tmp_path / relative
    _write_sqlite_graph(candidate, **metadata_updates)

    assert checker.check_artifacts(tmp_path) == 1
    assert expected in capsys.readouterr().out


def test_integrity_failure_is_rejected(tmp_path: Path, monkeypatch) -> None:
    class IntegrityCursor:
        def fetchone(self) -> tuple[str]:
            return ("not ok",)

    relative = "candidate.sqlite3"
    _write_project(tmp_path, graph=relative)
    candidate = tmp_path / relative
    _write_sqlite_graph(candidate)
    real_connect = sqlite3.connect

    class ConnectionProxy:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, sql: str, *args: Any):
            cursor = self.connection.execute(sql, *args)
            if sql == "PRAGMA integrity_check":
                return IntegrityCursor()
            return cursor

        def close(self) -> None:
            self.connection.close()

    monkeypatch.setattr(
        checker.sqlite3,
        "connect",
        lambda *args, **kwargs: ConnectionProxy(real_connect(*args, **kwargs)),
    )

    assert checker.check_artifacts(tmp_path) == 1


def test_cli_graph_override_returns_nonzero_for_bad_candidate(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _write_project(
        tmp_path,
        graph="data/processed/road_graphs/old/road_graph.json",
    )
    old_graph = tmp_path / "data/processed/road_graphs/old/road_graph.json"
    old_graph.parent.mkdir(parents=True)
    old_graph.write_text("legacy", encoding="utf-8")
    candidate = tmp_path / "candidate.sqlite3"
    _write_sqlite_graph(candidate, edge_count=2)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_beta_artifacts.py",
            "--project-root",
            str(tmp_path),
            "--graph",
            "candidate.sqlite3",
        ],
    )
    assert checker.main() == 1
    assert "edge_count" in capsys.readouterr().out


def test_active_checkpoint_resolves_registry_relative_and_root_relative(
    tmp_path: Path,
) -> None:
    root = tmp_path
    registry = root / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    sha = "ab" * 32

    # Registry-relative checkpoints/<sha>.pt resolves under the registry directory
    registry.write_text(
        json.dumps({"active": {"checkpoint": f"checkpoints/{sha}.pt"}}),
        encoding="utf-8",
    )
    reg_ckpt = registry.parent / "checkpoints" / f"{sha}.pt"
    reg_ckpt.parent.mkdir(parents=True)
    reg_ckpt.write_bytes(b"checkpoint")
    resolved, issues = checker._active_checkpoint(root)
    assert resolved == reg_ckpt
    assert issues == []

    # Project-root-relative models/... resolves against the deployment root
    registry.write_text(
        json.dumps({"active": {"checkpoint": "models/baseline.pt"}}),
        encoding="utf-8",
    )
    root_ckpt = root / "models" / "baseline.pt"
    root_ckpt.parent.mkdir(parents=True)
    root_ckpt.write_bytes(b"checkpoint")
    resolved, issues = checker._active_checkpoint(root)
    assert resolved == root_ckpt
    assert issues == []

    # Absolute container paths are preserved as-is
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(reg_ckpt)}}),
        encoding="utf-8",
    )
    resolved, issues = checker._active_checkpoint(root)
    assert resolved == reg_ckpt
    assert issues == []


def test_active_checkpoint_prefers_existing_and_reports_missing_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path
    registry = root / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    sha = "cd" * 32
    registry.write_text(
        json.dumps({"active": {"checkpoint": f"checkpoints/{sha}.pt"}}),
        encoding="utf-8",
    )

    # Only the project-root-relative copy exists -> prefer the existing file
    root_ckpt = root / "checkpoints" / f"{sha}.pt"
    root_ckpt.parent.mkdir(parents=True)
    root_ckpt.write_bytes(b"checkpoint")
    resolved, issues = checker._active_checkpoint(root)
    assert resolved == root_ckpt
    assert issues == []

    # Neither copy exists -> registry-relative candidate plus tried paths
    missing_sha = "ef" * 32
    registry.write_text(
        json.dumps({"active": {"checkpoint": f"checkpoints/{missing_sha}.pt"}}),
        encoding="utf-8",
    )
    resolved, issues = checker._active_checkpoint(root)
    assert resolved == registry.parent / "checkpoints" / f"{missing_sha}.pt"
    assert len(issues) == 1
    assert "tried:" in issues[0]
    assert str(registry.parent / "checkpoints" / f"{missing_sha}.pt") in issues[0]
    assert str(root / "checkpoints" / f"{missing_sha}.pt") in issues[0]
def test_valid_compact_graph_artifacts_pass(tmp_path: Path) -> None:
    relative_sqlite = "data/processed/road_graphs/candidate/road_graph.sqlite3"
    _write_project(tmp_path, graph=relative_sqlite)
    sqlite_path = tmp_path / relative_sqlite
    _write_sqlite_graph(sqlite_path)

    graph = RoadGraph.load(sqlite_path)
    graph.persist_edge_projection_index(sqlite_path)
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    write_compact_graph(sqlite_path, manifest_path)

    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 0


def test_corrupt_compact_payload_sha_rejected_by_checker(
    tmp_path: Path, capsys
) -> None:
    relative_sqlite = "data/processed/road_graphs/candidate/road_graph.sqlite3"
    _write_project(tmp_path, graph=relative_sqlite)
    sqlite_path = tmp_path / relative_sqlite
    _write_sqlite_graph(sqlite_path)

    graph = RoadGraph.load(sqlite_path)
    graph.persist_edge_projection_index(sqlite_path)
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    write_compact_graph(sqlite_path, manifest_path)

    bin_path = sqlite_path.with_name("road_graph.compact.bin")
    bin_bytes = bytearray(bin_path.read_bytes())
    bin_bytes[0] ^= 0xFF
    bin_path.write_bytes(bytes(bin_bytes))

    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 1
    output = capsys.readouterr().out
    assert "compact binary SHA-256 mismatch" in output


def test_corrupt_projection_sidecar_rejected_by_checker(
    tmp_path: Path, capsys
) -> None:
    relative_sqlite = "data/processed/road_graphs/candidate/road_graph.sqlite3"
    _write_project(tmp_path, graph=relative_sqlite)
    sqlite_path = tmp_path / relative_sqlite
    _write_sqlite_graph(sqlite_path)

    graph = RoadGraph.load(sqlite_path)
    graph.persist_edge_projection_index(sqlite_path)
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    write_compact_graph(sqlite_path, manifest_path)

    sidecar_path = sqlite_path.with_name("road_graph.sqlite3.edge_projection_index")
    sidecar_bytes = bytearray(sidecar_path.read_bytes())
    sidecar_bytes[-1] ^= 0xFF
    sidecar_path.write_bytes(bytes(sidecar_bytes))

    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 1
    output = capsys.readouterr().out
    assert "invalid:" in output
    assert "projection_index" in output or "edge_projection_index" in output


def test_multiple_enabled_regions_pass(tmp_path: Path) -> None:
    ne_graph = "data/processed/road_graphs/ne/road_graph.sqlite3"
    mw_graph = "data/processed/road_graphs/mw/road_graph.sqlite3"
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config/app_regions.json").write_text(
        json.dumps(
            {
                "default_region": "new_england_north",
                "regions": [
                    {
                        "region": "new_england_north",
                        "graph": ne_graph,
                        "run_name": "run_ne",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                    {
                        "region": "masswhites",
                        "graph": mw_graph,
                        "run_name": "run_mw",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_sqlite_graph(tmp_path / ne_graph)
    _write_sqlite_graph(tmp_path / mw_graph)
    for run in ("run_ne", "run_mw"):
        rep = tmp_path / "data/processed/heuristic_runs" / run / "report"
        rep.mkdir(parents=True)
        for name in ("report.json", "route.geojson", "route_metrics.json"):
            (rep / name).write_text("{}", encoding="utf-8")

    registry = tmp_path / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(checker.DEFAULT_CHECKPOINT_PATH)}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / checker.DEFAULT_CHECKPOINT_PATH
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert checker.check_artifacts(tmp_path) == 0


def test_disabled_region_skipped(tmp_path: Path) -> None:
    ne_graph = "data/processed/road_graphs/ne/road_graph.sqlite3"
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config/app_regions.json").write_text(
        json.dumps(
            {
                "default_region": "new_england_north",
                "regions": [
                    {
                        "region": "new_england_north",
                        "graph": ne_graph,
                        "run_name": "run_ne",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                    {
                        "region": "northeast_expanded",
                        "graph": None,
                        "route_planning": False,
                        "run_name": "run_expanded",
                        "bbox": BBOX,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_sqlite_graph(tmp_path / ne_graph)
    rep = tmp_path / "data/processed/heuristic_runs" / "run_ne" / "report"
    rep.mkdir(parents=True)
    for name in ("report.json", "route.geojson", "route_metrics.json"):
        (rep / name).write_text("{}", encoding="utf-8")

    registry = tmp_path / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(checker.DEFAULT_CHECKPOINT_PATH)}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / checker.DEFAULT_CHECKPOINT_PATH
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert checker.check_artifacts(tmp_path) == 0


def test_missing_second_region_artifact_rejected(
    tmp_path: Path, capsys
) -> None:
    ne_graph = "data/processed/road_graphs/ne/road_graph.sqlite3"
    mw_graph = "data/processed/road_graphs/mw/road_graph.sqlite3"
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config/app_regions.json").write_text(
        json.dumps(
            {
                "default_region": "new_england_north",
                "regions": [
                    {
                        "region": "new_england_north",
                        "graph": ne_graph,
                        "run_name": "run_ne",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                    {
                        "region": "masswhites",
                        "graph": mw_graph,
                        "run_name": "run_mw",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_sqlite_graph(tmp_path / ne_graph)
    _write_sqlite_graph(tmp_path / mw_graph)
    rep_ne = tmp_path / "data/processed/heuristic_runs" / "run_ne" / "report"
    rep_ne.mkdir(parents=True)
    for name in ("report.json", "route.geojson", "route_metrics.json"):
        (rep_ne / name).write_text("{}", encoding="utf-8")

    # Second region missing route_metrics.json
    rep_mw = tmp_path / "data/processed/heuristic_runs" / "run_mw" / "report"
    rep_mw.mkdir(parents=True)
    (rep_mw / "report.json").write_text("{}", encoding="utf-8")
    (rep_mw / "route.geojson").write_text("{}", encoding="utf-8")

    registry = tmp_path / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(checker.DEFAULT_CHECKPOINT_PATH)}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / checker.DEFAULT_CHECKPOINT_PATH
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert checker.check_artifacts(tmp_path) == 1
    output = capsys.readouterr().out
    assert "missing:" in output
    assert "run_mw/report/route_metrics.json" in output


def test_per_region_compact_validation_selection(tmp_path: Path) -> None:
    ne_graph = "data/processed/road_graphs/ne/road_graph.sqlite3"
    mw_sqlite = "data/processed/road_graphs/mw/road_graph.sqlite3"
    mw_compact = "data/processed/road_graphs/mw/road_graph.compact.json"
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config/app_regions.json").write_text(
        json.dumps(
            {
                "default_region": "new_england_north",
                "regions": [
                    {
                        "region": "new_england_north",
                        "graph": ne_graph,
                        "run_name": "run_ne",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                    {
                        "region": "masswhites",
                        "graph": mw_compact,
                        "run_name": "run_mw",
                        "max_route_snap_km": 1.0,
                        "bbox": BBOX,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_sqlite_graph(tmp_path / ne_graph)

    sqlite_path = tmp_path / mw_sqlite
    _write_sqlite_graph(sqlite_path)
    graph = RoadGraph.load(sqlite_path)
    graph.persist_edge_projection_index(sqlite_path)
    manifest_path = tmp_path / mw_compact
    write_compact_graph(sqlite_path, manifest_path)

    for run in ("run_ne", "run_mw"):
        rep = tmp_path / "data/processed/heuristic_runs" / run / "report"
        rep.mkdir(parents=True)
        for name in ("report.json", "route.geojson", "route_metrics.json"):
            (rep / name).write_text("{}", encoding="utf-8")

    registry = tmp_path / checker.REGISTRY_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(checker.DEFAULT_CHECKPOINT_PATH)}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / checker.DEFAULT_CHECKPOINT_PATH
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert checker.check_artifacts(tmp_path) == 0
def _make_valid_compact_project(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    relative_sqlite = "data/processed/road_graphs/candidate/road_graph.sqlite3"
    _write_project(tmp_path, graph=relative_sqlite)
    sqlite_path = tmp_path / relative_sqlite
    _write_sqlite_graph(sqlite_path)

    graph = RoadGraph.load(sqlite_path)
    graph.persist_edge_projection_index(sqlite_path)
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    write_compact_graph(sqlite_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sqlite_path, manifest_path, manifest


def _mutate_dict(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = d
    for key in path[:-1]:
        current = current[key]
    if value == "__DELETE__":
        current.pop(path[-1], None)
    else:
        current[path[-1]] = value


@pytest.mark.parametrize(
    "path,value,expected_issue",
    [
        (("format",), "invalid-format", "unsupported compact format"),
        (("schema_version",), True, "unsupported compact schema_version"),
        (("schema_version",), 2, "unsupported compact schema_version"),
        (("graph",), None, "missing graph counts mapping"),
        (("graph", "node_count"), -1, "invalid compact graph counts"),
        (("graph", "node_count"), True, "invalid compact graph counts"),
        (("graph", "node_count"), "five", "invalid compact graph counts"),
        (("source",), None, "missing source metadata"),
        (("source", "schema_version"), True, "unsupported compact source schema_version"),
        (("source", "schema_version"), 99, "unsupported compact source schema_version"),
        (("source", "node_count"), 999, "compact source node count mismatch"),
        (("source", "node_count"), True, "compact source node count mismatch"),
        (("source", "size_bytes"), True, "invalid source size_bytes"),
        (("source", "size_bytes"), "100", "invalid source size_bytes"),
        (("source", "sha256"), "short", "invalid or missing source sha256"),
        (("source", "sha256"), True, "invalid or missing source sha256"),
        (("bin_path",), "", "missing bin_path"),
        (("bin_size_bytes",), True, "invalid bin_size_bytes"),
        (("bin_size_bytes",), "100", "invalid bin_size_bytes"),
        (("bin_sha256",), "", "invalid or missing bin_sha256"),
        (("bin_sha256",), "invalid_hex", "invalid or missing bin_sha256"),
        (("sections",), None, "missing sections mapping"),
        (("scenic_byway_road_type",), "wrong", "scenic byway marker mismatch"),
        (("highway_road_types",), "motorway", "highway road-type mask is stale"),
        (("projection_index",), None, "missing projection_index metadata"),
        (("projection_index", "size_bytes"), True, "invalid projection sidecar size_bytes"),
        (("projection_index", "size_bytes"), "100", "invalid projection sidecar size_bytes"),
        (("projection_index", "sha256"), "short", "invalid or missing projection sidecar sha256"),
    ],
)
def test_compact_manifest_fields_rejected(
    tmp_path: Path, path: tuple[str, ...], value: Any, expected_issue: str, capsys
) -> None:
    _sqlite_path, manifest_path, manifest = _make_valid_compact_project(tmp_path)
    _mutate_dict(manifest, path, value)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 1
    output = capsys.readouterr().out
    assert expected_issue in output


def test_compact_manifest_missing_and_unknown_sections_rejected(
    tmp_path: Path, capsys
) -> None:
    _sqlite_path, manifest_path, manifest = _make_valid_compact_project(tmp_path)
    # Missing section
    manifest_missing = json.loads(json.dumps(manifest))
    del manifest_missing["sections"]["node_lat"]
    manifest_path.write_text(json.dumps(manifest_missing), encoding="utf-8")
    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 1
    assert "missing sections:" in capsys.readouterr().out

    # Unknown section
    manifest_unknown = json.loads(json.dumps(manifest))
    manifest_unknown["sections"]["extra_section"] = {"dtype": "i4", "offset": 0, "length": 4, "count": 1}
    manifest_path.write_text(json.dumps(manifest_unknown), encoding="utf-8")
    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 1
    assert "unknown section:" in capsys.readouterr().out


def test_compact_manifest_section_non_contiguous_rejected(
    tmp_path: Path, capsys
) -> None:
    _sqlite_path, manifest_path, manifest = _make_valid_compact_project(tmp_path)
    manifest["sections"]["node_lon"]["offset"] += 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert checker.check_artifacts(tmp_path, graph=manifest_path) == 1
    assert "offset is not contiguous" in capsys.readouterr().out


def test_config_app_regions_resolves_exactly_two_route_regions() -> None:
    root = Path(__file__).resolve().parent.parent
    regions, issues = checker._configured_route_regions(root)
    assert issues == []
    region_names = [r["region"] for r in regions]
    assert region_names == ["new_england_north", "northeast_expanded"]
    assert len(regions) == 2

    config_payload = json.loads((root / "config/app_regions.json").read_text(encoding="utf-8"))
    regions_by_name = {r["region"]: r for r in config_payload["regions"]}
    assert regions_by_name["masswhites"].get("route_planning") is False
    assert regions_by_name["philadelphia"].get("route_planning") is False
    assert regions_by_name["pittsfield"].get("route_planning") is False
    assert regions_by_name["northeast_expanded"].get("route_planning") is True
