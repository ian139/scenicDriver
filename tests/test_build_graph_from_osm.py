from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.routing import build_graph_from_osm as builder
from src.route_planner._edge_projection import EdgeProjectionIndex
from src.route_planner.graph import Edge, Node, RoadGraph


def _args(**overrides):
    values = {
        "min_lat": 42.0,
        "min_lon": -73.0,
        "max_lat": 43.0,
        "max_lon": -72.0,
        "output": None,
        "output_root": Path("data/processed/road_graphs"),
        "run_name": "test-run",
        "network": "drive",
        "max_query_area": None,
        "overpass_url": None,
        "timeout": 123,
        "source_pbf": [],
        "require_source_checksums": False,
        "graph_format": "json",
        "cache_folder": None,
        "attempts": 3,
        "coverage_probe": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parser_preserves_repeatable_sources_and_probes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--min-lat",
            "42",
            "--min-lon",
            "-73",
            "--max-lat",
            "43",
            "--max-lon",
            "-72",
            "--source-pbf",
            "a.pbf",
            "--source-pbf",
            "b.pbf",
            "--require-source-checksums",
            "--graph-format",
            "sqlite3",
            "--cache-folder",
            "cache/x",
            "--attempts",
            "5",
            "--coverage-probe",
            "start",
            "42.1",
            "-72.9",
        ],
    )

    args = builder.parse_args()

    assert [str(path) for path in args.source_pbf] == ["a.pbf", "b.pbf"]
    assert args.require_source_checksums is True
    assert args.graph_format == "sqlite3"
    assert args.attempts == 5
    assert args.coverage_probe == [["start", "42.1", "-72.9"]]


def test_graph_from_bbox_uses_west_south_east_north_order() -> None:
    calls = []

    class FakeOsmnx:
        @staticmethod
        def graph_from_bbox(*, bbox, network_type):
            calls.append((bbox, network_type))
            return "graph"

    assert builder._load_overpass_graph(
        FakeOsmnx,
        _args(
            min_lat=42.4,
            min_lon=-73.5,
            max_lat=47.5,
            max_lon=-66.7,
            attempts=1,
        ),
    ) == ("graph", [])
    assert calls == [((-73.5, 42.4, -66.7, 47.5), "drive")]


def test_osmnx_settings_use_requests_timeout_and_cache(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        cache_folder=None,
        use_cache=False,
        requests_timeout=None,
        max_query_area_size=1.0,
        overpass_url="default",
        overpass_rate_limit=False,
    )
    ox = SimpleNamespace(__version__="2.1.0", settings=settings)
    args = _args(
        cache_folder=tmp_path / "osmnx",
        max_query_area=1234.0,
        overpass_url="https://overpass.example/api",
        timeout=77,
    )

    record = builder._configure_osmnx(ox, args)

    assert settings.requests_timeout == 77
    assert settings.use_cache is True
    assert settings.overpass_rate_limit is True
    assert settings.cache_folder == str(tmp_path / "osmnx")
    assert settings.max_query_area_size == 1234.0
    assert settings.overpass_url == "https://overpass.example/api"
    assert record["requests_timeout"] == 77
    assert "timeout" not in record


def test_source_manifest_verifies_md5_and_records_sha256(tmp_path: Path) -> None:
    source = tmp_path / "source.osm.pbf"
    source.write_bytes(b"source-data")
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    source.with_name(source.name + ".md5").write_text(
        f"{digest}  {source.name}\n",
        encoding="utf-8",
    )

    manifest = builder._source_manifest([source], require_checksums=True)

    assert manifest[0]["verified_md5"] == digest
    assert manifest[0]["md5"] == digest
    assert manifest[0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_name(source.name + ".md5").write_text("0" * 32, encoding="utf-8")
    with pytest.raises(ValueError, match="MD5 checksum mismatch"):
        builder._source_manifest([source], require_checksums=True)


def test_derived_cache_digest_includes_bbox_and_filter_contract() -> None:
    manifest = [{"sha256": "source-digest"}]
    first = builder._derived_cache_digest(
        manifest,
        {"min_lat": 42.0, "min_lon": -73.0, "max_lat": 43.0, "max_lon": -72.0},
    )
    second = builder._derived_cache_digest(
        manifest,
        {"min_lat": 42.1, "min_lon": -73.0, "max_lat": 43.0, "max_lon": -72.0},
    )
    assert first != second
    assert builder._source_digests(manifest) != first


def test_merge_filter_command_order_and_common_timestamp(tmp_path: Path, monkeypatch) -> None:
    source_a = tmp_path / "a.pbf"
    source_b = tmp_path / "b.pbf"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    manifest = builder._source_manifest([source_a, source_b], require_checksums=False)
    calls: list[list[str]] = []

    def fake_run(command):
        calls.append(command)
        if command[1] == "fileinfo":
            return SimpleNamespace(stdout="osmosis_replication_timestamp=2026-07-15T00:00:00Z", stderr="")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(builder, "_run_command", fake_run)
    output, timestamps = builder._merge_and_filter_pbf(
        [source_a.resolve(), source_b.resolve()],
        manifest,
        {"min_lat": 42.0, "min_lon": -73.0, "max_lat": 43.0, "max_lon": -72.0},
        tmp_path / "cache",
    )

    assert output.name == "filtered.osm.bz2"
    assert len(timestamps) == 2
    commands = [command[1] for command in calls]
    assert commands == ["fileinfo", "fileinfo", "merge", "extract", "tags-filter", "check-refs", "cat"]
    assert calls.index(next(command for command in calls if command[1] == "merge")) < calls.index(next(command for command in calls if command[1] == "extract"))


def test_drive_filter_matches_osmnx_drive_exclusions() -> None:
    assert builder._drive_edge_allowed({"highway": "primary"})
    for data in (
        {"highway": "service"},
        {"highway": "track"},
        {"highway": "primary", "area": "yes"},
        {"highway": "primary", "access": "private"},
        {"highway": "primary", "motor_vehicle": "no"},
        {"highway": "primary", "motorcar": "no"},
        {"highway": "primary", "service": "driveway"},
    ):
        assert not builder._drive_edge_allowed(data)

def test_drive_filter_rejects_osmnx_compound_exclusions() -> None:
    for data in (
        {"highway": "residential;service"},
        {"highway": "primary", "access": "private;destination"},
        {"highway": "primary", "motor_vehicle": "no;destination"},
        {"highway": "primary", "motorcar": "none"},
        {"highway": "primary", "service": "driveway;parking"},
    ):
        assert not builder._drive_edge_allowed(data)



def test_overpass_attempts_retry_whole_download(monkeypatch) -> None:
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError(f"failure-{len(calls)}")
        return "graph"

    fake_ox = SimpleNamespace(graph_from_bbox=fake_download)
    graph, errors = builder._load_overpass_graph(
        fake_ox,
        _args(attempts=3),
    )

    assert graph == "graph"
    assert len(calls) == 3
    assert all(
        call == {"bbox": (-73.0, 42.0, -72.0, 43.0), "network_type": "drive"}
        for call in calls
    )
    assert [row["error_type"] for row in errors] == ["RuntimeError", "RuntimeError"]


def test_local_pipeline_runs_buffer_component_simplify_exact_component(
    monkeypatch,
) -> None:
    events: list[tuple[str, object]] = []
    graph = object()

    class FakeOx:
        settings = SimpleNamespace(useful_tags_way=["highway"])

        @staticmethod
        def graph_from_xml(path, **kwargs):
            events.append(("xml", kwargs))
            return graph

        @staticmethod
        def simplify_graph(value):
            events.append(("simplify", value))
            return value

    monkeypatch.setattr(
        builder,
        "_filter_drive_graph",
        lambda value, bbox: events.append(("buffer_filter", bbox)) or value,
    )
    monkeypatch.setattr(
        builder,
        "_largest_weak_component",
        lambda value: events.append(("component", value)) or value,
    )
    monkeypatch.setattr(
        builder,
        "_truncate_graph_to_bbox",
        lambda value, bbox: events.append(("exact_truncate", bbox)) or value,
    )

    exact_bbox = (-73.0, 42.0, -72.0, 43.0)
    result = builder._load_local_osm_graph(FakeOx, Path("filtered.osm.bz2"), exact_bbox)

    assert result is graph
    assert [name for name, _value in events] == [
        "xml",
        "buffer_filter",
        "component",
        "simplify",
        "exact_truncate",
        "component",
    ]
    assert events[0][1] == {
        "bidirectional": False,
        "simplify": False,
        "retain_all": True,
    }
    assert events[4][1] == exact_bbox
    assert "motor_vehicle" in FakeOx.settings.useful_tags_way
    assert "motorcar" in FakeOx.settings.useful_tags_way


def test_replication_timestamp_mismatch_fails_before_merge(tmp_path: Path, monkeypatch) -> None:
    source_a = tmp_path / "a.pbf"
    source_b = tmp_path / "b.pbf"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    manifest = builder._source_manifest([source_a, source_b], require_checksums=False)
    calls: list[list[str]] = []

    def fake_run(command):
        calls.append(command)
        if command[1] == "fileinfo":
            timestamp = "2026-07-15T00:00:00Z" if len(calls) == 1 else "2026-07-16T00:00:00Z"
            return SimpleNamespace(
                stdout=f"osmosis_replication_timestamp={timestamp}",
                stderr="",
            )
        raise AssertionError("merge must not run")

    monkeypatch.setattr(builder, "_run_command", fake_run)
    with pytest.raises(ValueError, match="timestamps differ"):
        builder._merge_and_filter_pbf(
            [source_a, source_b],
            manifest,
            {"min_lat": 42.0, "min_lon": -73.0, "max_lat": 43.0, "max_lon": -72.0},
            tmp_path / "cache",
        )
    assert [command[1] for command in calls] == ["fileinfo", "fileinfo"]


def test_sqlite_publication_failure_does_not_replace_existing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "road_graph.sqlite3"
    output.write_bytes(b"old-artifact")
    monkeypatch.setattr(
        builder,
        "_write_sqlite_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stream failed")),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        builder._publish_sqlite_graph(
            output,
            tmp_path,
            object(),
            {},
            {},
            [],
        )
    assert output.read_bytes() == b"old-artifact"


def test_sqlite_publication_emits_loadable_edge_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "road_graph.sqlite3"
    rows = (
        ("node", Node(id="a", lat=42.0, lon=-72.0)),
        ("node", Node(id="b", lat=42.1, lon=-72.0)),
        (
            "edge",
            Edge(
                id="ab",
                start_node_id="a",
                end_node_id="b",
                distance_km=12.0,
                scenic_score=5.0,
            ),
        ),
    )
    monkeypatch.setattr(
        builder,
        "_iter_osmnx_graph_rows",
        lambda _graph, _scores: iter(rows),
    )

    node_count, edge_count, probes = builder._publish_sqlite_graph(
        output,
        tmp_path,
        object(),
        {},
        {},
        [],
    )

    assert (node_count, edge_count, probes) == (2, 1, {})
    assert EdgeProjectionIndex.sidecar_path(output).is_file()
    loaded = RoadGraph.load(output)
    assert loaded.edge_projection_index_status["state"] == "loaded"
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        42.05,
        -72.0,
    )
    assert [projection.edge.id for projection in projections] == ["ab"]


def test_sqlite_sidecar_publication_failure_recovers_without_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "road_graph.sqlite3"
    output_sidecar = EdgeProjectionIndex.sidecar_path(output)
    rows = (
        ("node", Node(id="a", lat=42.0, lon=-72.0)),
        ("node", Node(id="b", lat=42.1, lon=-72.0)),
        (
            "edge",
            Edge(
                id="ab",
                start_node_id="a",
                end_node_id="b",
                distance_km=12.0,
                scenic_score=5.0,
            ),
        ),
    )
    monkeypatch.setattr(
        builder,
        "_iter_osmnx_graph_rows",
        lambda _graph, _scores: iter(rows),
    )
    replace = builder.os.replace

    def fail_output_sidecar(source: object, destination: object) -> None:
        if Path(destination) == output_sidecar:
            raise OSError("sidecar publication failed")
        replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_output_sidecar)
    with pytest.raises(OSError, match="sidecar publication failed"):
        builder._publish_sqlite_graph(
            output,
            tmp_path,
            object(),
            {},
            {},
            [],
        )

    candidate = tmp_path / ".road_graph.candidate.sqlite3"
    assert not candidate.exists()
    assert not EdgeProjectionIndex.sidecar_path(candidate).exists()
    loaded = RoadGraph.load(output)
    assert loaded.edge_projection_index_status["state"] == "missing"
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        42.05,
        -72.0,
    )
    assert [projection.edge.id for projection in projections] == ["ab"]
    assert loaded.edge_projection_index_status["state"] == "rebuilt"
