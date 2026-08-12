from __future__ import annotations

import argparse
import hashlib
import json
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
        "tile_report": None,
        "tile_buffer": 1,
        "output": None,
        "output_root": Path("data/processed/road_graphs"),
        "run_name": "test-run",
        "network": "drive",
        "source_pbf": [],
        "require_source_checksums": False,
        "graph_format": "json",
        "cache_folder": None,
        "coverage_probe": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parser_accepts_compact_graph_format(monkeypatch) -> None:
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
            "extract.pbf",
            "--graph-format",
            "compact",
            "--run-name",
            "test-run",
        ],
    )
    args = builder.parse_args()
    assert args.graph_format == "compact"
    output_path, run_dir, run_name = builder._resolve_output_paths(args)
    assert output_path.suffix == ".sqlite3"
    assert run_name == "test-run"


def test_publish_compact_graph_matches_sqlite_and_loads(tmp_path: Path) -> None:
    from src.route_planner.graph import CompactRoadGraph

    graph = RoadGraph()
    graph.add_node(Node("n0", 42.0, -72.0))
    graph.add_node(Node("n1", 42.1, -72.0))
    graph.add_edge(
        Edge(
            "e0",
            "n0",
            "n1",
            distance_km=11.1,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    sqlite_path = tmp_path / "road_graph.sqlite3"
    graph.save(sqlite_path)
    record = builder._publish_compact_graph(sqlite_path, tmp_path)
    assert record["node_count"] == 2
    assert record["edge_count"] == 1
    assert record["traversal_count"] == 1
    loaded = CompactRoadGraph.load(Path(record["manifest_path"]))
    assert len(loaded.nodes) == 2
    assert loaded.artifact_metadata["source"]["sha256"] == record["source_sha256"]
    assert loaded.edges["e0"].distance_km == 11.1
    assert loaded.edge_projection_index_status["state"] == "loaded"


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
    assert args.coverage_probe == [["start", "42.1", "-72.9"]]

def test_parser_requires_source_pbf(monkeypatch, capsys) -> None:
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
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        builder.parse_args()

    assert exc_info.value.code == 2
    assert "--source-pbf" in capsys.readouterr().err


def test_parser_rejects_bbox_and_tile_report_together(monkeypatch, capsys) -> None:
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
            "--tile-report",
            "report.json",
            "--source-pbf",
            "a.pbf",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        builder.parse_args()

    assert exc_info.value.code == 2
    assert "--tile-report" in capsys.readouterr().err


def test_parser_requires_exactly_one_input_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--source-pbf",
            "a.pbf",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        builder.parse_args()

    assert exc_info.value.code == 2
    assert "bbox mode requires" in capsys.readouterr().err


def test_parser_accepts_tile_report_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--tile-report",
            "report.json",
            "--tile-buffer",
            "2",
            "--source-pbf",
            "a.pbf",
        ],
    )

    args = builder.parse_args()

    assert args.tile_report == Path("report.json")
    assert args.tile_buffer == 2
    assert args.min_lat is None


def test_build_requires_source_pbf() -> None:
    with pytest.raises(ValueError, match="At least one --source-pbf is required"):
        builder._build(_args())




def test_osmnx_settings_use_local_cache(tmp_path: Path) -> None:
    settings = SimpleNamespace(cache_folder=None, use_cache=False)
    ox = SimpleNamespace(__version__="2.1.0", settings=settings)
    args = _args(cache_folder=tmp_path / "osmnx")

    record = builder._configure_osmnx(ox, args)

    assert settings.use_cache is True
    assert settings.cache_folder == str(tmp_path / "osmnx")
    assert record == {
        "version": "2.1.0",
        "cache_folder": str(tmp_path / "osmnx"),
        "use_cache": True,
    }


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


def test_parse_tile_report_validates_zoom_coords_and_deduplicates(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tiles": [
                    {"z": 14, "x": 1, "y": 2},
                    {"z": 14, "x": 1, "y": 2},
                    {"z": 14, "x": 3, "y": 4},
                ]
            }
        ),
        encoding="utf-8",
    )

    zoom, coords = builder._parse_tile_report(report)

    assert zoom == 14
    assert coords == [(1, 2), (3, 4)]


def test_parse_tile_report_rejects_malformed_and_mixed_zoom(tmp_path: Path) -> None:
    cases = (
        '{"tiles": []}',
        '{"tiles": [{"z": 14, "x": 1, "y": "2"}]}',
        '{"tiles": [{"z": 14.0, "x": 1, "y": 2}]}',
        '{"tiles": [{"z": 31, "x": 1, "y": 2}]}',
        '{"tiles": [{"z": 14, "x": 16384, "y": 2}]}',
        '{"tiles": [{"z": 14, "x": 1, "y": 2}, {"z": 15, "x": 2, "y": 3}]}',
        '{"summary": {}}',
    )
    for index, content in enumerate(cases):
        report = tmp_path / f"report_{index}.json"
        report.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            builder._parse_tile_report(report)


def test_compact_tile_rectangles_is_deterministic_and_gapless(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tiles": [
                    {"z": 2, "x": 1, "y": 1},
                    {"z": 2, "x": 2, "y": 1},
                    {"z": 2, "x": 2, "y": 2},
                    {"z": 2, "x": 3, "y": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    zoom, coords = builder._parse_tile_report(report)

    rectangles = builder._compact_tile_rectangles(zoom, coords, 0)
    rectangles_again = builder._compact_tile_rectangles(zoom, coords, 0)

    assert rectangles == rectangles_again
    assert rectangles == [(1, 2, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)]

    covered = {
        (x, y)
        for x_start, x_end, y_start, y_end in rectangles
        for x in range(x_start, x_end + 1)
        for y in range(y_start, y_end + 1)
    }
    assert covered == set(coords)


def test_compact_tile_rectangles_does_not_bridge_nonadjacent_matching_runs() -> None:
    coords = [(1, 1), (2, 2), (1, 3)]

    rectangles = builder._compact_tile_rectangles(zoom=3, coords=coords, buffer=0)

    covered = {
        (x, y)
        for x_start, x_end, y_start, y_end in rectangles
        for x in range(x_start, x_end + 1)
        for y in range(y_start, y_end + 1)
    }
    assert covered == set(coords)


def test_compact_tile_rectangles_expands_by_integer_buffer(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 2, "x": 1, "y": 1}]}),
        encoding="utf-8",
    )
    zoom, coords = builder._parse_tile_report(report)

    buffered = builder._compact_tile_rectangles(zoom, coords, 1)

    assert buffered == [(0, 2, 0, 2)]

    expanded = {
        (x, y)
        for x_start, x_end, y_start, y_end in buffered
        for x in range(x_start, x_end + 1)
        for y in range(y_start, y_end + 1)
    }
    assert expanded == {(x, y) for x in (0, 1, 2) for y in (0, 1, 2)}


def test_footprint_geojson_covers_buffered_tiles_without_gaps(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "tiles": [
                    {"z": 2, "x": 1, "y": 1},
                    {"z": 2, "x": 2, "y": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 1)
    feature = footprint["geojson"]["features"][0]
    assert feature["properties"]["zoom"] == 2
    assert feature["properties"]["rectangle_count"] == 3
    assert feature["geometry"]["type"] == "MultiPolygon"
    assert len(feature["geometry"]["coordinates"]) == 3

    ring_points = [
        point
        for polygon in feature["geometry"]["coordinates"]
        for ring in polygon
        for point in ring
    ]
    polygon_bounds = (
        min(point[0] for point in ring_points),
        min(point[1] for point in ring_points),
        max(point[0] for point in ring_points),
        max(point[1] for point in ring_points),
    )

    tiles = {
        (x, y)
        for x_start, x_end, y_start, y_end in footprint["rectangles"]
        for x in range(x_start, x_end + 1)
        for y in range(y_start, y_end + 1)
    }
    tile_bounds = [
        builder._tile_lonlat_bounds(x, y, 2) for x, y in tiles
    ]
    expected_bounds = (
        min(bounds[0] for bounds in tile_bounds),
        min(bounds[1] for bounds in tile_bounds),
        max(bounds[2] for bounds in tile_bounds),
        max(bounds[3] for bounds in tile_bounds),
    )
    assert polygon_bounds == expected_bounds


def test_footprint_digest_binds_report_buffer_and_cache_key(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 2, "x": 1, "y": 1}]}),
        encoding="utf-8",
    )
    first = builder._load_tile_footprint(report, 1)
    report.write_text(
        json.dumps({"tiles": [{"z": 2, "x": 1, "y": 2}]}),
        encoding="utf-8",
    )
    changed = builder._load_tile_footprint(report, 1)
    assert first["footprint_digest"] != changed["footprint_digest"]
    assert (
        builder._footprint_cache_digest(first)
        != builder._footprint_cache_digest(changed)
    )

    report.write_text(
        json.dumps({"tiles": [{"z": 2, "x": 1, "y": 1}]}),
        encoding="utf-8",
    )
    buffered = builder._load_tile_footprint(report, 2)
    assert (
        builder._footprint_cache_digest(first)
        != builder._footprint_cache_digest(buffered)
    )

    manifest = [{"sha256": "source-digest"}]
    bbox = {"min_lat": 42.0, "min_lon": -73.0, "max_lat": 43.0, "max_lon": -72.0}
    assert (
        builder._derived_cache_digest(manifest, bbox, footprint=first)
        != builder._derived_cache_digest(manifest, bbox)
    )


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


def test_merge_filter_uses_polygon_extract_in_tile_mode(tmp_path: Path, monkeypatch) -> None:
    source_a = tmp_path / "a.pbf"
    source_b = tmp_path / "b.pbf"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 14, "x": 2000, "y": 3000}]}),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 1)
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
        footprint["bbox"],
        tmp_path / "cache",
        footprint=footprint,
    )

    assert output.name == "filtered.osm.bz2"
    assert len(timestamps) == 2
    commands = [command[1] for command in calls]
    assert commands == ["fileinfo", "fileinfo", "merge", "extract", "tags-filter", "check-refs", "cat"]

    extract = next(command for command in calls if command[1] == "extract")
    assert "--polygon" in extract
    assert "--strategy" in extract
    assert extract[extract.index("--strategy") + 1] == "complete_ways"
    assert "--bbox" not in extract

    polygon_path = Path(extract[extract.index("--polygon") + 1])
    assert polygon_path.is_file()
    assert polygon_path.name == "extraction.geojson"
    geometry = json.loads(polygon_path.read_text(encoding="utf-8"))
    assert geometry["type"] == "FeatureCollection"
    assert geometry["features"][0]["geometry"]["type"] == "Polygon"


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


def test_footprint_filter_retains_crossing_edge_with_outside_endpoints(
    tmp_path: Path,
) -> None:
    import networkx as nx

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 14, "x": 2000, "y": 3000}]}),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 1)

    graph = nx.MultiDiGraph()
    for node_id, lon, lat in (
        ("a", -137.0, 74.46),
        ("b", -134.5, 74.46),
    ):
        graph.add_node(node_id, x=lon, y=lat)
    graph.add_edge(
        "a",
        "b",
        key=0,
        highway="primary",
        geometry={
            "type": "LineString",
            "coordinates": [
                [-137.0, 74.46],
                [-136.05, 74.46],
                [-136.05, 74.44],
                [-134.5, 74.44],
            ],
        },
    )
    graph.add_edge("a", "b", key=1, highway="primary")

    builder._truncate_graph_to_footprint(graph, footprint)

    assert graph.has_edge("a", "b", key=0)
    assert not graph.has_edge("a", "b", key=1)
    assert "a" in graph.nodes
    assert "b" in graph.nodes


def test_footprint_filter_removes_outside_isolated_nodes(tmp_path: Path) -> None:
    import networkx as nx

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 14, "x": 2000, "y": 3000}]}),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 1)

    graph = nx.MultiDiGraph()
    graph.add_node("inside", x=-136.05, y=74.44)
    graph.add_node("outside", x=-137.0, y=74.44)

    builder._truncate_graph_to_footprint(graph, footprint)

    assert "inside" in graph.nodes
    assert "outside" not in graph.nodes




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


def test_tile_pipeline_preserves_disconnected_components_while_simplifying(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 14, "x": 2000, "y": 3000}]}),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 1)
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
        lambda value, bbox: events.append(("bbox_filter", bbox)) or value,
    )
    monkeypatch.setattr(
        builder,
        "_truncate_graph_to_footprint",
        lambda value, fp: events.append(("footprint_filter", fp)) or value,
    )

    result = builder._load_local_osm_graph(FakeOx, Path("filtered.osm.bz2"), None, footprint=footprint)

    assert result is graph
    assert [name for name, _value in events] == [
        "xml",
        "bbox_filter",
        "footprint_filter",
        "simplify",
    ]
    assert events[1][1] == (
        footprint["bbox"]["min_lon"],
        footprint["bbox"]["min_lat"],
        footprint["bbox"]["max_lon"],
        footprint["bbox"]["max_lat"],
    )
    assert events[2][1] is footprint


def test_bbox_mode_regression_validation_and_digest() -> None:
    assert builder._bbox(_args()) == {
        "min_lat": 42.0,
        "min_lon": -73.0,
        "max_lat": 43.0,
        "max_lon": -72.0,
    }
    with pytest.raises(ValueError, match="Invalid bbox"):
        builder._bbox(_args(min_lat=50.0))

    manifest = [{"sha256": "source-digest"}]
    bbox = {"min_lat": 42.0, "min_lon": -73.0, "max_lat": 43.0, "max_lon": -72.0}
    digest = builder._derived_cache_digest(manifest, bbox)
    assert isinstance(digest, str)
    assert len(digest) == 24
    assert digest != builder._source_digests(manifest)


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
