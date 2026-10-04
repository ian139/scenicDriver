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

    builder._truncate_graph_to_region(graph, builder._footprint_geometry(footprint))

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

    builder._truncate_graph_to_region(graph, builder._footprint_geometry(footprint))

    assert "inside" in graph.nodes
    assert "outside" not in graph.nodes


class _ChainSimplifyingOx:
    """Fake OSMnx: loads a fixed raw graph and merges degree-2 chains.

    Simplification mirrors OSMnx: interstitial nodes are removed and the merged
    edge carries a shapely LineString through them.
    """

    settings = SimpleNamespace(useful_tags_way=["highway"])

    def __init__(self, graph) -> None:
        self.graph = graph

    def graph_from_xml(self, path, **kwargs):
        return self.graph

    @staticmethod
    def simplify_graph(G):
        from shapely.geometry import LineString

        interstitial = [
            node
            for node in G.nodes
            if G.in_degree(node) == 1 and G.out_degree(node) == 1
        ]
        for node in interstitial:
            (u, _node, in_data), = G.in_edges(node, data=True)
            (_node, v, out_data), = G.out_edges(node, data=True)
            points = [(G.nodes[u]["x"], G.nodes[u]["y"])]
            for data in (in_data, out_data):
                geometry = data.get("geometry")
                if geometry is not None:
                    points.extend(list(geometry.coords)[1:-1])
                if data is in_data:
                    points.append((G.nodes[node]["x"], G.nodes[node]["y"]))
            points.append((G.nodes[v]["x"], G.nodes[v]["y"]))
            G.remove_node(node)
            G.add_edge(u, v, highway="primary", geometry=LineString(points))
        return G


def _raw_chain(nodes):
    import networkx as nx

    graph = nx.MultiDiGraph()
    for node_id, lon, lat in nodes:
        graph.add_node(node_id, x=lon, y=lat)
    for (start, *_), (end, *_) in zip(nodes, nodes[1:]):
        graph.add_edge(start, end, highway="primary")
    return graph


def test_bbox_mode_keeps_simplified_edge_crossing_bbox_with_outside_endpoints() -> None:
    # A west->east road whose intersections lie outside the exact bbox but
    # inside the 500 m buffer; only its interstitial nodes are inside the bbox.
    exact_bbox = (-72.0, 42.0, -71.99, 42.01)
    raw = _raw_chain(
        [
            ("west", -72.002, 42.005),
            ("inside_a", -71.998, 42.005),
            ("inside_b", -71.992, 42.005),
            ("east", -71.988, 42.005),
        ]
    )
    raw.add_edge("east", "spur", highway="primary")
    raw.add_edge("east", "spur2", highway="primary")
    raw.add_node("spur", x=-71.987, y=42.006)
    raw.add_node("spur2", x=-71.987, y=42.004)
    raw.add_edge("before", "west", highway="primary")
    raw.add_edge("before2", "west", highway="primary")
    raw.add_node("before", x=-72.003, y=42.006)
    raw.add_node("before2", x=-72.003, y=42.004)

    graph = builder._load_local_osm_graph(
        _ChainSimplifyingOx(raw), Path("filtered.osm.bz2"), exact_bbox
    )

    assert graph.has_edge("west", "east")
    assert "inside_a" not in graph.nodes  # proves the edge came from simplification
    assert {"west", "east"} <= set(graph.nodes)
    assert not graph.has_edge("east", "spur")


def test_tile_mode_keeps_crossing_edge_whose_endpoint_is_beyond_footprint_envelope(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": 14, "x": 2000, "y": 3000}]}),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 0)
    west = footprint["bbox"]["min_lon"]
    east = footprint["bbox"]["max_lon"]
    lat = (footprint["bbox"]["min_lat"] + footprint["bbox"]["max_lat"]) / 2.0
    raw = _raw_chain(
        [
            ("far_west", west - 0.5, lat),
            ("inside", (west + east) / 2.0, lat),
            ("junction", (west + east) / 2.0 + 0.001, lat),
        ]
    )
    raw.add_edge("junction", "north", highway="primary")
    raw.add_edge("junction", "south", highway="primary")
    raw.add_node("north", x=(west + east) / 2.0 + 0.001, y=lat + 0.001)
    raw.add_node("south", x=(west + east) / 2.0 + 0.001, y=lat - 0.001)
    raw.add_edge("far_east", "far_east_2", highway="primary")
    raw.add_node("far_east", x=east + 1.0, y=lat)
    raw.add_node("far_east_2", x=east + 1.1, y=lat)

    graph = builder._load_local_osm_graph(
        _ChainSimplifyingOx(raw), Path("filtered.osm.bz2"), None, footprint=footprint
    )

    assert graph.has_edge("far_west", "junction")
    assert "far_west" in graph.nodes
    assert "far_east" not in graph.nodes
    assert not graph.has_edge("far_east", "far_east_2")


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


@pytest.mark.parametrize(
    "override",
    [
        {"min_lat": float("nan")},
        {"max_lon": float("inf")},
        {"min_lon": float("-inf")},
        {"max_lat": 90.5},
        {"min_lat": -91.0},
        {"min_lon": -181.0},
        {"max_lon": 180.5},
        {"min_lon": -72.0},
    ],
)
def test_bbox_rejects_nonfinite_out_of_world_and_unordered_bounds(override) -> None:
    with pytest.raises(ValueError, match="Invalid bbox"):
        builder._bbox(_args(**override))
    with pytest.raises(ValueError, match="Invalid bbox"):
        builder._validate_mode_flags(None, _args(**override))


def test_parser_rejects_nonfinite_bbox(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--min-lat", "nan",
            "--min-lon", "-73",
            "--max-lat", "43",
            "--max-lon", "-72",
            "--source-pbf", "extract.pbf",
        ],
    )
    with pytest.raises(SystemExit):
        builder.parse_args()
    assert "Invalid bbox" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("graph_format", "output", "valid"),
    [
        ("json", "out/graph.json", True),
        ("sqlite3", "out/graph.sqlite3", True),
        ("compact", "out/graph.sqlite3", True),
        ("json", "out/graph.sqlite3", False),
        ("json", "out/graph.compact.json", False),
        ("json", "out/graph.geojson", False),
        ("sqlite3", "out/graph.json", False),
        ("compact", "out/graph.compact.json", False),
        ("compact", "out/graph.db", False),
        ("sqlite3", "out/.sqlite3", False),
    ],
)
def test_explicit_output_suffix_must_match_graph_format(
    graph_format, output, valid
) -> None:
    args = _args(graph_format=graph_format, output=Path(output))
    if valid:
        builder._validate_mode_flags(None, args)
        assert builder._resolve_output_paths(args)[0] == Path(output)
        return
    with pytest.raises(ValueError, match="does not match --graph-format"):
        builder._validate_mode_flags(None, args)
    with pytest.raises(ValueError, match="does not match --graph-format"):
        builder._resolve_output_paths(args)


def test_tile_buffer_is_clamped_to_the_world_grid(tmp_path: Path) -> None:
    zoom = 3
    limit = 2**zoom - 1
    coords = [(0, 0), (limit, limit)]
    rectangles = builder._compact_tile_rectangles(zoom, coords, 2)
    covered = {
        (x, y)
        for x_start, x_end, y_start, y_end in rectangles
        for x in range(x_start, x_end + 1)
        for y in range(y_start, y_end + 1)
    }
    expected = {
        (x, y)
        for tile_x, tile_y in coords
        for x in range(max(0, tile_x - 2), min(limit, tile_x + 2) + 1)
        for y in range(max(0, tile_y - 2), min(limit, tile_y + 2) + 1)
    }
    assert covered == expected

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tiles": [{"z": zoom, "x": x, "y": y} for x, y in coords]}),
        encoding="utf-8",
    )
    footprint = builder._load_tile_footprint(report, 2)
    bbox = footprint["bbox"]
    assert bbox["min_lon"] == -180.0
    assert bbox["max_lon"] == 180.0
    assert -85.06 < bbox["min_lat"] < bbox["max_lat"] < 85.06



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

    node_count, edge_count, probes, compact = builder._publish_sqlite_graph(
        output,
        tmp_path,
        object(),
        {},
        {},
        [],
    )

    assert (node_count, edge_count, probes, compact) == (2, 1, {}, None)
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

    assert sorted(path.name for path in tmp_path.iterdir()) == ["road_graph.sqlite3"]
    loaded = RoadGraph.load(output)
    assert loaded.edge_projection_index_status["state"] == "missing"
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        42.05,
        -72.0,
    )
    assert [projection.edge.id for projection in projections] == ["ab"]
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def _old_generation(run_dir: Path) -> dict[str, bytes]:
    graph = RoadGraph()
    graph.add_node(Node("old_a", 40.0, -75.0))
    graph.add_node(Node("old_b", 40.1, -75.0))
    graph.add_edge(Edge("old", "old_a", "old_b", distance_km=11.0, scenic_score=1.0))
    output = run_dir / "road_graph.sqlite3"
    graph.save(output)
    builder._publish_compact_graph(output, run_dir)
    (run_dir / "run.json").write_text('{"generation": "old"}', encoding="utf-8")
    return {
        name: (run_dir / name).read_bytes()
        for name in (
            "road_graph.sqlite3",
            "road_graph.sqlite3.edge_projection_index",
            "road_graph.compact.json",
            "run.json",
        )
    }


def _new_rows():
    return (
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


def test_compact_publication_failure_keeps_graph_members_and_leaves_run_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _old_generation(tmp_path)
    monkeypatch.setattr(
        builder, "_iter_osmnx_graph_rows", lambda _graph, _scores: iter(_new_rows())
    )
    monkeypatch.setattr(
        builder,
        "write_compact_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compact failed")),
    )

    with pytest.raises(RuntimeError, match="compact failed"):
        builder._publish_sqlite_graph(
            tmp_path / "road_graph.sqlite3",
            tmp_path,
            object(),
            {},
            {},
            [],
            compact=True,
            withdraw=[tmp_path / "run.json"],
        )

    graph_members = {name: data for name, data in old.items() if name != "run.json"}
    assert {name: (tmp_path / name).read_bytes() for name in graph_members} == graph_members
    # The readiness gate was withdrawn before the first content swap: the run
    # reports not-ready instead of vouching for a set it may no longer describe.
    assert not (tmp_path / "run.json").exists()
    assert not any(".publish." in path.name for path in tmp_path.iterdir())


def test_compact_publication_never_exposes_mixed_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.route_planner.graph import CompactRoadGraph

    _old_generation(tmp_path)
    output = tmp_path / "road_graph.sqlite3"
    output_sidecar = EdgeProjectionIndex.sidecar_path(output)
    run_json = tmp_path / "run.json"
    monkeypatch.setattr(
        builder, "_iter_osmnx_graph_rows", lambda _graph, _scores: iter(_new_rows())
    )
    manifest_path = tmp_path / "road_graph.compact.json"
    visible = {manifest_path, output, output_sidecar}
    replace = builder.os.replace
    observed: list[tuple[str, bool]] = []
    at_graph_swap: list[dict[str, bool]] = []

    def observing_replace(source: object, destination: object) -> None:
        if Path(destination) in visible:
            observed.append((Path(destination).name, run_json.exists()))
        if Path(destination) == output:
            manifest = json.loads(manifest_path.read_text())
            at_graph_swap.append(
                {
                    "old_sidecar": output_sidecar.exists(),
                    "manifest_is_new": manifest["source"]["sha256"]
                    == hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                }
            )
        replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", observing_replace)
    node_count, edge_count, _probes, compact = builder._publish_sqlite_graph(
        output,
        tmp_path,
        object(),
        {},
        {},
        [],
        compact=True,
        withdraw=[run_json],
    )

    # Every consumer-visible swap happens while the readiness gate is absent.
    assert observed == [
        ("road_graph.compact.json", False),
        ("road_graph.sqlite3", False),
        ("road_graph.sqlite3.edge_projection_index", False),
    ]
    assert at_graph_swap == [{"old_sidecar": False, "manifest_is_new": True}]
    builder._verify_published_set(
        output,
        tmp_path,
        "compact",
        hashlib.sha256(output.read_bytes()).hexdigest(),
        compact,
    )
    assert (node_count, edge_count) == (2, 1)
    assert compact is not None
    assert compact["source_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    sqlite_graph = RoadGraph.load(output)
    assert sqlite_graph.edge_projection_index_status["state"] == "loaded"
    assert set(sqlite_graph.edges) == {"ab"}
    compact_graph = CompactRoadGraph.load(Path(compact["manifest_path"]))
    assert set(compact_graph.edges) == {"ab"}
    assert not any(".publish." in path.name for path in tmp_path.iterdir())


def test_published_set_verification_rejects_mixed_or_stale_members(
    tmp_path: Path,
) -> None:
    _old_generation(tmp_path)
    output = tmp_path / "road_graph.sqlite3"
    manifest = tmp_path / "road_graph.compact.json"
    record = {
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    graph_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    builder._verify_published_set(output, tmp_path, "compact", graph_sha256, record)

    with pytest.raises(ValueError, match="does not reference the published SQLite"):
        builder._verify_published_set(output, tmp_path, "compact", "0" * 64, record)
    (tmp_path / "road_graph.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale graph entry points"):
        builder._verify_published_set(output, tmp_path, "compact", graph_sha256, record)
    (tmp_path / "road_graph.json").unlink()
    EdgeProjectionIndex.sidecar_path(output).unlink()
    with pytest.raises(Exception, match="(?i)sidecar"):
        builder._verify_published_set(output, tmp_path, "sqlite3", graph_sha256, None)



def _generation_dirs(run_dir: Path) -> set[str]:
    return {
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and ".generation." in path.name
    }


def test_switch_to_sqlite3_retires_stale_json_and_compact_entry_points_at_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _old_generation(tmp_path)
    (tmp_path / "road_graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    generations = _generation_dirs(tmp_path)
    assert generations
    monkeypatch.setattr(
        builder, "_iter_osmnx_graph_rows", lambda _graph, _scores: iter(_new_rows())
    )

    # A failure before commit keeps every previous entry point.
    with pytest.raises(ValueError, match="must be finite"):
        builder._publish_sqlite_graph(
            tmp_path / "road_graph.sqlite3",
            tmp_path,
            object(),
            {},
            {},
            [{"name": "bad", "lat": float("nan"), "lon": -72.0}],
            withdraw=[tmp_path / "run.json"],
        )
    assert {"road_graph.json", "road_graph.compact.json", "run.json"} <= {
        path.name for path in tmp_path.iterdir()
    }

    builder._publish_sqlite_graph(
        tmp_path / "road_graph.sqlite3",
        tmp_path,
        object(),
        {},
        {},
        [],
        withdraw=[tmp_path / "run.json"],
    )

    names = {path.name for path in tmp_path.iterdir()}
    assert "road_graph.json" not in names
    assert "road_graph.compact.json" not in names
    assert _generation_dirs(tmp_path) == generations
    assert set(RoadGraph.load(tmp_path / "road_graph.sqlite3").edges) == {"ab"}


def test_switch_to_json_retires_stale_sqlite_and_compact_entry_points(
    tmp_path: Path,
) -> None:
    import networkx as nx

    _old_generation(tmp_path)
    generations = _generation_dirs(tmp_path)
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=-72.0, y=42.0)
    graph.add_node(2, x=-72.0, y=42.1)
    graph.add_edge(1, 2, 0, highway="primary", length=11_000.0, osmid=5)

    builder._publish_json_graph(
        tmp_path / "road_graph.json",
        tmp_path,
        graph,
        [],
        withdraw=[tmp_path / "run.json"],
    )

    names = {path.name for path in tmp_path.iterdir()}
    assert "road_graph.json" in names
    assert not names & {
        "road_graph.sqlite3",
        "road_graph.sqlite3.edge_projection_index",
        "road_graph.compact.json",
        "run.json",
    }
    assert _generation_dirs(tmp_path) == generations
    assert len(RoadGraph.load(tmp_path / "road_graph.json").edges) == 1
