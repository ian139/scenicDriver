from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import networkx as nx
import pytest

from src.route_planner.graph import (
    Edge,
    Node,
    RoadGraph,
    _graph_from_osmnx,
    _iter_osmnx_graph_rows,
    _write_sqlite_graph,
)

def _fake_osm_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(
        [
            ("B", {"x": -72.0, "y": 42.1}),
            ("A", {"x": -72.0, "y": 42.0}),
        ]
    )
    graph.add_edge(
        "A",
        "B",
        key=0,
        length=12_000.0,
        geometry=[(-72.0, 42.0), (-71.99, 42.05), (-72.0, 42.1)],
        highway="secondary",
        maxspeed="45 mph",
        name="Main Road",
        oneway=False,
        osmid=123,
    )
    return graph


def _basic_graph() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="B", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="AB",
            start_node_id="A",
            end_node_id="B",
            distance_km=12.0,
            scenic_score=7.5,
            road_name="Main Road",
            road_type="secondary",
            speed_limit_kmh=72,
            one_way=False,
        )
    )
    return graph


def test_sqlite_round_trip_preserves_graph_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "road_graph.sqlite3"
    graph = _basic_graph()
    graph.save(path, metadata={"bbox": {"west": -72.1}, "probe": {"distance_km": 0.0}})

    loaded = RoadGraph.load(path)

    assert list(loaded.nodes) == ["A", "B"]
    assert list(loaded.edges) == ["AB"]
    assert loaded.edges["AB"].road_name == "Main Road"
    assert loaded.edges["AB"].one_way is False
    assert loaded.artifact_metadata["graph_format"] == "scenic-roadgraph-sqlite"
    assert loaded.artifact_metadata["schema_version"] == 1
    assert loaded.artifact_metadata["node_count"] == 2
    assert loaded.artifact_metadata["edge_count"] == 1


def test_sqlite_rejects_unknown_format_and_schema_before_rows(tmp_path: Path) -> None:
    path = tmp_path / "road_graph.sqlite3"
    _basic_graph().save(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'graph_format'",
            (json.dumps("wrong-format"),),
        )
        connection.commit()

    with pytest.raises(ValueError, match="format"):
        RoadGraph.load(path)

    _basic_graph().save(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
            (json.dumps(99),),
        )
        connection.commit()

    with pytest.raises(ValueError, match="schema version"):
        RoadGraph.load(path)


def test_sqlite_rejects_unknown_nodes_and_duplicate_edges(tmp_path: Path) -> None:
    path = tmp_path / "malformed.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes(id TEXT PRIMARY KEY, lat REAL NOT NULL, lon REAL NOT NULL);
            CREATE TABLE edges(
                id TEXT, start_node_id TEXT NOT NULL, end_node_id TEXT NOT NULL,
                distance_km REAL NOT NULL, scenic_score REAL NOT NULL,
                road_name TEXT, road_type TEXT NOT NULL, speed_limit_kmh REAL,
                one_way INTEGER NOT NULL
            );
            """
        )
        metadata = {
            "graph_format": "scenic-roadgraph-sqlite",
            "schema_version": 1,
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        connection.executemany(
            "INSERT INTO nodes(id, lat, lon) VALUES (?, ?, ?)",
            [("A", 42.0, -72.0)],
        )
        connection.executemany(
            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("AB", "A", "missing", 1.0, 5.0, None, "secondary", 50.0, 1),
            ],
        )
        connection.commit()

    with pytest.raises(ValueError, match="Unknown end node"):
        RoadGraph.load(path)

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM edges")
        connection.executemany(
            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("AB", "A", "A", 1.0, 5.0, None, "secondary", 50.0, 1),
                ("AB", "A", "A", 1.0, 5.0, None, "secondary", 50.0, 1),
            ],
        )
        connection.commit()

    with pytest.raises(ValueError, match="Duplicate edge id"):
        RoadGraph.load(path)


def test_failed_sqlite_write_leaves_existing_artifact_untouched(tmp_path: Path) -> None:
    path = tmp_path / "road_graph.sqlite3"
    _basic_graph().save(path)
    original = path.read_bytes()

    def failing_rows():
        yield "node", Node(id="A", lat=42.0, lon=-72.0)
        raise RuntimeError("conversion failed")

    with pytest.raises(RuntimeError, match="conversion failed"):
        _write_sqlite_graph(path, failing_rows())

    assert path.read_bytes() == original


def test_streamed_osm_rows_match_roadgraph_conversion(tmp_path: Path) -> None:
    osm_graph = _fake_osm_graph()
    expected = _graph_from_osmnx(osm_graph, {"123": 8.0})
    path = tmp_path / "streamed.sqlite3"
    _write_sqlite_graph(
        path,
        _iter_osmnx_graph_rows(osm_graph, {"123": 8.0}),
        metadata={"source": "test"},
    )
    actual = RoadGraph.load(path)

    assert list(actual.nodes) == list(expected.nodes)
    assert list(actual.edges) == list(expected.edges)
    for edge_id, edge in expected.edges.items():
        loaded = actual.edges[edge_id]
        assert loaded.start_node_id == edge.start_node_id
        assert loaded.end_node_id == edge.end_node_id
        assert loaded.distance_km == pytest.approx(edge.distance_km)
        assert loaded.scenic_score == pytest.approx(edge.scenic_score)
        assert loaded.road_type == edge.road_type
        assert loaded.one_way is edge.one_way
        assert actual.nodes[loaded.start_node_id].coords == expected.nodes[edge.start_node_id].coords


def test_graph_replaced_during_load_marks_sidecar_stale(tmp_path: Path) -> None:
    import os

    from src.route_planner import graph as graph_module

    path = tmp_path / "road_graph.sqlite3"
    graph = _basic_graph()
    graph.save(path)

    replacement = tmp_path / "replacement.sqlite3"
    replacement_graph = RoadGraph()
    replacement_graph.add_node(Node(id="C", lat=10.0, lon=10.0))
    replacement_graph.add_node(Node(id="D", lat=10.1, lon=10.0))
    replacement_graph.add_edge(
        Edge(
            id="CD",
            start_node_id="C",
            end_node_id="D",
            distance_km=12.0,
            scenic_score=5.0,
        )
    )
    replacement_graph.save(replacement)

    original_stat = graph_module._path_identity(path.resolve())

    def swap_after_read_identity(*args, **kwargs):
        os.replace(replacement, path)
        return original_stat

    import unittest.mock as mock

    with mock.patch.object(
        graph_module, "_path_identity", side_effect=swap_after_read_identity
    ):
        loaded = RoadGraph.load(path)

    assert loaded.edge_projection_index_status["state"] == "invalid"
    assert (
        loaded.edge_projection_index_status["invalid_reason"]
        == "graph_replaced_during_load"
    )
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        10.05,
        10.0,
    )
    assert [projection.edge.id for projection in projections] == ["CD"]
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def _traversals(graph: RoadGraph, node_id: str) -> list[tuple[str, str, str]]:
    return [
        (item.id, item.start_node_id, item.end_node_id)
        for item in graph.iter_edges(node_id)
    ]


def _two_node_osm(**edges: dict) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(
        [("A", {"x": -72.0, "y": 42.0}), ("B", {"x": -72.0, "y": 42.1})]
    )
    for spec in edges.values():
        graph.add_edge(spec.pop("u"), spec.pop("v"), key=spec.pop("key", 0), **spec)
    return graph


def test_osm_reverse_oneway_emits_edges_and_geometry_against_digitization() -> None:
    osm = _two_node_osm(
        ab={
            "u": "A",
            "v": "B",
            "length": 12_000.0,
            "geometry": [(-72.0, 42.0), (-71.99, 42.05), (-72.0, 42.1)],
            "oneway": "-1",
            "osmid": 9,
        }
    )

    graph = _graph_from_osmnx(osm, {})

    segments = [graph.edges[f"A-B-0-segment-{index}"] for index in (0, 1)]
    assert [(e.start_node_id, e.end_node_id, e.one_way) for e in segments] == [
        ("B", "A-B-0-coord-1", True),
        ("A-B-0-coord-1", "A", True),
    ]
    assert graph.nodes["A-B-0-coord-1"].coords == (42.05, -71.99)
    assert _traversals(graph, "A") == []


def test_geojson_reverse_oneway_runs_against_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "roads.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                        },
                        "properties": {"id": "r", "oneway": "-1"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    graph = RoadGraph.from_geojson(path)
    edge = graph.edges["r_0"]

    assert edge.one_way is True
    assert graph.nodes[edge.start_node_id].coords == (42.1, -72.0)
    assert graph.nodes[edge.end_node_id].coords == (42.0, -72.0)


def test_canonical_loader_rejects_reverse_oneway_value(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "A", "lat": 42.0, "lon": -72.0},
                    {"id": "B", "lat": 42.1, "lon": -72.0},
                ],
                "edges": [
                    {"id": "AB", "start": "A", "end": "B", "distance_km": 1.0, "one_way": "-1"}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Reverse one-way"):
        RoadGraph.load(path)


def test_osm_reciprocal_requires_same_road_identity_and_reversed_geometry() -> None:
    forward = [(-72.0, 42.0), (-71.99, 42.05), (-72.0, 42.1)]
    matched = _graph_from_osmnx(
        _two_node_osm(
            ab={"u": "A", "v": "B", "oneway": False, "osmid": 1, "geometry": forward},
            ba={"u": "B", "v": "A", "oneway": False, "osmid": 1, "geometry": forward[::-1]},
        ),
        {},
    )
    assert {edge.one_way for edge in matched.edges.values()} == {True}

    other_road = _graph_from_osmnx(
        _two_node_osm(
            ab={"u": "A", "v": "B", "oneway": False, "osmid": 1},
            ba={"u": "B", "v": "A", "oneway": False, "osmid": 2},
        ),
        {},
    )
    assert {edge.one_way for edge in other_road.edges.values()} == {False}

    other_geometry = _graph_from_osmnx(
        _two_node_osm(
            ab={"u": "A", "v": "B", "oneway": False, "osmid": 1, "geometry": forward},
            ba={"u": "B", "v": "A", "oneway": False, "osmid": 1},
        ),
        {},
    )
    assert all(
        not edge.one_way
        for edge_id, edge in other_geometry.edges.items()
        if edge_id.startswith("A-B-0-")
    )


def test_osm_reciprocal_without_osmid_or_geometry_requires_same_key() -> None:
    parallel = _graph_from_osmnx(
        _two_node_osm(
            ab={"u": "A", "v": "B", "key": 0, "oneway": False},
            ba={"u": "B", "v": "A", "key": 1, "oneway": False},
        ),
        {},
    )
    assert {edge.one_way for edge in parallel.edges.values()} == {False}

    paired = _graph_from_osmnx(
        _two_node_osm(
            ab={"u": "A", "v": "B", "key": 0, "oneway": False},
            ba={"u": "B", "v": "A", "key": 0, "oneway": False},
        ),
        {},
    )
    assert {edge.one_way for edge in paired.edges.values()} == {True}


def test_edge_mapping_replace_and_delete_keep_adjacency_in_sync() -> None:
    graph = _basic_graph()
    graph.add_node(Node(id="C", lat=42.2, lon=-72.0))

    graph.edges["AB"] = Edge(
        id="AB",
        start_node_id="A",
        end_node_id="C",
        distance_km=1.0,
        scenic_score=5.0,
        one_way=True,
    )
    assert _traversals(graph, "A") == [("AB", "A", "C")]
    assert _traversals(graph, "B") == []
    assert _traversals(graph, "C") == []

    del graph.edges["AB"]
    assert all(_traversals(graph, node_id) == [] for node_id in graph.nodes)

    with pytest.raises(ValueError, match="does not match"):
        graph.edges["X"] = Edge(id="Y", start_node_id="A", end_node_id="B", distance_km=1.0, scenic_score=5.0)
    with pytest.raises(ValueError, match="Unknown end node"):
        graph.edges["Z"] = Edge(id="Z", start_node_id="A", end_node_id="missing", distance_km=1.0, scenic_score=5.0)
    assert "Z" not in graph.edges


def test_edge_topology_fields_are_immutable_but_scores_are_not() -> None:
    graph = _basic_graph()
    edge = graph.edges["AB"]

    for field, value in (("start_node_id", "B"), ("end_node_id", "A"), ("one_way", True), ("id", "BA")):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(edge, field, value)
    with pytest.raises(AttributeError, match="immutable"):
        graph.nodes["A"].id = "Z"

    edge.scenic_score = 9.0
    edge.speed_limit_kmh = 33.5
    assert graph.edges["AB"].scenic_score == 9.0
    assert _traversals(graph, "B") == [("AB::rev", "B", "A")]


def test_node_delete_rejects_referenced_nodes_and_replacement_keeps_topology() -> None:
    graph = RoadGraph()
    for node_id, lat in (("A", 42.0), ("B", 42.1), ("lonely", 42.2)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(id="AB", start_node_id="A", end_node_id="B", distance_km=1.0, scenic_score=5.0, one_way=True)
    )

    # B is only the target of a one-way edge, so it has no adjacency refs.
    for node_id in ("A", "B"):
        with pytest.raises(ValueError, match="referenced by edges"):
            del graph.nodes[node_id]
    with pytest.raises(ValueError, match="referenced by edges"):
        graph.nodes.clear()
    with pytest.raises(ValueError, match="does not match"):
        graph.nodes["A"] = Node(id="other", lat=0.0, lon=0.0)
    assert list(graph.nodes) == ["A", "B", "lonely"]

    graph.nodes["B"] = Node(id="B", lat=43.0, lon=-71.0)
    assert graph.nodes["B"].coords == (43.0, -71.0)
    assert _traversals(graph, "A") == [("AB", "A", "B")]

    del graph.nodes["lonely"]
    assert "lonely" not in graph.adjacency


def test_all_loaders_reject_duplicate_normalized_node_ids(tmp_path: Path) -> None:
    json_path = tmp_path / "graph.json"
    json_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 1, "lat": 42.0, "lon": -72.0},
                    {"id": "1", "lat": 43.0, "lon": -71.0},
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate node id: 1"):
        RoadGraph.load(json_path)

    sqlite_path = tmp_path / "graph.sqlite3"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes(id, lat REAL NOT NULL, lon REAL NOT NULL);
            CREATE TABLE edges(
                id TEXT, start_node_id TEXT, end_node_id TEXT, distance_km REAL,
                scenic_score REAL, road_name TEXT, road_type TEXT,
                speed_limit_kmh REAL, one_way INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [("graph_format", json.dumps("scenic-roadgraph-sqlite")), ("schema_version", "1")],
        )
        connection.executemany(
            "INSERT INTO nodes(id, lat, lon) VALUES (?, ?, ?)",
            [(1, 42.0, -72.0), ("1", 43.0, -71.0)],
        )
        connection.commit()
    with pytest.raises(ValueError, match="Duplicate node id: 1"):
        RoadGraph.load(sqlite_path)

    osm = nx.MultiDiGraph()
    osm.add_nodes_from([(1, {"x": -72.0, "y": 42.0}), ("1", {"x": -71.0, "y": 43.0})])
    with pytest.raises(ValueError, match="Duplicate node id: 1"):
        _graph_from_osmnx(osm, {})
    with pytest.raises(ValueError, match="Duplicate graph row id"):
        _write_sqlite_graph(tmp_path / "osm.sqlite3", _iter_osmnx_graph_rows(osm, {}))
    assert not (tmp_path / "osm.sqlite3").exists()


def test_add_node_rejects_duplicate_id_without_overwriting_coordinates() -> None:
    graph = _basic_graph()
    with pytest.raises(ValueError, match="Duplicate node id: A"):
        graph.add_node(Node(id="A", lat=0.0, lon=0.0))
    assert graph.nodes["A"].coords == (42.0, -72.0)


@pytest.mark.parametrize("suffix", [".json", ".sqlite3"])
def test_persisted_speed_round_trips_exactly(tmp_path: Path, suffix: str) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="B", lat=42.1, lon=-72.0))
    for edge_id, speed in (("slow", 5), ("fractional", 47.5), ("fast", 150)):
        graph.add_edge(
            Edge(
                id=edge_id,
                start_node_id="A",
                end_node_id="B",
                distance_km=1.0,
                scenic_score=5.0,
                speed_limit_kmh=speed,
                one_way=True,
            )
        )
    path = tmp_path / f"graph{suffix}"
    graph.save(path)

    loaded = RoadGraph.load(path)

    assert {edge_id: e.speed_limit_kmh for edge_id, e in loaded.edges.items()} == {
        "slow": 5.0,
        "fractional": 47.5,
        "fast": 150.0,
    }


def test_osm_maxspeed_parsing_stays_separate_from_persisted_speed() -> None:
    graph = _graph_from_osmnx(
        _two_node_osm(ab={"u": "A", "v": "B", "maxspeed": "5", "oneway": True, "osmid": 1}),
        {},
    )
    assert graph.edges["A-B-0-segment-0"].speed_limit_kmh == 10


@pytest.mark.parametrize("bad_speed", [0.0, -5.0, "fast"])
def test_sqlite_rejects_invalid_persisted_speed(tmp_path: Path, bad_speed: object) -> None:
    path = tmp_path / "graph.sqlite3"
    _basic_graph().save(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE edges SET speed_limit_kmh = ?", (bad_speed,))
        connection.commit()

    with pytest.raises(ValueError, match="Invalid speed limit"):
        RoadGraph.load(path)
