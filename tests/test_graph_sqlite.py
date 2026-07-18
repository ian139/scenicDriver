from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from src.route_planner.graph import (
    Edge,
    Node,
    RoadGraph,
    _graph_from_osmnx,
    _iter_osmnx_graph_rows,
    _write_sqlite_graph,
)


class _FakeOsmGraph:
    def nodes(self, data: bool = False):
        rows = [
            ("B", {"x": -72.0, "y": 42.1}),
            ("A", {"x": -72.0, "y": 42.0}),
        ]
        return rows if data else [node_id for node_id, _ in rows]

    def edges(self, keys: bool = False, data: bool = False):
        assert keys and data
        return [
            (
                "A",
                "B",
                0,
                {
                    "length": 12_000.0,
                    "geometry": [(-72.0, 42.0), (-71.99, 42.05), (-72.0, 42.1)],
                    "highway": "secondary",
                    "maxspeed": "45 mph",
                    "name": "Main Road",
                    "oneway": False,
                    "osmid": 123,
                },
            )
        ]

    def has_edge(self, start: str, end: str) -> bool:
        return False


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
    osm_graph = _FakeOsmGraph()
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
