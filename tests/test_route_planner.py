from __future__ import annotations

import json
from pathlib import Path

from src.route_planner.cost import ScenicCostFunction
from src.route_planner.graph import Edge, Node, RoadGraph
from src.route_planner.planner import ScenicRoutePlanner


def test_graph_bidirectional_edges() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="b", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="ab",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=6.0,
        )
    )

    forward = graph.get_edges("a")
    reverse = graph.get_edges("b")
    assert len(forward) == 1
    assert len(reverse) == 1
    assert forward[0].end_node_id == "b"
    assert reverse[0].end_node_id == "a"


def test_graph_respects_one_way_edges() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="b", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="ab_one_way",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=6.0,
            one_way=True,
        )
    )

    forward = graph.get_edges("a")
    reverse = graph.get_edges("b")
    assert len(forward) == 1
    assert len(reverse) == 0
    assert forward[0].end_node_id == "b"


def test_cost_function_weight_behavior() -> None:
    short_ugly = Edge(
        id="e1",
        start_node_id="a",
        end_node_id="b",
        distance_km=1.0,
        scenic_score=2.0,
        speed_limit_kmh=60,
    )
    long_scenic = Edge(
        id="e2",
        start_node_id="a",
        end_node_id="c",
        distance_km=4.0,
        scenic_score=9.0,
        speed_limit_kmh=60,
    )

    distance_only = ScenicCostFunction(scenic_weight=0.0)
    scenic_only = ScenicCostFunction(scenic_weight=1.0)

    assert distance_only.calculate(short_ugly) < distance_only.calculate(long_scenic)
    assert scenic_only.calculate(long_scenic) < scenic_only.calculate(short_ugly)


def test_simple_route_planning() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="B", lat=42.05, lon=-72.0))
    graph.add_node(Node(id="C", lat=42.10, lon=-72.0))

    graph.add_edge(
        Edge(
            id="AB",
            start_node_id="A",
            end_node_id="B",
            distance_km=5.0,
            scenic_score=5.0,
            speed_limit_kmh=50,
        )
    )
    graph.add_edge(
        Edge(
            id="BC",
            start_node_id="B",
            end_node_id="C",
            distance_km=5.0,
            scenic_score=8.0,
            speed_limit_kmh=50,
        )
    )

    planner = ScenicRoutePlanner(graph=graph)
    route = planner.find_scenic_route(
        start=(42.0, -72.0),
        end=(42.10, -72.0),
        scenic_weight=0.5,
    )

    assert len(route.segments) == 2
    assert route.total_distance_km == 10.0
    assert route.average_scenic_score > 6.0


def test_geojson_graph_load(tmp_path: Path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"road_type": "secondary", "scenic_score": 7.0},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                },
            }
        ],
    }
    path = tmp_path / "graph.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    graph = RoadGraph.from_geojson(path)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    first = next(iter(graph.edges.values()))
    assert first.one_way is True


def test_geojson_bidirectional_override(tmp_path: Path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "road_type": "secondary",
                    "scenic_score": 7.0,
                    "bidirectional": True,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                },
            }
        ],
    }
    path = tmp_path / "graph_bi.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    graph = RoadGraph.from_geojson(path)
    start_edges = graph.get_edges("n0")
    end_edges = graph.get_edges("n1")
    assert len(start_edges) == 1
    assert len(end_edges) == 1
