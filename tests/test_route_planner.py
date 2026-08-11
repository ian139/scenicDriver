from __future__ import annotations

import json
import math
import random
from pathlib import Path

import networkx as nx
import pytest

from src.route_planner.cost import CostWeights, ScenicCostFunction
from src.route_planner.graph import Edge, Node, RoadGraph
from src.route_planner.planner import ScenicRoutePlanner
from src.data_pipeline.web_mercator import lat_lon_to_tile
from src.route_planner.service import apply_tile_scores_to_graph


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


def test_scenic_byway_discount_stays_positive_and_duration_sensitive() -> None:
    short_byway = Edge(
        id="short-byway",
        start_node_id="a",
        end_node_id="b",
        distance_km=1.0,
        scenic_score=8.0,
        road_type="scenic_byway",
        speed_limit_kmh=60,
    )
    long_byway = Edge(
        id="long-byway",
        start_node_id="a",
        end_node_id="c",
        distance_km=4.0,
        scenic_score=8.0,
        road_type="scenic_byway",
        speed_limit_kmh=60,
    )

    for scenic_weight in (0.0, 0.5, 1.0):
        for bonus in (0.0, 1.0, 100.0):
            cost = ScenicCostFunction(
                scenic_weight=scenic_weight,
                weights=CostWeights(scenic_byway_bonus=bonus),
            )
            short_cost = cost.calculate(short_byway)
            long_cost = cost.calculate(long_byway)
            assert math.isfinite(short_cost) and short_cost >= 1e-6
            assert math.isfinite(long_cost) and long_cost >= 1e-6
            assert long_cost > short_cost

    uncapped = ScenicCostFunction(
        scenic_weight=0.5,
        weights=CostWeights(scenic_byway_bonus=100.0),
    )
    capped = ScenicCostFunction(
        scenic_weight=0.5,
        weights=CostWeights(scenic_byway_bonus=0.5),
    )
    assert uncapped.calculate(long_byway) == pytest.approx(capped.calculate(long_byway))


def test_perfect_scenic_byway_still_scales_with_duration() -> None:
    short_byway = Edge(
        id="short-perfect-byway",
        start_node_id="a",
        end_node_id="b",
        distance_km=1.0,
        scenic_score=10.0,
        road_type="scenic_byway",
        speed_limit_kmh=60,
    )
    long_byway = Edge(
        id="long-perfect-byway",
        start_node_id="a",
        end_node_id="c",
        distance_km=4.0,
        scenic_score=10.0,
        road_type="scenic_byway",
        speed_limit_kmh=60,
    )
    cost = ScenicCostFunction(
        scenic_weight=1.0,
        weights=CostWeights(scenic_byway_bonus=100.0),
    )

    short_cost = cost.calculate(short_byway)
    long_cost = cost.calculate(long_byway)
    assert short_cost >= 1e-6
    assert long_cost >= 1e-6
    assert math.isfinite(short_cost) and math.isfinite(long_cost)
    assert long_cost > short_cost


def test_intermediate_scenic_weights_cross_over_between_edge_choices() -> None:
    fast_ugly = Edge(
        id="fast-ugly",
        start_node_id="a",
        end_node_id="b",
        distance_km=10.0,
        scenic_score=2.0,
        speed_limit_kmh=60,
    )
    slow_scenic = Edge(
        id="slow-scenic",
        start_node_id="a",
        end_node_id="c",
        distance_km=20.0,
        scenic_score=10.0,
        speed_limit_kmh=60,
    )

    lower = ScenicCostFunction(scenic_weight=0.4)
    higher = ScenicCostFunction(scenic_weight=0.7)
    assert lower.calculate(fast_ugly) < lower.calculate(slow_scenic)
    assert higher.calculate(slow_scenic) < higher.calculate(fast_ugly)


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


def test_graph_load_cancellation_before_file_decode_preserves_exception(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    error = RuntimeError("cancelled")

    def check_cancelled() -> None:
        raise error

    with pytest.raises(RuntimeError) as raised:
        RoadGraph.load(path, check_cancelled=check_cancelled)
    assert raised.value is error

    with pytest.raises(RuntimeError) as raised:
        RoadGraph.from_geojson(path, check_cancelled=check_cancelled)
    assert raised.value is error


def test_geojson_load_mid_build_cancellation_returns_no_partial_graph(
    tmp_path: Path,
) -> None:
    coordinates = [[float(index), float(index)] for index in range(2049)]
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        ],
    }
    path = tmp_path / "large.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    error = RuntimeError("cancelled")
    calls = 0

    def check_cancelled() -> None:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise error

    with pytest.raises(RuntimeError) as raised:
        RoadGraph.from_geojson(path, check_cancelled=check_cancelled)

    assert raised.value is error
    assert calls == 7


def test_sqlite_load_mid_build_cancellation_returns_no_partial_graph(
    tmp_path: Path,
) -> None:
    source = RoadGraph()
    for index in range(2048):
        source.add_node(Node(id=f"node-{index}", lat=float(index), lon=float(index)))
    source.add_edge(
        Edge(
            id="edge",
            start_node_id="node-0",
            end_node_id="node-1",
            distance_km=1.0,
            scenic_score=5.0,
        )
    )
    path = tmp_path / "large.sqlite3"
    source.save(path)
    error = RuntimeError("cancelled")
    calls = 0

    def check_cancelled() -> None:
        nonlocal calls
        calls += 1
        if calls == 18:
            raise error

    with pytest.raises(RuntimeError) as raised:
        RoadGraph.load(path, check_cancelled=check_cancelled)

    assert raised.value is error
    assert calls == 18


def test_constrained_search_keeps_feasible_pareto_label() -> None:
    """A cheap long label can reach X but becomes infeasible only on its final edge."""
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("L", 42.01), ("X", 42.02), ("G", 42.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))

    # Both labels reach shared node X within the four-minute cap.  The cheap
    # label has lower cost but longer duration, so neither dominates the other.
    # Its final edge would exceed the cap; the costlier short label remains
    # feasible through that same final edge.
    long = Edge(
        id="long",
        start_node_id="S",
        end_node_id="X",
        distance_km=5.0,
        scenic_score=10.0,
        speed_limit_kmh=100,
        one_way=True,
    )
    short_1 = Edge(
        id="short-1",
        start_node_id="S",
        end_node_id="L",
        distance_km=2.0,
        scenic_score=1.0,
        speed_limit_kmh=120,
        one_way=True,
    )
    short_2 = Edge(
        id="short-2",
        start_node_id="L",
        end_node_id="X",
        distance_km=2.0,
        scenic_score=1.0,
        speed_limit_kmh=120,
        one_way=True,
    )
    finish = Edge(
        id="finish",
        start_node_id="X",
        end_node_id="G",
        distance_km=2.0,
        scenic_score=5.0,
        speed_limit_kmh=60,
        one_way=True,
    )
    for edge in (long, short_1, short_2, finish):
        graph.add_edge(edge)

    cap_minutes = 4.0
    cost = ScenicCostFunction(scenic_weight=1.0)
    short_duration = short_1.travel_time_minutes + short_2.travel_time_minutes
    assert long.travel_time_minutes <= cap_minutes
    assert short_duration <= cap_minutes
    assert long.travel_time_minutes > short_duration
    assert cost.calculate(long) < cost.calculate(short_1) + cost.calculate(short_2)
    assert long.travel_time_minutes + finish.travel_time_minutes > cap_minutes
    assert short_duration + finish.travel_time_minutes <= cap_minutes

    planner = ScenicRoutePlanner(graph=graph)
    edges = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=cap_minutes,
    )

    assert edges is not None
    assert [edge.id for edge in edges] == ["short-1", "short-2", "finish"]
    assert sum(edge.travel_time_minutes for edge in edges) <= cap_minutes


def test_scenic_weight_zero_matches_fastest_and_nonzero_selects_scenic() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("M", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(
            id="fast-ugly",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="scenic-1",
            start_node_id="S",
            end_node_id="M",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="scenic-2",
            start_node_id="M",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )

    planner = ScenicRoutePlanner(graph=graph)
    fastest = planner.find_fastest_route((42.0, -72.0), (42.02, -72.0))
    zero_weight = planner.find_scenic_route(
        (42.0, -72.0), (42.02, -72.0), scenic_weight=0.0
    )
    scenic = planner.find_scenic_route(
        (42.0, -72.0), (42.02, -72.0), scenic_weight=0.8, max_detour_factor=2.4
    )

    assert zero_weight.waypoints == fastest.waypoints
    assert zero_weight.total_distance_km == fastest.total_distance_km == 10.0
    assert scenic.total_distance_km == 12.0
    assert scenic.average_scenic_score == pytest.approx(10.0)
    assert scenic.estimated_duration_minutes <= fastest.estimated_duration_minutes * 2.4


def test_scenic_duration_cap_factor_one_cannot_exceed_fastest() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="A", lat=42.01, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.02, lon=-72.0))
    graph.add_edge(
        Edge(
            id="fast",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="slow-scenic-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="slow-scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )

    planner = ScenicRoutePlanner(graph=graph)
    fastest = planner.find_fastest_route((42.0, -72.0), (42.02, -72.0))
    route = planner.find_scenic_route(
        (42.0, -72.0),
        (42.02, -72.0),
        scenic_weight=1.0,
        max_detour_factor=1.0,
    )

    assert route.estimated_duration_minutes <= (
        fastest.estimated_duration_minutes + 1e-9
    )


def test_longer_distance_equal_duration_scenic_path_is_eligible() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="A", lat=42.01, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.02, lon=-72.0))
    graph.add_edge(
        Edge(
            id="fast-ugly",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="long-scenic-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=10.0,
            scenic_score=10.0,
            speed_limit_kmh=120,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="long-scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=10.0,
            speed_limit_kmh=120,
            one_way=True,
        )
    )

    route = ScenicRoutePlanner(graph=graph).find_scenic_route(
        (42.0, -72.0),
        (42.02, -72.0),
        scenic_weight=1.0,
        max_detour_factor=1.0,
    )

    assert [segment.road_name for segment in route.segments] == [None, None]
    assert route.total_distance_km == pytest.approx(20.0)
    assert route.estimated_duration_minutes == pytest.approx(10.0)


def test_over_duration_scenic_path_is_rejected_by_duration_cap() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="A", lat=42.01, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.02, lon=-72.0))
    graph.add_edge(
        Edge(
            id="fast-ugly",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="over-duration-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="over-duration-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )

    route = ScenicRoutePlanner(graph=graph).find_scenic_route(
        (42.0, -72.0),
        (42.02, -72.0),
        scenic_weight=1.0,
        max_detour_factor=1.0,
    )

    assert [segment.scenic_score for segment in route.segments] == [0.0]
    assert route.estimated_duration_minutes == pytest.approx(10.0)


def test_route_average_scenic_score_uses_distance_weighting() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="A", lat=42.01, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.02, lon=-72.0))
    first = Edge(
        id="short-scenic",
        start_node_id="S",
        end_node_id="A",
        distance_km=10.0,
        scenic_score=10.0,
        speed_limit_kmh=60,
        one_way=True,
    )
    second = Edge(
        id="long-ugly",
        start_node_id="A",
        end_node_id="G",
        distance_km=10.0,
        scenic_score=0.0,
        speed_limit_kmh=30,
        one_way=True,
    )
    graph.add_edge(first)
    graph.add_edge(second)

    route = ScenicRoutePlanner(graph=graph)._path_to_route([first, second])

    expected = (
        first.scenic_score * first.distance_km
        + second.scenic_score * second.distance_km
    ) / (first.distance_km + second.distance_km)
    assert route.average_scenic_score == pytest.approx(expected)


def test_scenic_cost_is_invariant_under_equivalent_edge_split() -> None:
    whole = Edge(
        id="whole",
        start_node_id="A",
        end_node_id="B",
        distance_km=10.0,
        scenic_score=7.0,
        road_type="highway",
        speed_limit_kmh=50,
    )
    first = Edge(
        id="first",
        start_node_id="A",
        end_node_id="M",
        distance_km=4.0,
        scenic_score=7.0,
        road_type="highway",
        speed_limit_kmh=50,
    )
    second = Edge(
        id="second",
        start_node_id="M",
        end_node_id="B",
        distance_km=6.0,
        scenic_score=7.0,
        road_type="highway",
        speed_limit_kmh=50,
    )
    cost = ScenicCostFunction(scenic_weight=0.7, avoid_highways=True)

    assert cost.calculate(whole) == pytest.approx(
        cost.calculate(first) + cost.calculate(second)
    )


def test_osm_linestring_intermediate_coordinates_survive_route_feature() -> None:
    from src.route_planner.graph import _graph_from_osmnx
    from src.route_planner.service import route_to_feature

    class _Geometry:
        coords = [(-72.0, 42.0), (-71.99, 42.005), (-71.98, 42.01)]

    osm_graph = nx.MultiDiGraph()
    osm_graph.add_nodes_from(
        [
            ("u", {"x": -72.0, "y": 42.0}),
            ("v", {"x": -71.98, "y": 42.01}),
        ]
    )
    osm_graph.add_edge(
        "u",
        "v",
        key=0,
        length=0.0,
        highway="secondary",
        oneway=True,
        geometry=_Geometry(),
    )

    graph = _graph_from_osmnx(osm_graph, {})
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    planner = ScenicRoutePlanner(graph=graph)
    start = graph.get_node("u")
    goal = graph.get_node("v")
    edges = planner._a_star(
        start,
        goal,
        cost_function=ScenicCostFunction(scenic_weight=0.0),
        max_path_minutes=None,
    )
    assert edges is not None
    feature = route_to_feature(planner._path_to_route(edges), "scenic")

    assert feature["geometry"]["coordinates"] == [
        [-72.0, 42.0],
        [-71.99, 42.005],
        [-71.98, 42.01],
    ]


def test_zero_edge_route_feature_uses_snapped_road_geometry() -> None:

    graph = RoadGraph()
    graph.add_node(Node(id="only", lat=42.0, lon=-72.0))
    planner = ScenicRoutePlanner(graph=graph)
    with pytest.raises(ValueError, match="No route found"):
        planner.find_fastest_route(
            (42.001, -72.001),
            (42.002, -72.002),
        )


def test_zero_edge_equal_endpoint_route_feature_has_two_positions() -> None:
    from src.route_planner.service import route_to_feature

    graph = RoadGraph()
    graph.add_node(Node(id="only", lat=42.0, lon=-72.0))
    route = ScenicRoutePlanner(graph=graph).find_fastest_route(
        (42.0, -72.0),
        (42.0, -72.0),
    )
    feature = route_to_feature(
        route,
        "baseline",
        requested_start=(42.0, -72.0),
        requested_end=(42.0, -72.0),
    )

    assert feature["geometry"]["coordinates"] == [
        [-72.0, 42.0],
        [-72.0, 42.0],
    ]


def test_osm_length_is_authoritative_for_curves_and_geometryless_edges() -> None:
    from src.route_planner.graph import _graph_from_osmnx, _haversine_km

    class _Geometry:
        coords = [
            (-72.0, 42.0),
            (-71.99, 42.02),
            (-71.98, 42.0),
        ]

    osm_graph = nx.MultiDiGraph()
    osm_graph.add_nodes_from(
        [
            ("u", {"x": -72.0, "y": 42.0}),
            ("v", {"x": -71.98, "y": 42.0}),
            ("c", {"x": -71.0, "y": 41.0}),
            ("d", {"x": -70.99, "y": 41.01}),
        ]
    )
    osm_graph.add_edge(
        "u",
        "v",
        key=0,
        length=2500.0,
        highway="secondary",
        oneway=True,
        geometry=_Geometry(),
    )
    osm_graph.add_edge(
        "c",
        "d",
        key=0,
        length=1800.0,
        highway="secondary",
        oneway=True,
    )

    graph = _graph_from_osmnx(osm_graph, {})
    curved = [edge for edge in graph.edges.values() if edge.id.startswith("u-v-0")]
    geometryless = graph.edges["c-d-0-segment-0"]

    assert sum(edge.distance_km for edge in curved) == pytest.approx(2.5)
    first_chord = _haversine_km(42.0, -72.0, 42.02, -71.99)
    second_chord = _haversine_km(42.02, -71.99, 42.0, -71.98)
    assert curved[0].distance_km / curved[1].distance_km == pytest.approx(
        first_chord / second_chord
    )
    assert geometryless.distance_km == pytest.approx(1.8)


def test_osm_invalid_length_falls_back_to_chord_total() -> None:
    from src.route_planner.graph import _graph_from_osmnx, _haversine_km

    osm_graph = nx.MultiDiGraph()
    osm_graph.add_nodes_from(
        [
            ("u", {"x": -72.0, "y": 42.0}),
            ("v", {"x": -71.98, "y": 42.01}),
        ]
    )
    osm_graph.add_edge(
        "u",
        "v",
        key=0,
        length="not-a-length",
        highway="secondary",
    )

    graph = _graph_from_osmnx(osm_graph, {})
    edge = graph.edges["u-v-0-segment-0"]
    assert edge.distance_km == pytest.approx(_haversine_km(42.0, -72.0, 42.01, -71.98))


def test_load_legacy_omitted_speed_uses_road_type_parser_default(
    tmp_path: Path,
) -> None:
    payload = {
        "nodes": [
            {"id": "A", "lat": 42.0, "lon": -72.0},
            {"id": "B", "lat": 42.1, "lon": -72.0},
        ],
        "edges": [
            {
                "id": "AB",
                "start": "A",
                "end": "B",
                "distance_km": 1.5,
                "road_type": "motorway",
            }
        ],
    }
    path = tmp_path / "legacy-motorway.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    graph = RoadGraph.load(path)

    assert graph.edges["AB"].speed_limit_kmh == 100


def test_missing_mapping_pop_is_a_cache_and_epoch_noop() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="b", lat=42.1, lon=-72.1))
    graph.add_edge(
        Edge(
            id="ab",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    graph.find_nearest_node(42.0, -72.0)
    index = graph._nearest_spatial_index
    stamp = graph._heuristic_cache_stamp()
    sentinel = object()

    assert graph.nodes.pop("missing", sentinel) is sentinel
    assert graph.edges.pop("missing", sentinel) is sentinel
    assert graph._nearest_spatial_index is index
    assert graph._heuristic_cache_stamp() == stamp


def test_kd_cutoff_preserves_exact_ties_and_compact_arrays(monkeypatch) -> None:
    import src.route_planner.graph as graph_module

    cutoff = graph_module._KD_SMALL_SUBTREE_CUTOFF
    calls = 0
    original_argpartition = graph_module.np.argpartition

    def counting_argpartition(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_argpartition(*args, **kwargs)

    monkeypatch.setattr(graph_module.np, "argpartition", counting_argpartition)

    for size in (cutoff, cutoff + 1, 2 * cutoff + 1):
        graph = RoadGraph()
        coordinates = []
        for index in range(size):
            coordinate = (float(index % 7), float((index * 3) % 11))
            if index in (size // 2, size - 1):
                coordinate = coordinates[0]
            coordinates.append(coordinate)
            graph.add_node(
                Node(
                    id=f"node-{index}",
                    lat=coordinate[0],
                    lon=coordinate[1],
                )
            )

        before_calls = calls
        index = graph._build_nearest_spatial_index()
        assert index[2].typecode == "d"
        assert index[3].typecode == "d"
        assert index[4].typecode == "i"
        if size <= cutoff:
            assert calls == before_calls
        else:
            assert calls > before_calls

        nodes = tuple(graph.nodes.values())
        for query in ((0.0, 0.0), (4.25, 6.5), (-1.0, 12.0)):
            expected = min(
                nodes,
                key=lambda node: (
                    (node.lat - query[0]) ** 2 + (node.lon - query[1]) ** 2,
                    nodes.index(node),
                ),
            )
            selected = graph.find_nearest_node(*query)
            assert selected.id == expected.id


def test_nearest_node_selects_exact_closest_coordinate() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="far", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="near", lat=42.05, lon=-72.04))
    graph.add_node(Node(id="other", lat=42.2, lon=-72.2))

    selected, snap_km = graph.find_nearest_node_with_distance(42.049, -72.039)

    assert selected.id == "near"
    assert snap_km > 0.0
    assert graph.find_nearest_node(42.049, -72.039).id == "near"


def test_nearest_node_ties_keep_node_insertion_order() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="first", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="second", lat=42.0, lon=-70.0))

    assert graph.find_nearest_node(42.0, -71.0).id == "first"


def test_nearest_node_cancellation_before_work_preserves_exception_and_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="node", lat=42.0, lon=-72.0))
    error = RuntimeError("cancelled")

    def check_cancelled() -> None:
        raise error

    with pytest.raises(RuntimeError) as raised:
        graph.find_nearest_node(42.0, -72.0, check_cancelled=check_cancelled)

    assert raised.value is error
    assert graph._nearest_spatial_index is None


def test_nearest_index_build_cancellation_does_not_publish_partial_cache() -> None:
    graph = RoadGraph()
    for index in range(2048):
        graph.add_node(Node(id=f"node-{index}", lat=float(index), lon=float(index)))
    error = RuntimeError("cancelled")
    calls = 0

    def check_cancelled() -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise error

    with pytest.raises(RuntimeError) as raised:
        graph.find_nearest_node_with_distance(
            0.0,
            0.0,
            check_cancelled=check_cancelled,
        )

    assert raised.value is error
    assert graph._nearest_spatial_index is None
    assert graph.find_nearest_node(0.0, 0.0).id == "node-0"
    assert graph._nearest_spatial_index is not None


def test_nearest_edge_projection_cancellation_before_work_preserves_exception() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=0.0, lon=0.0))
    graph.add_node(Node(id="b", lat=0.0, lon=1.0))
    graph.add_edge(
        Edge(
            id="ab",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=5.0,
        )
    )
    error = RuntimeError("cancelled")

    def check_cancelled() -> None:
        raise error

    with pytest.raises(RuntimeError) as raised:
        graph.find_nearest_edge_positions_with_distance(
            0.0,
            0.5,
            check_cancelled=check_cancelled,
        )

    assert raised.value is error
    assert graph._nearest_edge_projection_index is None


def test_nearest_edge_index_build_cancellation_does_not_publish_partial_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=0.0, lon=0.0))
    graph.add_node(Node(id="b", lat=0.0, lon=1.0))
    for index in range(2048):
        graph.add_edge(
            Edge(
                id=f"edge-{index}",
                start_node_id="a",
                end_node_id="b",
                distance_km=1.0,
                scenic_score=5.0,
            )
        )
    error = RuntimeError("cancelled")
    calls = 0

    def check_cancelled() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise error

    with pytest.raises(RuntimeError) as raised:
        graph.find_nearest_edge_positions_with_distance(
            0.0,
            0.5,
            check_cancelled=check_cancelled,
        )

    assert raised.value is error
    assert graph._nearest_edge_projection_index is None
    projections, distance = graph.find_nearest_edge_positions_with_distance(0.0, 0.5)
    assert len(projections) == 2048
    assert distance == pytest.approx(0.0)
    assert graph._nearest_edge_projection_index is not None


def test_nearest_edge_projection_cancellation_after_numpy_chunk_keeps_cache(
    monkeypatch,
) -> None:
    import src.route_planner.graph as graph_module

    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=0.0, lon=0.0))
    graph.add_node(Node(id="b", lat=0.0, lon=1.0))
    for index in range(4):
        graph.add_edge(
            Edge(
                id=f"edge-{index}",
                start_node_id="a",
                end_node_id="b",
                distance_km=1.0,
                scenic_score=5.0,
            )
        )
    graph.find_nearest_edge_positions_with_distance(0.0, 0.5)
    cached_index = graph._nearest_edge_projection_index
    original_chunk_size = graph_module._EDGE_PROJECTION_CHUNK_SIZE
    monkeypatch.setattr(graph_module, "_EDGE_PROJECTION_CHUNK_SIZE", 2)
    original_project = graph._project_edge_chunk
    chunks = 0
    error = RuntimeError("cancelled")

    def project_chunk(*args, **kwargs):
        nonlocal chunks
        chunks += 1
        return original_project(*args, **kwargs)

    def check_cancelled() -> None:
        if chunks >= 1:
            raise error

    monkeypatch.setattr(graph, "_project_edge_chunk", project_chunk)
    with pytest.raises(RuntimeError) as raised:
        graph.find_nearest_edge_positions_with_distance(
            0.0,
            0.5,
            check_cancelled=check_cancelled,
        )

    assert raised.value is error
    assert chunks == 1
    assert graph._nearest_edge_projection_index is cached_index
    assert original_chunk_size > 0


def test_nearest_edge_projection_ties_remain_deterministic() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=0.0, lon=0.0))
    graph.add_node(Node(id="b", lat=0.0, lon=1.0))
    graph.add_edge(
        Edge(
            id="second",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=5.0,
        )
    )
    graph.add_edge(
        Edge(
            id="first",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=5.0,
        )
    )

    projections, distance = graph.find_nearest_edge_positions_with_distance(0.0, 0.5)

    assert [projection.edge.id for projection in projections] == ["first", "second"]
    assert [projection.fraction for projection in projections] == [0.5, 0.5]
    assert distance == pytest.approx(0.0)


def test_nearest_node_kd_matches_seeded_bruteforce_with_duplicate_coordinates() -> None:
    rng = random.Random(20260710)
    graph = RoadGraph()
    coordinates: list[tuple[float, float]] = []
    duplicate_sources = {64: 7, 129: 42, 194: 42, 321: 129, 450: 7}

    for index in range(513):
        if index in duplicate_sources:
            lat, lon = coordinates[duplicate_sources[index]]
        else:
            lat = rng.uniform(-70.0, 70.0)
            lon = rng.uniform(-170.0, 170.0)
        coordinates.append((lat, lon))
        graph.add_node(Node(id=f"node-{index:03d}", lat=lat, lon=lon))

    queries = [coordinates[0], coordinates[64], coordinates[129]]
    queries.extend(
        (rng.uniform(-70.0, 70.0), rng.uniform(-170.0, 170.0)) for _ in range(256)
    )
    nodes = tuple(graph.nodes.values())

    for query_lat, query_lon in queries:
        _expected_rank, expected = min(
            enumerate(nodes),
            key=lambda item: (
                (item[1].lat - query_lat) ** 2 + (item[1].lon - query_lon) ** 2,
                item[0],
            ),
        )
        selected, snap_km = graph.find_nearest_node_with_distance(query_lat, query_lon)

        assert selected.id == expected.id

        dlat = math.radians(expected.lat - query_lat)
        dlon = math.radians(expected.lon - query_lon)
        haversine_a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(query_lat))
            * math.cos(math.radians(expected.lat))
            * math.sin(dlon / 2) ** 2
        )
        expected_snap_km = 6371.0 * 2 * math.asin(math.sqrt(haversine_a))
        assert snap_km == pytest.approx(expected_snap_km)


def test_nearest_node_empty_graph_is_rejected() -> None:
    graph = RoadGraph()

    with pytest.raises(ValueError):
        graph.find_nearest_node(42.0, -72.0)
    with pytest.raises(ValueError):
        graph.find_nearest_node_with_distance(42.0, -72.0)


def test_nearest_node_index_invalidates_after_add_and_overwrite() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="b", lat=42.1, lon=-72.1))
    query = (42.09, -72.09)

    assert graph.find_nearest_node(*query).id == "b"

    graph.add_node(Node(id="new", lat=42.089, lon=-72.089))
    assert graph.find_nearest_node(*query).id == "new"

    graph.add_node(Node(id="a", lat=42.091, lon=-72.091))
    assert graph.find_nearest_node(*query).id == "a"


def test_nearest_node_index_rebuilds_after_public_node_coordinate_mutation() -> None:
    graph = RoadGraph()
    moved = Node(id="moved", lat=42.0, lon=-72.0)
    stable = Node(id="stable", lat=42.01, lon=-72.01)
    graph.add_node(moved)
    graph.add_node(stable)
    query = (42.0, -72.0)

    assert graph.find_nearest_node(*query).id == "moved"

    graph.get_node("moved").lat = 42.2
    graph.get_node("moved").lon = -72.2

    assert graph.find_nearest_node(*query).id == "stable"
    assert graph.find_nearest_node(42.2, -72.2).id == "moved"


def test_tile_scores_apply_to_repeated_midpoint_tile_edges_and_fallback() -> None:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("a", 0.0, 0.0),
        ("b", 0.2, 0.1),
        ("c", 0.2, 0.0),
        ("d", 0.0, 0.1),
        ("e", 70.0, 100.0),
        ("f", 70.0, 100.1),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=lon))
    graph.add_edge(
        Edge(
            id="first",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=1.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="second",
            start_node_id="c",
            end_node_id="d",
            distance_km=1.0,
            scenic_score=2.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="fallback",
            start_node_id="e",
            end_node_id="f",
            distance_km=1.0,
            scenic_score=3.0,
            one_way=True,
        )
    )

    zoom = 4
    tile_x, tile_y = lat_lon_to_tile(0.1, 0.05, zoom)
    matched, total = apply_tile_scores_to_graph(
        graph,
        {(zoom, tile_x, tile_y): 7.25},
        zoom=zoom,
        fallback=2.5,
    )

    assert (matched, total) == (2, 3)
    assert graph.edges["first"].scenic_score == pytest.approx(7.25)
    assert graph.edges["second"].scenic_score == pytest.approx(7.25)
    assert graph.edges["fallback"].scenic_score == pytest.approx(2.5)


def test_constrained_search_matches_exhaustive_feasible_optimum() -> None:
    graph = RoadGraph()
    for node_id, lat in (
        ("S", 42.0),
        ("A", 42.01),
        ("B", 42.015),
        ("X", 42.02),
        ("G", 42.03),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    edges = (
        Edge(
            id="long",
            start_node_id="S",
            end_node_id="X",
            distance_km=4.0,
            scenic_score=5.0,
            speed_limit_kmh=240,
            one_way=True,
        ),
        Edge(
            id="short-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=5.0,
            speed_limit_kmh=20,
            one_way=True,
        ),
        Edge(
            id="short-2",
            start_node_id="A",
            end_node_id="X",
            distance_km=1.0,
            scenic_score=5.0,
            speed_limit_kmh=20,
            one_way=True,
        ),
        Edge(
            id="alternate-1",
            start_node_id="S",
            end_node_id="B",
            distance_km=0.5,
            scenic_score=5.0,
            speed_limit_kmh=10,
            one_way=True,
        ),
        Edge(
            id="alternate-2",
            start_node_id="B",
            end_node_id="G",
            distance_km=1.5,
            scenic_score=5.0,
            speed_limit_kmh=10,
            one_way=True,
        ),
        Edge(
            id="finish",
            start_node_id="X",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
    )
    for edge in edges:
        graph.add_edge(edge)

    cap_minutes = 8.0
    cost = ScenicCostFunction(scenic_weight=0.0)

    def enumerate_paths(
        node_id: str, path: list[Edge], duration_minutes: float
    ) -> list[list[Edge]]:
        if node_id == "G":
            return [path]
        paths: list[list[Edge]] = []
        for edge in graph.get_edges(node_id):
            if (
                edge.end_node_id in {item.start_node_id for item in path}
                or edge.end_node_id == "S"
            ):
                continue
            next_duration = duration_minutes + edge.travel_time_minutes
            if next_duration <= cap_minutes:
                paths.extend(
                    enumerate_paths(edge.end_node_id, [*path, edge], next_duration)
                )
        return paths

    feasible = enumerate_paths("S", [], 0.0)
    expected = min(
        feasible, key=lambda path: sum(cost.calculate(edge) for edge in path)
    )
    planner = ScenicRoutePlanner(graph=graph)
    actual = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=cap_minutes,
    )

    assert actual is not None
    assert [edge.id for edge in actual] == [edge.id for edge in expected]
    assert sum(cost.calculate(edge) for edge in actual) == pytest.approx(
        sum(cost.calculate(edge) for edge in expected)
    )


def test_scenic_route_keeps_incumbent_over_expensive_and_dead_branches() -> None:
    graph = RoadGraph()
    for node_id, lat in (
        ("S", 42.0),
        ("A", 42.01),
        ("D", 42.02),
        ("X", 42.03),
        ("G", 42.04),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))

    incumbent_edges = (
        Edge(
            id="incumbent-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=4.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="incumbent-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
    )
    expensive_edges = (
        Edge(
            id="expensive-1",
            start_node_id="S",
            end_node_id="D",
            distance_km=6.0,
            scenic_score=0.0,
            speed_limit_kmh=10,
            one_way=True,
        ),
        Edge(
            id="expensive-2",
            start_node_id="D",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=0.0,
            speed_limit_kmh=10,
            one_way=True,
        ),
    )
    dead_edges = (
        Edge(
            id="dead-1",
            start_node_id="S",
            end_node_id="X",
            distance_km=1.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="dead-2",
            start_node_id="X",
            end_node_id="D",
            distance_km=1.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
    )
    for edge in (*incumbent_edges, *expensive_edges, *dead_edges):
        graph.add_edge(edge)

    cost = ScenicCostFunction(scenic_weight=0.5)
    incumbent_cost = sum(cost.calculate(edge) for edge in incumbent_edges)
    expensive_cost = sum(cost.calculate(edge) for edge in expensive_edges)
    assert incumbent_cost < expensive_cost

    route = ScenicRoutePlanner(graph=graph).find_scenic_route(
        (42.0, -72.0),
        (42.04, -72.0),
        scenic_weight=0.5,
        max_detour_factor=1.8,
    )

    assert route.total_distance_km == pytest.approx(10.0)
    assert route.average_scenic_score == pytest.approx(5.0)


def test_scenic_route_keeps_a_path_equal_to_incumbent_cost() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))

    # The two-edge route is the shortest feasible incumbent (10 km).  The
    # direct route is 12 km and has the higher raw scenic score.  Under the
    # normalized utility the duration and scenic tie-breaks select it.
    graph.add_edge(
        Edge(
            id="equal-direct",
            start_node_id="S",
            end_node_id="G",
            distance_km=12.0,
            scenic_score=7.5,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="incumbent-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=5.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="incumbent-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=5.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )

    cost = ScenicCostFunction(scenic_weight=0.5)
    direct = graph.get_edges("S")[0]
    incumbent = graph.get_edges("S")[1]
    incumbent_tail = graph.get_edges("A")[0]
    assert cost.calculate(direct) == pytest.approx(
        cost.calculate(incumbent) + cost.calculate(incumbent_tail)
    )

    route = ScenicRoutePlanner(graph=graph).find_scenic_route(
        (42.0, -72.0),
        (42.02, -72.0),
        scenic_weight=0.5,
        max_detour_factor=1.8,
    )
    assert route.total_distance_km == pytest.approx(12.0)
    assert route.average_scenic_score == pytest.approx(7.5)


def test_reverse_cost_bound_handles_generated_reverse_and_literal_id_collision() -> (
    None
):
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("S", 0.0, 0.0),
        ("A", 0.0, 0.01),
        ("G", 0.0, 0.02),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=lon))

    graph.add_edge(
        Edge(
            id="junction::rev",
            start_node_id="A",
            end_node_id="S",
            distance_km=1.0,
            scenic_score=8.0,
            speed_limit_kmh=100,
            one_way=False,
        )
    )
    graph.add_edge(
        Edge(
            id="junction::rev::rev",
            start_node_id="S",
            end_node_id="G",
            distance_km=3.0,
            scenic_score=1.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=8.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )

    planner = ScenicRoutePlanner(graph=graph)
    path = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=ScenicCostFunction(scenic_weight=1.0),
        max_path_minutes=2.0,
        max_feasible_cost=1.0,
    )

    assert path is not None
    assert [edge.id for edge in path] == ["junction::rev::rev", "A-G"]
    assert sum(edge.travel_time_minutes for edge in path) == pytest.approx(1.2)


def test_custom_mutable_cost_changes_constrained_route_without_cost_bound_cache() -> (
    None
):
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("B", 42.02), ("G", 42.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    for edge in (
        Edge(
            id="A-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="A-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="B-1",
            start_node_id="S",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="B-2",
            start_node_id="B",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)

    class MutableCost:
        def __init__(self) -> None:
            self.prefer = "A"

        def calculate(self, edge: Edge) -> float:
            preferred = edge.id.startswith(self.prefer)
            return 1.0 if preferred else 4.0

    cost = MutableCost()
    planner = ScenicRoutePlanner(graph=graph)
    first = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=2.5,
        max_feasible_cost=2.0,
    )
    assert first is not None
    assert [edge.id for edge in first] == ["A-1", "A-2"]

    cost.prefer = "B"
    second = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=2.5,
        max_feasible_cost=2.0,
    )
    assert second is not None
    assert [edge.id for edge in second] == ["B-1", "B-2"]


def test_reverse_resource_bound_keeps_exact_one_way_optimum_at_cap() -> None:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("S", 0.0, 0.0),
        ("A", 0.0, 0.01),
        ("B", 0.01, 0.01),
        ("G", 0.01, 0.02),
        ("Z", 0.0, 0.015),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=lon))

    # The only S -> A traversal is the generated reverse view.  Its base ID
    # deliberately ends in ``::rev``, colliding with the generated ID of the
    # explicit S -> G edge; predecessor construction must use endpoints and
    # traversal direction rather than edge IDs.
    graph.add_edge(
        Edge(
            id="junction::rev",
            start_node_id="A",
            end_node_id="S",
            distance_km=1.0,
            scenic_score=8.0,
            speed_limit_kmh=100,
            one_way=False,
        )
    )
    graph.add_edge(
        Edge(
            id="junction::rev::rev",
            start_node_id="S",
            end_node_id="G",
            distance_km=3.0,
            scenic_score=1.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=8.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="S-B",
            start_node_id="S",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=1.0,
            speed_limit_kmh=10,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="B-G",
            start_node_id="B",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=1.0,
            speed_limit_kmh=10,
            one_way=True,
        )
    )
    # A zero-duration cycle must not create endlessly improving duplicate
    # labels while the cap-equal A -> G path remains feasible.
    graph.add_edge(
        Edge(
            id="A-Z",
            start_node_id="A",
            end_node_id="Z",
            distance_km=0.0,
            scenic_score=8.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="Z-A",
            start_node_id="Z",
            end_node_id="A",
            distance_km=0.0,
            scenic_score=8.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    # This one-way edge must not be treated as a usable G -> B predecessor.
    graph.add_edge(
        Edge(
            id="G-out",
            start_node_id="G",
            end_node_id="B",
            distance_km=0.1,
            scenic_score=10.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )

    class RouteCost:
        def calculate(self, edge: Edge) -> float:
            return (
                1.0 if edge.end_node_id == "G" and edge.start_node_id == "A" else 10.0
            )

    planner = ScenicRoutePlanner(graph=graph)
    path = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=RouteCost(),
        max_path_minutes=1.5,
    )

    assert path is not None
    assert [(edge.start_node_id, edge.end_node_id) for edge in path] == [
        ("S", "A"),
        ("A", "G"),
    ]
    assert sum(edge.travel_time_minutes for edge in path) == pytest.approx(1.2)


def test_reverse_resource_bound_prunes_geodesically_near_road_detour() -> None:
    graph = RoadGraph()
    for node_id, lon in (
        ("S", 0.0),
        ("N", 0.009),
        ("D", 0.018),
        ("G", 0.010),
        ("A", 0.006),
    ):
        graph.add_node(Node(id=node_id, lat=0.0, lon=lon))

    # N is only about a kilometre from G, but the directed road continuation
    # from N to G takes too long to fit after S -> N under the cap.
    graph.add_edge(
        Edge(
            id="S-N",
            start_node_id="S",
            end_node_id="N",
            distance_km=1.0,
            scenic_score=10.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="N-D",
            start_node_id="N",
            end_node_id="D",
            distance_km=3.0,
            scenic_score=10.0,
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="D-G",
            start_node_id="D",
            end_node_id="G",
            distance_km=3.0,
            scenic_score=10.0,
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="S-A",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.5,
            scenic_score=1.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=2.5,
            scenic_score=1.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )

    class RouteCost:
        def calculate(self, edge: Edge) -> float:
            return (
                0.0 if edge.start_node_id == "S" and edge.end_node_id == "N" else 10.0
            )

    path = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=RouteCost(),
        max_path_minutes=5.0,
    )

    assert path is not None
    assert [edge.id for edge in path] == ["S-A", "A-G"]
    assert sum(edge.travel_time_minutes for edge in path) == pytest.approx(4.8)


def test_constrained_duration_cap_keeps_feasible_path() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(
            id="over-cap",
            start_node_id="S",
            end_node_id="G",
            distance_km=5.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="feasible-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.2,
            scenic_score=1.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="feasible-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.2,
            scenic_score=1.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )

    planner = ScenicRoutePlanner(graph=graph)
    edges = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=ScenicCostFunction(scenic_weight=0.0),
        max_path_minutes=3.0,
    )

    assert edges is not None
    assert [edge.id for edge in edges] == ["feasible-1", "feasible-2"]
    assert sum(edge.travel_time_minutes for edge in edges) <= 3.0


def test_builtin_cost_ratio_scan_is_reused_without_graph_mutation() -> None:
    class CountingEdges(dict[str, Edge]):
        values_calls = 0

        def values(self):  # type: ignore[no-untyped-def]
            self.values_calls += 1
            return super().values()

    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=4.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    graph.edges = CountingEdges(graph.edges)
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.75)

    first = planner._minimum_cost_per_km(cost)
    values_calls_after_first = graph.edges.values_calls
    second = planner._minimum_cost_per_km(cost)

    assert first == second
    assert values_calls_after_first == 2
    assert graph.edges.values_calls == values_calls_after_first


def test_class_level_cost_monkeypatch_bypasses_ratio_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.5)
    calls = 0
    value = 1.0

    def counted_calculate(instance: ScenicCostFunction, edge: Edge) -> float:
        nonlocal calls
        calls += 1
        return value

    monkeypatch.setattr(ScenicCostFunction, "calculate", counted_calculate)

    first = planner._minimum_cost_per_km(cost)
    value = 3.0
    second = planner._minimum_cost_per_km(cost)

    assert first == pytest.approx(1.0 / 20.0)
    assert second == pytest.approx(3.0 / 20.0)
    assert calls == 2


@pytest.mark.parametrize(
    "cost_value", [-1.0, float("nan"), float("inf"), float("-inf")]
)
def test_constrained_search_rejects_invalid_edge_cost(cost_value: float) -> None:
    class InvalidCost:
        def calculate(self, edge: Edge) -> float:
            return cost_value

    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.01, lon=-72.0))
    graph.add_edge(
        Edge(
            id="invalid-cost",
            start_node_id="S",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )

    with pytest.raises(ValueError, match="cost"):
        ScenicRoutePlanner(graph=graph)._a_star(
            graph.get_node("S"),
            graph.get_node("G"),
            cost_function=InvalidCost(),
            max_path_minutes=2.0,
        )


@pytest.mark.parametrize(
    "max_path_minutes", [-1.0, float("nan"), float("inf"), float("-inf")]
)
def test_constrained_search_rejects_invalid_duration_cap(
    max_path_minutes: float,
) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.01, lon=-72.0))

    with pytest.raises(ValueError, match="max_path_minutes"):
        ScenicRoutePlanner(graph=graph)._a_star(
            graph.get_node("S"),
            graph.get_node("G"),
            cost_function=ScenicCostFunction(scenic_weight=0.0),
            max_path_minutes=max_path_minutes,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_feasible_cost", -1.0),
        ("max_feasible_cost", float("nan")),
        ("max_feasible_cost", float("inf")),
        ("max_feasible_cost", float("-inf")),
        ("shortest_duration_minutes", -1.0),
        ("shortest_duration_minutes", float("nan")),
        ("shortest_duration_minutes", float("inf")),
        ("shortest_duration_minutes", float("-inf")),
    ],
)
def test_constrained_search_rejects_invalid_optional_bounds(
    field: str, value: float
) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.01, lon=-72.0))
    graph.add_edge(
        Edge(
            id="valid",
            start_node_id="S",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    kwargs: dict[str, float] = {field: value}
    if field == "shortest_duration_minutes":
        kwargs["max_feasible_cost"] = 10.0

    with pytest.raises(ValueError, match=field):
        ScenicRoutePlanner(graph=graph)._a_star(
            graph.get_node("S"),
            graph.get_node("G"),
            cost_function=ScenicCostFunction(scenic_weight=0.0),
            max_path_minutes=2.0,
            **kwargs,
        )


@pytest.mark.parametrize("invalid_value", [-1.0, float("nan"), float("inf")])
def test_invalid_cost_is_rejected_before_cyclic_label_expansion(
    invalid_value: float,
) -> None:
    class CyclicCost:
        def calculate(self, edge: Edge) -> float:
            return invalid_value if edge.id.startswith("cycle") else 1.0

    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=0.0, lon=lon))
    graph.add_edge(
        Edge(
            id="cycle-forward",
            start_node_id="S",
            end_node_id="A",
            distance_km=0.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="cycle-back",
            start_node_id="A",
            end_node_id="S",
            distance_km=0.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="finish",
            start_node_id="S",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )

    with pytest.raises(ValueError, match="cost"):
        ScenicRoutePlanner(graph=graph)._a_star(
            graph.get_node("S"),
            graph.get_node("G"),
            cost_function=CyclicCost(),
            max_path_minutes=2.0,
        )


def test_infinitesimally_under_geodesic_distance_disables_positive_heuristic() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=0.0, lon=0.0))
    graph.add_node(Node(id="G", lat=0.0, lon=0.09))
    planner = ScenicRoutePlanner(graph=graph)
    geodesic_km = planner._haversine(0.0, 0.0, 0.0, 0.09)
    stored_distance_km = math.nextafter(geodesic_km, 0.0)
    graph.add_edge(
        Edge(
            id="under-geodesic",
            start_node_id="S",
            end_node_id="G",
            distance_km=stored_distance_km,
            scenic_score=5.0,
            one_way=True,
        )
    )

    assert stored_distance_km < geodesic_km
    assert planner._edge_distances_are_geodesic_lower_bounds() is False
    assert planner._minimum_cost_per_km(ScenicCostFunction(scenic_weight=0.0)) == 0.0


@pytest.mark.parametrize("bound", ["duration", "cost", "augmented"])
def test_reverse_bound_snapshot_is_disabled_by_graph_mutation(
    monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=0.0, lon=lon))
    graph.add_edge(
        Edge(
            id="S-A",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    original_get_edges = graph.get_edges
    mutated = False

    def mutate_during_enumeration(node_id: str) -> list[Edge]:
        nonlocal mutated
        edges = original_get_edges(node_id)
        if not mutated:
            mutated = True
            graph.edges["S-A"].distance_km = 1.1
        return edges

    monkeypatch.setattr(graph, "get_edges", mutate_during_enumeration)
    goal = graph.get_node("G")
    cost = ScenicCostFunction(scenic_weight=0.5)

    if bound == "duration":
        result = planner._reverse_duration_lower_bounds(goal, 3.0)
    elif bound == "cost":
        result = planner._reverse_cost_lower_bounds(goal, cost)
    else:
        result = planner._reverse_augmented_cost_lower_bounds(goal, cost, 1.0)

    assert result is None


def test_ratio_cache_is_bounded_over_many_weight_signatures() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)

    for index in range(40):
        planner._minimum_cost_per_km(ScenicCostFunction(scenic_weight=index / 39.0))

    assert len(planner._minimum_cost_per_km_cache) <= 8


def test_reverse_preprocessing_enumerates_traversals_once_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=0.0, lon=lon))
    graph.add_edge(
        Edge(
            id="S-A",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    original_iter_edges = graph.iter_edges
    setup_complete = False
    snapshot_calls = 0
    forward_calls = 0

    def counted_iter_edges(node_id: str):
        nonlocal forward_calls
        if setup_complete:
            forward_calls += 1
        return original_iter_edges(node_id)

    original_snapshot = planner._build_reverse_predecessor_snapshot

    def mark_setup_complete(
        cost_function: ScenicCostFunction | None = None,
    ):
        nonlocal setup_complete, snapshot_calls
        snapshot_calls += 1
        result = original_snapshot(cost_function)
        setup_complete = True
        return result

    monkeypatch.setattr(graph, "iter_edges", counted_iter_edges)
    monkeypatch.setattr(
        planner, "_build_reverse_predecessor_snapshot", mark_setup_complete
    )
    path = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=ScenicCostFunction(scenic_weight=0.5),
        max_path_minutes=3.0,
        max_feasible_cost=100.0,
        shortest_duration_minutes=2.4,
    )

    assert path is not None
    assert [edge.id for edge in path] == ["S-A", "A-G"]
    assert snapshot_calls == 1
    assert forward_calls > 0


def test_instance_shadowed_base_cost_calculate_bypasses_ratio_cache() -> None:
    class MutableCalculate:
        def __init__(self) -> None:
            self.value = 1.0
            self.calls = 0

        def __call__(self, edge: Edge) -> float:
            self.calls += 1
            return self.value

    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.5)
    calculate = MutableCalculate()
    setattr(cost, "calculate", calculate)

    first = planner._minimum_cost_per_km(cost)
    calculate.value = 3.0
    second = planner._minimum_cost_per_km(cost)

    assert first == pytest.approx(1.0 / 20.0)
    assert second == pytest.approx(3.0 / 20.0)
    assert calculate.calls == 2


def test_instance_shadowed_base_cost_road_adjustment_bypasses_ratio_cache() -> None:
    class MutableAdjustment:
        def __init__(self) -> None:
            self.value = 0.0
            self.calls = 0

        def __call__(self, road_type: str) -> float:
            self.calls += 1
            return self.value

    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.0)
    adjustment = MutableAdjustment()
    setattr(cost, "_road_type_adjustment", adjustment)

    first = planner._minimum_cost_per_km(cost)
    adjustment.value = 0.25
    second = planner._minimum_cost_per_km(cost)

    assert first == pytest.approx(24.0 / 20.0)
    assert second == pytest.approx(30.0 / 20.0)
    assert adjustment.calls == 2


def test_scenic_score_application_invalidates_cost_ratio_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=0.0, lon=0.0))
    graph.add_node(Node(id="G", lat=0.1, lon=0.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=2.0,
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=1.0)
    before = planner._minimum_cost_per_km(cost)

    zoom = 4
    tile_x, tile_y = lat_lon_to_tile(0.05, 0.0, zoom)
    matched, total = apply_tile_scores_to_graph(
        graph,
        {(zoom, tile_x, tile_y): 8.0},
        zoom=zoom,
        fallback=2.0,
    )
    after_application = planner._minimum_cost_per_km(cost)

    assert (matched, total) == (1, 1)
    assert after_application != before

    graph.edges["edge"].scenic_score = 6.0
    after_mutation = planner._minimum_cost_per_km(cost)
    assert after_mutation != after_application


def test_public_edge_mapping_replacement_and_deletion_invalidate_ratio_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.edges["edge"] = Edge(
        id="edge",
        start_node_id="S",
        end_node_id="G",
        distance_km=20.0,
        scenic_score=2.0,
        speed_limit_kmh=50,
        one_way=True,
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=1.0)
    before = planner._minimum_cost_per_km(cost)

    graph.edges["edge"] = Edge(
        id="edge",
        start_node_id="S",
        end_node_id="G",
        distance_km=20.0,
        scenic_score=8.0,
        speed_limit_kmh=50,
        one_way=True,
    )
    after_replacement = planner._minimum_cost_per_km(cost)
    del graph.edges["edge"]
    after_deletion = planner._minimum_cost_per_km(cost)

    assert after_replacement != before
    assert after_deletion == 0.0


def test_node_coordinate_mutation_invalidates_geodesic_safety_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)

    assert planner._edge_distances_are_geodesic_lower_bounds() is True
    graph.get_node("G").lon = -71.0
    assert planner._edge_distances_are_geodesic_lower_bounds() is False


def test_public_node_mapping_replacement_invalidates_geodesic_safety_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)

    assert planner._edge_distances_are_geodesic_lower_bounds() is True
    graph.nodes["G"] = Node(id="G", lat=42.0, lon=-71.0)
    assert planner._edge_distances_are_geodesic_lower_bounds() is False


def test_edge_distance_mutation_invalidates_geodesic_safety_cache() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)

    assert planner._edge_distances_are_geodesic_lower_bounds() is True
    graph.edges["edge"].distance_km = 1.0
    assert planner._edge_distances_are_geodesic_lower_bounds() is False


def test_custom_mutable_cost_function_is_not_ratio_cache_reused() -> None:
    class MutableCost:
        def __init__(self) -> None:
            self.value = 1.0
            self.calls = 0

        def calculate(self, edge: Edge) -> float:
            self.calls += 1
            return self.value

    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = MutableCost()

    first = planner._minimum_cost_per_km(cost)
    cost.value = 3.0
    second = planner._minimum_cost_per_km(cost)

    assert first != second
    assert cost.calls == 2


def test_geodesic_safety_scan_is_reused_without_graph_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="edge",
            start_node_id="S",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=5.0,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    haversine = planner._haversine
    calls = 0

    def counted_haversine(*args: float) -> float:
        nonlocal calls
        calls += 1
        return haversine(*args)

    monkeypatch.setattr(planner, "_haversine", counted_haversine)

    assert planner._edge_distances_are_geodesic_lower_bounds() is True
    calls_after_first = calls
    assert planner._edge_distances_are_geodesic_lower_bounds() is True

    assert calls_after_first == len(graph.edges)
    assert calls == calls_after_first


def test_manual_edge_distances_below_geodesic_disable_astar_heuristics() -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.045), ("G", 0.09)):
        graph.add_node(Node(id=node_id, lat=0.0, lon=lon))
    graph.add_edge(
        Edge(
            id="direct",
            start_node_id="S",
            end_node_id="G",
            distance_km=5.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="via-a-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=2.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="via-a-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.0)

    unconstrained = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=None,
    )
    constrained = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=4.0,
    )

    assert unconstrained is not None
    assert [edge.id for edge in unconstrained] == ["via-a-1", "via-a-2"]
    assert constrained is not None
    assert [edge.id for edge in constrained] == ["via-a-1", "via-a-2"]


def test_zero_distance_zero_cost_cycle_rejects_duplicate_resource_labels() -> None:
    class ZeroCost:
        def calculate(self, edge: Edge) -> float:
            return 0.0

    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=0.0, lon=0.0))
    graph.add_node(Node(id="A", lat=0.0, lon=0.001))
    graph.add_node(Node(id="G", lat=0.0, lon=0.005))
    graph.add_edge(
        Edge(
            id="to-cycle",
            start_node_id="S",
            end_node_id="A",
            distance_km=0.0,
            scenic_score=0.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="from-cycle",
            start_node_id="A",
            end_node_id="S",
            distance_km=0.0,
            scenic_score=0.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="finish",
            start_node_id="S",
            end_node_id="G",
            distance_km=0.5,
            scenic_score=0.0,
            one_way=True,
        )
    )

    path = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=ZeroCost(),
        max_path_minutes=1.0,
    )

    assert path is not None
    assert [edge.id for edge in path] == ["finish"]


def test_load_legacy_json_applies_all_omitted_edge_defaults(tmp_path: Path) -> None:
    payload = {
        "nodes": [
            {"id": "A", "lat": 42.0, "lon": -72.0},
            {"id": "B", "lat": 42.1, "lon": -72.0},
        ],
        "edges": [
            {
                "id": "AB",
                "start": "A",
                "end": "B",
                "distance_km": 1.5,
            }
        ],
    }
    path = tmp_path / "road_graph.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))

    graph = RoadGraph.load(path)
    edge = graph.edges["AB"]

    assert edge.scenic_score == 5.0
    assert edge.road_name is None
    assert edge.road_type == "secondary"
    assert edge.speed_limit_kmh == 60
    assert edge.one_way is True
    assert [item.id for item in graph.get_edges("A")] == ["AB"]
    assert graph.get_edges("B") == []


def test_load_legacy_json_preserves_order_and_reverse_views(tmp_path: Path) -> None:
    payload = {
        "nodes": [
            {"id": "A", "lat": 42.0, "lon": -72.0},
            {"id": "B", "lat": 42.1, "lon": -72.0},
            {"id": "C", "lat": 42.2, "lon": -72.0},
        ],
        "edges": [
            {
                "id": "AB",
                "start": "A",
                "end": "B",
                "distance_km": 1.0,
                "one_way": False,
            },
            {
                "id": "BC",
                "start": "B",
                "end": "C",
                "distance_km": 2.0,
                "one_way": True,
            },
        ],
    }
    path = tmp_path / "ordered_graph.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))

    graph = RoadGraph.load(path)

    assert list(graph.nodes) == ["A", "B", "C"]
    assert list(graph.edges) == ["AB", "BC"]
    middle_edges = graph.get_edges("B")
    assert [
        (edge.id, edge.start_node_id, edge.end_node_id) for edge in middle_edges
    ] == [
        ("AB::rev", "B", "A"),
        ("BC", "B", "C"),
    ]
    assert graph.get_edges("C") == []


@pytest.mark.parametrize(
    ("edge", "error_match"),
    [
        (
            {
                "id": "AB",
                "start": "missing",
                "end": "B",
                "distance_km": 1.0,
            },
            "Unknown start node",
        ),
        (
            {
                "id": "AB",
                "start": "A",
                "end": "missing",
                "distance_km": 1.0,
            },
            "Unknown end node",
        ),
        (
            {
                "id": "AB",
                "start": "A",
                "end": "B",
                "distance_km": 1.0,
            },
            "Duplicate edge id",
        ),
    ],
)
def test_load_legacy_json_rejects_bad_endpoints_and_duplicate_ids(
    tmp_path: Path, edge: dict[str, object], error_match: str
) -> None:
    edges = [edge]
    if error_match == "Duplicate edge id":
        edges.append(dict(edge))
    payload = {
        "nodes": [
            {"id": "A", "lat": 42.0, "lon": -72.0},
            {"id": "B", "lat": 42.1, "lon": -72.0},
        ],
        "edges": edges,
    }
    path = tmp_path / "invalid_graph.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))

    with pytest.raises(ValueError, match=error_match):
        RoadGraph.load(path)


def test_load_legacy_json_normalizes_historical_field_representations(
    tmp_path: Path,
) -> None:
    payload = {
        "nodes": [
            {"id": 1, "lat": "42.0", "lon": "-72.0"},
            {"id": "B", "lat": "42.1", "lon": "-72.0"},
        ],
        "edges": [
            {
                "id": 7,
                "start": 1,
                "end": "B",
                "distance_km": "1.25",
                "scenic_score": "8.5",
                "speed_limit_kmh": "35 mph",
                "one_way": "no",
            }
        ],
    }
    path = tmp_path / "historical_representations.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))

    graph = RoadGraph.load(path)
    edge = graph.edges["7"]

    assert list(graph.nodes) == ["1", "B"]
    assert graph.nodes["1"].coords == (42.0, -72.0)
    assert edge.start_node_id == "1"
    assert edge.distance_km == 1.25
    assert edge.scenic_score == 8.5
    assert edge.speed_limit_kmh == 56
    assert edge.one_way is False
    assert [
        (item.id, item.start_node_id, item.end_node_id) for item in graph.get_edges("B")
    ] == [("7::rev", "B", "1")]


def test_load_legacy_json_duplicate_node_overwrite_preserves_order_and_adjacency(
    tmp_path: Path,
) -> None:
    payload = {
        "nodes": [
            {"id": "A", "lat": 42.0, "lon": -72.0},
            {"id": "B", "lat": 42.1, "lon": -72.0},
            {"id": "A", "lat": 43.0, "lon": -71.0},
        ],
        "edges": [
            {
                "id": "AB",
                "start": "A",
                "end": "B",
                "distance_km": 1.0,
                "one_way": False,
            }
        ],
    }
    path = tmp_path / "duplicate_node.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))

    graph = RoadGraph.load(path)

    assert list(graph.nodes) == ["A", "B"]
    assert graph.nodes["A"].coords == (43.0, -71.0)
    assert [
        (item.id, item.start_node_id, item.end_node_id) for item in graph.get_edges("A")
    ] == [("AB", "A", "B")]
    assert [
        (item.id, item.start_node_id, item.end_node_id) for item in graph.get_edges("B")
    ] == [("AB::rev", "B", "A")]


@pytest.mark.parametrize(
    "field",
    ["distance_km", "scenic_score"],
)
def test_load_legacy_json_rejects_values_legacy_parser_cannot_normalize(
    tmp_path: Path, field: str
) -> None:
    edge: dict[str, object] = {
        "id": "AB",
        "start": "A",
        "end": "B",
        "distance_km": 1.0,
    }
    edge[field] = "not-a-number"
    payload = {
        "nodes": [
            {"id": "A", "lat": 42.0, "lon": -72.0},
            {"id": "B", "lat": 42.1, "lon": -72.0},
        ],
        "edges": [edge],
    }
    path = tmp_path / "typed_invalid_graph.json"
    path.write_bytes(json.dumps(payload).encode("utf-8"))

    with pytest.raises(ValueError):
        RoadGraph.load(path)


@pytest.mark.parametrize("reverse_first_leg", [False, True])
def test_builtin_lagrangian_bound_matches_exhaustive_optimum_in_both_orientations(
    reverse_first_leg: bool,
) -> None:
    """Eligible built-in costs preserve the exact constrained optimum.

    The bidirectional case reaches ``A`` through the generated reverse view of
    ``A -> S``; this keeps reverse-potential traversal aligned with
    ``RoadGraph.get_edges`` rather than relying on stored edge orientation.
    """
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("B", 42.02), ("G", 42.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    first_leg = Edge(
        id="S-A" if not reverse_first_leg else "A-S",
        start_node_id="S" if not reverse_first_leg else "A",
        end_node_id="A" if not reverse_first_leg else "S",
        distance_km=2.0,
        scenic_score=10.0,
        speed_limit_kmh=60,
        one_way=not reverse_first_leg,
    )
    edges = (
        first_leg,
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="S-B",
            start_node_id="S",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="B-G",
            start_node_id="B",
            end_node_id="G",
            distance_km=4.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
    )
    for edge in edges:
        graph.add_edge(edge)

    cap_km = 5.0
    cost = ScenicCostFunction(scenic_weight=0.7)

    def enumerate_paths(
        node_id: str, path: list[Edge], distance_km: float
    ) -> list[list[Edge]]:
        if node_id == "G":
            return [path]
        paths: list[list[Edge]] = []
        visited = {item.start_node_id for item in path}
        for edge in graph.get_edges(node_id):
            if edge.end_node_id in visited or edge.end_node_id == "S":
                continue
            next_distance = distance_km + edge.distance_km
            if next_distance <= cap_km:
                paths.extend(
                    enumerate_paths(edge.end_node_id, [*path, edge], next_distance)
                )
        return paths

    feasible = enumerate_paths("S", [], 0.0)
    expected = min(
        feasible, key=lambda path: sum(cost.calculate(edge) for edge in path)
    )
    expected_cost = sum(cost.calculate(edge) for edge in expected)
    actual = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=cap_km,
        max_feasible_cost=expected_cost,
        shortest_duration_minutes=4.0,
    )

    assert actual is not None
    assert [edge.id for edge in actual] == [edge.id for edge in expected]
    assert sum(edge.distance_km for edge in actual) <= cap_km
    assert sum(cost.calculate(edge) for edge in actual) == pytest.approx(expected_cost)


def test_builtin_lagrangian_bound_keeps_cost_and_distance_cap_equal_candidate() -> None:
    """A candidate exactly equal to incumbent cost and cap remains feasible."""
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    candidate = (
        Edge(
            id="cap-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=2.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="cap-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=3.0,
            scenic_score=5.0,
            speed_limit_kmh=90,
            one_way=True,
        ),
    )
    for edge in candidate:
        graph.add_edge(edge)
    cost = ScenicCostFunction(scenic_weight=0.7)
    incumbent_cost = sum(cost.calculate(edge) for edge in candidate)

    actual = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=5.0,
        max_feasible_cost=incumbent_cost,
        shortest_duration_minutes=5.0,
    )

    assert actual is not None
    assert [edge.id for edge in actual] == ["cap-1", "cap-2"]
    assert sum(edge.distance_km for edge in actual) == pytest.approx(5.0)
    assert sum(cost.calculate(edge) for edge in actual) == pytest.approx(incumbent_cost)


def test_mutable_custom_cost_remains_exact_fallback_with_constrained_bound() -> None:
    class MutableCost:
        def __init__(self) -> None:
            self.prefer = "A"

        def calculate(self, edge: Edge) -> float:
            return 1.0 if edge.id.startswith(self.prefer) else 4.0

    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("B", 42.02), ("G", 42.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    for edge in (
        Edge(
            id="A-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="A-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="B-1",
            start_node_id="S",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="B-2",
            start_node_id="B",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)
    cost = MutableCost()
    planner = ScenicRoutePlanner(graph=graph)

    first = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=2.5,
        max_feasible_cost=2.0,
        shortest_duration_minutes=2.4,
    )
    assert first is not None
    assert [edge.id for edge in first] == ["A-1", "A-2"]

    cost.prefer = "B"
    second = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=2.5,
        max_feasible_cost=2.0,
        shortest_duration_minutes=2.4,
    )
    assert second is not None
    assert [edge.id for edge in second] == ["B-1", "B-2"]


def test_subclassed_builtin_cost_remains_exact_fallback() -> None:
    class SubclassCost(ScenicCostFunction):
        def __init__(self) -> None:
            super().__init__(scenic_weight=0.5)
            self.prefer = "A"

        def calculate(self, edge: Edge) -> float:
            return 1.0 if edge.id.startswith(self.prefer) else 4.0

    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("B", 42.02), ("G", 42.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    for edge in (
        Edge(
            id="A-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="A-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="B-1",
            start_node_id="S",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
        Edge(
            id="B-2",
            start_node_id="B",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=5.0,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)
    cost = SubclassCost()
    path = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=2.5,
        max_feasible_cost=2.0,
        shortest_duration_minutes=2.4,
    )

    assert path is not None
    assert [edge.id for edge in path] == ["A-1", "A-2"]


def test_duration_constrained_search_matches_exhaustive_feasible_optimum() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("B", 42.02), ("G", 42.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    for edge in (
        Edge(
            id="fast-ugly",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=100,
            one_way=True,
        ),
        Edge(
            id="best-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=4.0,
            scenic_score=9.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="best-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=4.0,
            scenic_score=9.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="over-1",
            start_node_id="S",
            end_node_id="B",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=40,
            one_way=True,
        ),
        Edge(
            id="over-2",
            start_node_id="B",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=40,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)
    cost = ScenicCostFunction(scenic_weight=1.0)
    cap_minutes = 8.0

    def enumerate_paths(node_id: str, path: list[Edge]) -> list[list[Edge]]:
        if node_id == "G":
            return [path]
        paths: list[list[Edge]] = []
        for edge in graph.get_edges(node_id):
            candidate = [*path, edge]
            if sum(item.travel_time_minutes for item in candidate) <= cap_minutes:
                paths.extend(enumerate_paths(edge.end_node_id, candidate))
        return paths

    feasible = enumerate_paths("S", [])
    expected = min(
        feasible, key=lambda path: sum(cost.calculate(edge) for edge in path)
    )
    actual = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=cap_minutes,
        shortest_duration_minutes=6.0,
    )

    assert actual is not None
    assert [edge.id for edge in actual] == [edge.id for edge in expected]


def test_fastest_baseline_does_not_discount_scenic_byways() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(
            id="ordinary",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="byway-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=7.5,
            scenic_score=10.0,
            road_type="scenic_byway",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="byway-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=7.5,
            scenic_score=10.0,
            road_type="scenic_byway",
            speed_limit_kmh=60,
            one_way=True,
        )
    )

    fastest = ScenicRoutePlanner(graph=graph).find_fastest_route(
        (42.0, -72.0), (42.02, -72.0)
    )
    capped = ScenicRoutePlanner(graph=graph).find_scenic_route(
        (42.0, -72.0), (42.02, -72.0), scenic_weight=1.0, max_detour_factor=1.0
    )

    assert [segment.road_type for segment in fastest.segments] == ["secondary"]
    assert capped.estimated_duration_minutes == pytest.approx(10.0)


def test_duration_mode_keeps_equal_cost_faster_label() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("X", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(
            id="slow-byway",
            start_node_id="S",
            end_node_id="X",
            distance_km=2.0,
            scenic_score=5.0,
            road_type="scenic_byway",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="fast-ordinary",
            start_node_id="S",
            end_node_id="X",
            distance_km=2.0,
            scenic_score=5.0,
            speed_limit_kmh=120,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="finish",
            start_node_id="X",
            end_node_id="G",
            distance_km=3.0,
            scenic_score=5.0,
            speed_limit_kmh=90,
            one_way=True,
        )
    )
    for index in range(257):
        start_id = f"d{index}a"
        end_id = f"d{index}b"
        graph.add_node(Node(id=start_id, lat=0.0, lon=float(index) / 1000))
        graph.add_node(Node(id=end_id, lat=0.0, lon=float(index + 1) / 1000))
        graph.add_edge(
            Edge(
                id=f"dummy-{index}",
                start_node_id=start_id,
                end_node_id=end_id,
                distance_km=1.0,
                scenic_score=5.0,
                one_way=True,
            )
        )

    path = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=ScenicCostFunction(scenic_weight=0.0),
        max_path_minutes=3.1,
        shortest_duration_minutes=3.0,
    )

    assert path is not None
    assert [edge.id for edge in path] == ["fast-ordinary", "finish"]


def test_avoid_highways_is_hard_filter_in_forward_and_reverse_search() -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=42.0, lon=-72.0 + lon))
    graph.add_edge(
        Edge(
            id="fast-motorway",
            start_node_id="S",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=0.0,
            road_type="motorway",
            speed_limit_kmh=120,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="secondary-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=2.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="secondary-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    cost = ScenicCostFunction(scenic_weight=0.8, avoid_highways=True)
    planner = ScenicRoutePlanner(graph=graph, cost_function=cost)

    reverse_bounds = planner._reverse_duration_lower_bounds(
        graph.get_node("G"), max_path_minutes=20.0
    )
    assert reverse_bounds is not None
    assert reverse_bounds["S"] == pytest.approx(8.0)

    path = planner._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=8.0,
    )
    assert path is not None
    assert [edge.id for edge in path] == ["secondary-1", "secondary-2"]
    assert all(edge.road_type not in {"highway", "motorway", "trunk"} for edge in path)


def test_fastest_route_ignores_custom_scenic_weights() -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=42.0, lon=-72.0 + lon))
    graph.add_edge(
        Edge(
            id="fast-ugly",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="slow-scenic-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="slow-scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(
        graph=graph,
        cost_function=ScenicCostFunction(
            scenic_weight=1.0,
            weights=CostWeights(
                travel_time=0.0,
                scenic_reward=100.0,
                highway_penalty=100.0,
                scenic_byway_bonus=0.5,
            ),
        ),
    )

    fastest = planner.find_fastest_route((42.0, -72.0), (42.0, -71.98))
    assert [segment.road_name for segment in fastest.segments] == [None]
    assert fastest.estimated_duration_minutes == pytest.approx(10.0)


def test_duration_cap_uses_true_fastest_eligible_baseline() -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=42.0, lon=-72.0 + lon))
    graph.add_edge(
        Edge(
            id="ineligible-trunk",
            start_node_id="S",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=0.0,
            road_type="trunk",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="eligible-scenic-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=2.0,
            scenic_score=10.0,
            road_type="secondary",
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="eligible-scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=10.0,
            road_type="secondary",
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)

    fastest = planner.find_fastest_route(
        (42.0, -72.0), (42.0, -71.98), avoid_highways=True
    )
    route = planner.find_scenic_route(
        (42.0, -72.0),
        (42.0, -71.98),
        scenic_weight=1.0,
        avoid_highways=True,
        max_detour_factor=1.1,
    )

    assert fastest.estimated_duration_minutes == pytest.approx(8.0)
    assert route.estimated_duration_minutes == pytest.approx(8.0)
    assert [segment.road_type for segment in route.segments] == [
        "secondary",
        "secondary",
    ]


def test_duration_oracle_matches_exact_scenic_optimum_with_hard_filter() -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("B", 0.02), ("G", 0.03)):
        graph.add_node(Node(id=node_id, lat=42.0, lon=-72.0 + lon))
    edges = [
        Edge(
            id="forbidden-highway",
            start_node_id="S",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=10.0,
            road_type="highway",
            speed_limit_kmh=120,
            one_way=True,
        ),
        Edge(
            id="A-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=2.0,
            scenic_score=2.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="A-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=2.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="B-1",
            start_node_id="S",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=9.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
        Edge(
            id="B-2",
            start_node_id="B",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=9.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
    ]
    for edge in edges:
        graph.add_edge(edge)

    cost = ScenicCostFunction(scenic_weight=0.8, avoid_highways=True)
    feasible_paths = [
        [edges[1], edges[2]],
        [edges[3], edges[4]],
    ]
    expected = min(
        feasible_paths,
        key=lambda path: sum(cost.calculate(edge) for edge in path),
    )
    actual = ScenicRoutePlanner(graph=graph)._a_star(
        graph.get_node("S"),
        graph.get_node("G"),
        cost_function=cost,
        max_path_minutes=4.0,
    )

    assert actual is not None
    assert [edge.id for edge in actual] == [edge.id for edge in expected]
    assert sum(edge.travel_time_minutes for edge in actual) <= 4.0
    assert sum(cost.calculate(edge) for edge in actual) == pytest.approx(
        sum(cost.calculate(edge) for edge in expected)
    )


def test_weight_one_maximizes_distance_weighted_scenic_score() -> None:
    """Weight one maximizes the normalized distance-weighted scenic score."""
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=42.0, lon=-72.0 + lon))
    graph.add_edge(
        Edge(
            id="direct",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=8.0,
            road_name="direct",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="detour-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=10.0,
            scenic_score=8.5,
            road_name="detour",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="detour-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=8.5,
            road_name="detour",
            speed_limit_kmh=60,
            one_way=True,
        )
    )

    route = ScenicRoutePlanner(graph=graph).find_scenic_route(
        (42.0, -72.0),
        (42.0, -71.98),
        scenic_weight=1.0,
        max_detour_factor=2.0,
    )

    assert route.average_scenic_score == pytest.approx(8.5)
    assert route.estimated_duration_minutes == pytest.approx(20.0)
    assert [segment.road_name for segment in route.segments] == [
        "detour",
        "detour",
    ]


def test_compiled_csr_handles_reverse_traversals_and_hard_highway_filter() -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=42.0, lon=-72.0 + lon))
    graph.add_edge(
        Edge(
            id="A-S",
            start_node_id="A",
            end_node_id="S",
            distance_km=1.0,
            scenic_score=8.0,
            speed_limit_kmh=60,
            one_way=False,
        )
    )
    graph.add_edge(
        Edge(
            id="A-G",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=8.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="motorway",
            start_node_id="S",
            end_node_id="G",
            distance_km=0.1,
            scenic_score=0.0,
            road_type="motorway",
            speed_limit_kmh=120,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.5, avoid_highways=True)

    path = planner._compiled_builtin_path(
        graph.get_node("S"), graph.get_node("G"), cost
    )

    assert path is not None
    assert [edge.id for edge in path] == ["A-S::rev", "A-G"]
    assert [(edge.start_node_id, edge.end_node_id) for edge in path] == [
        ("S", "A"),
        ("A", "G"),
    ]
    assert (
        planner._compiled_builtin_path(graph.get_node("G"), graph.get_node("S"), cost)
        is None
    )


def test_compiled_csr_cache_invalidates_on_graph_stamp_and_cost_signature() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
    edge = Edge(
        id="direct",
        start_node_id="S",
        end_node_id="G",
        distance_km=2.0,
        scenic_score=2.0,
        speed_limit_kmh=60,
        one_way=True,
    )
    graph.add_edge(edge)
    planner = ScenicRoutePlanner(graph=graph)
    cost = ScenicCostFunction(scenic_weight=0.5)

    assert planner._compiled_builtin_path(
        graph.get_node("S"), graph.get_node("G"), cost
    )
    first_topology = planner._csr_topology_cache
    first_data = next(iter(planner._csr_data_cache.values()))

    cost.scenic_weight = 0.8
    assert planner._compiled_builtin_path(
        graph.get_node("S"), graph.get_node("G"), cost
    )
    second_data = next(
        item for item in planner._csr_data_cache.values() if item.signature[0] == 0.8
    )
    assert planner._csr_topology_cache is first_topology
    assert second_data is not first_data

    edge.scenic_score = 9.0
    assert planner._compiled_builtin_path(
        graph.get_node("S"), graph.get_node("G"), cost
    )
    assert planner._csr_topology_cache is not first_topology


def test_scenic_shortcut_skips_resource_solver_when_unconstrained_is_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(
            id="direct",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="scenic-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)

    def fail_resource_solver(*args: object, **kwargs: object) -> None:
        raise AssertionError("feasible unconstrained optimum should shortcut")

    monkeypatch.setattr(planner, "_resource_constrained_path", fail_resource_solver)
    route = planner.find_scenic_route(
        (42.0, -72.0),
        (42.02, -72.0),
        scenic_weight=1.0,
        max_detour_factor=1.3,
    )

    assert [segment.road_name for segment in route.segments] == [None, None]
    assert route.estimated_duration_minutes == pytest.approx(12.0)


def test_scenic_factor_one_uses_shortest_duration_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 42.0), ("A", 42.01), ("G", 42.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=-72.0))
    graph.add_edge(
        Edge(
            id="direct",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="over-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="over-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=6.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph=graph)
    original_solver = planner._resource_constrained_path
    calls = 0

    def count_resource_solver(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(planner, "_resource_constrained_path", count_resource_solver)
    route = planner.find_scenic_route(
        (42.0, -72.0),
        (42.02, -72.0),
        scenic_weight=1.0,
        max_detour_factor=1.0,
    )

    assert calls == 0
    assert [segment.road_name for segment in route.segments] == [None]
    assert route.estimated_duration_minutes == pytest.approx(10.0)


def test_shared_compiled_caches_isolate_graphs_and_mutations() -> None:
    def make_graph(edge_id: str, distance_km: float) -> RoadGraph:
        graph = RoadGraph()
        graph.add_node(Node(id="S", lat=42.0, lon=-72.0))
        graph.add_node(Node(id="G", lat=42.1, lon=-72.0))
        graph.add_edge(
            Edge(
                id=edge_id,
                start_node_id="S",
                end_node_id="G",
                distance_km=distance_km,
                scenic_score=0.0,
                speed_limit_kmh=60,
                one_way=True,
            )
        )
        return graph

    first_graph = make_graph("first", 1.0)
    first_planner = ScenicRoutePlanner(graph=first_graph)
    first_route = first_planner.find_fastest_route((42.0, -72.0), (42.1, -72.0))
    assert first_route.total_distance_km == pytest.approx(1.0)

    second_graph = make_graph("second", 10.0)
    second_planner = ScenicRoutePlanner(graph=second_graph)
    second_route = second_planner.find_fastest_route((42.0, -72.0), (42.1, -72.0))
    assert second_route.total_distance_km == pytest.approx(10.0)
    assert [segment.road_name for segment in second_route.segments] == [None]

    first_graph.edges["first"].distance_km = 2.0
    refreshed_planner = ScenicRoutePlanner(graph=first_graph)
    refreshed_route = refreshed_planner.find_fastest_route((42.0, -72.0), (42.1, -72.0))
    assert refreshed_route.total_distance_km == pytest.approx(2.0)
