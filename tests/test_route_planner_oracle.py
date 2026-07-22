from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import math
from threading import Lock
import time
import tracemalloc

import pytest
from src.route_planner.cancellation import RoutingTimeout

from src.route_planner.cost import evaluate_path, resolve_routing_policy
from src.route_planner.graph import Edge, EndpointRoadGraph, Node, RoadGraph
from src.route_planner.planner import _FrontierLabel, ScenicRoutePlanner

_SEARCH_DIAGNOSTIC_KEYS = {
    "time_limit_seconds",
    "labels_generated",
    "labels_expanded",
    "labels_pruned",
    "max_frontier_size",
    "remaining_frontier_size",
    "deadline_reached",
    "elapsed_ms",
    "mode",
}


def _tradeoff_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, lat in (
        ("S", 0.0),
        ("M", 0.01),
        ("A", 0.02),
        ("G", 0.03),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    edges = (
        # True fastest baseline: six minutes, score zero.
        Edge(
            id="fast",
            start_node_id="S",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=0.0,
            speed_limit_kmh=100,
            one_way=True,
        ),
        # Intermediate: twelve minutes, score six.
        Edge(
            id="mid-1",
            start_node_id="S",
            end_node_id="M",
            distance_km=4.0,
            scenic_score=6.0,
            speed_limit_kmh=40,
            one_way=True,
        ),
        Edge(
            id="mid-2",
            start_node_id="M",
            end_node_id="G",
            distance_km=4.0,
            scenic_score=6.0,
            speed_limit_kmh=40,
            one_way=True,
        ),
        # Scenic detour: twenty minutes, score ten.
        Edge(
            id="scenic-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=5.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
        Edge(
            id="scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=5.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
        # A tempting cycle must never become a feasible route prefix.
        Edge(
            id="cycle",
            start_node_id="A",
            end_node_id="S",
            distance_km=0.01,
            scenic_score=10.0,
            speed_limit_kmh=1,
            one_way=True,
        ),
    )
    for edge in edges:
        graph.add_edge(edge)
    return graph


def test_large_graph_fastest_uses_reverse_same_edge_access() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=0.0, lon=0.0))
    graph.add_node(Node(id="B", lat=0.0, lon=1.0))
    graph.add_edge(
        Edge(
            id="two-way",
            start_node_id="A",
            end_node_id="B",
            distance_km=10.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=False,
        )
    )
    for index in range(2_001):
        graph.add_node(Node(id=f"isolated-{index}", lat=10.0 + index, lon=10.0))

    planner = ScenicRoutePlanner(graph)
    route = planner.find_fastest_route((0.0, 0.8), (0.0, 0.2))

    assert route.edge_ids == ("two-way",)
    assert route.segments[0].direction == "reverse"
    assert route.estimated_duration_minutes == pytest.approx(6.0)
    assert route.waypoints == [(0.0, 0.8), (0.0, 0.2)]




def test_large_graph_fastest_checks_all_tied_edge_projections() -> None:
    graph = RoadGraph()
    for node_id in ("A", "B", "C", "D"):
        graph.add_node(Node(id=node_id, lat=0.0, lon=0.0 if node_id in ("A", "C") else 1.0))
    graph.add_edge(
        Edge(
            id="a-bad",
            start_node_id="A",
            end_node_id="B",
            distance_km=10.0,
            scenic_score=1.0,
            speed_limit_kmh=10,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="z-good",
            start_node_id="C",
            end_node_id="D",
            distance_km=10.0,
            scenic_score=1.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    for index in range(2_001):
        graph.add_node(Node(id=f"isolated-{index}", lat=10.0 + index, lon=10.0))

    planner = ScenicRoutePlanner(graph)
    route = planner.find_fastest_route((0.0, 0.2), (0.0, 0.8))

    assert route.edge_ids == ("z-good",)
    assert route.estimated_duration_minutes == pytest.approx(3.6)


def test_scenic_endpoint_overlay_keeps_all_tied_boundary_projections() -> None:
    graph = RoadGraph()
    for node_id in ("A", "B", "C", "D"):
        graph.add_node(
            Node(
                id=node_id,
                lat=0.0 if node_id in ("A", "B") else 0.002,
                lon=0.0 if node_id in ("A", "C") else 1.0,
            )
        )
    graph.add_edge(
        Edge(
            id="a-slow",
            start_node_id="A",
            end_node_id="B",
            distance_km=10.0,
            scenic_score=1.0,
            speed_limit_kmh=10,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="z-fast",
            start_node_id="C",
            end_node_id="D",
            distance_km=10.0,
            scenic_score=1.0,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    for index in range(2_001):
        graph.add_node(
            Node(id=f"isolated-{index}", lat=10.0 + index, lon=10.0)
        )

    coordinate_epoch = Node._coordinate_mutation_epoch
    planner = ScenicRoutePlanner(graph)
    scenic = planner.find_scenic_route(
        (0.001, 0.0),
        (0.001, 1.0),
        q=0.0,
        kappa=1.0,
        scenic_priority=True,
    )
    baseline = planner.find_fastest_route((0.001, 0.0), (0.001, 1.0))

    assert scenic.edge_ids == baseline.edge_ids == ("z-fast",)
    assert scenic.waypoints[0] == pytest.approx((0.002, 0.0))
    assert scenic.waypoints[-1] == pytest.approx((0.002, 1.0))
    assert Node._coordinate_mutation_epoch == coordinate_epoch
    assert scenic.estimated_duration_minutes == pytest.approx(
        baseline.estimated_duration_minutes
    )
    assert scenic.estimated_duration_minutes <= scenic.duration_cap_minutes


def _endpoint_overlay_graph(extra_nodes: int = 0) -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=0.0, lon=0.0))
    graph.add_node(Node(id="B", lat=0.0, lon=1.0))
    graph.add_edge(
        Edge(
            id="base-edge",
            start_node_id="A",
            end_node_id="B",
            distance_km=10.0,
            scenic_score=7.0,
            speed_limit_kmh=60,
            one_way=False,
        )
    )
    for index in range(extra_nodes):
        graph.add_node(
            Node(id=f"unused-{index}", lat=20.0 + index, lon=20.0)
        )
    graph.find_nearest_edge_positions_with_distance(0.0, 0.25)
    return graph


def test_endpoint_overlay_structurally_shares_frozen_base_graph() -> None:
    graph = _endpoint_overlay_graph(extra_nodes=32)
    planner = ScenicRoutePlanner(graph)
    nodes_id = id(graph.nodes)
    edges_id = id(graph.edges)
    adjacency_id = id(graph.adjacency)
    adjacency_lists = {
        node_id: id(traversals)
        for node_id, traversals in graph.adjacency.items()
    }
    stamp = graph._heuristic_cache_stamp()

    overlay = planner._endpoint_graph((0.0, 0.25), (0.0, 0.75))

    assert isinstance(overlay, EndpointRoadGraph)
    assert overlay.base_graph is graph
    assert overlay.nodes["A"] is graph.nodes["A"]
    assert overlay.edges["base-edge"] is graph.edges["base-edge"]
    assert id(graph.nodes) == nodes_id
    assert id(graph.edges) == edges_id
    assert id(graph.adjacency) == adjacency_id
    assert {
        node_id: id(traversals)
        for node_id, traversals in graph.adjacency.items()
    } == adjacency_lists
    assert graph._heuristic_cache_stamp() == stamp
    assert not hasattr(overlay.adjacency["A"], "append")
    with pytest.raises(RuntimeError, match="frozen"):
        overlay.add_node(Node(id="late", lat=1.0, lon=1.0))
    with pytest.raises(TypeError):
        overlay.nodes["late"] = Node(id="late", lat=1.0, lon=1.0)  # type: ignore[index]


def _endpoint_setup_peak_bytes(graph: RoadGraph) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        overlay = ScenicRoutePlanner(graph)._endpoint_graph(
            (0.0, 0.25), (0.0, 0.75)
        )
        assert isinstance(overlay, EndpointRoadGraph)
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def test_endpoint_overlay_allocation_does_not_scale_with_base_nodes() -> None:
    small_peak = _endpoint_setup_peak_bytes(_endpoint_overlay_graph())
    large_peak = _endpoint_setup_peak_bytes(
        _endpoint_overlay_graph(extra_nodes=10_000)
    )

    assert large_peak <= small_peak + 128 * 1024


def test_scenic_hot_path_uses_structural_endpoint_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_endpoint_overlay_graph())
    original = planner._endpoint_graph
    observed: list[EndpointRoadGraph] = []

    def capture(*args, **kwargs):
        overlay = original(*args, **kwargs)
        observed.append(overlay)
        return overlay

    monkeypatch.setattr(planner, "_endpoint_graph", capture)
    route = planner.find_scenic_route(
        (0.0, 0.25),
        (0.0, 0.75),
        q=0.0,
        kappa=1.0,
    )

    assert route.edge_ids == ("base-edge",)
    assert len(observed) == 1
    assert observed[0].base_graph is planner.graph

def test_large_scenic_and_baseline_reuse_endpoint_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _endpoint_overlay_graph(extra_nodes=2_001)
    planner = ScenicRoutePlanner(graph)
    original = graph.find_nearest_edge_positions_with_distance
    calls = 0

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "find_nearest_edge_positions_with_distance", capture)
    original_search = planner._multi_access_builtin_path
    searches = 0

    def capture_search(*args, **kwargs):
        nonlocal searches
        searches += 1
        return original_search(*args, **kwargs)

    monkeypatch.setattr(planner, "_multi_access_builtin_path", capture_search)
    scenic = planner.find_scenic_route(
        (0.0, 0.25), (0.0, 0.75), q=0.0, kappa=1.0
    )
    baseline = planner.find_fastest_route((0.0, 0.25), (0.0, 0.75))

    assert scenic.edge_ids == baseline.edge_ids == ("base-edge",)
    assert calls == 2
    assert searches == 1


def test_same_planner_serializes_concurrent_endpoint_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _endpoint_overlay_graph()
    planner = ScenicRoutePlanner(graph)
    original = planner._path_to_route
    state_lock = Lock()
    active = 0
    max_active = 0

    def observed_path_to_route(*args, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return original(*args, **kwargs)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(planner, "_path_to_route", observed_path_to_route)
    with ThreadPoolExecutor(max_workers=2) as executor:
        routes = list(
            executor.map(
                lambda _: planner.find_scenic_route(
                    (0.0, 0.25),
                    (0.0, 0.75),
                    q=0.0,
                    kappa=1.0,
                ),
                range(2),
            )
        )

    assert [route.edge_ids for route in routes] == [
        ("base-edge",),
        ("base-edge",),
    ]
    assert max_active == 1
    assert planner.graph is graph

def _route(planner: ScenicRoutePlanner, q: float, kappa: float):
    return planner.find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        q=q,
        kappa=kappa,
    )


def _pairwise_large_fastest_oracle(
    planner: ScenicRoutePlanner,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    avoid_highways: bool = False,
):
    """Test-only copy of the former endpoint-access candidate enumeration."""
    base = planner.graph
    assert base is not None
    excluded = (
        frozenset({"motorway", "trunk", "primary", "secondary", "tertiary"})
        if avoid_highways
        else frozenset()
    )
    planner.cost_function.strict_highways = bool(avoid_highways)
    planner.cost_function.avoid_highways = bool(avoid_highways)
    planner.cost_function.highway_preference = 0.0
    starts, _ = base.find_nearest_edge_positions_with_distance(
        *start, excluded_road_types=excluded
    )
    ends, _ = base.find_nearest_edge_positions_with_distance(
        *end, excluded_road_types=excluded
    )

    def partial(source, edge_id, from_id, to_id, fraction, direction):
        edge = Edge(
            edge_id,
            from_id,
            to_id,
            float(source.distance_km) * max(0.0, min(1.0, float(fraction))),
            float(source.scenic_score),
            road_name=source.road_name,
            road_type=source.road_type,
            speed_limit_kmh=source.speed_limit_kmh,
            one_way=True,
        )
        edge.canonical_edge_id = str(source.id)
        edge.direction = direction
        edge.source_fraction = float(fraction)
        return edge

    best = None
    for start_index, start_projection in enumerate(starts):
        prefixes = [
            (
                str(start_projection.edge.end_node_id),
                partial(
                    start_projection.edge,
                    f"oracle-start:{start_index}:forward",
                    "__oracle-start__",
                    str(start_projection.edge.end_node_id),
                    1.0 - float(start_projection.fraction),
                    "forward",
                ),
            )
        ]
        if not start_projection.edge.one_way:
            prefixes.append(
                (
                    str(start_projection.edge.start_node_id),
                    partial(
                        start_projection.edge,
                        f"oracle-start:{start_index}:reverse",
                        "__oracle-start__",
                        str(start_projection.edge.start_node_id),
                        float(start_projection.fraction),
                        "reverse",
                    ),
                )
            )
        for end_index, end_projection in enumerate(ends):
            suffixes = [
                (
                    str(end_projection.edge.start_node_id),
                    partial(
                        end_projection.edge,
                        f"oracle-end:{end_index}:forward",
                        str(end_projection.edge.start_node_id),
                        "__oracle-end__",
                        float(end_projection.fraction),
                        "forward",
                    ),
                )
            ]
            if not end_projection.edge.one_way:
                suffixes.append(
                    (
                        str(end_projection.edge.end_node_id),
                        partial(
                            end_projection.edge,
                            f"oracle-end:{end_index}:reverse",
                            str(end_projection.edge.end_node_id),
                            "__oracle-end__",
                            1.0 - float(end_projection.fraction),
                            "reverse",
                        ),
                    )
                )
            for prefix_node, prefix in prefixes:
                for suffix_node, suffix in suffixes:
                    middle = planner._cached_fastest_edges(
                        base.get_node(prefix_node),
                        base.get_node(suffix_node),
                        avoid_highways,
                        0.0,
                    )
                    if middle is None:
                        continue
                    candidate = [prefix, *middle, suffix]
                    duration = planner._path_duration_minutes(candidate)
                    if best is None or duration < best[0]:
                        best = (duration, candidate, start_projection, end_projection)
            if str(start_projection.edge.id) != str(end_projection.edge.id):
                continue
            start_fraction = float(start_projection.fraction)
            end_fraction = float(end_projection.fraction)
            if start_fraction <= end_fraction:
                direct = partial(
                    start_projection.edge,
                    f"oracle-direct:{start_index}:{end_index}:forward",
                    "__oracle-start__",
                    "__oracle-end__",
                    end_fraction - start_fraction,
                    "forward",
                )
                duration = planner._path_duration_minutes([direct])
                if best is None or duration < best[0]:
                    best = (duration, [direct], start_projection, end_projection)
            if not start_projection.edge.one_way and start_fraction >= end_fraction:
                direct = partial(
                    start_projection.edge,
                    f"oracle-direct:{start_index}:{end_index}:reverse",
                    "__oracle-start__",
                    "__oracle-end__",
                    start_fraction - end_fraction,
                    "reverse",
                )
                duration = planner._path_duration_minutes([direct])
                if best is None or duration < best[0]:
                    best = (duration, [direct], start_projection, end_projection)
    return best

def _edge_direction(edge: Edge) -> str:
    direction = getattr(edge, "direction", None)
    if direction is not None:
        return str(direction)
    return "reverse" if getattr(edge, "_is_reverse_traversal", False) else "forward"


def _route_signature(route):
    return (
        route.estimated_duration_minutes,
        route.edge_ids,
        route.traversal_ids,
        tuple(segment.direction for segment in route.segments),
    )

def test_multi_access_matches_pairwise_oracle_for_tied_mixed_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for index in range(6):
        graph.add_node(Node(str(index), index * 0.01, 0.0))
    graph.add_edge(
        Edge("a-fast", "0", "1", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    graph.add_edge(
        Edge("z-slow", "0", "1", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    graph.add_edge(
        Edge("one-way", "1", "2", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("two-way", "2", "3", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    graph.add_edge(
        Edge("out", "3", "4", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("last", "4", "5", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_COLLAPSED_ACCESS_NODE_THRESHOLD", 0)
    start = (0.002, 0.0)
    end = (0.048, 0.0)
    oracle = _pairwise_large_fastest_oracle(planner, start, end)
    assert oracle is not None
    actual = planner.find_fastest_route(start, end)
    expected_ids = tuple(
        str(getattr(edge, "canonical_edge_id", edge.id)) for edge in oracle[1]
    )
    expected_directions = tuple(
        _edge_direction(edge) for edge in oracle[1]
    )
    assert actual.estimated_duration_minutes == pytest.approx(oracle[0])
    assert actual.edge_ids == expected_ids
    assert tuple(segment.direction for segment in actual.segments) == expected_directions


def test_multi_access_matches_pairwise_oracle_with_highway_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for index in range(4):
        graph.add_node(Node(str(index), index * 0.01, 0.0))
    graph.add_edge(
        Edge("highway", "0", "3", 1.0, 0.0, road_type="motorway", speed_limit_kmh=120)
    )
    for start_id, end_id, edge_id in (("0", "1", "r1"), ("1", "2", "r2"), ("2", "3", "r3")):
        graph.add_edge(
            Edge(edge_id, start_id, end_id, 1.0, 0.0, road_type="residential", speed_limit_kmh=30)
        )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    start = (0.001, 0.0)
    end = (0.029, 0.0)
    oracle = _pairwise_large_fastest_oracle(
        planner, start, end, avoid_highways=True
    )
    assert oracle is not None
    actual = planner.find_fastest_route(start, end, avoid_highways=True)
    assert actual.edge_ids == tuple(
        str(getattr(edge, "canonical_edge_id", edge.id)) for edge in oracle[1]
    )
    assert "highway" not in actual.edge_ids


def test_multi_access_preserves_unreachable_access_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    graph.add_node(Node("0", 0.0, 0.0))
    graph.add_node(Node("1", 0.01, 0.0))
    graph.add_node(Node("2", 0.02, 0.0))
    graph.add_node(Node("3", 0.03, 0.0))
    graph.add_edge(Edge("left", "0", "1", 1.0, 0.0, speed_limit_kmh=60))
    graph.add_edge(Edge("right", "2", "3", 1.0, 0.0, speed_limit_kmh=60))
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    assert _pairwise_large_fastest_oracle(planner, (0.002, 0.0), (0.028, 0.0)) is None
    with pytest.raises(ValueError, match="No route found"):
        planner.find_fastest_route((0.002, 0.0), (0.028, 0.0))


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((0.0, 0.0), (0.01, 0.0)),
        ((0.01, 0.0), (0.0, 0.0)),
        ((0.005, 0.0), (0.005, 0.0)),
    ],
)
def test_multi_access_boundary_and_zero_length_routes(
    monkeypatch: pytest.MonkeyPatch,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    graph = RoadGraph()
    graph.add_node(Node("A", 0.0, 0.0))
    graph.add_node(Node("B", 0.01, 0.0))
    graph.add_edge(
        Edge("boundary", "A", "B", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_COLLAPSED_ACCESS_NODE_THRESHOLD", 0)
    oracle = _pairwise_large_fastest_oracle(planner, start, end)
    assert oracle is not None
    route = planner.find_fastest_route(start, end)
    assert route.estimated_duration_minutes == pytest.approx(oracle[0])
    if oracle[0] == 0.0:
        assert route.segments
        return
    oracle_edges = [edge for edge in oracle[1] if edge.distance_km > 1e-12]
    expected = [
        (
            str(getattr(edge, "canonical_edge_id", edge.id)),
            _edge_direction(edge),
        )
        for edge in oracle_edges
    ]
    actual = [
        (edge_id, segment.direction)
        for edge_id, segment in zip(route.edge_ids, route.segments)
        if segment.duration_minutes > 1e-12
    ]
    assert actual == expected

def test_large_graph_fastest_route_uses_one_ranked_access_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for index in range(5):
        graph.add_node(Node(str(index), index * 0.01, 0.0))
        if index:
            graph.add_edge(
                Edge(
                    f"edge-{index - 1}",
                    str(index - 1),
                    str(index),
                    1.0,
                    0.0,
                    speed_limit_kmh=60.0,
                    one_way=False,
                )
            )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    calls = 0
    original = planner._multi_access_builtin_path

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "_multi_access_builtin_path", counted)
    route = planner.find_fastest_route((0.002, 0.0), (0.038, 0.0))
    assert calls == 1
    assert route.edge_ids == ("edge-0", "edge-1", "edge-2", "edge-3")
    assert route.traversal_ids == (
        "0:forward:edge-0",
        "1:forward:edge-1",
        "2:forward:edge-2",
        "3:forward:edge-3",
    )


def test_large_graph_multi_access_preserves_reverse_direct_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    graph.add_node(Node("A", 0.0, 0.0))
    graph.add_node(Node("B", 0.01, 0.0))
    graph.add_edge(
        Edge(
            "two-way",
            "A",
            "B",
            10.0,
            0.0,
            speed_limit_kmh=60.0,
            one_way=False,
        )
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    route = planner.find_fastest_route((0.008, 0.0), (0.002, 0.0))
    assert route.edge_ids == ("two-way",)
    assert route.traversal_ids == ("0:reverse:two-way",)
    assert route.segments[0].direction == "reverse"
def test_large_graph_tie_breaks_middle_by_canonical_edge_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("S", 0.0, 0.0),
        ("A", 0.01, 0.0),
        ("B", 0.02, 0.01),
        ("C", 0.02, -0.01),
        ("D", 0.03, 0.0),
        ("G", 0.04, 0.0),
    ):
        graph.add_node(Node(node_id, lat, lon))
    for index in range(25):
        graph.add_node(Node(f"tie-isolated-{index}", 30.0 + index, 30.0))
    graph.add_edge(
        Edge("start", "S", "A", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("z-diamond", "A", "B", 1.5, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("a-diamond", "A", "C", 1.5, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("z-merge", "B", "D", 1.5, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("a-merge", "C", "D", 1.5, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("finish", "D", "G", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)

    route = planner.find_fastest_route((0.005, 0.0), (0.035, 0.0))

    assert route.edge_ids == ("start", "a-diamond", "a-merge", "finish")


def test_collapsed_ties_rank_by_canonical_edge_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("S", 0.0, 0.0),
        ("A", 0.01, 0.0),
        ("B", 0.02, 0.01),
        ("C", 0.02, -0.01),
        ("D", 0.03, 0.0),
        ("G", 0.04, 0.0),
    ):
        graph.add_node(Node(node_id, lat, lon))
    graph.add_edge(
        Edge("start", "S", "A", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    z_diamond = Edge(
        "z-storage", "A", "B", 1.5, 0.0, speed_limit_kmh=60, one_way=True
    )
    z_diamond.canonical_edge_id = "a-canonical"
    graph.add_edge(z_diamond)
    a_diamond = Edge(
        "a-storage", "A", "C", 1.5, 0.0, speed_limit_kmh=60, one_way=True
    )
    a_diamond.canonical_edge_id = "z-canonical"
    graph.add_edge(a_diamond)
    z_merge = Edge(
        "z-merge-storage", "B", "D", 1.5, 0.0, speed_limit_kmh=60, one_way=True
    )
    z_merge.canonical_edge_id = "a-merge-canonical"
    graph.add_edge(z_merge)
    a_merge = Edge(
        "a-merge-storage", "C", "D", 1.5, 0.0, speed_limit_kmh=60, one_way=True
    )
    a_merge.canonical_edge_id = "z-merge-canonical"
    graph.add_edge(a_merge)
    graph.add_edge(
        Edge("finish", "D", "G", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_COLLAPSED_ACCESS_NODE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)

    route = planner.find_fastest_route((0.005, 0.0), (0.035, 0.0))

    assert route.edge_ids == (
        "start",
        "a-canonical",
        "a-merge-canonical",
        "finish",
    )
def test_forward_large_access_ties_use_canonical_edge_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("S", 0.0, 0.0),
        ("X", 0.01, 0.0),
        ("A", 0.02, 0.0),
        ("B", 0.02, 0.0),
        ("G", 0.03, 0.0),
    ):
        graph.add_node(Node(node_id, lat, lon))
    graph.add_edge(
        Edge("start", "S", "X", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    z_storage = Edge(
        "z-storage", "X", "A", 1.0, 0.0, speed_limit_kmh=60, one_way=True
    )
    z_storage.canonical_edge_id = "a-canonical"
    graph.add_edge(z_storage)
    a_storage = Edge(
        "a-storage", "X", "A", 1.0, 0.0, speed_limit_kmh=60, one_way=True
    )
    a_storage.canonical_edge_id = "z-canonical"
    graph.add_edge(a_storage)
    graph.add_edge(
        Edge("x-b", "X", "B", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("a-g", "A", "G", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    graph.add_edge(
        Edge("b-g", "B", "G", 1.0, 0.0, speed_limit_kmh=60, one_way=False)
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)

    route = planner.find_fastest_route((0.005, 0.0), (0.025, 0.0))

    assert route.edge_ids == ("start", "a-canonical", "a-g")


def test_large_graph_tie_breaks_reversed_parallel_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.03)):
        graph.add_node(Node(node_id, lat, 0.0))
    for index in range(25):
        graph.add_node(Node(f"parallel-isolated-{index}", 30.0 + index, 30.0))
    graph.add_edge(
        Edge("start", "S", "A", 1.0, 0.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge(
            "z-parallel",
            "A",
            "G",
            2.0,
            0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "a-parallel",
            "A",
            "G",
            2.0,
            0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)

    route = planner.find_fastest_route((0.005, 0.0), (0.02, 0.0))

    assert route.edge_ids == ("start", "a-parallel")

def test_oracle_fastest_and_scenic_differ_and_detour_unlocks() -> None:
    planner = ScenicRoutePlanner(_tradeoff_graph())
    fastest = _route(planner, q=0.0, kappa=4.0)
    scenic = _route(planner, q=1.0, kappa=4.0)

    assert fastest.edge_ids == ("fast",)
    assert scenic.edge_ids == ("scenic-1", "scenic-2")
    assert scenic.actual_duration_ratio == pytest.approx(20.0 / 6.0)
    assert scenic.applied_max_detour_factor == pytest.approx(4.0)
    assert scenic.exactness_status == "exact"
    assert scenic.optimality_gap is None


def test_oracle_q_zero_is_exact_fastest_and_q_one_is_scenic_optimum() -> None:
    planner = ScenicRoutePlanner(_tradeoff_graph())
    q0 = _route(planner, q=0.0, kappa=1.0)
    q1 = _route(planner, q=1.0, kappa=4.0)

    assert q0.edge_ids == ("fast",)
    assert q0.objective_value == pytest.approx(1.0)
    assert q1.average_scenic_score == pytest.approx(10.0)
    assert q1.normalized_scenic_score == pytest.approx(1.0)
    assert q1.objective_value == pytest.approx(1.0)


def test_resolved_policy_additive_preference_preserves_legacy_objective() -> None:
    edge = Edge(
        id="h",
        start_node_id="S",
        end_node_id="G",
        distance_km=2.0,
        scenic_score=5.0,
        road_type="motorway",
        speed_limit_kmh=60,
        one_way=True,
    )
    legacy_policy = resolve_routing_policy(scenic_weight=0.5, kappa=2.0)
    preferred_policy = resolve_routing_policy(
        scenic_weight=0.5, kappa=2.0, highway_preference=0.25
    )
    legacy = evaluate_path(
        [edge],
        q=0.5,
        kappa=2.0,
        fastest_duration_minutes=edge.travel_time_minutes,
        policy=legacy_policy,
    )
    preferred = evaluate_path(
        [edge],
        q=0.5,
        kappa=2.0,
        fastest_duration_minutes=edge.travel_time_minutes,
        policy=preferred_policy,
    )
    assert legacy.objective == pytest.approx(0.75)
    assert preferred.highway_cost == pytest.approx(0.25)
    assert preferred.objective == pytest.approx(legacy.objective - 0.25)
    assert preferred.policy != legacy.policy


def test_oracle_intermediate_q_selects_tradeoff_path() -> None:
    planner = ScenicRoutePlanner(_tradeoff_graph())
    route = _route(planner, q=0.5, kappa=4.0)

    assert route.edge_ids == ("mid-1", "mid-2")
    assert route.average_scenic_score == pytest.approx(6.0)
    assert route.objective_value > 0.6
    assert route.objective_value < 0.7


def test_oracle_simple_paths_do_not_exploit_cycles() -> None:
    route = _route(ScenicRoutePlanner(_tradeoff_graph()), q=1.0, kappa=20.0)
    node_ids = [route.segments[0].start[0]] + [segment.end[0] for segment in route.segments]
    assert len(node_ids) == len(set(node_ids))
    assert route.edge_ids == ("scenic-1", "scenic-2")


@pytest.mark.parametrize(
    "road_type", ["motorway_link", "trunk_link", "primary", "primary_link"]
)
def test_oracle_highway_filter_is_hard_for_baseline_and_scenic(
    road_type: str,
) -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    graph.add_edge(
        Edge(
            id=road_type,
            start_node_id="S",
            end_node_id="G",
            distance_km=5.0,
            scenic_score=0.0,
            road_type=road_type,
            speed_limit_kmh=120,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="secondary-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=5.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="secondary-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=5.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)

    assert _route_for_graph(planner, q=0.0, kappa=4.0, avoid_highways=True).edge_ids == (
        "secondary-1",
        "secondary-2",
    )
    scenic = planner.find_scenic_route(
        (0.0, 0.0), (0.02, 0.0), q=1.0, kappa=4.0, avoid_highways=True
    )
    assert scenic.edge_ids == ("secondary-1", "secondary-2")
    assert scenic.highway_count == 0


def test_primary_shortcut_is_excluded_by_checked_filter() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("R", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    graph.add_edge(
        Edge(
            id="primary-shortcut",
            start_node_id="S",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=0.0,
            road_type="primary",
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="secondary-detour",
            start_node_id="S",
            end_node_id="R",
            distance_km=1.0,
            scenic_score=8.0,
            road_type="secondary",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="residential-detour",
            start_node_id="R",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=9.0,
            road_type="residential",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)

    assert planner.find_fastest_route(
        (0.0, 0.0), (0.02, 0.0), avoid_highways=False
    ).edge_ids == ("primary-shortcut",)
    assert planner.find_scenic_route(
        (0.0, 0.0),
        (0.02, 0.0),
        q=0.0,
        kappa=1.0,
        avoid_highways=False,
    ).edge_ids == ("primary-shortcut",)

    fastest = planner.find_fastest_route(
        (0.0, 0.0), (0.02, 0.0), avoid_highways=True
    )
    scenic = planner.find_scenic_route(
        (0.0, 0.0),
        (0.02, 0.0),
        q=1.0,
        kappa=4.0,
        avoid_highways=True,
    )
    assert fastest.edge_ids == ("secondary-detour", "residential-detour")
    assert scenic.edge_ids == ("secondary-detour", "residential-detour")
    assert all(segment.road_type != "primary" for segment in scenic.segments)
    assert scenic.highway_count == 0

def test_large_search_fallback_preserves_highway_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("R", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    graph.add_edge(
        Edge(
            id="primary-shortcut",
            start_node_id="S",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=0.0,
            road_type="primary",
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="residential-detour",
            start_node_id="S",
            end_node_id="R",
            distance_km=1.0,
            scenic_score=0.0,
            road_type="residential",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="residential-finish",
            start_node_id="R",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=0.0,
            road_type="residential",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(
        planner,
        "_large_graph_multi_access_path",
        lambda *args, **kwargs: None,
    )

    route = planner.find_fastest_route(
        (0.0, 0.0), (0.02, 0.0), avoid_highways=True
    )

    assert route.edge_ids == ("residential-detour", "residential-finish")


@pytest.mark.parametrize("road_type", ["motorway", "primary"])
def test_avoid_highways_rejects_highway_only_connectivity_but_keeps_residential_path(
    road_type: str,
) -> None:
    highway_only = RoadGraph()
    highway_only.add_node(Node(id="S", lat=0.0, lon=0.0))
    highway_only.add_node(Node(id="G", lat=0.02, lon=0.0))
    highway_only.add_edge(
        Edge(
            id="only-highway",
            start_node_id="S",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=0.0,
            road_type=road_type,
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    highway_planner = ScenicRoutePlanner(highway_only)
    with pytest.raises(ValueError, match="No route found between"):
        highway_planner.find_fastest_route((0.0, 0.0), (0.02, 0.0), avoid_highways=True)
    with pytest.raises(ValueError, match="No route found between"):
        highway_planner.find_scenic_route(
            (0.0, 0.0),
            (0.02, 0.0),
            scenic_weight=1.0,
            avoid_highways=True,
        )

    connected = RoadGraph()
    for node_id, lat in (("S", 0.0), ("R", 0.01), ("G", 0.02)):
        connected.add_node(Node(id=node_id, lat=lat, lon=0.0))
    connected.add_edge(
        Edge(
            id="highway",
            start_node_id="S",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=0.0,
            road_type="motorway",
            speed_limit_kmh=100,
            one_way=True,
        )
    )
    connected.add_edge(
        Edge(
            id="residential",
            start_node_id="S",
            end_node_id="R",
            distance_km=1.0,
            scenic_score=8.0,
            road_type="residential",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    connected.add_edge(
        Edge(
            id="service",
            start_node_id="R",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=9.0,
            road_type="service",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(connected)

    fastest = planner.find_fastest_route((0.0, 0.0), (0.02, 0.0), avoid_highways=True)
    scenic = planner.find_scenic_route(
        (0.0, 0.0),
        (0.02, 0.0),
        scenic_weight=1.0,
        max_detour_factor=4.0,
        avoid_highways=True,
    )
    assert fastest.edge_ids == ("residential", "service")
    assert scenic.edge_ids == ("residential", "service")
    assert scenic.highway_count == 0
    assert all(
        segment.road_type not in {"motorway", "trunk"} for segment in scenic.segments
    )

def _route_for_graph(
    planner: ScenicRoutePlanner,
    *,
    q: float,
    kappa: float,
    avoid_highways: bool,
):
    return planner.find_scenic_route(
        (0.0, 0.0), (0.02, 0.0), q=q, kappa=kappa, avoid_highways=avoid_highways
    )


def test_oracle_cap_accepts_exact_boundary_and_rejects_over_cap() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
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
            id="boundary-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=7.5,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="boundary-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=7.5,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)
    exact = planner.find_scenic_route(
        (0.0, 0.0), (0.02, 0.0), q=1.0, kappa=1.5
    )
    over = planner.find_scenic_route(
        (0.0, 0.0), (0.02, 0.0), q=1.0, kappa=1.4
    )

    assert exact.edge_ids == ("boundary-1", "boundary-2")
    assert exact.estimated_duration_minutes == pytest.approx(
        exact.duration_cap_minutes
    )
    assert over.edge_ids == ("fast",)
    assert over.estimated_duration_minutes <= over.duration_cap_minutes
def test_production_frontier_reports_exact_completion_certificate() -> None:
    graph = _tradeoff_graph()
    for index in range(25):
        graph.add_node(
            Node(id=f"isolated-{index}", lat=10.0 + index, lon=10.0)
        )

    route = _route(ScenicRoutePlanner(graph), q=0.5, kappa=4.0)

    assert route.exact is True
    assert route.exactness_status == "exact"
    assert route.certified_upper_bound == pytest.approx(route.objective_value)
    assert route.optimality_gap == pytest.approx(0.0)


def test_production_frontier_keeps_lower_highway_cost_label() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("X", 0.01), ("G", 0.03)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    graph.add_edge(
        Edge(
            id="high-prefix",
            start_node_id="S",
            end_node_id="X",
            distance_km=1.0,
            scenic_score=10.0,
            road_type="motorway",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="low-prefix",
            start_node_id="S",
            end_node_id="X",
            distance_km=2.0,
            scenic_score=5.0,
            road_type="residential",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="suffix",
            start_node_id="X",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    for index in range(25):
        graph.add_node(Node(id=f"isolated-{index}", lat=10.0 + index, lon=10.0))
    route = ScenicRoutePlanner(graph).find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        q=0.5,
        kappa=4.0,
        highway_preference=1.0,
    )
    assert route.exactness_status == "exact"
    assert route.edge_ids == ("low-prefix", "suffix")
    assert route.certified_upper_bound == pytest.approx(route.objective_value)
def test_reverse_collision_uses_canonical_and_positional_traversal_identity() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    graph.add_edge(
        Edge(
            id="foo",
            start_node_id="A",
            end_node_id="S",
            distance_km=1.0,
            scenic_score=1.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="foo::rev",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=9.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)
    reverse = planner._edge_from_reverse_index("foo", True)
    tail = graph.edges["foo::rev"]
    route = planner._path_to_route([reverse, tail])

    assert route.edge_ids == ("foo", "foo::rev")
    assert route.traversal_ids == (
        "0:reverse:foo",
        "1:forward:foo::rev",
    )
    assert [segment.edge_id for segment in route.segments] == [
        "foo",
        "foo::rev",
    ]
    assert [segment.direction for segment in route.segments] == [
        "reverse",
        "forward",
    ]
    assert [segment.traversal_id for segment in route.segments] == list(
        route.traversal_ids
    )
    assert tuple(edge_id for edge_id, _ in route.score_run) == route.traversal_ids
    assert [segment.scenic_score for segment in route.segments] == [
        pytest.approx(1.0),
        pytest.approx(9.0),
    ]
    assert route.average_scenic_score == pytest.approx(5.0)


def test_production_near_tie_over_cap_is_rejected_with_strict_tolerance() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
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
            id="near-1",
            start_node_id="S",
            end_node_id="A",
            distance_km=5.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="near-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=5.00000000005,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    # Force production mode while keeping all unrelated edges disconnected.
    for index in range(25):
        graph.add_node(Node(id=f"orphan-{index}", lat=20.0 + index, lon=20.0))
    for index in range(190):
        graph.add_edge(
            Edge(
                id=f"orphan-edge-{index}",
                start_node_id=f"orphan-{index % 25}",
                end_node_id=f"orphan-{(index + 1) % 25}",
                distance_km=1.0,
                scenic_score=10.0,
                speed_limit_kmh=60,
                one_way=True,
            )
        )

    route = ScenicRoutePlanner(graph).find_scenic_route(
        (0.0, 0.0), (0.02, 0.0), q=1.0, kappa=1.0
    )

    assert route.edge_ids == ("fast",)
    assert route.estimated_duration_minutes <= route.duration_cap_minutes
def test_reverse_identity_is_stable_across_compiled_threshold() -> None:
    def make_graph(extra: bool) -> RoadGraph:
        graph = RoadGraph()
        for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
            graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
        graph.add_edge(
            Edge(
                id="foo",
                start_node_id="A",
                end_node_id="S",
                distance_km=1.0,
                scenic_score=1.0,
                speed_limit_kmh=60,
                one_way=False,
            )
        )
        graph.add_edge(
            Edge(
                id="foo::rev",
                start_node_id="A",
                end_node_id="G",
                distance_km=1.0,
                scenic_score=9.0,
                speed_limit_kmh=60,
                one_way=True,
            )
        )
        if extra:
            for index in range(25):
                graph.add_node(
                    Node(id=f"threshold-{index}", lat=20.0 + index, lon=20.0)
                )
            for index in range(190):
                graph.add_edge(
                    Edge(
                        id=f"threshold-edge-{index}",
                        start_node_id=f"threshold-{index % 25}",
                        end_node_id=f"threshold-{(index + 1) % 25}",
                        distance_km=1.0,
                        scenic_score=5.0,
                        speed_limit_kmh=60,
                        one_way=True,
                    )
                )
        return graph

    bounded = ScenicRoutePlanner(make_graph(False)).find_fastest_route(
        (0.0, 0.0), (0.02, 0.0)
    )
    compiled = ScenicRoutePlanner(make_graph(True)).find_fastest_route(
        (0.0, 0.0), (0.02, 0.0)
    )

    assert compiled.edge_ids == bounded.edge_ids == ("foo", "foo::rev")
    assert [
        segment.direction for segment in compiled.segments
    ] == [segment.direction for segment in bounded.segments]
    assert compiled.traversal_ids == bounded.traversal_ids

def test_compiled_unreachable_cache_short_circuits_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=0.0, lon=0.0))
    graph.add_node(Node(id="A", lat=0.01, lon=0.0))
    graph.add_node(Node(id="G", lat=0.02, lon=0.0))
    graph.add_edge(
        Edge(
            id="blocked",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=0.0,
            road_type="motorway",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    planner = ScenicRoutePlanner(graph)
    planner.cost_function.avoid_highways = True
    cost_function = planner._make_fastest_cost_function()

    assert planner._a_star(
        graph.nodes["S"],
        graph.nodes["G"],
        cost_function=cost_function,
    ) is None

    def fail_if_recomputed(*args: object, **kwargs: object) -> None:
        raise AssertionError("compiled reachability miss was not cached")

    monkeypatch.setattr(planner, "_compiled_builtin_path", fail_if_recomputed)
    assert planner._a_star(
        graph.nodes["S"],
        graph.nodes["G"],
        cost_function=cost_function,
    ) is None


def test_reachability_cache_drops_previous_graph_epoch() -> None:
    first_graph = _tradeoff_graph()
    second_graph = _tradeoff_graph()
    ScenicRoutePlanner.clear_shared_caches()
    first = ScenicRoutePlanner(first_graph)
    second = ScenicRoutePlanner(second_graph)

    first._record_compiled_reachability(
        first_graph.nodes["S"],
        first_graph.nodes["G"],
        first.cost_function,
        first_graph._heuristic_cache_stamp(),
        False,
    )
    second._record_compiled_reachability(
        second_graph.nodes["S"],
        second_graph.nodes["G"],
        second.cost_function,
        second_graph._heuristic_cache_stamp(),
        False,
    )

    assert ScenicRoutePlanner._ELIGIBLE_REACHABILITY_SHARED_GRAPH is second_graph
    assert len(ScenicRoutePlanner._ELIGIBLE_REACHABILITY_SHARED_CACHE) == 1
    assert all(
        key[0] is second_graph
        for key in ScenicRoutePlanner._ELIGIBLE_REACHABILITY_SHARED_CACHE
    )


def test_parallel_edges_do_not_corrupt_compiled_shortest_path() -> None:
    graph = RoadGraph()
    for node_id, lat in (("S", 0.0), ("A", 0.01), ("G", 0.02)):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    graph.add_edge(
        Edge(
            id="zero",
            start_node_id="S",
            end_node_id="A",
            distance_km=0.0,
            scenic_score=0.0,
            speed_limit_kmh=60.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="one",
            start_node_id="S",
            end_node_id="A",
            distance_km=1.0,
            scenic_score=0.0,
            speed_limit_kmh=60.0,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="finish",
            start_node_id="A",
            end_node_id="G",
            distance_km=1.0,
            scenic_score=0.0,
            speed_limit_kmh=60.0,
            one_way=True,
        )
    )

    planner = ScenicRoutePlanner(graph)
    route = planner.find_fastest_route((0.0, 0.0), (0.02, 0.0))

    assert route.edge_ids == ("zero", "finish")
    assert route.estimated_duration_minutes == pytest.approx(1.0)


def test_compiled_path_positions_preserve_reverse_traversal_identity() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=0.0, lon=0.0))
    graph.add_node(Node(id="B", lat=0.01, lon=0.0))
    graph.add_edge(
        Edge(
            id="two-way",
            start_node_id="A",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
            one_way=False,
        )
    )
    planner = ScenicRoutePlanner(graph)
    topology = planner._csr_topology(False)
    assert topology is not None

    result = planner._compiled_weighted_path_with_positions(
        topology,
        topology.node_index["B"],
        topology.node_index["A"],
        topology.travel_time_minutes,
    )

    assert result is not None
    path, positions = result
    assert len(path) == len(positions) == 1
    assert path[0].start_node_id == "B"
    assert path[0].end_node_id == "A"
    assert topology.edge_refs[int(positions[0])] == ("two-way", True)


def test_same_node_route_has_renderable_zero_length_geometry() -> None:
    graph = _tradeoff_graph()
    planner = ScenicRoutePlanner(graph)

    route = planner.find_fastest_route((0.0, 0.0), (0.0, 0.0))

    assert route.edge_ids == ()
    assert route.waypoints == [(0.0, 0.0), (0.0, 0.0)]


def _force_production(graph: RoadGraph) -> RoadGraph:
    for index in range(25):
        graph.add_node(
            Node(
                id=f"frontier-isolated-{index}",
                lat=30.0 + index,
                lon=30.0,
            )
        )
    return graph


def test_endpoint_duration_bound_validates_speed_and_virtual_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for node_id, lon in (("S", -0.01), ("A", 0.0), ("B", 0.01)):
        graph.add_node(Node(id=node_id, lat=0.0, lon=lon))
    planner = ScenicRoutePlanner(graph)
    edge_km = planner._haversine(0.0, -0.01, 0.0, 0.0)
    for edge_id, start_node_id, end_node_id in (
        ("sa", "S", "A"),
        ("ab", "A", "B"),
    ):
        graph.add_edge(
            Edge(
                edge_id,
                start_node_id,
                end_node_id,
                edge_km,
                1.0,
                speed_limit_kmh=140,
                one_way=True,
            )
        )

    request = planner._build_endpoint_access_request(
        (0.0, -0.01),
        (0.0, 0.005),
    )
    setattr(request.overlay, "_route_access_request", request)
    planner.graph = request.overlay
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)

    goal = planner.graph.get_node(request.end_node_id)
    bounds = planner._frontier_reverse_duration_lower_bounds(
        goal,
        False,
        10.0,
    )

    expected = edge_km * 1.5 * 60.0 / 140.0
    assert bounds.get("S") == pytest.approx(expected)
    assert bounds.get(request.end_node_id) == 0.0


def _production_side_road_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, lat in (
        ("S", 0.0),
        ("R1", 0.01),
        ("R2", 0.02),
        ("G", 0.03),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    for index in range(25):
        graph.add_node(
            Node(
                id=f"production-isolated-{index}",
                lat=30.0 + index,
                lon=30.0,
            )
        )
    for edge in (
        Edge(
            id="highway",
            start_node_id="S",
            end_node_id="G",
            distance_km=8.0,
            scenic_score=0.0,
            road_type="motorway",
            speed_limit_kmh=80,
            one_way=True,
        ),
        Edge(
            id="residential-1",
            start_node_id="S",
            end_node_id="R1",
            distance_km=2.0,
            scenic_score=10.0,
            road_type="residential",
            speed_limit_kmh=30,
            one_way=True,
        ),
        Edge(
            id="service",
            start_node_id="R1",
            end_node_id="R2",
            distance_km=1.5,
            scenic_score=10.0,
            road_type="service",
            speed_limit_kmh=25,
            one_way=True,
        ),
        Edge(
            id="residential-2",
            start_node_id="R2",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=10.0,
            road_type="residential",
            speed_limit_kmh=30,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)
    return graph


def _production_lower_prefix_multiedge_detour_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, lat in (
        ("S", 0.0),
        ("A", 0.01),
        ("B", 0.02),
        ("C", 0.03),
        ("G", 0.04),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    for index in range(25):
        graph.add_node(
            Node(
                id=f"lower-prefix-isolated-{index}",
                lat=30.0 + index,
                lon=30.0,
            )
        )
    for edge in (
        Edge(
            id="fast",
            start_node_id="S",
            end_node_id="G",
            distance_km=8.0,
            scenic_score=0.0,
            speed_limit_kmh=80,
            one_way=True,
        ),
        Edge(
            id="slow-first",
            start_node_id="S",
            end_node_id="A",
            distance_km=2.0,
            scenic_score=1.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
        Edge(
            id="slow-finish",
            start_node_id="A",
            end_node_id="G",
            distance_km=3.0,
            scenic_score=2.0,
            speed_limit_kmh=45,
            one_way=True,
        ),
        Edge(
            id="detour-1",
            start_node_id="A",
            end_node_id="B",
            distance_km=1.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
        Edge(
            id="detour-2",
            start_node_id="B",
            end_node_id="C",
            distance_km=1.0,
            scenic_score=10.0,
            speed_limit_kmh=30,
            one_way=True,
        ),
        Edge(
            id="detour-3",
            start_node_id="C",
            end_node_id="G",
            distance_km=2.0,
            scenic_score=10.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)
    return graph


def _production_corridor_diversity_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, lat in (
        ("S", 0.0),
        ("C", 0.01),
        ("A", 0.02),
        ("G", 0.03),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=0.0))
    for index in range(25):
        graph.add_node(
            Node(
                id=f"corridor-isolated-{index}",
                lat=30.0 + index,
                lon=30.0,
            )
        )
    for edge in (
        Edge(
            id="shared-connector",
            start_node_id="S",
            end_node_id="C",
            distance_km=1.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        ),
        Edge(
            id="direct",
            start_node_id="C",
            end_node_id="G",
            distance_km=10.0,
            scenic_score=6.0,
            speed_limit_kmh=600,
            one_way=True,
        ),
        Edge(
            id="scenic-1",
            start_node_id="C",
            end_node_id="A",
            distance_km=20.0,
            scenic_score=9.0,
            speed_limit_kmh=600,
            one_way=True,
        ),
        Edge(
            id="scenic-2",
            start_node_id="A",
            end_node_id="G",
            distance_km=20.0,
            scenic_score=9.0,
            speed_limit_kmh=600,
            one_way=True,
        ),
    ):
        graph.add_edge(edge)
    return graph


def _shared_root_beam_graph() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=0.0, lon=0.0))
    graph.add_node(Node(id="C", lat=0.01, lon=0.0))
    graph.add_node(Node(id="G", lat=0.04, lon=0.0))
    graph.add_edge(
        Edge(
            id="shared-root",
            start_node_id="S",
            end_node_id="C",
            distance_km=1.0,
            scenic_score=0.0,
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    for index in range(20):
        branch = f"B{index:02d}"
        graph.add_node(
            Node(id=branch, lat=0.02, lon=index * 0.001)
        )
        graph.add_edge(
            Edge(
                id=f"branch-{index:02d}-in",
                start_node_id="C",
                end_node_id=branch,
                distance_km=1.0,
                scenic_score=float(index) / 2.0,
                speed_limit_kmh=60,
                one_way=True,
            )
        )
        graph.add_edge(
            Edge(
                id=f"branch-{index:02d}-out",
                start_node_id=branch,
                end_node_id="G",
                distance_km=1.0,
                scenic_score=float(index) / 2.0,
                speed_limit_kmh=60,
                one_way=True,
            )
        )
    return graph


def test_corridor_warm_start_finds_material_detour_and_respects_cap() -> None:
    graph = _production_corridor_diversity_graph()
    planner = ScenicRoutePlanner(
        graph,
        frontier_time_limit_seconds=0.0,
    )

    fastest = planner.find_fastest_route((0.0, 0.0), (0.03, 0.0))
    scenic = planner.find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        scenic_weight=1.0,
        max_detour_factor=3.0,
        scenic_priority=True,
    )
    with pytest.raises(
        RoutingTimeout, match="without a scenic route"
    ):
        planner.find_scenic_route(
            (0.0, 0.0),
            (0.03, 0.0),
            scenic_weight=1.0,
            max_detour_factor=2.4,
            scenic_priority=True,
        )

    assert fastest.edge_ids == ("shared-connector", "direct")
    assert scenic.edge_ids == (
        "shared-connector",
        "scenic-1",
        "scenic-2",
    )
    assert (
        scenic.normalized_scenic_score
        >= fastest.normalized_scenic_score + 0.3
    )
    assert 1.0 < scenic.actual_duration_ratio <= 3.0 + 1e-12
    assert len(scenic.waypoints) == len(set(scenic.waypoints))


def test_beam_uses_capacity_when_paths_share_first_edge() -> None:
    graph = _shared_root_beam_graph()
    planner = ScenicRoutePlanner(graph)
    bounds = planner._frontier_reverse_duration_lower_bounds(
        graph.nodes["G"],
        False,
        10.0,
    )

    paths = planner._frontier_beam_warm_start_paths(
        graph.nodes["S"],
        graph.nodes["G"],
        False,
        None,
        10.0,
        bounds,
        1.0,
        0.0,
    )

    identities = {
        tuple(edge.id for edge in path)
        for path in paths
    }
    assert len(identities) == 20
    assert all(path[0] == "shared-root" for path in identities)


def test_local_detour_budget_counts_final_replacement_edge() -> None:
    corridor = _production_corridor_diversity_graph()
    corridor_paths = ScenicRoutePlanner(
        corridor
    )._frontier_local_detour_paths(
        [[corridor.edges["direct"]]],
        avoid_highways=False,
        duration_cap_minutes=10.0,
        budget=2,
    )
    lower_prefix = _production_lower_prefix_multiedge_detour_graph()
    lower_prefix_paths = ScenicRoutePlanner(
        lower_prefix
    )._frontier_local_detour_paths(
        [[lower_prefix.edges["slow-finish"]]],
        avoid_highways=False,
        duration_cap_minutes=10.0,
        budget=3,
    )

    assert any(
        tuple(edge.id for edge in path) == ("scenic-1", "scenic-2")
        for path in corridor_paths
    )
    assert any(
        tuple(edge.id for edge in path)
        == ("detour-1", "detour-2", "detour-3")
        for path in lower_prefix_paths
    )


def test_production_frontier_finds_lower_prefix_multiedge_detour() -> None:
    graph = _production_lower_prefix_multiedge_detour_graph()
    planner = ScenicRoutePlanner(graph)

    fastest = planner.find_fastest_route((0.0, 0.0), (0.04, 0.0))
    scenic = planner.find_scenic_route(
        (0.0, 0.0),
        (0.04, 0.0),
        scenic_weight=1.0,
        max_detour_factor=1.8,
        scenic_priority=True,
    )

    assert fastest.edge_ids == ("fast",)
    assert scenic.edge_ids == (
        "slow-first",
        "detour-1",
        "detour-2",
        "detour-3",
    )
    assert scenic.normalized_scenic_score > fastest.normalized_scenic_score
    assert scenic.actual_duration_ratio <= 1.8 + 1e-12
    assert scenic.actual_duration_ratio > 1.0
    assert scenic.segments[0].scenic_score < max(
        segment.scenic_score for segment in scenic.segments[1:]
    )
    node_ids = [scenic.segments[0].start[0]] + [
        segment.end[0] for segment in scenic.segments
    ]
    assert len(node_ids) == len(set(node_ids))


def test_production_frontier_selects_feasible_residential_service_detour() -> None:
    graph = _production_side_road_graph()
    planner = ScenicRoutePlanner(graph)

    fastest = planner.find_fastest_route((0.0, 0.0), (0.03, 0.0))
    scenic = planner.find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        scenic_weight=1.0,
        max_detour_factor=3.0,
    )

    assert fastest.edge_ids == ("highway",)
    assert scenic.edge_ids == ("residential-1", "service", "residential-2")
    assert [segment.road_type for segment in scenic.segments] == [
        "residential",
        "service",
        "residential",
    ]
    assert scenic.algorithm == "production-multilabel-frontier"
    assert scenic.average_scenic_score == pytest.approx(10.0)
    assert scenic.objective_value == pytest.approx(1.0)
    assert scenic.estimated_duration_minutes <= scenic.duration_cap_minutes


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_exact_routes_report_zero_frontier_diagnostics(q: float) -> None:
    route = _route(ScenicRoutePlanner(_tradeoff_graph()), q=q, kappa=4.0)

    diagnostics = route.search_diagnostics
    assert set(diagnostics) == _SEARCH_DIAGNOSTIC_KEYS
    assert diagnostics["time_limit_seconds"] == pytest.approx(0.0)
    assert diagnostics["labels_generated"] == 0
    assert diagnostics["labels_expanded"] == 0
    assert diagnostics["labels_pruned"] == 0
    assert diagnostics["max_frontier_size"] == 0
    assert diagnostics["remaining_frontier_size"] == 0
    assert diagnostics["deadline_reached"] is False
    assert diagnostics["elapsed_ms"] == pytest.approx(0.0)
    assert diagnostics["mode"] == "exact"

def test_production_frontier_finds_optimum_and_completes_exactly() -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))

    route = _route(planner, q=0.5, kappa=4.0)

    assert route.edge_ids == ("mid-1", "mid-2")
    assert route.exact is True
    assert route.exactness_status == "exact"
    assert route.algorithm == "production-multilabel-frontier"
    assert route.certified_upper_bound == pytest.approx(route.objective_value)
    assert route.optimality_gap == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("compiled_threshold", "expected_algorithm"),
    [
        (0, "compiled-lagrangian-endpoint-search"),
        (100_000, "production-multilabel-frontier"),
    ],
)
def test_production_routes_support_multiple_divergences_and_rejoins(
    monkeypatch: pytest.MonkeyPatch,
    compiled_threshold: int,
    expected_algorithm: str,
) -> None:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("S", 0.0, 0.00),
        ("A", 0.0, 0.01),
        ("X", 0.01, 0.015),
        ("B", 0.0, 0.02),
        ("C", 0.0, 0.03),
        ("Y", 0.01, 0.035),
        ("D", 0.0, 0.04),
        ("G", 0.0, 0.05),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=lon))

    def add_edge(
        edge_id: str,
        start: str,
        end: str,
        *,
        scenic_score: float = 0.0,
    ) -> None:
        graph.add_edge(
            Edge(
                id=edge_id,
                start_node_id=start,
                end_node_id=end,
                distance_km=1.0,
                scenic_score=scenic_score,
                speed_limit_kmh=60.0,
                one_way=True,
            )
        )

    add_edge("main-s-a", "S", "A")
    add_edge("main-a-b", "A", "B")
    add_edge("main-b-c", "B", "C")
    add_edge("main-c-d", "C", "D")
    add_edge("main-d-g", "D", "G")
    add_edge("scenic-a-x", "A", "X", scenic_score=10.0)
    add_edge("scenic-x-b", "X", "B", scenic_score=10.0)
    add_edge("scenic-c-y", "C", "Y", scenic_score=10.0)
    add_edge("scenic-y-d", "Y", "D", scenic_score=10.0)

    planner = ScenicRoutePlanner(_force_production(graph))
    monkeypatch.setattr(
        planner, "_COMPILED_SCENIC_MIN_NODES", compiled_threshold
    )
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_COLLAPSED_ACCESS_NODE_THRESHOLD", 0)
    route = planner.find_scenic_route(
        (0.0, 0.0),
        (0.0, 0.05),
        scenic_weight=1.0,
        max_detour_factor=1.4,
        scenic_priority=True,
    )

    assert route.edge_ids == (
        "main-s-a",
        "scenic-a-x",
        "scenic-x-b",
        "main-b-c",
        "scenic-c-y",
        "scenic-y-d",
        "main-d-g",
    )
    assert route.algorithm == expected_algorithm
    assert route.exact is (expected_algorithm == "production-multilabel-frontier")


def _label(
    *,
    visited: set[str],
    duration: float,
    distance: float,
    exposure: float,
) -> _FrontierLabel:
    return _FrontierLabel(
        0,
        "N",
        duration,
        distance,
        exposure,
        None,
        None,
        frozenset(visited),
        (),
        root_traversal_id="",
    )


def test_frontier_dominance_rejects_unsafe_ancestry() -> None:
    planner = ScenicRoutePlanner()
    first = _label(
        visited={"S", "N", "blocked"},
        duration=1.0,
        distance=1.0,
        exposure=1.0,
    )
    second = _label(
        visited={"S", "N"},
        duration=2.0,
        distance=2.0,
        exposure=0.0,
    )

    assert not planner._frontier_label_dominates(first, second)


def test_frontier_dominance_rejects_ratio_unsafe_label() -> None:
    planner = ScenicRoutePlanner()
    first = _label(
        visited={"S", "N"},
        duration=1.0,
        distance=1.0,
        exposure=0.1,
    )
    second = _label(
        visited={"S", "N"},
        duration=2.0,
        distance=2.0,
        exposure=1.0,
    )

    assert not planner._frontier_label_dominates(first, second)


def test_production_frontier_rejects_cycles_and_enforces_reverse_cap() -> None:
    graph = _force_production(_tradeoff_graph())
    planner = ScenicRoutePlanner(graph)

    bounds = planner._frontier_reverse_duration_lower_bounds(
        graph.nodes["G"], False, 6.0
    )
    assert bounds["S"] == pytest.approx(6.0)
    route = planner.find_scenic_route(
        (0.0, 0.0), (0.03, 0.0), q=1.0, kappa=1.0
    )
    node_ids = [route.segments[0].start[0]] + [
        segment.end[0] for segment in route.segments
    ]
    assert len(node_ids) == len(set(node_ids))
    assert route.edge_ids == ("fast",)
    assert route.estimated_duration_minutes <= route.duration_cap_minutes


def test_production_frontier_timeout_during_preprocessing_certifies_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    planner._frontier_time_limit_seconds = 1.0
    ticks = [0.0, 0.0, 2.0]
    monkeypatch.setattr(
        planner,
        "_monotonic",
        lambda: ticks.pop(0) if ticks else 2.0,
    )

    route = _route(planner, q=1.0, kappa=4.0)

    assert route.exact is False
    assert route.exactness_status == "approximate-certified"
    assert route.certified_upper_bound >= route.objective_value
    assert route.search_diagnostics["deadline_reached"] is True


def test_production_frontier_timeout_without_candidate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    planner._frontier_time_limit_seconds = 1.0
    monkeypatch.setattr(
        planner, "_frontier_warm_start_paths", lambda *_args: []
    )
    ticks = [0.0, 0.0, 0.0]
    monkeypatch.setattr(
        planner,
        "_monotonic",
        lambda: ticks.pop(0) if ticks else 2.0,
    )

    with pytest.raises(
        RoutingTimeout, match="without a scenic route"
    ):
        _route(planner, q=1.0, kappa=4.0)

def test_production_frontier_timeout_with_scenic_candidate_certifies_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    planner._frontier_time_limit_seconds = 1.0
    ticks = [0.0, 0.0, 0.0]
    monkeypatch.setattr(
        planner,
        "_monotonic",
        lambda: ticks.pop(0) if ticks else 2.0,
    )

    route = _route(planner, q=1.0, kappa=4.0)

    assert route.exact is False
    assert route.exactness_status == "approximate-certified"
    assert math.isfinite(route.certified_upper_bound)
    assert route.certified_upper_bound >= route.objective_value
    diagnostics = route.search_diagnostics
    assert set(diagnostics) == _SEARCH_DIAGNOSTIC_KEYS
    assert diagnostics["time_limit_seconds"] == pytest.approx(1.0)
    assert diagnostics["labels_generated"] >= diagnostics["labels_expanded"] >= 0
    assert diagnostics["labels_pruned"] >= 0
    assert (
        diagnostics["max_frontier_size"]
        >= diagnostics["remaining_frontier_size"]
        >= 0
    )
    assert diagnostics["remaining_frontier_size"] <= diagnostics["labels_generated"]
    assert diagnostics["deadline_reached"] is True
    assert diagnostics["elapsed_ms"] >= 0.0
    assert diagnostics["mode"] == "frontier"


def test_scenic_priority_maximizes_score_under_duration_cap() -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    route = planner.find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        q=0.5,
        kappa=4.0,
        scenic_priority=True,
    )

    assert route.edge_ids == ("scenic-1", "scenic-2")
    assert route.normalized_scenic_score == pytest.approx(1.0)
    assert route.estimated_duration_minutes <= route.duration_cap_minutes
    node_ids = [route.segments[0].start[0]] + [
        segment.end[0] for segment in route.segments
    ]
    assert len(node_ids) == len(set(node_ids))

def test_compiled_endpoint_scenic_priority_searches_scenic_accesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    monkeypatch.setattr(planner, "_COMPILED_SCENIC_MIN_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_COLLAPSED_ACCESS_NODE_THRESHOLD", 0)

    route = planner.find_scenic_route(
        (0.001, 0.0),
        (0.029, 0.0),
        q=0.0,
        kappa=4.0,
        scenic_priority=True,
    )

    assert route.algorithm == "compiled-lagrangian-endpoint-search"
    assert route.average_scenic_score == pytest.approx(10.0)
    assert route.estimated_duration_minutes <= route.duration_cap_minutes

def test_production_frontier_preserves_interior_endpoint_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    monkeypatch.setattr(planner, "_COMPILED_SCENIC_MIN_NODES", 100_000)

    route = planner.find_scenic_route(
        (0.001, 0.0),
        (0.029, 0.0),
        q=0.0,
        kappa=4.0,
        scenic_priority=True,
    )

    assert route.algorithm == "production-multilabel-frontier"
    assert route.edge_ids == ("scenic-1", "scenic-2")
    assert route.traversal_ids == (
        "0:forward:scenic-1",
        "1:forward:scenic-2",
    )
    assert route.waypoints == [
        (0.001, 0.0),
        (0.02, 0.0),
        (0.029, 0.0),
    ]
    assert route.average_scenic_score == pytest.approx(10.0)
    assert route.estimated_duration_minutes <= route.duration_cap_minutes
    assert route.exact is True


def test_compiled_endpoint_collapsed_boundaries_preserve_fastest_and_scenic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(_force_production(_tradeoff_graph()))
    monkeypatch.setattr(planner, "_ENDPOINT_OVERLAY_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_LARGE_GRAPH_EDGE_THRESHOLD", 0)
    monkeypatch.setattr(planner, "_COLLAPSED_ACCESS_NODE_THRESHOLD", 0)

    fastest = planner.find_fastest_route((0.0, 0.0), (0.03, 0.0))
    assert fastest.edge_ids == ("fast",)
    assert fastest.traversal_ids == ("0:forward:fast",)
    assert fastest.waypoints == [(0.0, 0.0), (0.03, 0.0)]

    scenic = planner.find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        q=0.0,
        kappa=4.0,
        scenic_priority=True,
    )
    assert scenic.edge_ids == ("scenic-1", "scenic-2")
    assert scenic.traversal_ids == (
        "0:forward:scenic-1",
        "1:forward:scenic-2",
    )
    assert scenic.waypoints == [
        (0.0, 0.0),
        (0.02, 0.0),
        (0.03, 0.0),
    ]
    assert scenic.estimated_duration_minutes <= scenic.duration_cap_minutes

def test_scenic_priority_overrides_zero_weight_shortcut() -> None:
    route = ScenicRoutePlanner(_tradeoff_graph()).find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        q=0.0,
        kappa=4.0,
        scenic_priority=True,
    )

    assert route.edge_ids == ("scenic-1", "scenic-2")
    assert route.estimated_duration_minutes <= route.duration_cap_minutes

def test_compact_reverse_csr_has_numeric_positions_and_inbound_mapping() -> None:
    graph = RoadGraph()
    for node_id in ("S", "A", "G"):
        graph.add_node(Node(node_id, 0.0, 0.0))
    graph.add_edge(
        Edge("forward", "S", "A", 1.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("finish", "A", "G", 1.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    planner = ScenicRoutePlanner(graph)
    topology = planner._csr_topology()
    assert topology is not None
    assert topology.reverse_indptr.shape == (len(topology.node_ids) + 1,)
    assert topology.reverse_indices.shape == topology.indices.shape
    assert topology.reverse_positions.shape == topology.indices.shape
    assert topology.reverse_indptr.dtype.kind in "iu"
    assert topology.reverse_indices.dtype.kind in "iu"
    assert topology.reverse_positions.dtype.kind in "iu"
    assert topology.reverse_indptr.nbytes + topology.reverse_indices.nbytes + topology.reverse_positions.nbytes <= (
        32 * (len(topology.node_ids) + len(topology.indices))
    )
    goal_index = topology.node_index["G"]
    reverse_start = int(topology.reverse_indptr[goal_index])
    reverse_position = int(topology.reverse_positions[reverse_start])
    assert topology.edge_refs[reverse_position] == ("finish", False)
    assert int(topology.reverse_indices[reverse_start]) == topology.node_index["A"]


def test_endpoint_scenic_search_reuses_prewarmed_base_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = ScenicRoutePlanner(
        _production_corridor_diversity_graph(),
        frontier_time_limit_seconds=0.0,
    )
    planner._LARGE_GRAPH_EDGE_THRESHOLD = 0
    planner.prewarm_routing_cache()

    def fail_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("endpoint query rebuilt base CSR")

    monkeypatch.setattr(planner, "_build_csr_topology", fail_rebuild)
    route = planner.find_scenic_route(
        (0.0, 0.0),
        (0.03, 0.0),
        scenic_weight=1.0,
        max_detour_factor=3.0,
        scenic_priority=True,
    )
    assert route.edge_ids == (
        "shared-connector",
        "scenic-1",
        "scenic-2",
    )


def test_target_bounded_builtin_search_matches_canonical_without_reverse_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for index in range(20):
        graph.add_node(Node(str(index), 0.0, float(index)))
    graph.add_edge(
        Edge("z-direct", "0", "19", 2.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("a-first", "0", "1", 1.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("a-last", "1", "19", 1.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    graph.add_edge(
        Edge("wrong-way", "19", "18", 1.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    planner = ScenicRoutePlanner(graph)
    planner._LARGE_GRAPH_EDGE_THRESHOLD = 0
    monkeypatch.setattr(
        planner,
        "_cached_reverse_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy reverse map used")
        ),
    )
    fastest = planner._make_fastest_cost_function()
    expected = planner._canonical_fastest_edges(
        graph.get_node("0"), graph.get_node("19"), False
    )
    actual = planner._bidirectional_builtin_path(
        graph.get_node("0"), graph.get_node("19"), fastest
    )
    assert [edge.id for edge in actual or []] == [
        edge.id for edge in expected or []
    ]
    assert (
        planner._bidirectional_builtin_path(
            graph.get_node("19"), graph.get_node("0"), fastest
        )
        is None
    )


def test_target_bounded_query_has_no_dense_node_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RoadGraph()
    for index in range(2_000):
        graph.add_node(Node(str(index), 0.0, float(index)))
    graph.add_edge(
        Edge("only", "0", "1", 1.0, 5.0, speed_limit_kmh=60, one_way=True)
    )
    planner = ScenicRoutePlanner(graph)
    planner._LARGE_GRAPH_EDGE_THRESHOLD = 0
    planner.prewarm_routing_cache()
    monkeypatch.setattr(
        "src.route_planner.planner.np.full",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("query allocated dense node state")
        ),
    )
    path = planner._bidirectional_builtin_path(
        graph.get_node("0"),
        graph.get_node("1"),
        planner._make_fastest_cost_function(),
    )
    assert [edge.id for edge in path or []] == ["only"]
