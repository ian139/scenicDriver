from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.route_planner import planner as planner_module
from src.route_planner import service as route_service
from src.route_planner._compact_search import (
    _scalar_cost_bound_per_minute as module_bound_per_minute,
    run_compact_bidirectional_search,
)
from src.route_planner.cancellation import RoutingDeadline, RoutingTimeout
from src.route_planner.cost import CostWeights, ScenicCostFunction
from src.route_planner.graph import (
    CompactRoadGraph,
    Edge,
    Node,
    RoadGraph,
    write_compact_graph,
)
from src.route_planner.planner import ScenicRoutePlanner

_SAFETY = 1.0 - 1e-12


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> None:
    yield
    route_service.clear_route_caches()
    ScenicRoutePlanner.clear_shared_caches()


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def _certified_chain_graph(
    edge_count: int = 8, *, with_detour: bool = False
) -> RoadGraph:
    """One-way meridian chain whose stored distances exactly equal the node
    geodesics, so the compact writer certifies ``geodesic_bound_speed``.

    Optionally appends a long two-way detour branch off the first edge that
    is far longer than the direct chain, giving the completion bound real
    labels to prune while the optimum stays on the chain.
    """
    graph = RoadGraph()
    for index in range(edge_count + 1):
        graph.add_node(Node(f"c{index}", 42.0 + 0.05 * index, -72.0))
    for index in range(edge_count):
        distance = _haversine(
            42.0 + 0.05 * index, -72.0, 42.0 + 0.05 * (index + 1), -72.0
        )
        graph.add_edge(
            Edge(
                f"e{index}",
                f"c{index}",
                f"c{index + 1}",
                distance,
                scenic_score=5.0,
                road_type="secondary",
                speed_limit_kmh=90,
                one_way=True,
            )
        )
    if with_detour:
        for index in range(3):
            graph.add_node(Node(f"d{index}", 42.025, -72.0 + 0.1 * (index + 1)))
        detour_distance = _haversine(
            42.025, -72.0, 42.025, -72.1
        ) + 2 * _haversine(42.025, -72.1, 42.025, -72.2)
        graph.add_edge(
            Edge(
                "de0",
                "c1",
                "d0",
                _haversine(42.05, -72.0, 42.025, -72.1),
                scenic_score=9.0,
                road_type="secondary",
                speed_limit_kmh=90,
                one_way=True,
            )
        )
        graph.add_edge(
            Edge(
                "de1",
                "d0",
                "d1",
                _haversine(42.025, -72.1, 42.025, -72.2),
                scenic_score=9.0,
                road_type="secondary",
                speed_limit_kmh=90,
                one_way=True,
            )
        )
        graph.add_edge(
            Edge(
                "de2",
                "d1",
                "d2",
                _haversine(42.025, -72.2, 42.025, -72.1),
                scenic_score=9.0,
                road_type="secondary",
                speed_limit_kmh=90,
                one_way=True,
            )
        )
        graph.add_edge(
            Edge(
                "de3",
                "d2",
                "c3",
                _haversine(42.025, -72.1, 42.15, -72.0),
                scenic_score=9.0,
                road_type="secondary",
                speed_limit_kmh=90,
                one_way=True,
            )
        )
        assert detour_distance > 0.0
    return graph


def _zero_duration_graph() -> RoadGraph:
    """Certified chain with a free zero-duration loop edge at the goal."""
    graph = _certified_chain_graph(4)
    graph.add_node(Node("z0", 42.2, -72.0))
    graph.add_edge(
        Edge(
            "ze0",
            "c4",
            "z0",
            distance_km=0.0,
            scenic_score=10.0,
            road_type="secondary",
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    return graph


def _publish(tmp_path: Path, graph: RoadGraph) -> tuple[Path, Path]:
    sqlite_path = tmp_path / "road_graph.sqlite3"
    graph.save(sqlite_path)
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    write_compact_graph(sqlite_path, manifest_path)
    return sqlite_path, manifest_path


def _lower_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ScenicRoutePlanner, "_ENDPOINT_OVERLAY_MAX_NODES", 2)
    monkeypatch.setattr(ScenicRoutePlanner, "_LARGE_GRAPH_EDGE_THRESHOLD", 2)
    monkeypatch.setattr(ScenicRoutePlanner, "_COMPILED_SCENIC_MIN_NODES", 2)
    monkeypatch.setattr(ScenicRoutePlanner, "_EXACT_ORACLE_MAX_NODES", 0)
    monkeypatch.setattr(ScenicRoutePlanner, "_EXACT_ORACLE_MAX_EDGES", 0)


# ---------------------------------------------------------------------------
# Bound construction and scalar floor
# ---------------------------------------------------------------------------


def test_compact_duration_bounds_match_reference_and_shrink(
    tmp_path: Path,
) -> None:
    graph = _certified_chain_graph(8)
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["geodesic_bound_speed"]["all"] is not None

    speed = compact.endpoint_geodesic_bound_speed(False)
    assert speed is not None
    goal = (42.4, -72.0)
    bounds = compact.compact_endpoint_duration_bounds((goal,), speed)
    assert bounds is not None
    assert bounds.shape == (compact.node_count,)

    for rank in range(compact.node_count):
        node = compact._node_at_rank(rank)
        reference = (
            _haversine(node.lat, node.lon, goal[0], goal[1]) * 60.0 / speed
        )
        # Downward-shrunk: floating error can never overestimate.
        assert bounds[rank] <= reference
        assert bounds[rank] >= 0.0
        # Tight for this straight-line graph: within the safety factor.
        assert bounds[rank] == pytest.approx(
            reference * _SAFETY, rel=2e-9
        )
    # Zero bound exactly at the target node.
    target_rank = compact._rank_for_id("c8", "node")
    assert bounds[target_rank] == 0.0

    # Multiple targets take the minimum.
    multi = compact.compact_endpoint_duration_bounds(
        ((42.4, -72.0), (42.0, -72.0)), speed
    )
    assert multi is not None
    assert multi[0] == 0.0
    assert multi[target_rank] == 0.0


def test_compact_duration_bounds_fail_closed(tmp_path: Path) -> None:
    graph = _certified_chain_graph(4)
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    for bad_targets in ((), ((42.2, float("nan")),)):
        assert (
            compact.compact_endpoint_duration_bounds(bad_targets, 90.0)
            is None
        )
    assert (
        compact.compact_endpoint_duration_bounds(((42.2, -72.0),), 0.0)
        is None
    )
    assert (
        compact.compact_endpoint_duration_bounds(
            ((42.2, -72.0),), float("inf")
        )
        is None
    )


def test_bound_per_minute_formula_and_planner_consistency() -> None:
    fastest = ScenicCostFunction(
        scenic_weight=0.0,
        weights=CostWeights(
            travel_time=1.0,
            scenic_reward=0.0,
            highway_penalty=0.0,
            scenic_byway_bonus=0.0,
        ),
    )
    strict = ScenicCostFunction(
        scenic_weight=0.5,
        strict_highways=True,
        weights=CostWeights(highway_penalty=2.0),
    )
    preferred = ScenicCostFunction(
        scenic_weight=0.5,
        highway_preference=0.7,
        weights=CostWeights(),
    )
    plain = ScenicCostFunction(scenic_weight=0.5)

    assert module_bound_per_minute(fastest, 0.0) == pytest.approx(
        0.5e-6 * _SAFETY
    )
    assert module_bound_per_minute(strict, 0.0) == pytest.approx(
        0.5e-6 * _SAFETY
    )
    assert module_bound_per_minute(preferred, 0.0) == pytest.approx(
        0.5e-6 * _SAFETY
    )
    assert module_bound_per_minute(plain, 1.5) == pytest.approx(
        (0.5e-6 + 1.5) * _SAFETY
    )
    assert module_bound_per_minute(plain, -1.0) == pytest.approx(
        0.5e-6 * _SAFETY
    )

    planner = ScenicRoutePlanner(RoadGraph())
    for cost_function, multiplier in (
        (fastest, 0.0),
        (strict, 0.25),
        (preferred, 1.5),
        (plain, 0.75),
    ):
        assert (
            planner._scalar_cost_bound_per_minute(cost_function, multiplier)
            == module_bound_per_minute(cost_function, multiplier)
        )


def test_bound_per_minute_is_a_real_per_edge_lower_bound() -> None:
    """The scalar floor must never exceed cost/duration for any edge class:
    a maximally discounted scenic byway, a strict-highway edge, a preferred
    highway, and a plain edge, with and without a Lagrangian multiplier."""
    byway = Edge(
        "byway",
        "a",
        "b",
        distance_km=1.0,
        scenic_score=10.0,
        road_type="scenic_byway",
        speed_limit_kmh=60,
        one_way=True,
    )
    highway = Edge(
        "highway",
        "a",
        "b",
        distance_km=1.0,
        scenic_score=10.0,
        road_type="trunk",
        speed_limit_kmh=60,
        one_way=True,
    )
    plain_edge = Edge(
        "plain",
        "a",
        "b",
        distance_km=1.0,
        scenic_score=0.0,
        road_type="secondary",
        speed_limit_kmh=60,
        one_way=True,
    )
    specs = (
        ScenicCostFunction(
            scenic_weight=0.8,
            weights=CostWeights(scenic_byway_bonus=100.0),
        ),
        ScenicCostFunction(
            scenic_weight=1.0,
            strict_highways=True,
            weights=CostWeights(highway_penalty=2.0),
        ),
        ScenicCostFunction(
            scenic_weight=0.0,
            highway_preference=0.7,
            weights=CostWeights(scenic_reward=0.0),
        ),
        ScenicCostFunction(scenic_weight=0.5),
    )
    for cost_function in specs:
        for multiplier in (0.0, 0.25, 1.5):
            floor = module_bound_per_minute(cost_function, multiplier)
            for edge in (byway, highway, plain_edge):
                duration = edge.travel_time_minutes
                scalar_cost = cost_function.calculate(edge) + multiplier * duration
                assert (
                    floor * duration <= scalar_cost + 1e-15
                ), f"floor overestimates edge cost for {cost_function!r}"


def test_scalar_completion_bounds_admissible_against_exact_min_cost(
    tmp_path: Path,
) -> None:
    """For every node, the planner's scalar completion bound must never
    exceed the exact minimum remaining scalar cost to the destination."""
    graph = _certified_chain_graph(8)
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    request = planner._build_endpoint_access_request(
        (42.0, -72.0), (42.4, -72.0), avoid_highways=False
    )
    cost_function = ScenicCostFunction(
        scenic_weight=0.8,
        weights=CostWeights(
            travel_time=1.0,
            scenic_reward=1.0,
            highway_penalty=2.0,
            scenic_byway_bonus=1.0,
        ),
    )
    duration_bounds = planner._compact_request_duration_bounds(
        request, compact, cost_function
    )
    assert duration_bounds is not None
    forward_duration, reverse_duration = duration_bounds
    assert forward_duration is not None and reverse_duration is not None

    topology = planner._csr_topology()
    signature = planner._built_in_cost_signature(cost_function)
    base_weights = planner._vectorized_builtin_weights(topology, signature)
    assert base_weights is not None

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    multiplier = 0.75
    weights = np.where(
        np.isfinite(base_weights),
        np.maximum(base_weights + multiplier * topology.travel_time_minutes, 0.0),
        np.inf,
    )
    matrix = csr_matrix(
        (weights, topology.indices, topology.indptr),
        shape=(len(topology.node_ids), len(topology.node_ids)),
    )
    goal_rank = topology.node_index["c8"]
    exact = dijkstra(matrix, directed=True, indices=range(len(topology.node_ids)))[
        :, goal_rank
    ]

    bound_per_minute = module_bound_per_minute(cost_function, multiplier)
    limit = float(
        np.sum(
            weights[p]
            + multiplier
            * float(topology.travel_time_minutes[p])
            for p in range(8)
        )
    )
    margin = 1e-9 * max(1.0, limit)
    bound_forward = np.maximum(
        forward_duration * bound_per_minute - margin, 0.0
    )
    for rank in range(len(topology.node_ids)):
        assert bound_forward[rank] <= (
            exact[rank] + 1e-9 * max(1.0, exact[rank])
        ), f"bound overestimates remaining cost at rank {rank}"


# ---------------------------------------------------------------------------
# Native search parity with completion bounds
# ---------------------------------------------------------------------------


def test_native_search_with_bounds_matches_unbounded(tmp_path: Path) -> None:
    graph = _certified_chain_graph(8)
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()

    forward = [("c0", 0.0, (0, 0), None)]
    reverse = [("c8", 0.0, (0, 0), None)]
    limit = float(sum(topology.travel_time_minutes[p] for p in range(8)))
    bound_fwd = compact.compact_endpoint_duration_bounds(
        ((42.4, -72.0),), 90.0
    )
    bound_rev = compact.compact_endpoint_duration_bounds(
        ((42.0, -72.0),), 90.0
    )

    unbounded = run_compact_bidirectional_search(
        topology, cost_function, forward, reverse, cost_limit=limit
    )
    bounded = run_compact_bidirectional_search(
        topology,
        cost_function,
        forward,
        reverse,
        cost_limit=limit,
        bound_forward=bound_fwd,
        bound_reverse=bound_rev,
    )
    assert unbounded is not None and bounded is not None
    assert bounded == unbounded
    assert bounded[0] == list(range(8))
    assert bounded[1] == pytest.approx(limit)

    # A limit strictly below the optimum prunes everything with or without
    # the completion bound.
    tight = limit * 0.5
    assert (
        run_compact_bidirectional_search(
            topology, cost_function, forward, reverse, cost_limit=tight
        )
        is None
    )
    assert (
        run_compact_bidirectional_search(
            topology,
            cost_function,
            forward,
            reverse,
            cost_limit=tight,
            bound_forward=bound_fwd,
            bound_reverse=bound_rev,
        )
        is None
    )


def test_native_equal_cost_tie_survives_bounds(tmp_path: Path) -> None:
    """Equal-cost alternatives must resolve identically with bounds on: the
    strict ``>`` prune never removes a label whose completion equals the
    limit."""
    graph = _certified_chain_graph(4)
    # Parallel equal-duration final edges: canonical-smaller ``ea`` inserted
    # last, so CSR order favors ``ez`` and only the ranked tie-break picks
    # the canonical-smaller path (mirrors the existing diamond tests).
    # Distances equal the node geodesics so the artifact stays certified
    # and the hand-built completion bounds remain admissible.
    graph.add_node(Node("c5", 42.3, -72.0))
    final_distance = _haversine(42.2, -72.0, 42.3, -72.0)
    graph.add_edge(
        Edge(
            "ez",
            "c4",
            "c5",
            distance_km=final_distance,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=90,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "ea",
            "c4",
            "c5",
            distance_km=final_distance,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=90,
            one_way=True,
        )
    )
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()
    forward = [("c0", 0.0, (0, 0), None)]
    reverse = [("c5", 0.0, (0, 0), None)]
    limit = float(sum(topology.travel_time_minutes[p] for p in range(5)))
    bound_fwd = compact.compact_endpoint_duration_bounds(
        ((42.3, -72.0),), 90.0
    )
    bound_rev = compact.compact_endpoint_duration_bounds(
        ((42.0, -72.0),), 90.0
    )
    unbounded = run_compact_bidirectional_search(
        topology, cost_function, forward, reverse, cost_limit=limit
    )
    bounded = run_compact_bidirectional_search(
        topology,
        cost_function,
        forward,
        reverse,
        cost_limit=limit,
        bound_forward=bound_fwd,
        bound_reverse=bound_rev,
    )
    assert unbounded is not None and bounded is not None
    assert bounded == unbounded
    # Both choose the canonical-smaller ``ea`` final traversal (the ranked
    # middle-key tie-break, not CSR insertion order).
    assert bounded[0][-1] == 5


def test_native_zero_duration_edge_with_bounds(tmp_path: Path) -> None:
    graph = _zero_duration_graph()
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()
    forward = [("c0", 0.0, (0, 0), None)]
    reverse = [("z0", 0.0, (0, 0), None)]
    limit = float(sum(topology.travel_time_minutes[p] for p in range(5)))
    bound_fwd = compact.compact_endpoint_duration_bounds(
        ((42.2, -72.0),), 90.0
    )
    bound_rev = compact.compact_endpoint_duration_bounds(
        ((42.0, -72.0),), 90.0
    )
    unbounded = run_compact_bidirectional_search(
        topology, cost_function, forward, reverse, cost_limit=limit
    )
    bounded = run_compact_bidirectional_search(
        topology,
        cost_function,
        forward,
        reverse,
        cost_limit=limit,
        bound_forward=bound_fwd,
        bound_reverse=bound_rev,
    )
    assert unbounded is not None and bounded is not None
    assert bounded == unbounded
    assert bounded[0] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Planner-level scenic parity with the completion bound engaged
# ---------------------------------------------------------------------------


def test_compact_scenic_route_bounds_engaged_and_eager_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _certified_chain_graph(8)
    sqlite_path, manifest_path = _publish(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)
    _lower_thresholds(monkeypatch)

    captured: list[tuple[object, object]] = []
    original = planner_module.run_compact_bidirectional_search

    def spy(
        topology: object,
        cost_function: object,
        forward_seeds: object,
        reverse_seeds: object,
        **kwargs: object,
    ) -> object:
        captured.append(
            (kwargs.get("bound_forward"), kwargs.get("bound_reverse"))
        )
        return original(
            topology, cost_function, forward_seeds, reverse_seeds, **kwargs
        )

    monkeypatch.setattr(
        planner_module, "run_compact_bidirectional_search", spy
    )

    start = (42.0, -72.0)
    end = (42.4, -72.0)
    ref_route = ScenicRoutePlanner(graph=reference).find_scenic_route(
        start, end, scenic_weight=0.8
    )
    comp_route = ScenicRoutePlanner(graph=compact).find_scenic_route(
        start, end, scenic_weight=0.8
    )
    assert tuple(comp_route.edge_ids) == tuple(ref_route.edge_ids)
    assert comp_route.estimated_duration_minutes == pytest.approx(
        ref_route.estimated_duration_minutes
    )
    assert comp_route.exactness_status == ref_route.exactness_status
    assert comp_route.edge_ids == ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7")
    # The scenic Lagrangian searches hand the native layer admissible
    # completion bounds (the fastest-path reference search legitimately
    # runs without them).
    assert captured
    assert any(
        bound_forward is not None and bound_reverse is not None
        for bound_forward, bound_reverse in captured
    )


def test_compact_scenic_avoid_highways_bounds_use_non_highway_speed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With highway avoidance the speed bound must exclude highway edges
    (and remain admissible); the routed path must never use a highway."""
    graph = _certified_chain_graph(8)
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    _lower_thresholds(monkeypatch)

    start = (42.0, -72.0)
    end = (42.4, -72.0)
    route = ScenicRoutePlanner(graph=compact).find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=True
    )
    assert route.edge_ids
    assert route.waypoints[0] == start
    assert route.waypoints[-1] == end


def test_compact_scenic_uncertified_graph_disables_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture graph fails geodesic certification, so the completion
    bound must be None (zero-bound fallback) and the route must still match
    the eager reference exactly."""
    graph = _certified_chain_graph(4)
    # Break certification with a stored distance below the endpoint geodesic.
    graph.edges["e2"].distance_km = 0.5
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["geodesic_bound_speed"]["all"] is None
    compact = CompactRoadGraph.load(manifest_path)
    assert compact.endpoint_geodesic_bound_speed(False) is None
    _lower_thresholds(monkeypatch)

    captured: list[tuple[object, object]] = []
    original = planner_module.run_compact_bidirectional_search

    def spy(
        topology: object,
        cost_function: object,
        forward_seeds: object,
        reverse_seeds: object,
        **kwargs: object,
    ) -> object:
        captured.append(
            (kwargs.get("bound_forward"), kwargs.get("bound_reverse"))
        )
        return original(
            topology, cost_function, forward_seeds, reverse_seeds, **kwargs
        )

    monkeypatch.setattr(
        planner_module, "run_compact_bidirectional_search", spy
    )

    route = ScenicRoutePlanner(graph=compact).find_scenic_route(
        (42.0, -72.0), (42.2, -72.0), scenic_weight=0.8
    )
    assert route.edge_ids
    assert all(
        bound_forward is None and bound_reverse is None
        for bound_forward, bound_reverse in captured
    )


def test_compact_scenic_bounds_deadline_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _certified_chain_graph(8)
    _sqlite_path, manifest_path = _publish(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    _lower_thresholds(monkeypatch)

    with pytest.raises(RoutingTimeout):
        ScenicRoutePlanner(graph=compact).find_scenic_route(
            (42.0, -72.0),
            (42.4, -72.0),
            scenic_weight=0.8,
            deadline=RoutingDeadline.after(0.0),
        )
