"""Deterministic in-memory benchmark for scenic-priority routing.

The benchmark includes bounded exact-oracle cases and a deterministic
production-frontier stress case. Every score and invariant is recomputed from
the returned route segments before a metric is emitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.cost import is_highway_road_type  # noqa: E402
from src.route_planner.graph import Edge, Node, RoadGraph  # noqa: E402
from src.route_planner.cancellation import RoutingTimeout  # noqa: E402
from src.route_planner.planner import ScenicRoutePlanner  # noqa: E402


_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BenchmarkCase:
    """One fixed graph and one routing policy for the benchmark."""

    name: str
    graph: RoadGraph
    start: tuple[float, float]
    end: tuple[float, float]
    max_detour_factor: float
    avoid_highways: bool = False
    frontier_call_budget: int | None = None


class _CallBudgetFrontierPlanner(ScenicRoutePlanner):
    """Run the production frontier under a deterministic logical budget."""
    _MAX_FRONTIER_TIME_LIMIT_SECONDS = 120.0
    _CLOCK_SCALE = 1000.0

    def __init__(self, graph: RoadGraph, frontier_call_budget: int) -> None:
        if frontier_call_budget <= 0:
            raise ValueError("frontier call budget must be positive")
        self._frontier_call_budget = float(frontier_call_budget)
        self._logical_clock_started = False
        self._frontier_clock_calls = 0
        super().__init__(
            graph=graph,
            frontier_time_limit_seconds=(
                self._frontier_call_budget / self._CLOCK_SCALE + 0.0005
            ),
        )
        # The production search still executes unchanged. Only its deadline
        # source is replaced so preprocessing and warm starts are unmetered,
        # while frontier expansion receives a fixed call budget.
        self._monotonic = self._logical_monotonic

    def _logical_monotonic(self) -> float:
        if not self._logical_clock_started:
            return 0.0
        self._frontier_clock_calls += 1
        return self._frontier_clock_calls / self._CLOCK_SCALE

    def _frontier_warm_start_paths(self, *args: Any, **kwargs: Any) -> list[list[Edge]]:
        paths = super()._frontier_warm_start_paths(*args, **kwargs)
        self._logical_clock_started = True
        return paths


@dataclass(frozen=True)
class RouteMetrics:
    """Metrics independently recomputed from a planner route."""

    distance_km: float
    duration_minutes: float
    raw_scenic_score: float
    normalized_scenic_score: float
    node_ids: tuple[str, ...]


def _add_nodes(
    graph: RoadGraph,
    coordinates: dict[str, tuple[float, float]],
) -> None:
    for node_id, (lat, lon) in coordinates.items():
        graph.add_node(Node(id=node_id, lat=lat, lon=lon))


def _add_edge(
    graph: RoadGraph,
    edge_id: str,
    start_node_id: str,
    end_node_id: str,
    distance_km: float,
    scenic_score: float,
    speed_limit_kmh: int,
    *,
    road_type: str = "secondary",
) -> None:
    graph.add_edge(
        Edge(
            id=edge_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            distance_km=distance_km,
            scenic_score=scenic_score,
            road_type=road_type,
            speed_limit_kmh=speed_limit_kmh,
            one_way=True,
        )
    )


def build_fast_ugly_detour_graph() -> RoadGraph:
    """Build a fast low-score route beside a slower high-score detour."""

    graph = RoadGraph()
    _add_nodes(
        graph,
        {
            "S": (42.0000, -72.0000),
            "A": (42.0100, -72.0100),
            "G": (42.0200, -72.0000),
        },
    )
    _add_edge(graph, "fast-ugly", "S", "G", 10.0, 0.0, 100)
    _add_edge(graph, "scenic-west", "S", "A", 4.0, 9.0, 40)
    _add_edge(graph, "scenic-east", "A", "G", 4.0, 9.0, 40)
    # A high-score edge invites a backtracking prefix but must never be used
    # because it returns to S, which is already in the path.
    _add_edge(graph, "backtrack-to-start", "A", "S", 0.01, 10.0, 60)
    return graph


def build_competing_scenic_detours_graph() -> RoadGraph:
    """Build two feasible scenic detours and one slower competing option."""

    graph = RoadGraph()
    _add_nodes(
        graph,
        {
            "S": (43.0000, -72.0000),
            "A": (43.0100, -72.0100),
            "B": (43.0100, -71.9900),
            "C": (43.0200, -72.0150),
            "D": (43.0200, -71.9850),
            "G": (43.0300, -72.0000),
        },
    )
    # The highway is faster in the unconstrained graph, while the benchmark
    # asks for avoid_highways=True so the secondary-edge baseline is used.
    _add_edge(graph, "highway-shortcut", "S", "G", 8.0, 0.0, 80, road_type="motorway")
    _add_edge(graph, "eligible-fast", "S", "G", 10.0, 1.0, 100)
    _add_edge(graph, "detour-a-1", "S", "A", 3.0, 7.0, 30)
    _add_edge(graph, "detour-a-2", "A", "G", 3.0, 7.0, 30)
    _add_edge(graph, "detour-b-1", "S", "B", 4.0, 9.0, 40)
    _add_edge(graph, "detour-b-2", "B", "G", 4.0, 9.0, 40)
    _add_edge(graph, "detour-c-1", "S", "C", 2.5, 10.0, 25)
    _add_edge(graph, "detour-c-2", "C", "D", 2.5, 10.0, 25)
    _add_edge(graph, "detour-c-3", "D", "G", 2.5, 10.0, 25)
    _add_edge(graph, "detour-backtrack", "B", "S", 0.01, 10.0, 60)
    # Force this representative case through the production multi-label
    # frontier instead of the bounded exact oracle.
    for index in range(13):
        graph.add_node(
            Node(
                id=f"isolated-{index}",
                lat=50.0 + index,
                lon=-80.0,
            )
        )
    return graph


def build_frontier_timeout_stress_graph(
    stages: int = 30,
    *,
    latitude: float = 46.0,
) -> RoadGraph:
    """Build many non-dominated alternatives for the production frontier."""

    stages = int(stages)
    options = 4
    graph = RoadGraph()
    graph.add_node(Node(id="S", lat=latitude, lon=-72.0))
    for stage in range(stages):
        graph.add_node(
            Node(
                id=f"L{stage + 1}",
                lat=latitude + 0.01 * (stage + 1),
                lon=-72.0,
            )
        )
    graph.add_node(
        Node(
            id="G",
            lat=latitude + 0.01 * (stages + 1),
            lon=-72.0,
        )
    )
    nodes = ["S"] + [f"L{index}" for index in range(1, stages + 1)] + ["G"]
    for stage in range(stages + 1):
        for option in range(options):
            distance_km = (
                1.0
                + 0.35 * option
                + 0.04 * ((stage * 17 + option * 11) % 11) / 10.0
            )
            speed_limit_kmh = 60 - 5 * option - ((stage * 3 + option * 5) % 4)
            scenic_score = max(
                0.0,
                min(
                    10.0,
                    1.0
                    + 2.3 * option
                    + 0.14 * (((stage * 7 + option * 3) % 11) - 5),
                ),
            )
            _add_edge(
                graph,
                f"stress-{stage}-{option}",
                nodes[stage],
                nodes[stage + 1],
                distance_km,
                scenic_score,
                speed_limit_kmh,
            )
    return graph


def build_hard_cap_graph() -> RoadGraph:
    """Build a 1.1-cap case with a high-score detour outside the cap."""

    graph = RoadGraph()
    _add_nodes(
        graph,
        {
            "S": (44.0000, -72.0000),
            "M": (44.0100, -72.0100),
            "A": (44.0100, -71.9900),
            "G": (44.0200, -72.0000),
        },
    )
    _add_edge(graph, "cap-fast", "S", "G", 10.0, 0.0, 100)
    # 6.3 minutes, which is feasible under the 6.6-minute hard cap.
    _add_edge(graph, "cap-feasible-1", "S", "M", 2.1, 4.0, 40)
    _add_edge(graph, "cap-feasible-2", "M", "G", 2.1, 4.0, 40)
    # 12 minutes and score 10: attractive, but outside 1.1 * 6 minutes.
    _add_edge(graph, "cap-too-slow-1", "S", "A", 5.0, 10.0, 50)
    _add_edge(graph, "cap-too-slow-2", "A", "G", 5.0, 10.0, 50)
    _add_edge(graph, "cap-backtrack", "A", "S", 0.01, 10.0, 60)
    return graph


def build_cycle_backtracking_graph() -> RoadGraph:
    """Build a scenic route with high-score cycle edges around it."""

    graph = RoadGraph()
    _add_nodes(
        graph,
        {
            "S": (45.0000, -72.0000),
            "A": (45.0100, -72.0100),
            "B": (45.0200, -71.9900),
            "G": (45.0300, -72.0000),
        },
    )
    _add_edge(graph, "cycle-fast", "S", "G", 8.0, 0.0, 80)
    _add_edge(graph, "cycle-scenic-1", "S", "A", 3.0, 8.0, 30)
    _add_edge(graph, "cycle-scenic-2", "A", "B", 3.0, 8.0, 30)
    _add_edge(graph, "cycle-scenic-3", "B", "G", 3.0, 8.0, 30)
    # Both edges are tempting high-score backtracking moves.  A valid route
    # must not revisit A or S after leaving them.
    _add_edge(graph, "cycle-back-to-a", "B", "A", 0.01, 10.0, 60)
    _add_edge(graph, "cycle-back-to-s", "A", "S", 0.01, 10.0, 60)
    return graph


def build_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return all fixed benchmark cases in stable output order."""

    return (
        BenchmarkCase(
            name="fast_ugly_detour",
            graph=build_fast_ugly_detour_graph(),
            start=(42.0000, -72.0000),
            end=(42.0200, -72.0000),
            max_detour_factor=2.0,
        ),
        BenchmarkCase(
            name="competing_scenic_detours",
            graph=build_competing_scenic_detours_graph(),
            start=(43.0000, -72.0000),
            end=(43.0300, -72.0000),
            max_detour_factor=2.1,
            avoid_highways=True,
        ),
        BenchmarkCase(
            name="frontier_timeout_stress",
            graph=build_frontier_timeout_stress_graph(),
            start=(46.0, -72.0),
            end=(46.31, -72.0),
            max_detour_factor=1.1,
            frontier_call_budget=30000,
        ),
        BenchmarkCase(
            name="frontier_extended_stress",
            graph=build_frontier_timeout_stress_graph(
                stages=40,
                latitude=47.0,
            ),
            start=(47.0, -72.0),
            frontier_call_budget=100000,
            end=(47.41, -72.0),
            max_detour_factor=1.1,
        ),
        BenchmarkCase(
            name="hard_cap_1_1",
            graph=build_hard_cap_graph(),
            start=(44.0000, -72.0000),
            end=(44.0200, -72.0000),
            max_detour_factor=1.1,
        ),
        BenchmarkCase(
            name="cycle_backtracking",
            graph=build_cycle_backtracking_graph(),
            start=(45.0000, -72.0000),
            end=(45.0300, -72.0000),
            max_detour_factor=3.0,
        ),
    )


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _finite_coordinate(value: object, label: str) -> tuple[float, float]:
    try:
        if len(value) != 2:  # type: ignore[arg-type]
            raise ValueError
        latitude = float(value[0])  # type: ignore[index]
        longitude = float(value[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError, KeyError, OverflowError) as exc:
        raise ValueError(f"{label} must contain two finite coordinates") from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError(f"{label} must contain two finite coordinates")
    return latitude, longitude


def _coords_match(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return all(
        math.isclose(left[index], right[index], rel_tol=0.0, abs_tol=_TOLERANCE)
        for index in range(2)
    )


def _node_id_for_coords(graph: RoadGraph, coordinates: tuple[float, float]) -> str:
    for node in graph.nodes.values():
        if _coords_match((float(node.lat), float(node.lon)), coordinates):
            return str(node.id)
    raise ValueError(f"route coordinate does not identify a graph node: {coordinates!r}")


def _close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=_TOLERANCE)


def recompute_route_metrics(
    graph: RoadGraph,
    route: Any,
    expected_start: tuple[float, float],
    expected_end: tuple[float, float],
    *,
    avoid_highways: bool,
) -> RouteMetrics:
    """Validate a route and independently recompute its aggregate metrics."""

    segments = getattr(route, "segments", None)
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("route must contain at least one segment")

    first_start = _finite_coordinate(getattr(segments[0], "start", None), "route start")
    last_end = _finite_coordinate(getattr(segments[-1], "end", None), "route end")
    if not _coords_match(first_start, expected_start):
        raise ValueError("route start endpoint does not match the case")
    if not _coords_match(last_end, expected_end):
        raise ValueError("route end endpoint does not match the case")

    distance_km = 0.0
    duration_minutes = 0.0
    weighted_scenic_score = 0.0
    node_ids: list[str] = [_node_id_for_coords(graph, first_start)]
    edge_ids: list[str] = []
    traversal_ids: list[str] = []
    previous_end: tuple[float, float] | None = None

    for index, segment in enumerate(segments):
        start = _finite_coordinate(getattr(segment, "start", None), f"segment {index} start")
        end = _finite_coordinate(getattr(segment, "end", None), f"segment {index} end")
        if previous_end is not None and not _coords_match(previous_end, start):
            raise ValueError("route segments are not continuous")
        previous_end = end

        distance = _finite_nonnegative(
            getattr(segment, "distance_km", None), f"segment {index} distance"
        )
        duration = _finite_nonnegative(
            getattr(segment, "duration_minutes", None), f"segment {index} duration"
        )
        scenic_score = _finite_nonnegative(
            getattr(segment, "scenic_score", None), f"segment {index} scenic score"
        )
        if scenic_score > 10.0:
            raise ValueError(f"segment {index} scenic score is outside [0, 10]")

        start_id = _node_id_for_coords(graph, start)
        end_id = _node_id_for_coords(graph, end)
        if start_id != node_ids[-1]:
            raise ValueError("route segment starts at the wrong graph node")
        node_ids.append(end_id)

        edge_id = str(getattr(segment, "edge_id", ""))
        if not edge_id or edge_id not in graph.edges:
            raise ValueError(f"route segment references unknown edge: {edge_id!r}")
        edge = graph.edges[edge_id]
        direction = str(getattr(segment, "direction", "")).lower()
        if direction == "forward":
            expected_edge_nodes = (str(edge.start_node_id), str(edge.end_node_id))
        elif direction == "reverse":
            if bool(edge.one_way):
                raise ValueError("route uses a reverse traversal of a one-way edge")
            expected_edge_nodes = (str(edge.end_node_id), str(edge.start_node_id))
        else:
            raise ValueError(f"segment {index} has an invalid traversal direction")
        if (start_id, end_id) != expected_edge_nodes:
            raise ValueError("route segment endpoints do not match its edge")
        if not _close_enough(distance, float(edge.distance_km)):
            raise ValueError("route segment distance disagrees with the graph edge")
        if not _close_enough(scenic_score, float(edge.scenic_score)):
            raise ValueError("route segment scenic score disagrees with the graph edge")
        if not _close_enough(duration, float(edge.travel_time_minutes)):
            raise ValueError("route segment duration disagrees with the graph edge")
        if avoid_highways and is_highway_road_type(edge.road_type):
            raise ValueError("route contains a prohibited highway edge")

        traversal_id = str(getattr(segment, "traversal_id", ""))
        if not traversal_id:
            raise ValueError("route segment is missing traversal identity")
        edge_ids.append(edge_id)
        traversal_ids.append(traversal_id)
        distance_km += distance
        duration_minutes += duration
        weighted_scenic_score += distance * scenic_score

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("route backtracks or cycles through a graph node")
    if distance_km <= 0.0:
        raise ValueError("route distance must be positive")

    reported_edge_ids = tuple(getattr(route, "edge_ids", ()))
    if reported_edge_ids != tuple(edge_ids):
        raise ValueError("route edge_ids disagree with its segments")
    reported_traversal_ids = tuple(getattr(route, "traversal_ids", ()))
    if reported_traversal_ids != tuple(traversal_ids):
        raise ValueError("route traversal_ids disagree with its segments")

    raw_scenic_score = weighted_scenic_score / distance_km
    normalized_scenic_score = raw_scenic_score / 10.0
    reported_distance = _finite_nonnegative(
        getattr(route, "total_distance_km", None), "route distance"
    )
    reported_duration = _finite_nonnegative(
        getattr(route, "estimated_duration_minutes", None), "route duration"
    )
    reported_raw_score = _finite_nonnegative(
        getattr(route, "average_scenic_score", None), "route scenic score"
    )
    reported_normalized_score = _finite_nonnegative(
        getattr(route, "normalized_scenic_score", None),
        "route normalized scenic score",
    )
    if reported_normalized_score > 1.0:
        raise ValueError("route normalized scenic score is outside [0, 1]")
    if not _close_enough(reported_distance, distance_km):
        raise ValueError("route distance disagrees with its segments")
    if not _close_enough(reported_duration, duration_minutes):
        raise ValueError("route duration disagrees with its segments")
    if not _close_enough(reported_raw_score, raw_scenic_score):
        raise ValueError("route scenic score disagrees with its segments")
    if not _close_enough(reported_normalized_score, normalized_scenic_score):
        raise ValueError("route normalized scenic score disagrees with its segments")

    return RouteMetrics(
        distance_km=distance_km,
        duration_minutes=duration_minutes,
        raw_scenic_score=raw_scenic_score,
        normalized_scenic_score=normalized_scenic_score,
        node_ids=tuple(node_ids),
    )


def _run_case(case: BenchmarkCase) -> tuple[float, float, float]:
    max_detour_factor = _finite_nonnegative(
        case.max_detour_factor, f"{case.name} max detour factor"
    )
    if max_detour_factor < 1.0:
        raise ValueError(f"{case.name} max detour factor must be at least one")

    planner: ScenicRoutePlanner
    if case.frontier_call_budget is None:
        planner = ScenicRoutePlanner(graph=case.graph)
    else:
        planner = _CallBudgetFrontierPlanner(
            graph=case.graph,
            frontier_call_budget=case.frontier_call_budget,
        )

    fastest_route = planner.find_fastest_route(
        case.start,
        case.end,
        avoid_highways=case.avoid_highways,
    )
    fastest = recompute_route_metrics(
        case.graph,
        fastest_route,
        case.start,
        case.end,
        avoid_highways=case.avoid_highways,
    )
    try:
        scenic_route = planner.find_scenic_route(
            case.start,
            case.end,
            scenic_weight=1.0,
            avoid_highways=case.avoid_highways,
            max_detour_factor=max_detour_factor,
            scenic_priority=True,
        )
    except RoutingTimeout:
        scenic_route = fastest_route
    scenic = recompute_route_metrics(
        case.graph,
        scenic_route,
        case.start,
        case.end,
        avoid_highways=case.avoid_highways,
    )

    if fastest.duration_minutes <= 0.0:
        raise ValueError(f"{case.name} fastest duration must be positive")
    duration_cap = fastest.duration_minutes * max_detour_factor
    if not math.isfinite(duration_cap):
        raise ValueError(f"{case.name} duration cap is not finite")
    if fastest.duration_minutes > duration_cap + _TOLERANCE:
        raise ValueError(f"{case.name} fastest route exceeds its own duration cap")
    if scenic.duration_minutes > duration_cap + _TOLERANCE:
        raise ValueError(f"{case.name} scenic route exceeds its duration cap")
    if scenic.normalized_scenic_score + _TOLERANCE < fastest.normalized_scenic_score:
        raise ValueError(f"{case.name} scenic route is less scenic than fastest route")

    return (
        scenic.normalized_scenic_score,
        scenic.normalized_scenic_score - fastest.normalized_scenic_score,
        scenic.duration_minutes / fastest.duration_minutes,
    )


def run_benchmark() -> tuple[float, float, float, int]:
    """Run every case and return deterministic aggregate metrics."""

    cases = build_benchmark_cases()
    if not cases:
        raise ValueError("benchmark must contain at least one case")
    results = [_run_case(case) for case in cases]
    case_count = len(results)
    normalized_score = sum(row[0] for row in results) / case_count
    scenic_uplift = sum(row[1] for row in results) / case_count
    duration_ratio_max = max(row[2] for row in results)
    values = (normalized_score, scenic_uplift, duration_ratio_max)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("benchmark aggregates must be finite and nonnegative")
    return normalized_score, scenic_uplift, duration_ratio_max, case_count


def main() -> None:
    normalized_score, scenic_uplift, duration_ratio_max, case_count = run_benchmark()
    # Keep stdout machine-parseable and deterministic: no case logs or timing.
    print(f"METRIC normalized_scenic_score={normalized_score:.6f}")
    print(f"METRIC scenic_uplift={scenic_uplift:.6f}")
    print(f"METRIC duration_ratio_max={duration_ratio_max:.6f}")
    print(f"METRIC cases={case_count}")


if __name__ == "__main__":
    main()
