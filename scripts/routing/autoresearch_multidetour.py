"""Deterministic oracle benchmark for separated scenic detours.

The workload compares the production multi-label and compiled large-graph
searches with an independent exhaustive oracle on fixed staged graphs.  Every
oracle-optimal route contains multiple geographically separated divergences
and rejoins under one shared duration cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter, process_time_ns
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.routing.benchmark_scenic_routing import (  # noqa: E402
    recompute_route_metrics,
)
from src.route_planner.cost import (  # noqa: E402
    PathEvaluation,
    compare_path_evaluations,
    evaluate_path,
    resolve_routing_policy,
)
from src.route_planner.graph import Edge, Node, RoadGraph  # noqa: E402
from src.route_planner.planner import ScenicRoutePlanner  # noqa: E402


_TOLERANCE = 1e-9
_SCENIC_WEIGHT = 0.8
_COMPILED_REPETITIONS = 5
_DIRECT_DURATION_MINUTES = 1.0
_CONNECTOR_DURATION_MINUTES = 0.5


@dataclass(frozen=True)
class DetourSpec:
    duration_minutes: float
    scenic_score: float
    main_scenic_score: float = 0.0
    main_road_type: str = "secondary"
    detour_road_type: str = "secondary"


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    detours: tuple[DetourSpec, ...]
    max_detour_factor: float
    expected_oracle_detours: int
    scenic_weight: float = _SCENIC_WEIGHT
    scenic_priority: bool = True
    highway_preference: float = 0.0


@dataclass(frozen=True)
class OracleResult:
    evaluation: PathEvaluation
    edge_ids: tuple[str, ...]
    fastest_duration_minutes: float
    detour_indices: frozenset[int]


@dataclass(frozen=True)
class RouteResult:
    route: Any
    evaluation: PathEvaluation
    edge_ids: tuple[str, ...]
    detour_indices: frozenset[int]
    process_us: float


class _FrontierPlanner(ScenicRoutePlanner):
    """Force a bounded fixture through the production multi-label frontier."""

    _EXACT_ORACLE_MAX_NODES = 0
    _EXACT_ORACLE_MAX_EDGES = 0
    _COMPILED_SCENIC_MIN_NODES = 100_000


class _CompiledPlanner(ScenicRoutePlanner):
    """Force a bounded fixture through the production large-graph path."""

    _EXACT_ORACLE_MAX_NODES = 0
    _EXACT_ORACLE_MAX_EDGES = 0
    _ENDPOINT_OVERLAY_MAX_NODES = 0
    _LARGE_GRAPH_EDGE_THRESHOLD = 0
    _COLLAPSED_ACCESS_NODE_THRESHOLD = 0
    _COMPILED_SCENIC_MIN_NODES = 0


def _detours(*values: tuple[Any, ...]) -> tuple[DetourSpec, ...]:
    return tuple(DetourSpec(*value) for value in values)


_FIXTURES = (
    FixtureSpec(
        "symmetric_three_choose_two",
        _detours((2.5, 10.0), (2.5, 10.0), (2.5, 10.0)),
        1.75,
        2,
    ),
    FixtureSpec(
        "symmetric_four_choose_two",
        _detours((2.5, 10.0), (2.5, 10.0), (2.5, 10.0), (2.5, 10.0)),
        1.55,
        2,
    ),
    FixtureSpec(
        "symmetric_four_choose_three",
        _detours((2.5, 10.0), (2.5, 10.0), (2.5, 10.0), (2.5, 10.0)),
        1.82,
        3,
    ),
    FixtureSpec(
        "graded_three_a",
        _detours((1.5, 6.0), (2.0, 8.0), (2.5, 10.0)),
        1.65,
        2,
    ),
    FixtureSpec(
        "graded_three_b",
        _detours((1.4, 8.0), (2.2, 9.0), (2.8, 10.0)),
        1.70,
        2,
    ),
    FixtureSpec(
        "graded_four_a",
        _detours((1.5, 7.0), (1.8, 8.0), (2.2, 9.0), (2.6, 10.0)),
        1.60,
        3,
    ),
    FixtureSpec(
        "graded_four_b",
        _detours((2.8, 10.0), (2.2, 9.0), (1.8, 8.0), (1.5, 7.0)),
        1.75,
        3,
    ),
    FixtureSpec(
        "mixed_three_a",
        _detours((2.4, 10.0), (2.0, 7.0), (1.4, 5.0)),
        1.60,
        2,
    ),
    FixtureSpec(
        "mixed_three_b",
        _detours((3.0, 10.0), (1.8, 9.0), (1.6, 6.0)),
        1.70,
        2,
    ),
    FixtureSpec(
        "mixed_four_a",
        _detours((3.0, 10.0), (2.6, 9.0), (1.6, 7.0), (1.4, 6.0)),
        1.60,
        3,
    ),
    FixtureSpec(
        "plateau_three_choose_two",
        _detours((2.2, 9.0), (2.2, 9.0), (2.2, 9.0)),
        1.65,
        2,
    ),
    FixtureSpec(
        "plateau_four_choose_three",
        _detours((2.0, 8.0), (2.0, 8.0), (2.0, 8.0), (2.0, 8.0)),
        1.60,
        3,
    ),
    FixtureSpec(
        "route_gain_beats_absolute_segment_score",
        _detours(
            (1.05, 10.0, 9.0),
            (1.05, 9.0, 0.0),
            (1.05, 9.0, 0.0),
        ),
        1.025,
        2,
    ),
    FixtureSpec(
        "equal_scenery_highway_avoidance",
        _detours(
            (1.5, 5.0, 5.0, "motorway", "secondary"),
            (1.5, 5.0, 5.0, "motorway", "secondary"),
            (1.5, 5.0, 5.0, "motorway", "secondary"),
        ),
        1.25,
        2,
        scenic_weight=0.5,
        scenic_priority=False,
        highway_preference=2.0,
    ),
)


def _add_edge(
    graph: RoadGraph,
    edge_id: str,
    start_node_id: str,
    end_node_id: str,
    duration_minutes: float,
    scenic_score: float,
    *,
    road_type: str = "secondary",
) -> None:
    graph.add_edge(
        Edge(
            id=edge_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            distance_km=duration_minutes,
            scenic_score=scenic_score,
            road_type=road_type,
            speed_limit_kmh=60,
            one_way=True,
        )
    )


def _build_graph(fixture: FixtureSpec) -> RoadGraph:
    graph = RoadGraph()
    for index in range(len(fixture.detours)):
        longitude = index * 0.03
        graph.add_node(Node(id=f"A{index}", lat=0.0, lon=longitude))
        graph.add_node(Node(id=f"B{index}", lat=0.0, lon=longitude + 0.01))
        graph.add_node(Node(id=f"X{index}", lat=0.01, lon=longitude + 0.005))

    for index, detour in enumerate(fixture.detours):
        _add_edge(
            graph,
            f"main-{index}",
            f"A{index}",
            f"B{index}",
            _DIRECT_DURATION_MINUTES,
            detour.main_scenic_score,
            road_type=detour.main_road_type,
        )
        half_duration = detour.duration_minutes / 2.0
        _add_edge(
            graph,
            f"detour-{index}-in",
            f"A{index}",
            f"X{index}",
            half_duration,
            detour.scenic_score,
            road_type=detour.detour_road_type,
        )
        _add_edge(
            graph,
            f"detour-{index}-out",
            f"X{index}",
            f"B{index}",
            half_duration,
            detour.scenic_score,
            road_type=detour.detour_road_type,
        )
        if index + 1 < len(fixture.detours):
            _add_edge(
                graph,
                f"connector-{index}",
                f"B{index}",
                f"A{index + 1}",
                _CONNECTOR_DURATION_MINUTES,
                0.0,
            )
    return graph


def _endpoints(fixture: FixtureSpec) -> tuple[tuple[float, float], tuple[float, float]]:
    last = len(fixture.detours) - 1
    return (0.0, 0.0), (0.0, last * 0.03 + 0.01)


def _path_for_choices(
    graph: RoadGraph, choices: tuple[bool, ...]
) -> list[Edge]:
    path: list[Edge] = []
    for index, use_detour in enumerate(choices):
        if use_detour:
            path.extend(
                (
                    graph.edges[f"detour-{index}-in"],
                    graph.edges[f"detour-{index}-out"],
                )
            )
        else:
            path.append(graph.edges[f"main-{index}"])
        if index + 1 < len(choices):
            path.append(graph.edges[f"connector-{index}"])
    return path


def _detour_indices(edge_ids: tuple[str, ...]) -> frozenset[int]:
    indices: set[int] = set()
    for edge_id in edge_ids:
        parts = edge_id.split("-")
        if len(parts) == 3 and parts[0] == "detour" and parts[2] == "in":
            indices.add(int(parts[1]))
    return frozenset(indices)


def _independent_oracle(fixture: FixtureSpec) -> OracleResult:
    graph = _build_graph(fixture)
    fastest_path = _path_for_choices(graph, (False,) * len(fixture.detours))
    fastest_duration = sum(edge.travel_time_minutes for edge in fastest_path)
    duration_cap = fastest_duration * fixture.max_detour_factor
    policy = resolve_routing_policy(
        scenic_weight=fixture.scenic_weight,
        kappa=fixture.max_detour_factor,
        highway_preference=fixture.highway_preference,
        scenic_priority=fixture.scenic_priority,
    )
    best_path: list[Edge] | None = None
    best_evaluation: PathEvaluation | None = None
    for choices in itertools.product((False, True), repeat=len(fixture.detours)):
        path = _path_for_choices(graph, choices)
        evaluation = evaluate_path(
            path,
            q=fixture.scenic_weight,
            kappa=fixture.max_detour_factor,
            fastest_duration_minutes=fastest_duration,
            policy=policy,
        )
        if evaluation.duration_minutes > duration_cap + _TOLERANCE:
            continue
        if best_evaluation is None or compare_path_evaluations(
            evaluation, best_evaluation
        ) > 0:
            best_path = path
            best_evaluation = evaluation
    if best_path is None or best_evaluation is None:
        raise RuntimeError(f"{fixture.name}: exhaustive oracle found no route")
    edge_ids = tuple(edge.id for edge in best_path)
    detours = _detour_indices(edge_ids)
    if len(detours) != fixture.expected_oracle_detours:
        raise RuntimeError(
            f"{fixture.name}: expected {fixture.expected_oracle_detours} oracle "
            f"detours, observed {len(detours)}"
        )
    return OracleResult(
        evaluation=best_evaluation,
        edge_ids=edge_ids,
        fastest_duration_minutes=fastest_duration,
        detour_indices=detours,
    )


def _close_enough(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-10, abs_tol=_TOLERANCE)


def _run_mode(fixture: FixtureSpec, mode: str, oracle: OracleResult) -> RouteResult:
    ScenicRoutePlanner.clear_shared_caches()
    graph = _build_graph(fixture)
    if mode == "exact":
        planner: ScenicRoutePlanner = ScenicRoutePlanner(graph)
    elif mode == "frontier":
        planner = _FrontierPlanner(graph, frontier_time_limit_seconds=60.0)
    elif mode == "compiled":
        planner = _CompiledPlanner(graph)
    else:  # pragma: no cover - internal harness misuse
        raise ValueError(f"unknown routing mode: {mode}")

    start, end = _endpoints(fixture)
    started = process_time_ns()
    route = planner.find_scenic_route(
        start,
        end,
        scenic_weight=fixture.scenic_weight,
        max_detour_factor=fixture.max_detour_factor,
        highway_preference=fixture.highway_preference,
        scenic_priority=fixture.scenic_priority,
    )
    process_us = (process_time_ns() - started) / 1_000.0
    metrics = recompute_route_metrics(
        graph,
        route,
        start,
        end,
        avoid_highways=False,
    )
    edge_ids = tuple(str(edge_id) for edge_id in route.edge_ids)
    edges = [graph.edges[edge_id] for edge_id in edge_ids]
    policy = resolve_routing_policy(
        scenic_weight=fixture.scenic_weight,
        kappa=fixture.max_detour_factor,
        highway_preference=fixture.highway_preference,
        scenic_priority=fixture.scenic_priority,
    )
    evaluation = evaluate_path(
        edges,
        q=fixture.scenic_weight,
        kappa=fixture.max_detour_factor,
        fastest_duration_minutes=oracle.fastest_duration_minutes,
        policy=policy,
    )
    duration_cap = oracle.fastest_duration_minutes * fixture.max_detour_factor
    if metrics.duration_minutes > duration_cap + _TOLERANCE:
        raise RuntimeError(f"{fixture.name}/{mode}: route exceeds duration cap")
    if not _close_enough(
        float(route.fastest_duration_minutes), oracle.fastest_duration_minutes
    ):
        raise RuntimeError(f"{fixture.name}/{mode}: fastest duration changed")
    if not _close_enough(float(route.duration_cap_minutes), duration_cap):
        raise RuntimeError(f"{fixture.name}/{mode}: declared cap changed")
    if not _close_enough(float(route.objective_value), evaluation.objective):
        raise RuntimeError(f"{fixture.name}/{mode}: objective recomputation failed")
    if not _close_enough(
        metrics.normalized_scenic_score, evaluation.normalized_scenic_score
    ):
        raise RuntimeError(f"{fixture.name}/{mode}: score recomputation failed")
    if compare_path_evaluations(evaluation, oracle.evaluation) > 0:
        raise RuntimeError(f"{fixture.name}/{mode}: route exceeded exhaustive oracle")

    if mode == "exact":
        if route.algorithm != "exact-simple-path-oracle" or not route.exact:
            raise RuntimeError(f"{fixture.name}: bounded exact mode was not exercised")
    elif mode == "frontier":
        if route.algorithm != "production-multilabel-frontier" or not route.exact:
            raise RuntimeError(f"{fixture.name}: exact frontier mode was not exercised")
    else:
        if route.exact or route.exactness_status != "approximate-certified":
            raise RuntimeError(
                f"{fixture.name}: compiled approximate mode was not exercised"
            )
        if route.certified_upper_bound is None or route.optimality_gap is None:
            raise RuntimeError(
                f"{fixture.name}: compiled certification is missing"
            )
        upper_bound = float(route.certified_upper_bound)
        gap = float(route.optimality_gap)
        if upper_bound + _TOLERANCE < evaluation.objective or not _close_enough(
            gap, max(0.0, upper_bound - evaluation.objective)
        ):
            raise RuntimeError(
                f"{fixture.name}: compiled certification is inconsistent"
            )

    return RouteResult(
        route=route,
        evaluation=evaluation,
        edge_ids=edge_ids,
        detour_indices=_detour_indices(edge_ids),
        process_us=process_us,
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _fixture_digest() -> str:
    encoded = json.dumps(
        [asdict(fixture) for fixture in _FIXTURES],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    total_started = perf_counter()
    if len(_FIXTURES) != 14:
        raise RuntimeError("multi-detour fixture denominator changed")

    objective_regrets_pp: list[float] = []
    scenic_regrets_pp: list[float] = []
    detour_recalls: list[float] = []
    compiled_oracle_matches = 0
    compiled_multiple_detour_cases = 0
    exact_oracle_matches = 0
    frontier_oracle_matches = 0
    exact_times_us: list[float] = []
    frontier_times_us: list[float] = []
    compiled_times_us: list[float] = []

    for fixture in _FIXTURES:
        oracle = _independent_oracle(fixture)
        exact = _run_mode(fixture, "exact", oracle)
        frontier = _run_mode(fixture, "frontier", oracle)
        compiled_runs = [
            _run_mode(fixture, "compiled", oracle)
            for _ in range(_COMPILED_REPETITIONS)
        ]
        compiled = compiled_runs[0]

        if exact.edge_ids != oracle.edge_ids or compare_path_evaluations(
            exact.evaluation, oracle.evaluation
        ) != 0:
            raise RuntimeError(f"{fixture.name}: planner exact route disagrees with oracle")
        exact_oracle_matches += 1
        if frontier.edge_ids != oracle.edge_ids or compare_path_evaluations(
            frontier.evaluation, oracle.evaluation
        ) != 0:
            raise RuntimeError(f"{fixture.name}: frontier route disagrees with oracle")
        frontier_oracle_matches += 1

        for repeated in compiled_runs[1:]:
            if repeated.edge_ids != compiled.edge_ids or not _close_enough(
                repeated.evaluation.objective, compiled.evaluation.objective
            ):
                raise RuntimeError(f"{fixture.name}: compiled route is nondeterministic")

        objective_regret = max(
            0.0,
            oracle.evaluation.objective - compiled.evaluation.objective,
        )
        objective_regrets_pp.append(objective_regret * 100.0)
        scenic_regret = max(
            0.0,
            oracle.evaluation.normalized_scenic_score
            - compiled.evaluation.normalized_scenic_score,
        )
        scenic_regrets_pp.append(scenic_regret * 100.0)
        if compiled.edge_ids == oracle.edge_ids and compare_path_evaluations(
            compiled.evaluation, oracle.evaluation
        ) == 0:
            compiled_oracle_matches += 1
        if len(compiled.detour_indices) >= 2:
            compiled_multiple_detour_cases += 1
        detour_recalls.append(
            len(compiled.detour_indices & oracle.detour_indices)
            / len(oracle.detour_indices)
        )
        exact_times_us.append(exact.process_us)
        frontier_times_us.append(frontier.process_us)
        compiled_times_us.extend(result.process_us for result in compiled_runs)

    denominator = len(_FIXTURES)
    mean_objective_regret_pp = sum(objective_regrets_pp) / denominator
    worst_objective_regret_pp = max(objective_regrets_pp)
    mean_scenic_regret_pp = sum(scenic_regrets_pp) / denominator
    worst_scenic_regret_pp = max(scenic_regrets_pp)
    oracle_match_rate = compiled_oracle_matches / denominator
    multi_detour_rate = compiled_multiple_detour_cases / denominator
    detour_recall_rate = sum(detour_recalls) / denominator

    # A non-scenic-priority q=0 request must retain the fastest identity.
    control_fixture = _FIXTURES[0]
    control_graph = _build_graph(control_fixture)
    control_start, control_end = _endpoints(control_fixture)
    ScenicRoutePlanner.clear_shared_caches()
    control_planner = _CompiledPlanner(control_graph)
    control_route = control_planner.find_scenic_route(
        control_start,
        control_end,
        q=0.0,
        kappa=control_fixture.max_detour_factor,
        scenic_priority=False,
    )
    expected_fastest_ids = tuple(
        edge.id
        for edge in _path_for_choices(
            control_graph, (False,) * len(control_fixture.detours)
        )
    )
    if tuple(control_route.edge_ids) != expected_fastest_ids:
        raise RuntimeError("q=0 control route changed from canonical fastest")

    print("ASI benchmark=staged_separated_multi_detour_oracle")
    print(f"ASI fixture_digest={_fixture_digest()}")
    print(f"ASI denominator={denominator}")
    print("ASI policy_fixture_cases=2")
    print("ASI seed=none_no_rng")
    print("ASI workers=1")
    print(f"ASI compiled_repetitions={_COMPILED_REPETITIONS}")
    print("ASI cache_policy=fresh_graph_and_planner_clear_shared_caches_per_mode")
    print(
        "METRIC compiled_mean_objective_regret_pp="
        f"{mean_objective_regret_pp:.12f}"
    )
    print(
        "METRIC compiled_worst_objective_regret_pp="
        f"{worst_objective_regret_pp:.12f}"
    )
    print(
        "METRIC compiled_mean_scenic_regret_pp="
        f"{mean_scenic_regret_pp:.12f}"
    )
    print(
        "METRIC compiled_worst_scenic_regret_pp="
        f"{worst_scenic_regret_pp:.12f}"
    )
    print(f"METRIC compiled_oracle_match_rate={oracle_match_rate:.12f}")
    print(f"METRIC compiled_multi_detour_rate={multi_detour_rate:.12f}")
    print(f"METRIC compiled_detour_recall_rate={detour_recall_rate:.12f}")
    print(f"METRIC exact_oracle_match_rate={exact_oracle_matches / denominator:.12f}")
    print(
        "METRIC frontier_oracle_match_rate="
        f"{frontier_oracle_matches / denominator:.12f}"
    )
    print(f"METRIC compiled_case_median_us={median(compiled_times_us):.6f}")
    print(f"METRIC compiled_case_p95_us={_p95(compiled_times_us):.6f}")
    print(f"METRIC frontier_case_median_us={median(frontier_times_us):.6f}")
    print(f"METRIC exact_case_median_us={median(exact_times_us):.6f}")
    print(f"METRIC multi_detour_cases={denominator}")
    print("METRIC q0_fastest_pass=1")
    print("METRIC correctness_failures=0")
    print(f"METRIC total_wall_ms={(perf_counter() - total_started) * 1000.0:.6f}")


if __name__ == "__main__":
    main()
