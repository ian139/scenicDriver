from __future__ import annotations


import pytest

from src.route_planner.cost import (
    PathEvaluation,
    clamp_scenic_score,
    combined_utility,
    compare_path_evaluations,
    distance_weighted_scenic_score,
    duration_component,
    evaluate_path,
    normalize_scenic_score,
)
from src.route_planner.graph import Edge


def edge(edge_id: str, distance: float, score: float, speed: float = 60.0, road_type: str = "secondary") -> Edge:
    return Edge(
        id=edge_id,
        start_node_id=f"{edge_id}-s",
        end_node_id=f"{edge_id}-t",
        distance_km=distance,
        scenic_score=score,
        speed_limit_kmh=int(speed),
        road_type=road_type,
    )


def test_distance_weighting_clamps_scores_and_normalizes_stably() -> None:
    path = [edge("a", 2.0, -4.0), edge("b", 1.0, 20.0)]

    assert distance_weighted_scenic_score(path) == pytest.approx(10.0 / 3.0)
    assert distance_weighted_scenic_score([edge("a", 4.0, -4.0), edge("b", 2.0, 20.0)]) == pytest.approx(
        distance_weighted_scenic_score(path)
    )
    assert clamp_scenic_score(-1) == 0.0
    assert clamp_scenic_score(12) == 10.0
    assert normalize_scenic_score(5) == pytest.approx(0.5)
    assert normalize_scenic_score(15) == 1.0


def test_duration_component_handles_kappa_one_and_endpoints() -> None:
    assert duration_component(10.0, 10.0, 1.0) == 1.0
    assert duration_component(10.0001, 10.0, 1.0) == 0.0
    assert duration_component(20.0, 10.0, 2.0) == 0.0
    assert duration_component(10.0, 10.0, 2.0) == 1.0
    assert combined_utility(0.0, 2.0, 10.0, 10.0, 0.0) == 1.0
    assert combined_utility(1.0, 2.0, 20.0, 10.0, 0.75) == pytest.approx(0.75)


def test_invalid_objective_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_scenic_score(float("nan"))
    with pytest.raises(ValueError):
        duration_component(1.0, 1.0, 0.9)
    with pytest.raises(ValueError):
        combined_utility(1.1, 2.0, 1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        distance_weighted_scenic_score([edge("bad", -1.0, 5.0)])


def test_finite_but_overflowing_aggregates_are_rejected() -> None:
    huge = edge("huge", 1e308, 10.0)

    with pytest.raises(ValueError):
        distance_weighted_scenic_score([huge])
    with pytest.raises(ValueError):
        evaluate_path(
            [huge], q=1.0, kappa=1.0, fastest_duration_minutes=1e308
        )
    with pytest.raises(ValueError):
        duration_component(1.0, 1e308, 1e308)


def test_evaluation_is_independently_recomputable_and_reports_diagnostics() -> None:
    path = [edge("a", 2.0, 4.0, speed=60.0), edge("b", 1.0, 10.0, speed=30.0, road_type="motorway_link")]
    evaluation = evaluate_path(path, q=0.25, kappa=2.0, fastest_duration_minutes=2.0)

    expected_raw = (2.0 * 4.0 + 1.0 * 10.0) / 3.0
    expected_duration = (2.0 + 2.0)
    expected_duration_utility = (4.0 - expected_duration) / 2.0
    expected_objective = 0.75 * expected_duration_utility + 0.25 * (expected_raw / 10.0)
    assert evaluation.edge_ids == ("a", "b")
    assert evaluation.total_distance_km == pytest.approx(3.0)
    assert evaluation.duration_minutes == pytest.approx(expected_duration)
    assert evaluation.raw_scenic_score == pytest.approx(expected_raw)
    assert evaluation.normalized_scenic_score == pytest.approx(expected_raw / 10.0)
    assert evaluation.duration_utility == pytest.approx(expected_duration_utility)
    assert evaluation.objective == pytest.approx(expected_objective)
    assert evaluation.highway_count == 1
    assert evaluation.score_coverage == 1.0
    assert evaluation.score_run == (("a", 4.0), ("b", 10.0))
    assert evaluation.normalization_version == "linear-v1"


def test_evaluation_prefers_explicit_traversal_identity() -> None:
    reverse_view = edge("road::rev", 1.0, 6.0)
    reverse_view.traversal_id = "road|reverse"

    evaluation = evaluate_path(
        [reverse_view], q=1.0, kappa=1.0, fastest_duration_minutes=1.0
    )

    assert evaluation.edge_ids == ("road|reverse",)
    assert evaluation.score_run == (("road|reverse", 6.0),)


def test_empty_traversal_identity_falls_back_to_edge_id() -> None:
    view = edge("road::rev", 1.0, 6.0)
    view.traversal_id = ""

    evaluation = evaluate_path(
        [view], q=1.0, kappa=1.0, fastest_duration_minutes=1.0
    )

    assert evaluation.edge_ids == ("road::rev",)


def test_deterministic_comparison_uses_objective_score_duration_then_ids() -> None:
    def evaluated(edge_id: str, duration: float, score: float, objective: float) -> PathEvaluation:
        return PathEvaluation((edge_id,), 1.0, duration, score, score / 10.0, 0.0, objective, 0, 1.0, ((edge_id, score),))

    assert compare_path_evaluations(evaluated("z", 1, 5, 0.5), evaluated("a", 1, 5, 0.5)) == -1
    assert compare_path_evaluations(evaluated("a", 1, 6, 0.5), evaluated("b", 2, 5, 0.5)) == 1
    assert compare_path_evaluations(evaluated("a", 1, 5, 0.5), evaluated("b", 1, 5, 0.5)) == 1
    assert compare_path_evaluations(evaluated("a", 1, 5, 0.5), evaluated("a", 1, 5, 0.5)) == 0
