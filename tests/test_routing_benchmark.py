from __future__ import annotations


from copy import deepcopy
import math
import pytest

from types import SimpleNamespace

from scripts.routing.production_benchmark import (
    KAPPA_VALUES,
    Q_VALUES,
    _case_specs,
    _classify_pairs,
    _invariant_summary,
    _evaluations_match,
    evaluate_service_response,
    recompute_feature_metrics,
)
from scripts.routing.benchmark_scenic_routing import (
    _run_case,
    build_benchmark_cases,
)

def _feature(
    kind: str,
    *,
    score: float,
    duration: float,
    edge_ids: tuple[str, ...] = ("e0",),
    road_types: tuple[str, ...] = ("secondary",),
    objective: float | None = None,
) -> dict[str, object]:
    per_segment_duration = duration / len(edge_ids)
    segments = [
        {
            "edge_id": edge_id,
            "canonical_edge_id": edge_id,
            "traversal_id": f"{index}:forward:{edge_id}",
            "direction": "forward",
            "start": [42.0 + index * 0.01, -72.0],
            "end": [42.0 + (index + 1) * 0.01, -72.0],
            "distance_km": 2.0,
            "duration_minutes": per_segment_duration,
            "scenic_score": segment_score,
            "normalized_scenic_score": segment_score / 10.0,
            "road_name": None,
            "road_type": road_type,
        }
        for index, (edge_id, road_type, segment_score) in enumerate(
            zip(edge_ids, road_types, (score,) * len(edge_ids))
        )
    ]
    properties: dict[str, object] = {
        "route_kind": kind,
        "edge_ids": list(edge_ids),
        "traversal_ids": [row["traversal_id"] for row in segments],
        "segment_identity": segments,
        "total_distance_km": 2.0 * len(segments),
        "average_scenic_score": score,
        "raw_scenic_score": score,
        "normalized_scenic_score": score / 10.0,
        "estimated_duration_minutes": duration,
        "highway_count": sum(road_type in {"highway", "motorway", "trunk"} for road_type in road_types),
        "exactness_status": "exact",
        "optimality_gap": 0.0,
        "certified_upper_bound": None,
    }
    if objective is not None:
        properties["objective_value"] = objective
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "LineString",
            "coordinates": [[-72.0, 42.0 + index * 0.01] for index in range(len(segments) + 1)],
        },
    }


def _response(*, q: float = 0.5, kappa: float = 1.8, avoid: bool = False) -> dict[str, object]:
    # Scenic duration=12, baseline duration=10, scenic score=8, baseline=4.
    # objective=(1-q)*.75+q*.8=.775 for q=.5,kappa=1.8.
    scenic_objective = (1.0 - q) * 0.75 + q * 0.8
    scenic = _feature(
        "scenic",
        score=8.0,
        duration=12.0,
        edge_ids=("s0", "s1"),
        road_types=("secondary", "residential"),
        objective=scenic_objective,
    )
    baseline = _feature(
        "baseline",
        score=4.0,
        duration=10.0,
        edge_ids=("b0",),
        road_types=("secondary",),
        objective=1.0,
    )
    for feature in (scenic, baseline):
        feature["properties"].update(
            {
                "requested_scenic_weight": q,
                "applied_scenic_weight": q,
                "requested_max_detour_factor": kappa,
                "applied_max_detour_factor": kappa,
            }
        )
    scenic["properties"].update(
        {
            "scenic_score_delta_absolute": 4.0,
            "scenic_score_delta_relative": 1.0,
            "same_route": False,
            "no_better_route_reason": None,
        }
    )
    baseline["properties"].update(
        {
            "scenic_score_delta_absolute": None,
            "scenic_score_delta_relative": None,
            "same_route": False,
            "no_better_route_reason": None,
        }
    )
    return {
        "request": {
            "scenic_weight": q,
            "max_detour_factor": kappa,
            "avoid_highways": avoid,
            "include_baseline": True,
        },
        "diagnostics": {
            "graph_cache_hit": True,
            "tile_score_cache_hit": True,
            "scored_graph_cache_hit": True,
            "requested_scenic_weight": q,
            "applied_scenic_weight": q,
            "requested_max_detour_factor": kappa,
            "applied_max_detour_factor": kappa,
            "avoid_highways_applied": avoid,
            "certified_upper_bound": None,
            "optimality_gap": 0.0,
            "exactness_status": "exact",
            "normalized_scenic_score": 0.8,
            "scenic_score_delta_absolute": 4.0,
            "scenic_score_delta_relative": 1.0,
            "same_route": False,
            "no_better_route_reason": None,
        },
        "score_mapping": {"report_signature": "report-a", "graph_signature": "graph-a"},
        "routes": [
            {"route_kind": "scenic", "metrics": scenic["properties"]},
            {"route_kind": "baseline", "metrics": baseline["properties"]},
        ],
        "geojson": {"type": "FeatureCollection", "features": [scenic, baseline]},
    }


def test_recompute_feature_metrics_uses_segment_identity() -> None:
    feature = _feature(
        "scenic",
        score=7.0,
        duration=3.0,
        edge_ids=("e0", "e1"),
        road_types=("trunk", "secondary"),
    )
    metrics = recompute_feature_metrics(feature)
    assert metrics["distance_km"] == pytest.approx(4.0)
    assert metrics["raw_scenic_score"] == pytest.approx(7.0)
    assert metrics["segment_identity_first"] == [
        "0:forward:e0",
        "1:forward:e1",
    ]
    assert metrics["geometry_point_count"] == 3
    assert metrics["segment_identity_sha256"] != metrics["geometry_sha256"]


def test_recompute_feature_metrics_includes_requested_endpoints() -> None:
    feature = _feature(
        "scenic",
        score=7.0,
        duration=3.0,
        edge_ids=("e0", "e1"),
        road_types=("secondary", "residential"),
    )
    edge_index = {
        "e0": SimpleNamespace(
            start_node_id="n0",
            end_node_id="n1",
            distance_km=2.0,
            scenic_score=7.0,
            travel_time_minutes=1.5,
            road_type="secondary",
        ),
        "e1": SimpleNamespace(
            start_node_id="n1",
            end_node_id="n2",
            distance_km=2.0,
            scenic_score=7.0,
            travel_time_minutes=1.5,
            road_type="residential",
        ),
    }
    metrics = recompute_feature_metrics(
        feature,
        edge_index=edge_index,
        requested_start=(41.99, -72.0),
        requested_end=(42.02, -72.0),
    )

    assert metrics["edge_metric_consistency"]["geometry_sequence"] is False



@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        ([[-72.0, 42.0]], False),
        ([[-72.0, 42.0], [-72.0, 42.0]], True),
    ],
)
def test_recompute_feature_metrics_validates_zero_edge_equal_endpoint_geometry(
    coordinates: list[list[float]], expected: bool
) -> None:
    feature = {
        "properties": {
            "edge_ids": [],
            "traversal_ids": [],
            "segment_identity": [],
            "total_distance_km": 0.0,
            "raw_scenic_score": 0.0,
            "average_scenic_score": 0.0,
            "normalized_scenic_score": 0.0,
            "estimated_duration_minutes": 0.0,
            "highway_count": 0,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
    }

    metrics = recompute_feature_metrics(
        feature,
        requested_start=(42.0, -72.0),
        requested_end=(42.0, -72.0),
    )

    assert metrics["edge_metric_consistency"]["geometry_sequence"] is expected


def _zero_edge_response(
    coordinates: list[list[float]],
) -> dict[str, object]:
    response = _response()
    features = response["geojson"]["features"]
    assert isinstance(features, list)
    for feature in features:
        assert isinstance(feature, dict)
        properties = feature["properties"]
        assert isinstance(properties, dict)
        properties.update(
            {
                "edge_ids": [],
                "traversal_ids": [],
                "segment_identity": [],
                "total_distance_km": 0.0,
                "average_scenic_score": 0.0,
                "raw_scenic_score": 0.0,
                "normalized_scenic_score": 0.0,
                "estimated_duration_minutes": 0.0,
                "duration_utility": 1.0,
                "actual_duration_ratio": 1.0,
                "scenic_score_delta_absolute": 0.0,
                "scenic_score_delta_relative": None,
                "same_route": True,
                "no_better_route_reason": "same_route",
                "objective_value": 0.5,
            }
        )
        geometry = feature["geometry"]
        assert isinstance(geometry, dict)
        geometry["coordinates"] = coordinates
    routes = response["routes"]
    assert isinstance(routes, list)
    for route, feature in zip(routes, features):
        assert isinstance(route, dict)
        assert isinstance(feature, dict)
        route["metrics"] = feature["properties"]
    diagnostics = response["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics.update(
        {
            "normalized_scenic_score": 0.0,
            "scenic_score_delta_absolute": 0.0,
            "scenic_score_delta_relative": None,
            "same_route": True,
            "no_better_route_reason": "same_route",
        }
    )
    return response


@pytest.mark.parametrize(
    ("coordinates", "expected_status"),
    [
        ([[-72.0, 42.0]], "invalid"),
        ([[-72.0, 42.0], [-72.0, 42.0]], "ok"),
    ],
)
def test_evaluate_service_response_validates_zero_edge_geometry(
    coordinates: list[list[float]], expected_status: str
) -> None:
    evaluation = evaluate_service_response(
        _zero_edge_response(coordinates),
        q=0.5,
        kappa=1.8,
        avoid_highways=False,
        requested_start=(42.0, -72.0),
        requested_end=(42.0, -72.0),
    )

    assert evaluation["status"] == expected_status
    if expected_status == "ok":
        assert evaluation["failed_invariants"] == []
        assert evaluation["duration_minutes"] == pytest.approx(0.0)
        assert evaluation["objective"]["recomputed"] == pytest.approx(0.5)
    else:
        assert "edge_metric_recomputation" in evaluation["failed_invariants"]



def test_evaluate_service_response_rejects_zero_edge_distinct_endpoints() -> None:
    evaluation = evaluate_service_response(
        _zero_edge_response([[-72.0, 42.0], [-71.99, 42.0]]),
        q=0.5,
        kappa=1.8,
        avoid_highways=False,
        requested_start=(42.0, -72.0),
        requested_end=(42.0, -71.99),
    )

    assert evaluation["status"] == "invalid"
    assert "edge_metric_recomputation" in evaluation["failed_invariants"]

def test_recompute_feature_metrics_uses_canonical_edge_duration() -> None:
    feature = _feature(
        "scenic",
        score=7.0,
        duration=3.0,
        edge_ids=("e0", "e1"),
        road_types=("trunk", "secondary"),
    )
    edge_index = {
        "e0": SimpleNamespace(
            distance_km=2.0,
            scenic_score=7.0,
            travel_time_minutes=1.5,
            road_type="trunk",
        ),
        "e1": SimpleNamespace(
            distance_km=2.0,
            scenic_score=7.0,
            travel_time_minutes=1.5,
            road_type="secondary",
        ),
    }
    metrics = recompute_feature_metrics(feature, edge_index=edge_index)
    assert metrics["duration_minutes_recomputed"] == pytest.approx(3.0)
    assert metrics["duration_minutes"] == pytest.approx(3.0)
    assert metrics["edge_metric_consistency"] == {
        "top_level_edge_ids": True,
        "segments": True,
        "continuity": True,
        "simple_path": True,
        "traversal_identity": True,
        "geometry_sequence": True,
        "distance": True,
        "raw_scenic_score": True,
        "average_scenic_score": True,
        "normalized_scenic_score": True,
        "duration": True,
        "highway_count": True,
    }




def test_evaluate_service_response_rejects_missing_scenic_feature() -> None:
    response = _response()
    response["geojson"]["features"] = [
        response["geojson"]["features"][1]
    ]

    with pytest.raises(ValueError, match="did not contain a scenic feature"):
        evaluate_service_response(
            response,
            q=0.5,
            kappa=1.8,
            avoid_highways=False,
        )

def test_evaluate_service_response_recomputes_objective_and_cap() -> None:
    evaluation = evaluate_service_response(_response(), q=0.5, kappa=1.8, avoid_highways=False)
    assert evaluation["status"] == "ok", evaluation["failed_invariants"]
    assert evaluation["distance_km"] == pytest.approx(4.0)
    assert evaluation["duration_ratio"] == pytest.approx(1.2)
    assert evaluation["objective"]["recomputed"] == pytest.approx(0.775)
    assert evaluation["objective"]["absolute_error"] == pytest.approx(0.0)
    assert evaluation["uplift_absolute"] == pytest.approx(4.0)
    assert evaluation["invariants"]["duration_cap"] is True
    assert evaluation["invariants"]["objective_recomputation"] is True


def test_evaluate_service_response_flags_prohibited_highway_and_q0_violation() -> None:
    response = _response(q=0.0, kappa=1.0)
    scenic = response["geojson"]["features"][0]
    scenic["properties"]["estimated_duration_minutes"] = 12.0
    scenic["properties"]["segment_identity"][0]["road_type"] = "trunk"
    scenic["properties"]["highway_count"] = 1
    evaluation = evaluate_service_response(response, q=0.0, kappa=1.0, avoid_highways=True)
    assert evaluation["invariants"]["prohibited_highways"] is False
    assert evaluation["invariants"]["q0_fastest"] is False
    assert evaluation["invariants"]["duration_cap"] is False


def test_invariant_summary_records_cache_boundary() -> None:
    evaluation = evaluate_service_response(_response(), q=0.5, kappa=1.8, avoid_highways=False)
    rows = [
        {
            "pair_id": "p",
            "q": 0.5,
            "kappa": 1.8,
            "avoid_highways": False,
            "evaluation": evaluation,
        }
    ]
    summary = _invariant_summary(rows)
    assert summary["cache_boundaries"] == {"pass": 1}
    assert summary["cache_isolation"] == {"not_proven_single_variant": 1}
    assert summary["warm_cache_hits"] == {"skip": 1}


def test_case_matrix_is_explicit_for_all_pairs_and_settings() -> None:
    pairs = [
        {"id": f"pair-{index}", "start": [42.0, -72.0], "end": [42.01, -72.0]}
        for index in range(22)
    ]
    specs = _case_specs({"pairs": pairs})
    assert len(specs) == 22 * len(Q_VALUES) * len(KAPPA_VALUES) * 2
    assert {spec.q for spec in specs} == set(Q_VALUES)
    assert {spec.kappa for spec in specs} == set(KAPPA_VALUES)
    assert {spec.avoid_highways for spec in specs} == {False, True}


def test_categories_are_empirical_and_screenshot_is_not_fabricated() -> None:
    pair = {"id": "p", "start": [44.0, -72.0], "end": [44.01, -72.0]}
    corpus = {"pairs": [pair], "historical_screenshot_coordinates": None}
    response = evaluate_service_response(_response(), q=0.5, kappa=1.8, avoid_highways=False)
    rows = [
        {
            "pair_id": "p",
            "q": 0.5,
            "kappa": 1.8,
            "avoid_highways": False,
            "reason": None,
            "evaluation": response,
        }
    ]
    discovered, missing = _classify_pairs(corpus, rows)
    assert discovered["short urban"] == ["p"]
    assert discovered["obvious scenic candidate"] == ["p"]
    assert "obvious scenic candidate" not in missing
    assert "checked-in default reproduction" in missing
    assert corpus["historical_screenshot_coordinates"] is None


@pytest.mark.parametrize(
    ("mutate", "failed_invariant"),
    [
        (
            lambda response: response["geojson"]["features"][0]["properties"][
                "segment_identity"
            ][0].pop("duration_minutes"),
            "edge_metric_recomputation",
        ),
        (
            lambda response: response["geojson"]["features"][0]["properties"].update(
                {"edge_ids": ["wrong", "s1"]}
            ),
            "edge_metric_recomputation",
        ),
        (
            lambda response: response["routes"][0].__setitem__(
                "metrics",
                {
                    **response["routes"][0]["metrics"],
                    "raw_scenic_score": 9.0,
                },
            ),
            "route_metrics_consistency",
        ),
        (
            lambda response: response["geojson"]["features"][0]["properties"].update(
                {"average_scenic_score": 9.0}
            ),
            "edge_metric_recomputation",
        ),
        (
            lambda response: response["request"].update({"scenic_weight": 0.4}),
            "settings_consistency",
        ),
        (
            lambda response: response["diagnostics"].update(
                {"applied_max_detour_factor": 1.4}
            ),
            "settings_consistency",
        ),
        (
            lambda response: response["geojson"]["features"][0]["properties"].update(
                {
                    "exactness_status": "approximate-certified",
                    "optimality_gap": 0.2,
                    "certified_upper_bound": None,
                }
            ),
            "certification_consistency",
        ),
        (
            lambda response: response["diagnostics"].update({"optimality_gap": 0.2}),
            "diagnostics_consistency",
        ),
        (
            lambda response: (
                response["routes"].pop(),
                response["geojson"]["features"].pop(),
            ),
            "baseline_present",
        ),
    ],
)
def test_evaluator_rejects_contract_mutations(mutate, failed_invariant: str) -> None:
    response = deepcopy(_response())
    mutate(response)
    evaluation = evaluate_service_response(
        response, q=0.5, kappa=1.8, avoid_highways=False
    )
    assert evaluation["status"] == "invalid"
    assert failed_invariant in evaluation["failed_invariants"]


def test_evaluator_rejects_positive_gap_same_route_proof() -> None:
    response = deepcopy(_response())
    objective = response["geojson"]["features"][0]["properties"]["objective_value"]
    for container in (
        response["geojson"]["features"][0]["properties"],
        response["routes"][0]["metrics"],
    ):
        container.update(
            {
                "exactness_status": "approximate-certified",
                "optimality_gap": 0.2,
                "certified_upper_bound": objective + 0.2,
                "same_route": True,
                "scenic_score_delta_absolute": 0.0,
                "no_better_route_reason": "same_route",
            }
        )
    response["diagnostics"].update(
        {
            "exactness_status": "approximate-certified",
            "optimality_gap": 0.2,
            "same_route": True,
            "scenic_score_delta_absolute": 0.0,
            "no_better_route_reason": "same_route",
        }
    )
    evaluation = evaluate_service_response(
        response, q=0.5, kappa=1.8, avoid_highways=False
    )
    assert evaluation["status"] == "invalid"
    assert "no_better_reason_consistency" in evaluation["failed_invariants"]


def test_cache_invariant_requires_both_signatures() -> None:
    evaluation = evaluate_service_response(
        _response(), q=0.5, kappa=1.8, avoid_highways=False
    )
    evaluation["score_mapping"].pop("graph_signature")
    summary = _invariant_summary(
        [
            {
                "pair_id": "p",
                "q": 0.5,
                "kappa": 1.8,
                "avoid_highways": False,
                "evaluation": evaluation,
            }
        ]
    )
    assert summary["cache_boundaries"] == {"fail": 1}


def test_kappa_one_and_zero_duration_objective_boundaries() -> None:
    shorter = {
        "duration_minutes_recomputed": 9.0,
        "raw_scenic_score": 8.0,
    }
    baseline = {
        "duration_minutes_recomputed": 10.0,
        "raw_scenic_score": 4.0,
    }
    from scripts.routing.production_benchmark import recompute_objective

    result = recompute_objective(q=0.5, kappa=1.0, scenic=shorter, baseline=baseline)
    assert result["duration_utility"] == 0.0

    zero = {
        "duration_minutes_recomputed": 0.0,
        "raw_scenic_score": 5.0,
    }
    zero_result = recompute_objective(q=0.5, kappa=1.0, scenic=zero, baseline=zero)
    assert zero_result["actual_duration_ratio"] == 1.0


def test_strict_direct_parity_checks_route_identity_and_objective() -> None:
    strict = evaluate_service_response(
        _response(), q=0.5, kappa=1.8, avoid_highways=False
    )
    direct = deepcopy(strict)
    assert all(_evaluations_match(strict, direct).values())

    direct["routes"]["scenic"]["traversal_ids"] = ["different"]
    checks = _evaluations_match(strict, direct)
    assert checks["scenic_traversal_ids"] is False

def test_frontier_extended_stress_beats_recorded_baseline() -> None:
    case = next(
        item
        for item in build_benchmark_cases()
        if item.name == "frontier_extended_stress"
    )
    normalized_score, scenic_uplift, duration_ratio = _run_case(case)

    assert math.isfinite(normalized_score)
    assert math.isfinite(scenic_uplift)
    assert math.isfinite(duration_ratio)
    assert normalized_score >= 0.18672137028069238 - 1e-12
    assert duration_ratio <= 1.1 + 1e-12
