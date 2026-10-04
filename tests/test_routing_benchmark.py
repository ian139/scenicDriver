from __future__ import annotations


from copy import deepcopy
import json
import math
import os
import signal
import time
from pathlib import Path
import sys
import pytest
from types import SimpleNamespace
from typing import Any
import shutil
import subprocess

import scripts.routing.benchmark_scenic_routing as synthetic_benchmark
import scripts.routing.northeast_expanded_autoresearch_benchmark as autoresearch_benchmark
from src.route_planner.graph import Edge, Node, RoadGraph, write_compact_graph
from src.data_pipeline.web_mercator import lat_lon_to_tile

import scripts.routing.production_benchmark as production_benchmark
from scripts.routing.production_benchmark import (
    CaseSpec,
    KAPPA_VALUES,
    Q_VALUES,
    _case_specs,
    _classify_pairs,
    _invariant_summary,
    _evaluations_match,
    evaluate_service_response,
    parse_args,
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
        "requested_start": [42.0, -72.0],
        "requested_end": [42.0 + len(segments) * 0.01, -72.0],
        "snapped_start": [42.0, -72.0],
        "snapped_end": [42.0 + len(segments) * 0.01, -72.0],
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
            "highway_avoidance_fallback": False,
            "highway_avoidance_fallback_reason": None,
            "highway_avoidance_mode": "strict" if avoid else "off",
            "highway_preference": 0.0,
            "highway_avoidance_cost": 0.0,
            "hard_highway_count": 0,
            "optimization_mode": "scenic_score_under_duration_cap",
            "detour_reference_duration_minutes": 10.0,
            "duration_cap_minutes": 10.0 * kappa,
            "duration_cap_satisfied": 12.0 <= 10.0 * kappa + 1e-9,
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

    assert metrics["edge_metric_consistency"]["geometry_sequence"] is True
    assert metrics["edge_metric_consistency"]["requested_endpoints"] is False



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
                "requested_start": [42.0, -72.0],
                "requested_end": [42.0, -72.0],
                "snapped_start": [42.0, -72.0],
                "snapped_end": [42.0, -72.0],
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
            "detour_reference_duration_minutes": 0.0,
            "duration_cap_minutes": 0.0,
            "duration_cap_satisfied": True,
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



def test_evaluate_service_response_accepts_zero_edge_distinct_endpoints() -> None:
    response = _zero_edge_response(
        [[-72.0, 42.0], [-72.0, 42.0]]
    )
    features = response["geojson"]["features"]
    assert isinstance(features, list)
    for feature in features:
        assert isinstance(feature, dict)
        properties = feature["properties"]
        assert isinstance(properties, dict)
        properties["requested_end"] = [42.0, -71.99]

    evaluation = evaluate_service_response(
        response,
        q=0.5,
        kappa=1.8,
        avoid_highways=False,
        requested_start=(42.0, -72.0),
        requested_end=(42.0, -71.99),
    )

    assert evaluation["status"] == "ok"
    assert evaluation["failed_invariants"] == []

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
        "requested_endpoints": True,
        "snapped_endpoints": True,
        "distance": True,
        "raw_scenic_score": True,
        "average_scenic_score": True,
        "normalized_scenic_score": True,
        "duration": True,
        "highway_count": True,
    }


def test_recompute_feature_metrics_uses_trusted_partial_edge_metrics() -> None:
    feature = _feature(
        "scenic",
        score=7.0,
        duration=2.5,
        edge_ids=("e0", "e1"),
        road_types=("trunk", "secondary"),
    )
    rows = feature["properties"]["segment_identity"]
    assert isinstance(rows, list)
    rows[0].update(
        {
            "source_edge_id": "e0",
            "source_fraction": 0.0,
            "distance_km": 0.0,
            "duration_minutes": 0.0,
        }
    )
    rows[1].update(
        {
            "source_edge_id": "e1",
            "source_fraction": 0.25,
            "distance_km": 2.5,
            "duration_minutes": 2.5,
        }
    )
    rows[0]["end"] = list(rows[0]["start"])
    rows[1]["start"] = list(rows[0]["end"])
    rows[1]["end"] = [41.995, -72.0]
    feature["geometry"]["coordinates"] = [
        [-72.0, 42.0],
        [-72.0, 41.995],
    ]
    feature["properties"].update(
        {
            "total_distance_km": 2.5,
            "estimated_duration_minutes": 2.5,
            "highway_count": 1,
            "requested_end": [41.995, -72.0],
            "snapped_end": [41.995, -72.0],
        }
    )
    edge_index = {
        "e0": SimpleNamespace(
            id="e0",
            distance_km=10.0,
            scenic_score=7.0,
            travel_time_minutes=10.0,
            start_node_id="A",
            end_node_id="B",
            road_type="trunk",
        ),
        "e1": SimpleNamespace(
            id="e1",
            distance_km=10.0,
            scenic_score=7.0,
            travel_time_minutes=10.0,
            road_type="secondary",
            start_node_id="B",
            end_node_id="A",
        ),
    }
    node_index = {
        "A": SimpleNamespace(id="A", lat=41.98, lon=-72.0),
        "B": SimpleNamespace(id="B", lat=42.0, lon=-72.0),
    }

    metrics = recompute_feature_metrics(
        feature, edge_index=edge_index, node_index=node_index
    )

    assert metrics["distance_km"] == pytest.approx(2.5)
    assert metrics["raw_scenic_score"] == pytest.approx(7.0)
    assert metrics["duration_minutes_recomputed"] == pytest.approx(2.5)
    assert metrics["highway_count"] == 1
    assert all(metrics["edge_metric_consistency"].values())

    discontinuous = deepcopy(feature)
    discontinuous_rows = discontinuous["properties"]["segment_identity"]
    assert isinstance(discontinuous_rows, list)
    discontinuous_rows[1]["start"] = [41.99, -71.99]
    invalid = recompute_feature_metrics(
        discontinuous, edge_index=edge_index, node_index=node_index
    )
    assert invalid["edge_metric_consistency"]["segments"] is False
    assert invalid["edge_metric_consistency"]["geometry_sequence"] is False

def test_zero_distance_partial_edge_accepts_full_source_fraction() -> None:
    feature = _feature(
        "scenic",
        score=0.0,
        duration=0.0,
        edge_ids=("e0",),
        road_types=("secondary",),
    )
    row = feature["properties"]["segment_identity"][0]
    row.update(
        {
            "source_edge_id": "e0",
            "source_fraction": 1.0,
            "distance_km": 0.0,
            "duration_minutes": 0.0,
            "start": [42.0, -72.0],
            "end": [42.0, -72.0],
        }
    )
    feature["geometry"]["coordinates"] = [
        [-72.0, 42.0],
        [-72.0, 42.0],
    ]
    feature["properties"].update(
        {
            "total_distance_km": 0.0,
            "requested_end": [42.0, -72.0],
            "snapped_end": [42.0, -72.0],
        }
    )
    edge_index = {
        "e0": SimpleNamespace(
            id="e0",
            distance_km=0.0,
            scenic_score=0.0,
            travel_time_minutes=0.0,
            road_type="secondary",
            start_node_id="A",
            end_node_id="B",
        )
    }
    node_index = {
        "A": SimpleNamespace(id="A", lat=42.0, lon=-72.0),
        "B": SimpleNamespace(id="B", lat=42.0, lon=-72.0),
    }

    metrics = recompute_feature_metrics(
        feature, edge_index=edge_index, node_index=node_index
    )

    assert all(metrics["edge_metric_consistency"].values())



def test_partial_rows_cannot_join_at_unconnected_edge_intersections() -> None:
    feature = _feature(
        "scenic",
        score=7.0,
        duration=10.0,
        edge_ids=("e0", "e1"),
        road_types=("secondary", "secondary"),
    )
    rows = feature["properties"]["segment_identity"]
    assert isinstance(rows, list)
    rows[0].update(
        {
            "source_edge_id": "e0",
            "source_fraction": 0.5,
            "distance_km": 2.0,
            "duration_minutes": 5.0,
            "start": [41.98, -72.0],
            "end": [41.99, -72.0],
        }
    )
    rows[1].update(
        {
            "source_edge_id": "e1",
            "source_fraction": 0.5,
            "distance_km": 2.0,
            "duration_minutes": 5.0,
            "start": [41.99, -72.0],
            "end": [42.0, -72.0],
        }
    )
    feature["geometry"]["coordinates"] = [
        [-72.0, 41.98],
        [-72.0, 41.99],
        [-72.0, 42.0],
    ]
    feature["properties"].update(
        {
            "requested_start": [41.98, -72.0],
            "requested_end": [42.0, -72.0],
            "snapped_start": [41.98, -72.0],
            "snapped_end": [42.0, -72.0],
        }
    )
    edge_index = {
        edge_id: SimpleNamespace(
            id=edge_id,
            distance_km=4.0,
            scenic_score=7.0,
            travel_time_minutes=10.0,
            road_type="secondary",
            start_node_id=start_id,
            end_node_id=end_id,
        )
        for edge_id, start_id, end_id in (
            ("e0", "A", "B"),
            ("e1", "C", "D"),
        )
    }
    node_index = {
        "A": SimpleNamespace(id="A", lat=41.98, lon=-72.0),
        "B": SimpleNamespace(id="B", lat=42.0, lon=-72.0),
        "C": SimpleNamespace(id="C", lat=41.98, lon=-72.0),
        "D": SimpleNamespace(id="D", lat=42.0, lon=-72.0),
    }

    metrics = recompute_feature_metrics(
        feature, edge_index=edge_index, node_index=node_index
    )

    assert metrics["edge_metric_consistency"]["segments"] is True
    assert metrics["edge_metric_consistency"]["geometry_sequence"] is True
    assert metrics["edge_metric_consistency"]["continuity"] is False


def test_recompute_feature_metrics_keeps_full_edges_strict() -> None:
    feature = _feature(
        "scenic",
        score=7.0,
        duration=3.0,
        edge_ids=("e0", "e1"),
        road_types=("trunk", "secondary"),
    )
    rows = feature["properties"]["segment_identity"]
    assert isinstance(rows, list)
    rows[0]["distance_km"] = 1.0
    edge_index = {
        "e0": SimpleNamespace(
            id="e0",
            distance_km=2.0,
            scenic_score=7.0,
            travel_time_minutes=1.5,
            road_type="trunk",
        ),
        "e1": SimpleNamespace(
            id="e1",
            distance_km=2.0,
            scenic_score=7.0,
            travel_time_minutes=1.5,
            road_type="secondary",
        ),
    }

    metrics = recompute_feature_metrics(feature, edge_index=edge_index)

    assert metrics["distance_km"] == pytest.approx(4.0)
    assert metrics["edge_metric_consistency"]["segments"] is False
    assert metrics["edge_metric_consistency"]["distance"] is False




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


def test_avoid_highways_invariants_allow_unrestricted_baseline() -> None:
    response = _response(q=1.0, kappa=1.5, avoid=True)
    scenic = response["geojson"]["features"][0]
    baseline = response["geojson"]["features"][1]
    scenic["properties"]["fastest_duration_minutes"] = 10.0
    scenic["properties"]["duration_cap_minutes"] = 15.0
    baseline["properties"]["estimated_duration_minutes"] = 8.0
    baseline["properties"]["segment_identity"][0]["road_type"] = "trunk"
    baseline["properties"]["highway_count"] = 1

    evaluation = evaluate_service_response(
        response,
        q=1.0,
        kappa=1.5,
        avoid_highways=True,
    )

    assert evaluation["invariants"]["prohibited_highways"] is True
    assert evaluation["invariants"]["duration_cap"] is True
    assert evaluation["duration_ratio"] == pytest.approx(1.2)


def test_best_effort_fallback_recomputes_highway_penalty() -> None:
    response = _response(q=1.0, kappa=1.5, avoid=True)
    scenic = response["geojson"]["features"][0]
    scenic_properties = scenic["properties"]
    for row in scenic_properties["segment_identity"]:
        row["road_type"] = "trunk"
    scenic_properties["highway_count"] = 2
    scenic_properties["fastest_duration_minutes"] = 10.0
    scenic_properties["duration_cap_minutes"] = 15.0
    scenic_properties["objective_value"] = -1.6
    diagnostics = response["diagnostics"]
    diagnostics["avoid_highways_applied"] = False
    diagnostics["highway_avoidance_fallback"] = True
    diagnostics["highway_avoidance_fallback_reason"] = "strict_no_route"
    diagnostics["highway_avoidance_mode"] = "best_effort_fallback"
    diagnostics["highway_preference"] = 2.0
    diagnostics["highway_avoidance_cost"] = 2.4
    diagnostics["hard_highway_count"] = 2
    diagnostics["optimization_mode"] = (
        "scenic_score_with_best_effort_highway_avoidance_under_duration_cap"
    )

    evaluation = evaluate_service_response(
        response,
        q=1.0,
        kappa=1.5,
        avoid_highways=True,
        scenic_priority=True,
    )

    assert evaluation["status"] == "ok", evaluation["failed_invariants"]
    assert evaluation["invariants"]["highway_avoidance_reporting"] is True
    assert evaluation["invariants"]["prohibited_highways"] is True
    assert evaluation["objective"]["recomputed"] == pytest.approx(-1.6)


def test_evaluate_service_response_flags_prohibited_highway_and_q0_violation() -> None:
    response = _response(q=0.0, kappa=1.0)
    scenic = response["geojson"]["features"][0]
    scenic["properties"]["estimated_duration_minutes"] = 12.0
    scenic["properties"]["segment_identity"][0]["road_type"] = "trunk"
    scenic["properties"]["highway_count"] = 1
    evaluation = evaluate_service_response(
        response,
        q=0.0,
        kappa=1.0,
        avoid_highways=True,
        scenic_priority=False,
    )
    assert evaluation["invariants"]["prohibited_highways"] is False
    assert evaluation["invariants"]["q0_fastest"] is False
    assert evaluation["invariants"]["duration_cap"] is False


def test_evaluate_service_response_allows_q0_scenic_priority() -> None:
    response = _response(q=0.0, kappa=3.0)
    response["geojson"]["features"][0]["properties"][
        "estimated_duration_minutes"
    ] = 12.0
    response["geojson"]["features"][0]["properties"]["objective_value"] = 0.8
    evaluation = evaluate_service_response(
        response,
        q=0.0,
        kappa=3.0,
        avoid_highways=False,
        scenic_priority=True,
    )
    assert evaluation["invariants"]["q0_fastest"] is True
    assert evaluation["status"] == "ok"
    assert evaluation["failed_invariants"] == []


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


def test_case_matrix_includes_required_full_bbox_strict_service_probe() -> None:
    pair = {
        "id": "full_bbox_rutland_lisbon",
        "start": [43.60784414, -72.98226538],
        "end": [44.02516775, -70.10003245],
    }
    specs = _case_specs({"pairs": [pair]})
    probes = [
        spec
        for spec in specs
        if spec.q == pytest.approx(0.8)
        and spec.kappa == pytest.approx(1.8)
        and spec.avoid_highways is False
    ]

    assert len(specs) == len(Q_VALUES) * len(KAPPA_VALUES) * 2 + 1
    assert len(probes) == 1
    case_id = production_benchmark._case_id(probes[0])
    assert case_id in production_benchmark.STRICT_SERVICE_CASE_IDS


def test_latency_sla_uses_configured_timeout_for_every_case() -> None:
    assert production_benchmark._row_within_case_deadline(
        {
            "wall_ms": 9_000.0,
            "case_timeout_seconds": 1_800.0,
            "reason": None,
        },
        10.0,
    )
    assert not production_benchmark._row_within_case_deadline(
        {
            "wall_ms": 249_000.0,
            "case_timeout_seconds": 1_800.0,
            "reason": None,
        },
        10.0,
    )
    assert not production_benchmark._row_within_case_deadline(
        {"wall_ms": 9_000.0, "reason": "timeout"},
        10.0,
    )


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


def test_no_highway_free_category_accepts_successful_fallback() -> None:
    pair = {"id": "p", "start": [44.0, -72.0], "end": [44.01, -72.0]}
    corpus = {"pairs": [pair], "historical_screenshot_coordinates": None}
    unrestricted = evaluate_service_response(
        _response(), q=0.5, kappa=1.8, avoid_highways=False
    )
    fallback = deepcopy(unrestricted)
    fallback["diagnostics"].update(
        {
            "highway_avoidance_fallback": True,
            "highway_avoidance_fallback_reason": "strict_no_route",
        }
    )
    rows = [
        {
            "pair_id": "p",
            "q": 0.5,
            "kappa": 1.8,
            "avoid_highways": False,
            "reason": None,
            "evaluation": unrestricted,
        },
        {
            "pair_id": "p",
            "q": 0.5,
            "kappa": 1.8,
            "avoid_highways": True,
            "reason": None,
            "evaluation": fallback,
        },
    ]

    discovered, _ = _classify_pairs(corpus, rows)

    assert discovered["no-highway-free"] == ["p"]


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
            lambda response: response["diagnostics"].update(
                {"highway_preference": 2.0}
            ),
            "diagnostics_consistency",
        ),
        (
            lambda response: response["diagnostics"].update(
                {"highway_avoidance_cost": 1.0}
            ),
            "diagnostics_consistency",
        ),
        (
            lambda response: response["diagnostics"].update(
                {"hard_highway_count": 1}
            ),
            "diagnostics_consistency",
        ),
        (
            lambda response: response["diagnostics"].update(
                {"duration_cap_minutes": 999.0}
            ),
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


def _checkpoint_specs() -> list[CaseSpec]:
    return [
        CaseSpec(f"pair-{index}", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
        for index in range(3)
    ]


def _checkpoint_row(index: int, spec: CaseSpec) -> dict[str, object]:
    return {
        "case_index": index,
        "case_id": production_benchmark._case_id(spec),
        "pair_id": spec.pair_id,
        "evaluation": {
            "status": "ok",
            "uplift_absolute": 0.0,
            "uplift_relative": 0.0,
            "optimality_gap": None,
            "exactness_status": "exact",
            "objective": {"recomputed": 1.0},
        },
        "wall_ms": 1.0,
        "reason": None,
        "execution_mode": "direct_planner",
    }


def _patch_checkpoint_runtime(monkeypatch: pytest.MonkeyPatch) -> list[CaseSpec]:
    specs = _checkpoint_specs()
    monkeypatch.setattr(production_benchmark, "_load_corpus", lambda path: {"pairs": []})
    monkeypatch.setattr(production_benchmark, "_case_specs", lambda corpus: specs)
    monkeypatch.setattr(
        production_benchmark,
        "_prepare_benchmark_context",
        lambda graph, report: (
            {"score_mapping": {}},
            0.0,
            SimpleNamespace(),
            {},
            {},
            {"prewarmed": True},
        ),
    )
    monkeypatch.setattr(production_benchmark, "_classify_pairs", lambda corpus, rows: ({}, {}))
    monkeypatch.setattr(production_benchmark, "_invariant_summary", lambda rows: {})
    return specs


def test_benchmark_resume_skips_stable_case_keys_and_orders_results(
    tmp_path, monkeypatch
) -> None:
    specs = _patch_checkpoint_runtime(monkeypatch)
    calls: list[str] = []

    def execute(**kwargs):
        calls.append(kwargs["spec"].pair_id)
        return _checkpoint_row(kwargs["index"], kwargs["spec"])

    monkeypatch.setattr(production_benchmark, "_execute_case", execute)
    output = tmp_path / "benchmark.json"
    first = production_benchmark.run_benchmark(
        corpus_path=tmp_path / "corpus.json",
        graph_path=tmp_path / "graph",
        report_path=tmp_path / "report",
        output_path=output,
        group_size=1,
    )
    assert first["matrix"]["all_cases_persisted"] is True
    assert [row["pair_id"] for row in json.loads(output.read_text())["results"]] == [
        spec.pair_id for spec in specs
    ]
    calls.clear()
    second = production_benchmark.run_benchmark(
        corpus_path=tmp_path / "corpus.json",
        graph_path=tmp_path / "graph",
        report_path=tmp_path / "report",
        output_path=output,
        resume=True,
        group_size=1,
    )
    assert calls == []
    assert second["matrix"]["all_cases_persisted"] is True


def test_benchmark_interruption_preserves_checkpoint_and_final_json(
    tmp_path, monkeypatch
) -> None:
    specs = _patch_checkpoint_runtime(monkeypatch)
    output = tmp_path / "benchmark.json"
    output.write_text("valid-final")
    calls = 0

    def interrupting(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _checkpoint_row(kwargs["index"], kwargs["spec"])

    monkeypatch.setattr(production_benchmark, "_execute_case", interrupting)
    with pytest.raises(KeyboardInterrupt):
        production_benchmark.run_benchmark(
            corpus_path=tmp_path / "corpus.json",
            graph_path=tmp_path / "graph",
            report_path=tmp_path / "report",
            output_path=output,
            group_size=1,
        )
    assert output.read_text() == "valid-final"
    checkpoint_rows = output.with_suffix(".jsonl").read_text().splitlines()
    assert len(checkpoint_rows) == 1
    assert json.loads(checkpoint_rows[0])["case_id"] == production_benchmark._case_id(
        specs[0]
    )


def test_sqlite_benchmark_requires_mmap_loaded_edge_projection_index() -> None:
    loaded = {
        "edge_projection_index": {
            "state": "loaded",
            "mmap_read_only": True,
        }
    }
    production_benchmark._require_loaded_edge_projection_index(
        Path("graph.sqlite3"),
        loaded,
    )

    with pytest.raises(RuntimeError, match="mmap-loaded"):
        production_benchmark._require_loaded_edge_projection_index(
            Path("graph.sqlite3"),
            {
                "edge_projection_index": {
                    "state": "rebuilt",
                    "mmap_read_only": False,
                }
            },
        )


@pytest.mark.parametrize("source_name", ["_compact_search.py", "_compact_search_c.c"])
def test_checkpoint_fingerprint_changes_with_compact_search_source(
    tmp_path, monkeypatch, source_name
) -> None:
    paths = {
        name: tmp_path / path.name
        for name, path in production_benchmark._BENCHMARK_IMPLEMENTATION_PATHS.items()
    }
    for path in paths.values():
        path.write_text("implementation")
    source = tmp_path / source_name
    source.write_text("original compact search")
    monkeypatch.setattr(production_benchmark, "_BENCHMARK_IMPLEMENTATION_PATHS", paths)

    def fingerprint():
        return production_benchmark._checkpoint_fingerprint(
            corpus_path=tmp_path / "corpus.json",
            corpus={"pairs": []},
            graph_path=tmp_path / "graph",
            report_path=tmp_path / "report",
            case_timeout_seconds=10.0,
            strict_service_full=False,
            workers=1,
            group_size=1,
        )

    before = fingerprint()
    source.write_text("modified compact search")
    assert fingerprint() != before


def test_benchmark_resume_rejects_incompatible_checkpoint(
    tmp_path, monkeypatch
) -> None:
    _patch_checkpoint_runtime(monkeypatch)
    monkeypatch.setattr(
        production_benchmark,
        "_execute_case",
        lambda **kwargs: _checkpoint_row(kwargs["index"], kwargs["spec"]),
    )
    output = tmp_path / "benchmark.json"
    production_benchmark.run_benchmark(
        corpus_path=tmp_path / "corpus.json",
        graph_path=tmp_path / "graph",
        report_path=tmp_path / "report",
        output_path=output,
        group_size=1,
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        production_benchmark.run_benchmark(
            corpus_path=tmp_path / "corpus.json",
            graph_path=tmp_path / "graph",
            report_path=tmp_path / "report",
            output_path=output,
            resume=True,
            group_size=2,
        )


def test_benchmark_resume_repairs_partial_checkpoint_tail(
    tmp_path, monkeypatch
) -> None:
    specs = _patch_checkpoint_runtime(monkeypatch)
    calls = 0

    def interrupting(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _checkpoint_row(kwargs["index"], kwargs["spec"])

    monkeypatch.setattr(production_benchmark, "_execute_case", interrupting)
    output = tmp_path / "benchmark.json"
    with pytest.raises(KeyboardInterrupt):
        production_benchmark.run_benchmark(
            corpus_path=tmp_path / "corpus.json",
            graph_path=tmp_path / "graph",
            report_path=tmp_path / "report",
            output_path=output,
            group_size=1,
        )
    checkpoint = output.with_suffix(".jsonl")
    with checkpoint.open("ab") as stream:
        stream.write(b'{"case_id":"partial')

    monkeypatch.setattr(
        production_benchmark,
        "_execute_case",
        lambda **kwargs: _checkpoint_row(kwargs["index"], kwargs["spec"]),
    )
    result = production_benchmark.run_benchmark(
        corpus_path=tmp_path / "corpus.json",
        graph_path=tmp_path / "graph",
        report_path=tmp_path / "report",
        output_path=output,
        resume=True,
        group_size=1,
    )
    assert result["matrix"]["all_cases_persisted"] is True
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert [row["case_id"] for row in rows] == [
        production_benchmark._case_id(spec) for spec in specs
    ]


@pytest.mark.parametrize(
    ("args", "attribute"),
    [
        (["--workers", "0"], "workers"),
        (["--group-size", "-1"], "group_size"),
    ],
)
def test_benchmark_cli_rejects_invalid_bounds(args, attribute, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["production_benchmark", *args])
    with pytest.raises(SystemExit):
        parse_args()

def test_case_deadline_alarms_classify_as_timeout() -> None:
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM unavailable")
    import time

    with pytest.raises(production_benchmark.RoutingTimeout):
        with production_benchmark._case_deadline(0.05) as deadline:
            assert isinstance(deadline, production_benchmark.RoutingDeadline)
            time.sleep(0.5)
    error = production_benchmark.CaseTimeout("alarm test")
    assert production_benchmark._error_reason(error) == "timeout"
    assert production_benchmark._error_reason(
        production_benchmark.RoutingTimeout("routing test")
    ) == "timeout"
    assert production_benchmark._error_reason(
        production_benchmark.RoutingCancelled("cancel test")
    ) == "cancelled"


def test_direct_planner_response_forwards_one_deadline_identity(monkeypatch) -> None:
    deadlines: list[tuple[str, Any]] = []

    class FakeRoute:
        average_scenic_score = 5.0
        estimated_duration_minutes = 10.0
        fastest_duration_minutes = 0.0

    class FakePlanner:
        graph = SimpleNamespace(nodes={}, edges={})

        def find_scenic_route(self, **kwargs):
            deadlines.append(("scenic", kwargs.get("deadline")))
            assert kwargs["detour_reference_duration_minutes"] == 10.0
            return FakeRoute()

        def find_fastest_route(self, **kwargs):
            deadlines.append(("baseline", kwargs.get("deadline")))
            return FakeRoute()

    def fake_objective(request, scenic, baseline, *, deadline=None, **kwargs):
        deadlines.append(("objective", deadline))
        return {
            "objective_value": 0.5,
            "highway_preference": 0.0,
            "highway_avoidance_cost": 0.0,
            "optimization_mode": "scenic_score_under_duration_cap",
        }

    def fake_route_to_feature(route, kind, *, deadline=None, **kwargs):
        deadlines.append((kind, deadline))
        return {
            "properties": {
                "exactness_status": "exact",
                "optimality_gap": None,
                "certified_upper_bound": None,
                "normalized_scenic_score": 0.5,
                "scenic_score_delta_absolute": None,
                "scenic_score_delta_relative": None,
                "same_route": None,
                "no_better_route_reason": None,
            },
            "type": "Feature",
        }

    monkeypatch.setattr(
        production_benchmark.route_service, "_objective_components", fake_objective
    )
    monkeypatch.setattr(
        production_benchmark.route_service, "route_to_feature", fake_route_to_feature
    )

    planner = FakePlanner()
    request = production_benchmark.RouteRequest(
        graph_geojson="g.geojson",
        start=(1.0, 2.0),
        end=(1.1, 2.1),
        scenic_weight=0.5,
        max_detour_factor=1.8,
        avoid_highways=False,
        include_baseline=True,
        tile_scores_json="r.json",
    )
    deadline = production_benchmark.RoutingDeadline.after(10.0)
    response = production_benchmark._direct_planner_response(
        planner,
        request,
        score_mapping={},
        deadline=deadline,
    )
    assert response is not None
    assert len(deadlines) == 5
    assert [label for label, _ in deadlines] == [
        "baseline",
        "scenic",
        "objective",
        "scenic",
        "baseline",
    ]
    assert all(d is deadline for _, d in deadlines)
    diagnostics = response["diagnostics"]
    assert diagnostics["baseline_avoid_highways_applied"] is False
    assert diagnostics["scenic_fastest_duration_ratio"] == pytest.approx(1.0)
    assert diagnostics["detour_reference_duration_minutes"] == pytest.approx(
        10.0
    )
    assert diagnostics["duration_cap_minutes"] == pytest.approx(18.0)
    assert diagnostics["duration_cap_satisfied"] is True


def test_persistent_planning_child_reuses_context_and_survives_cache_mutation(
    monkeypatch,
) -> None:
    cache: dict[tuple[str, bool], tuple[str, str]] = {}

    def fake_execute(*, spec, index, route_error_cache, **kwargs):
        key = (spec.pair_id, spec.avoid_highways)
        hit = key in route_error_cache
        if not hit:
            route_error_cache[key] = ("no_route", "fake")
        return {
            "case_index": index,
            "case_id": production_benchmark._case_id(spec),
            "reason": "no_route",
            "route_error_cache_hit": hit,
            "cache_len": len(route_error_cache),
        }

    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    context = ({}, SimpleNamespace(), {}, {})
    supervisor = production_benchmark._PersistentPlanningChild(
        context, cache, grace_seconds=0.1
    )
    original_pid = supervisor._child_pid
    try:
        spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
        row0 = supervisor.run(spec, 0, Path("g"), Path("r"), 10.0, False)
        assert row0["route_error_cache_hit"] is False
        assert row0["cache_len"] == 1
        row1 = supervisor.run(spec, 1, Path("g"), Path("r"), 10.0, False)
        assert row1["route_error_cache_hit"] is True
        assert row1["cache_len"] == 1
        assert supervisor._child_pid == original_pid
    finally:
        supervisor.close()
    assert supervisor._child_pid is None


def test_persistent_planning_child_replaces_unresponsive_child(monkeypatch) -> None:
    def fake_execute(*, spec, index, **kwargs):
        if index == 0:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            r, _ = os.pipe()
            os.read(r, 1)
        return _checkpoint_row(index, spec)

    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    context = ({}, SimpleNamespace(), {}, {})
    supervisor = production_benchmark._PersistentPlanningChild(
        context, {}, grace_seconds=0.1
    )
    original_pid = supervisor._child_pid
    try:
        spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
        with pytest.raises(production_benchmark.CaseTimeout):
            supervisor.run(spec, 0, Path("g"), Path("r"), 0.01, False)
        assert supervisor._child_pid != original_pid
        assert supervisor.context is context
        row = supervisor.run(spec, 1, Path("g"), Path("r"), 10.0, False)
        assert row["case_id"] == production_benchmark._case_id(spec)
        assert row["case_index"] == 1
    finally:
        supervisor.close()


def test_persistent_planning_child_cleans_up_no_zombies(monkeypatch) -> None:
    def fake_execute(*, spec, index, **kwargs):
        return _checkpoint_row(index, spec)

    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    context = ({}, SimpleNamespace(), {}, {})
    supervisor = production_benchmark._PersistentPlanningChild(
        context, {}, grace_seconds=0.1
    )
    original_pid = supervisor._child_pid
    spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
    supervisor.run(spec, 0, Path("g"), Path("r"), 10.0, False)
    supervisor.close()
    assert supervisor._child_pid is None
    with pytest.raises(ChildProcessError):
        os.waitpid(original_pid, 0)


def test_persistent_planning_child_sends_large_payloads(monkeypatch) -> None:
    def fake_execute(*, spec, index, **kwargs):
        return {
            **_checkpoint_row(index, spec),
            "payload": "x" * 1_000_000,
        }

    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    context = ({}, SimpleNamespace(), {}, {})
    supervisor = production_benchmark._PersistentPlanningChild(
        context, {}, grace_seconds=0.1
    )
    try:
        spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
        row = supervisor.run(spec, 0, Path("g"), Path("r"), 10.0, False)
        assert len(row["payload"]) == 1_000_000
    finally:
        supervisor.close()


def test_persistent_planning_child_does_not_reload_graph(monkeypatch) -> None:

    def fake_execute(*, spec, index, **kwargs):
        return _checkpoint_row(index, spec)

    def raising_preload(*args, **kwargs):
        raise AssertionError("graph preload called in child")

    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    context = ({"score_mapping": {}}, SimpleNamespace(), {}, {})
    supervisor = production_benchmark._PersistentPlanningChild(
        context, {}, grace_seconds=0.1
    )
    try:
        # If the child tried to preload, it would hit this; the child only uses
        # the inherited context.
        monkeypatch.setattr(
            production_benchmark, "_prepare_benchmark_context", raising_preload
        )
        spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
        row = supervisor.run(spec, 0, Path("g"), Path("r"), 10.0, False)
        assert row["case_id"] == production_benchmark._case_id(spec)
    finally:
        supervisor.close()


def test_run_case_worker_executes_in_process_when_supervisor_unavailable(
    monkeypatch,
) -> None:
    def fake_execute(**kwargs):
        return {
            "case_id": production_benchmark._case_id(kwargs["spec"]),
            "marker": "in_process",
            "wall_ms": 1.0,
        }

    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    monkeypatch.setattr(
        production_benchmark, "_WORKER_CONTEXT", ({}, SimpleNamespace(), {}, {})
    )
    monkeypatch.setattr(production_benchmark, "_WORKER_ERRORS", {})
    monkeypatch.setattr(production_benchmark, "_WORKER_SUPERVISOR", None)
    spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
    row = production_benchmark._run_case_worker(
        (0, spec, Path("g"), Path("r"), 10.0, False)
    )
    assert row["marker"] == "in_process"


def test_run_case_worker_hard_timeout_wall_ms_is_nonzero(monkeypatch) -> None:
    class FakeSupervisor:
        def run(self, *args, **kwargs):
            time.sleep(0.05)
            raise production_benchmark.CaseTimeout("fake timeout")

    monkeypatch.setattr(
        production_benchmark, "_WORKER_CONTEXT", ({}, SimpleNamespace(), {}, {})
    )
    monkeypatch.setattr(production_benchmark, "_WORKER_ERRORS", {})
    monkeypatch.setattr(production_benchmark, "_WORKER_SUPERVISOR", FakeSupervisor())
    spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)
    row = production_benchmark._run_case_worker(
        (0, spec, Path("g"), Path("r"), 10.0, False)
    )
    assert row["reason"] == "timeout"
    assert row["wall_ms"] >= 50.0


def test_run_benchmark_hard_timeout_wall_ms_is_nonzero(tmp_path, monkeypatch) -> None:
    spec = CaseSpec("p", (1.0, 2.0), (1.1, 2.1), 0.0, 1.0, False)

    def fake_execute(*, index, **kwargs):
        if index == 0:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            r, _ = os.pipe()
            os.read(r, 1)
        return _checkpoint_row(index, kwargs["spec"])

    monkeypatch.setattr(production_benchmark, "_SUPERVISOR_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(production_benchmark, "_load_corpus", lambda path: {"pairs": []})
    monkeypatch.setattr(production_benchmark, "_case_specs", lambda corpus: [spec])
    monkeypatch.setattr(
        production_benchmark,
        "_prepare_benchmark_context",
        lambda graph, report: ({}, 0.0, SimpleNamespace(), {}, {}, {}),
    )
    monkeypatch.setattr(production_benchmark, "_classify_pairs", lambda corpus, rows: ({}, {}))
    monkeypatch.setattr(production_benchmark, "_invariant_summary", lambda rows: {})
    monkeypatch.setattr(production_benchmark, "_execute_case", fake_execute)
    output = tmp_path / "benchmark.json"
    result = production_benchmark.run_benchmark(
        corpus_path=tmp_path / "corpus.json",
        graph_path=tmp_path / "graph",
        report_path=tmp_path / "report",
        output_path=output,
        case_timeout_seconds=0.01,
    )
    assert result["matrix"]["all_cases_persisted"] is True
    rows = json.loads(output.read_text())["results"]
    assert len(rows) == 1
    row = rows[0]
    assert row["reason"] == "timeout"
    assert row["wall_ms"] >= 50.0


def _highway_budget_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, latitude, longitude in (
        ("S", 42.0, -72.0),
        ("A", 42.01, -72.01),
        ("G", 42.02, -72.0),
    ):
        graph.add_node(Node(node_id, latitude, longitude))
    for edge_id, start, end, distance, score, road_type in (
        ("highway", "S", "G", 6.0, 0.0, "motorway"),
        ("eligible", "S", "G", 9.0, 4.0, "secondary"),
        ("scenic-west", "S", "A", 6.0, 9.0, "secondary"),
        ("scenic-east", "A", "G", 6.0, 9.0, "secondary"),
    ):
        graph.add_edge(Edge(
            edge_id, start, end, distance_km=distance, scenic_score=score,
            road_type=road_type, speed_limit_kmh=60, one_way=True,
        ))
    return graph


def test_synthetic_avoidance_uses_unrestricted_cap_and_restricted_uplift() -> None:
    case = synthetic_benchmark.BenchmarkCase(
        name="distinct-highway-reference",
        graph=_highway_budget_graph(),
        start=(42.0, -72.0),
        end=(42.02, -72.0),
        max_detour_factor=2.0,
        avoid_highways=True,
    )
    score, uplift, ratio = _run_case(case)
    assert score == pytest.approx(0.9)
    assert uplift == pytest.approx(0.5)
    assert ratio == pytest.approx(2.0)


def test_synthetic_rejects_detour_outside_unrestricted_highway_budget(monkeypatch) -> None:
    graph = _highway_budget_graph()
    graph.add_node(Node("B", 42.01, -71.99))
    for edge_id, start, end in (("over-west", "S", "B"), ("over-east", "B", "G")):
        graph.add_edge(Edge(
            edge_id, start, end, distance_km=9.0, scenic_score=10.0,
            road_type="secondary", speed_limit_kmh=60, one_way=True,
        ))
    case = synthetic_benchmark.BenchmarkCase(
        name="outside-unrestricted-budget", graph=graph,
        start=(42.0, -72.0), end=(42.02, -72.0),
        max_detour_factor=2.0, avoid_highways=True,
    )
    # A genuine highway-free route fits the restricted 18-minute reference
    # budget but exceeds the public contract's unrestricted 12-minute cap.
    route = synthetic_benchmark.ScenicRoutePlanner(graph=graph).find_scenic_route(
        case.start, case.end, scenic_weight=1.0, avoid_highways=True,
        max_detour_factor=3.0, scenic_priority=True,
    )
    assert route.estimated_duration_minutes == pytest.approx(18.0)
    monkeypatch.setattr(
        synthetic_benchmark.ScenicRoutePlanner, "find_scenic_route",
        lambda *args, **kwargs: route,
    )
    with pytest.raises(ValueError, match="scenic route exceeds its duration cap"):
        _run_case(case)


def _autoresearch_artifacts(tmp_path, monkeypatch):
    graph = _highway_budget_graph()
    source = tmp_path / "road_graph.sqlite3"
    graph.save(source)
    manifest_path = tmp_path / "road_graph.compact.json"
    write_compact_graph(source, manifest_path)
    report = tmp_path / "report.json"
    rows = []
    for edge in graph.edges.values():
        start = graph.nodes[edge.start_node_id]
        end = graph.nodes[edge.end_node_id]
        x, y = lat_lon_to_tile((start.lat + end.lat) / 2, (start.lon + end.lon) / 2, 14)
        rows.append({"x": x, "y": y, "z": 14, "scenic_score": edge.scenic_score})
    report.write_text(json.dumps({"tiles": rows}))
    monkeypatch.setattr(autoresearch_benchmark, "GRAPH_PATH", manifest_path)
    monkeypatch.setattr(autoresearch_benchmark, "REPORT_PATH", report)
    monkeypatch.setattr(
        autoresearch_benchmark, "EXPECTED_GRAPH_SHA256",
        autoresearch_benchmark._sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        autoresearch_benchmark, "EXPECTED_REPORT_SHA256",
        autoresearch_benchmark._sha256_file(report),
    )
    monkeypatch.setattr(autoresearch_benchmark, "ORACLE_PATH", tmp_path / "absent-oracle.json")
    return manifest_path, report


@pytest.mark.parametrize("payload", ["binary", "source"])
@pytest.mark.parametrize("corruption", ["digest", "size"])
def test_autoresearch_rejects_corrupt_payload_before_timing(
    tmp_path, monkeypatch, payload, corruption
) -> None:
    manifest_path, _ = _autoresearch_artifacts(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / (
        manifest["bin_path"] if payload == "binary" else manifest["source"]["path"]
    )
    data = bytearray(path.read_bytes())
    if corruption == "digest":
        data[-1] ^= 1
    else:
        data.extend(b"trailing")
    path.write_bytes(data)
    monkeypatch.setattr(
        autoresearch_benchmark, "_run_preload_worker",
        lambda: pytest.fail("timing started with corrupt artifacts"),
    )
    with pytest.raises(ValueError, match="SHA-256|size mismatch"):
        autoresearch_benchmark.run_benchmark(timed_runs=1, preload_runs=1)
    assert not autoresearch_benchmark.ORACLE_PATH.exists()


def test_autoresearch_rejects_wrong_declared_source_size(tmp_path, monkeypatch) -> None:
    manifest_path, _ = _autoresearch_artifacts(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        autoresearch_benchmark, "EXPECTED_GRAPH_SHA256",
        autoresearch_benchmark._sha256_file(manifest_path),
    )
    with pytest.raises(ValueError, match="size mismatch"):
        autoresearch_benchmark.run_benchmark(timed_runs=1, preload_runs=1)


def test_autoresearch_missing_oracle_never_seeds_or_starts_timing(
    tmp_path, monkeypatch
) -> None:
    _autoresearch_artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(autoresearch_benchmark, "LATEST_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(
        autoresearch_benchmark, "_run_preload_worker",
        lambda: pytest.fail("timing started without the immutable oracle"),
    )
    with pytest.raises(FileNotFoundError, match="must not be self-seeded"):
        autoresearch_benchmark.run_benchmark(timed_runs=1, preload_runs=1)
    assert not autoresearch_benchmark.ORACLE_PATH.exists()
    assert not autoresearch_benchmark.LATEST_PATH.exists()


def test_autoresearch_parent_and_preload_worker_ignore_cwd(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    script = root / "scripts/routing/northeast_expanded_autoresearch_benchmark.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(autoresearch_benchmark.__file__, script)
    graph_path = root / autoresearch_benchmark.GRAPH_PATH.relative_to(
        autoresearch_benchmark.PROJECT_ROOT
    )
    report_path = root / autoresearch_benchmark.REPORT_PATH.relative_to(
        autoresearch_benchmark.PROJECT_ROOT
    )
    graph_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)
    manifest_path, report = _autoresearch_artifacts(graph_path.parent, monkeypatch)
    manifest_path.rename(graph_path)
    report.rename(report_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    code = (
        "import json,runpy,sys;"
        "ns=runpy.run_path(sys.argv[1]);"
        "g=ns['_artifact_identity'].__globals__;"
        f"g['EXPECTED_GRAPH_SHA256']={autoresearch_benchmark.EXPECTED_GRAPH_SHA256!r};"
        f"g['EXPECTED_REPORT_SHA256']={autoresearch_benchmark.EXPECTED_REPORT_SHA256!r};"
        "ns['_preload_worker_main']() if sys.argv[2]=='worker' else None;"
        "print('IDENTITY '+json.dumps(ns['_artifact_identity'](),sort_keys=True))"
    )
    env = {**os.environ, "PYTHONPATH": str(autoresearch_benchmark.PROJECT_ROOT)}
    identities = []
    for mode, cwd in (("parent", root), ("parent", elsewhere), ("worker", elsewhere)):
        completed = subprocess.run(
            [sys.executable, "-c", code, str(script), mode],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=60, check=True,
        )
        identities.append(json.loads(next(
            line.removeprefix("IDENTITY ") for line in completed.stdout.splitlines()
            if line.startswith("IDENTITY ")
        )))
        if mode == "worker":
            preload = json.loads(next(
                line.removeprefix("PRELOAD_RESULT ") for line in completed.stdout.splitlines()
                if line.startswith("PRELOAD_RESULT ")
            ))
            assert preload["diagnostics"]["graph_cache_hit"] is False
            assert preload["peak_rss_bytes"] > 0
    assert identities[0] == identities[1] == identities[2]


@pytest.mark.parametrize(
    "exceed_phase,expected_phases",
    [
        ("initial", []),
        ("session", ["fresh", "session"]),
        ("warmup", ["fresh", "session", "warmup"]),
        ("timed", ["fresh", "session", "warmup", "timed"]),
    ],
)
def test_autoresearch_rss_stop_prevents_later_planning(
    monkeypatch, exceed_phase, expected_phases
) -> None:
    phases = []
    rss = autoresearch_benchmark.MAX_RSS_BYTES + 1 if exceed_phase == "initial" else 1
    monkeypatch.setattr(autoresearch_benchmark, "_peak_rss_bytes", lambda: rss)
    monkeypatch.setattr(autoresearch_benchmark, "_artifact_identity", lambda: {})
    monkeypatch.setattr(autoresearch_benchmark, "_load_pinned_oracle", lambda: ({}, {}))
    monkeypatch.setattr(autoresearch_benchmark, "clear_route_caches", lambda: None)
    monkeypatch.setattr(autoresearch_benchmark, "clear_route_response_cache", lambda: None)
    monkeypatch.setattr(autoresearch_benchmark, "_assert_response_cache_state", lambda *a, **kw: None)
    monkeypatch.setattr(autoresearch_benchmark, "_validate_response_semantics", lambda *a, **kw: "fixture")

    def preload(phase):
        nonlocal rss
        phases.append(phase)
        if exceed_phase == phase:
            rss = autoresearch_benchmark.MAX_RSS_BYTES + 1
        return {"wall_seconds": 1.0, "cpu_seconds": 1.0, "peak_rss_bytes": rss}

    def request(_):
        nonlocal rss
        phase = "warmup" if "warmup" not in phases else "timed"
        phases.append(phase)
        if exceed_phase == phase:
            rss = autoresearch_benchmark.MAX_RSS_BYTES + 1
        return {}, 1.0, 1.0

    monkeypatch.setattr(autoresearch_benchmark, "_run_preload_worker", lambda: preload("fresh"))
    monkeypatch.setattr(autoresearch_benchmark, "_measure_preload", lambda: preload("session"))
    monkeypatch.setattr(autoresearch_benchmark, "_timed_request", request)
    with pytest.raises(MemoryError, match="hard stop"):
        autoresearch_benchmark.run_benchmark(timed_runs=2, preload_runs=1)
    assert phases == expected_phases
