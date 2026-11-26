from __future__ import annotations

import json
from copy import deepcopy
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
import pytest

from scenicdriver.data_pipeline.web_mercator import lat_lon_to_tile
from scenicdriver.route_planner import service as route_service
from scenicdriver.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from scenicdriver.route_planner.graph import Edge, Node, RoadGraph
from scenicdriver.route_planner.planner import Route, RouteSegment, ScenicRoutePlanner

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


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_promote_regression_model_smoke(tmp_path: Path) -> None:
    import torch
    from scenicdriver.active_learning.common import sha256_file
    from scenicdriver.scenic_scorer.regression import ScenicRegressionModel

    decision_json = tmp_path / "decision.json"
    registry_json = tmp_path / "model_registry.json"
    candidate_ckpt = tmp_path / "candidate.pt"

    model = ScenicRegressionModel()
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_state": "completed",
            "model_state_dict": model.state_dict(),
            "vit_dim": 768,
            "terrain_dim": 6,
            "num_classes": 45,
            "hidden_dim": 256,
            "val_loss": 0.1,
        },
        candidate_ckpt,
    )
    cand_sha = sha256_file(candidate_ckpt)

    initial_registry = {
        "schema_version": 1,
        "active": {
            "run_name": "baseline",
            "checkpoint": str(candidate_ckpt),
            "sha256": cand_sha,
        },
        "history": [],
    }
    registry_json.write_text(json.dumps(initial_registry), encoding="utf-8")
    reg_sha = sha256_file(registry_json)
    decision_payload = {
        "all_gates_pass": True,
        "gates": {
            "integrity_pass": True,
            "expanded_benchmark_corr_pass": True,
            "expanded_benchmark_mse_pass": True,
            "expanded_benchmark_improvement_pass": True,
            "control_benchmark_non_regression_pass": True,
            "control_benchmark_corr_pass": True,
            "worst_slice_mse_pass": True,
            "calibration_pass": True,
            "distribution_pass": True,
            "route_evidence_pass": True,
        },
        "reason": "Test promotion",
        "baseline": {"sha256": cand_sha},
        "candidate": {"sha256": cand_sha},
        "expanded_human_benchmark": {
            "candidate_metrics": {"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 100}
        },
    }
    decision_json.write_text(json.dumps(decision_payload), encoding="utf-8")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/modeling/promote_regression_model.py"),
        "--decision",
        str(decision_json),
        "--candidate-checkpoint",
        str(candidate_ckpt),
        "--registry",
        str(registry_json),
        "--expected-registry-sha256",
        reg_sha,
        "--run-name",
        "smoke_vx",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    assert registry["active"] is not None
    assert registry["active"]["run_name"] == "smoke_vx"
    assert registry["active"]["sha256"] == cand_sha


def test_route_compare_service_smoke(tmp_path: Path) -> None:
    graph = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "e0",
                    "road_type": "secondary",
                    "scenic_score": 5.0,
                    "one_way": True,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                },
            }
        ],
    }

    root = tmp_path
    graph_path = root / "graph.geojson"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    run_name = "smoke_run"
    report_dir = root / "data/processed/heuristic_runs" / run_name / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tiles": [
            {"z": 14, "x": 4915, "y": 6078, "scenic_score": 6.5},
        ]
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/routing/route_compare_service.py"),
        "--start",
        "42.0",
        "-72.0",
        "--end",
        "42.1",
        "-72.0",
        "--run-name",
        run_name,
        "--graph-geojson",
        str(graph_path),
    ]
    subprocess.run(cmd, check=True, cwd=root)

    route_geojson = report_dir / "route.geojson"
    route_metrics = report_dir / "route_metrics.json"
    assert route_geojson.exists()
    assert route_metrics.exists()
    metrics = json.loads(route_metrics.read_text(encoding="utf-8"))
    assert "scenic" in metrics
    assert "score_mapping" in metrics
    for payload in (metrics["request"], metrics["scenic"], metrics["baseline"]):
        assert not any("scenic_weight" in key for key in payload)


def test_route_compare_service_cli_rejects_removed_weight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.routing import route_compare_service

    argv = [
        "route_compare_service.py",
        "--start", "42.0", "-72.0",
        "--end", "42.1", "-72.0",
        "--run-name", "smoke_run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = route_compare_service.parse_args()
    assert not any("weight" in key for key in vars(args))
    monkeypatch.setattr(sys, "argv", [*argv, "--scenic-weight", "0.5"])
    with pytest.raises(SystemExit) as error:
        route_compare_service.parse_args()
    assert error.value.code == 2
    assert "unrecognized arguments: --scenic-weight" in capsys.readouterr().err


def test_route_request_serializes_only_applied_settings() -> None:
    payload = {
        "graph_geojson": "graph.geojson",
        "start": [42.0, -72.0],
        "end": [42.1, -72.0],
        "avoid_highways": True,
        "max_detour_factor": 1.5,
        "include_baseline": False,
        "tile_scores_json": None,
        "tile_score_zoom": None,
        "tile_score_fallback": None,
        "max_snap_distance_km": None,
    }
    request = route_service.RouteRequest.from_dict(payload)
    assert request.to_dict() == payload
    assert set(vars(request)) == set(payload)


def test_plan_routes_uses_unrestricted_baseline_for_filtered_scenic_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "edge", "road_type": "secondary"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                },
            }
        ],
    }
    graph_path = tmp_path / "graph.geojson"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    def route(duration: float, distance: float, score: float) -> Route:
        segment = RouteSegment(
            edge_id="fake-edge",
            traversal_id="fake-edge::forward",
            direction="forward",
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            distance_km=distance,
            scenic_score=score,
            road_name=None,
            road_type="secondary",
        )
        return Route(
            segments=[segment],
            total_distance_km=distance,
            average_scenic_score=score,
            estimated_duration_minutes=duration,
            waypoints=[(42.0, -72.0), (42.1, -72.0)],
        )

    class FakePlanner:
        calls: list[tuple[str, bool]] = []

        def __init__(self, *, graph: object) -> None:
            del graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            self.calls.append(("scenic", bool(kwargs["avoid_highways"])))
            assert kwargs["detour_reference_duration_minutes"] == pytest.approx(10.0)
            assert kwargs["scenic_priority"] is True
            assert not any("weight" in key for key in kwargs)
            result = route(12.0, 10.0, 8.0)
            result.fastest_duration_minutes = 10.0
            result.duration_cap_minutes = 15.0
            return result

        def find_fastest_route(self, **kwargs: object) -> Route:
            self.calls.append(("fastest", bool(kwargs["avoid_highways"])))
            return route(10.0, 8.0, 4.0)

    route_service.clear_route_caches()
    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    request = route_service.RouteRequest(
        graph_geojson=str(graph_path),
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        avoid_highways=True,
        max_detour_factor=1.5,
        include_baseline=True,
    )

    result = route_service.plan_routes(request)
    diagnostics = result["diagnostics"]

    assert FakePlanner.calls == [("fastest", False), ("scenic", True)]
    assert diagnostics["requested_max_detour_factor"] == pytest.approx(1.5)
    assert diagnostics["applied_max_detour_factor"] == pytest.approx(1.5)
    assert diagnostics["scenic_fastest_duration_ratio"] == pytest.approx(1.2)
    assert diagnostics["detour_reference_duration_minutes"] == pytest.approx(10.0)
    assert diagnostics["duration_cap_minutes"] == pytest.approx(15.0)
    assert diagnostics["duration_cap_satisfied"] is True
    assert diagnostics["scenic_fastest_distance_ratio"] == pytest.approx(1.25)
    assert diagnostics["avoid_highways_applied"] is True
    assert diagnostics["highway_avoidance_fallback"] is False
    assert diagnostics["highway_avoidance_mode"] == "strict"
    assert diagnostics["baseline_avoid_highways_applied"] is False
    assert diagnostics["score_mapping_coverage"] == pytest.approx(
        result["score_mapping"]["matched_ratio"]
    )
    assert diagnostics["score_mapping_coverage"] == pytest.approx(0.0)
    for payload in (request.to_dict(), diagnostics):
        assert not any("scenic_weight" in key for key in payload)
    scenic_metrics = result["routes"][0]["metrics"]
    baseline_metrics = result["routes"][1]["metrics"]
    for row, feature in zip(result["routes"], result["geojson"]["features"]):
        metrics = row["metrics"]
        assert feature["properties"] == {
            "route_kind": row["route_kind"],
            "snapped_start": metrics["snapped_start"],
            "snapped_end": metrics["snapped_end"],
        }
        assert feature["geometry"] is metrics["geometry"]
        assert feature["properties"]["snapped_start"] is metrics["snapped_start"]
        assert feature["properties"]["snapped_end"] is metrics["snapped_end"]
    for metrics in (scenic_metrics, baseline_metrics):
        assert not any("scenic_weight" in key for key in metrics)
        assert not any("scenic_weight" in key for key in metrics["objective_components"])
    assert scenic_metrics["objective_value"] != pytest.approx(
        baseline_metrics["objective_value"]
    )
    assert scenic_metrics["objective"] == pytest.approx(
        scenic_metrics["objective_value"]
    )
    assert baseline_metrics["objective"] == pytest.approx(
        baseline_metrics["objective_value"]
    )
    assert isinstance(scenic_metrics["objective_components"], dict)
    assert scenic_metrics["objective_components"]["objective_value"] == pytest.approx(
        scenic_metrics["objective_value"]
    )
    assert baseline_metrics["objective_components"]["objective_value"] == pytest.approx(
        baseline_metrics["objective_value"]
    )
    assert diagnostics["planning_elapsed_ms"] >= 0.0
    search_diagnostics = diagnostics["search_diagnostics"]
    assert set(search_diagnostics) == _SEARCH_DIAGNOSTIC_KEYS
    assert scenic_metrics["search_diagnostics"] == search_diagnostics
    assert set(baseline_metrics["search_diagnostics"]) == _SEARCH_DIAGNOSTIC_KEYS
    assert scenic_metrics["requested_start"] == list(request.start)
    assert scenic_metrics["requested_end"] == list(request.end)
    assert baseline_metrics["requested_start"] == list(request.start)
    assert baseline_metrics["requested_end"] == list(request.end)


def test_best_effort_avoidance_retries_with_highway_penalty() -> None:
    segment = RouteSegment(
        edge_id="required-trunk",
        traversal_id="required-trunk::forward",
        direction="forward",
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        distance_km=10.0,
        scenic_score=8.0,
        road_name="Required road",
        road_type="trunk",
        duration_minutes=10.0,
    )
    fallback_route = Route(
        segments=[segment],
        total_distance_km=10.0,
        average_scenic_score=8.0,
        estimated_duration_minutes=12.0,
        waypoints=[segment.start, segment.end],
        objective_value=-1.2,
        fastest_duration_minutes=10.0,
        duration_cap_minutes=18.0,
        highway_count=1,
    )

    class FakePlanner:
        calls: list[dict[str, object]] = []

        def find_scenic_route(self, **kwargs: object) -> Route:
            self.calls.append(dict(kwargs))
            if kwargs["avoid_highways"] is True:
                raise ValueError("No route found between the given coordinates.")
            return fallback_route

    request = route_service.RouteRequest(
        graph_geojson="unused.geojson",
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        avoid_highways=True,
        max_detour_factor=1.8,
    )
    route, strict_applied, fallback_reason = (
        route_service._find_scenic_route_with_best_effort_avoidance(
            FakePlanner(),  # type: ignore[arg-type]
            request,
            detour_reference_duration_minutes=10.0,
            deadline=None,
        )
    )
    objective = route_service._objective_components(
        request,
        route,
        None,
        highway_avoidance_fallback=True,
    )

    assert strict_applied is False
    assert fallback_reason == "strict_no_route"
    assert route is fallback_route
    assert [call["avoid_highways"] for call in FakePlanner.calls] == [
        True,
        False,
    ]
    assert FakePlanner.calls[1]["highway_preference"] == pytest.approx(2.0)
    assert FakePlanner.calls[1]["scenic_priority"] is True
    assert objective["objective_value"] == pytest.approx(-1.2)
    assert objective["highway_avoidance_cost"] == pytest.approx(2.0)
    assert objective["optimization_mode"] == (
        "scenic_score_with_best_effort_highway_avoidance_under_duration_cap"
    )


def test_best_effort_avoidance_retries_strict_over_unrestricted_cap() -> None:
    fallback_route = _cache_test_route(8.0)
    deadline = RoutingDeadline.after(10.0)

    class FakePlanner:
        calls: list[dict[str, object]] = []

        def find_scenic_route(self, **kwargs: object) -> Route:
            self.calls.append(dict(kwargs))
            if kwargs["avoid_highways"] is True:
                raise ValueError("No route satisfies the requested duration cap.")
            return fallback_route

    request = route_service.RouteRequest(
        graph_geojson="unused.geojson",
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        avoid_highways=True,
        max_detour_factor=1.8,
    )

    route, strict_applied, fallback_reason = (
        route_service._find_scenic_route_with_best_effort_avoidance(
            FakePlanner(),  # type: ignore[arg-type]
            request,
            detour_reference_duration_minutes=10.0,
            deadline=deadline,
        )
    )

    assert route is fallback_route
    assert strict_applied is False
    assert fallback_reason == "strict_over_unrestricted_cap"
    assert [call["avoid_highways"] for call in FakePlanner.calls] == [
        True,
        False,
    ]
    assert all(call["deadline"] is deadline for call in FakePlanner.calls)
    assert all(
        call["detour_reference_duration_minutes"] == pytest.approx(10.0)
        for call in FakePlanner.calls
    )
    assert all(
        call["max_detour_factor"] == pytest.approx(1.8) for call in FakePlanner.calls
    )
    assert FakePlanner.calls[1]["highway_preference"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_detour_factor", 3.1),
        ("max_detour_factor", float("nan")),
        ("start", [91.0, 0.0]),
        ("end", [0.0, 181.0]),
    ],
)
def test_route_request_rejects_nonfinite_and_out_of_contract_values(
    field: str, value: object
) -> None:
    payload: dict[str, object] = {
        "graph_geojson": "graph.geojson",
        "start": [42.0, -72.0],
        "end": [42.1, -72.0],
    }
    payload[field] = value
    with pytest.raises(ValueError):
        route_service.RouteRequest.from_dict(payload)


def test_route_request_snap_limit_serializes_and_validates() -> None:
    request = route_service.RouteRequest(
        graph_geojson="graph.geojson",
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        max_snap_distance_km=1.25,
    )
    payload = request.to_dict()
    assert payload["max_snap_distance_km"] == pytest.approx(1.25)
    restored = route_service.RouteRequest.from_dict(payload)
    assert restored.max_snap_distance_km == pytest.approx(1.25)

    for value in (float("nan"), float("inf"), -1.0):
        with pytest.raises(
            ValueError, match="max_snap_distance_km must be finite and nonnegative"
        ):
            route_service.RouteRequest(
                graph_geojson="graph.geojson",
                start=(42.0, -72.0),
                end=(42.1, -72.0),
                max_snap_distance_km=value,
            )


def test_service_uses_canonical_normalization_version() -> None:
    assert route_service._NORMALIZATION_VERSION == "linear-v1"


def test_plan_routes_without_baseline_uses_route_fastest_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            del graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            result = _cache_test_route(8.0)
            result.estimated_duration_minutes = 12.0
            result.fastest_duration_minutes = 10.0
            return result

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    result = route_service.plan_routes(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            include_baseline=False,
            max_detour_factor=1.5,
        )
    )
    diagnostics = result["diagnostics"]
    assert diagnostics["scenic_fastest_duration_ratio"] == pytest.approx(1.2)
    assert result["routes"][0]["metrics"]["actual_duration_ratio"] == pytest.approx(1.2)


def test_empty_route_comparison_preserves_same_route_identity() -> None:
    request = route_service.RouteRequest(
        graph_geojson="unused",
        start=(42.0, -72.0),
        end=(42.0, -72.0),
    )
    empty_route = Route(
        segments=[],
        total_distance_km=0.0,
        average_scenic_score=0.0,
        estimated_duration_minutes=0.0,
        waypoints=[request.start, request.end],
    )

    objective = route_service._objective_components(
        request,
        empty_route,
        empty_route,
    )

    assert objective["same_route"] is True


def test_objective_route_identity_includes_traversal_direction() -> None:
    request = route_service.RouteRequest(
        graph_geojson="unused",
        start=(42.0, -72.0),
        end=(42.1, -72.0),
    )

    def route(direction: str, traversal_id: str) -> Route:
        segment = RouteSegment(
            edge_id="two-way-road",
            traversal_id=traversal_id,
            direction=direction,
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            distance_km=10.0,
            scenic_score=5.0,
            road_name=None,
            road_type="secondary",
        )
        return Route(
            segments=[segment],
            total_distance_km=10.0,
            average_scenic_score=5.0,
            estimated_duration_minutes=10.0,
            waypoints=[segment.start, segment.end],
            edge_ids=("two-way-road",),
            traversal_ids=(traversal_id,),
            exact=True,
        )

    objective = route_service._objective_components(
        request,
        route("forward", "0:forward:two-way-road"),
        route("reverse", "0:reverse:two-way-road"),
    )

    assert objective["same_route"] is False
    assert objective["no_better_route_reason"] == "no_better_route"


def _write_cache_graph(path: Path, scenic_score: float = 1.0) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "edge",
                    "road_type": "secondary",
                    "scenic_score": scenic_score,
                    "one_way": True,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _LargeNodeSet:
    def __len__(self) -> int:
        return route_service._ENDPOINT_NODE_DIAGNOSTIC_MAX_NODES + 1


class _LargeEndpointGraph:
    """Fake graph large enough to skip nearest-node diagnostic indexing.

    Configurable: a finite ``projection_distance`` makes edge projections
    succeed; ``float('inf')`` makes them fail while recording every
    ``excluded_road_types`` query.
    """

    def __init__(self, *, projection_distance: float = 0.25) -> None:
        self.projection_distance = projection_distance
        self.edge_queries: list[frozenset[str]] = []
        self.nodes = _LargeNodeSet()

    def find_nearest_edge_positions_with_distance(
        self, *args: object, **kwargs: object
    ) -> tuple[list[object], float]:
        self.edge_queries.append(kwargs["excluded_road_types"])
        if math.isfinite(self.projection_distance):
            return [object()], self.projection_distance
        return [], self.projection_distance

    def find_nearest_node_with_distance(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("large endpoint graph must not build nearest-node index")


def test_endpoint_snap_diagnostics_use_edge_projections(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    graph = route_service._load_graph(graph_path)
    request = route_service.RouteRequest(
        graph_geojson=str(graph_path),
        start=(42.05, -72.001),
        end=(42.05, -72.001),
        max_snap_distance_km=1.0,
    )

    diagnostics = route_service._endpoint_snap_diagnostics(graph, request)

    assert diagnostics["start_snap_km"] == pytest.approx(0.08257, rel=1e-3)
    assert diagnostics["end_snap_km"] == pytest.approx(0.08257, rel=1e-3)
    assert diagnostics["start_snap_km"] < 1.0
    assert diagnostics["start_all_road_snap_km"] is None


def test_endpoint_snap_diagnostics_skip_large_nearest_node_index() -> None:
    graph = _LargeEndpointGraph()
    request = route_service.RouteRequest(
        graph_geojson="unused",
        start=(42.05, -72.001),
        end=(42.05, -72.001),
    )

    diagnostics = route_service._endpoint_snap_diagnostics(graph, request)  # type: ignore[arg-type]

    assert diagnostics["start_snap_km"] == pytest.approx(0.25)
    assert diagnostics["end_snap_km"] == pytest.approx(0.25)
    assert diagnostics["start_node_id"] is None
    assert diagnostics["end_node_id"] is None


def test_endpoint_snap_diagnostics_bound_large_projection_failure() -> None:
    graph = _LargeEndpointGraph(projection_distance=float("inf"))
    request = route_service.RouteRequest(
        graph_geojson="unused",
        start=(42.05, -72.001),
        end=(42.05, -72.001),
        avoid_highways=True,
        max_snap_distance_km=1.0,
    )

    diagnostics = route_service._endpoint_snap_diagnostics(graph, request)  # type: ignore[arg-type]

    assert len(graph.edge_queries) == 4
    assert graph.edge_queries[0] == route_service.HIGHWAY_ROAD_TYPES
    assert graph.edge_queries[1] == frozenset()
    assert graph.edge_queries[2] == route_service.HIGHWAY_ROAD_TYPES
    assert graph.edge_queries[3] == frozenset()
    assert diagnostics["start_snap_km"] is None
    assert diagnostics["end_snap_km"] is None
    assert diagnostics["start_all_road_snap_km"] is None
    assert diagnostics["end_all_road_snap_km"] is None
    assert diagnostics["start_node_id"] is None
    assert diagnostics["end_node_id"] is None


def _write_cache_tiles(path: Path, score: float) -> None:
    x, y = lat_lon_to_tile(42.05, -72.0, 14)
    path.write_text(
        json.dumps({"tiles": [{"z": 14, "x": x, "y": y, "scenic_score": score}]}),
        encoding="utf-8",
    )


def _cache_test_route(score: float) -> Route:
    segment = RouteSegment(
        edge_id="cache-edge",
        traversal_id="cache-edge::forward",
        direction="forward",
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        distance_km=10.0,
        scenic_score=score,
        road_name=None,
        road_type="secondary",
    )
    return Route(
        segments=[segment],
        total_distance_km=10.0,
        average_scenic_score=score,
        estimated_duration_minutes=12.0,
        waypoints=[(42.0, -72.0), (42.1, -72.0)],
    )


def test_frontier_time_limit_constructor_validates_finite_bounds() -> None:
    assert ScenicRoutePlanner()._frontier_time_limit_seconds == pytest.approx(4.0)
    assert ScenicRoutePlanner(
        frontier_time_limit_seconds=0.0
    )._frontier_time_limit_seconds == pytest.approx(0.0)
    assert ScenicRoutePlanner(
        frontier_time_limit_seconds=60.0
    )._frontier_time_limit_seconds == pytest.approx(60.0)
    for value in ("nan", "inf", "-inf", -0.1, 60.1, "not-a-number"):
        with pytest.raises(ValueError, match="finite and between 0 and 60"):
            ScenicRoutePlanner(frontier_time_limit_seconds=value)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("0", 0.0),
        ("2.5", 2.5),
        ("60", 60.0),
    ],
)
def test_frontier_time_limit_env_parsing(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
    expected: float | None,
) -> None:
    if raw_value is None:
        monkeypatch.delenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", raw_value)

    parsed = route_service._frontier_time_limit_from_env()
    if expected is None:
        assert parsed is None
    else:
        assert parsed == pytest.approx(expected)


@pytest.mark.parametrize("raw_value", ["nan", "inf", "-1", "60.1", "invalid"])
def test_frontier_time_limit_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    monkeypatch.setenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", raw_value)

    with pytest.raises(route_service.RouteConfigurationError):
        route_service._frontier_time_limit_from_env()


def test_frontier_time_limit_env_wires_plan_and_preload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    route_service.clear_route_caches()
    real_planner = route_service.ScenicRoutePlanner
    constructor_limits: list[float | None] = []

    class FakePlanner:
        validate_frontier_time_limit_seconds = (
            real_planner.validate_frontier_time_limit_seconds
        )

        def __init__(
            self, *, graph: object, frontier_time_limit_seconds: float | None = None
        ) -> None:
            del graph
            constructor_limits.append(frontier_time_limit_seconds)

        def prewarm_routing_cache(self) -> dict[str, object]:
            return {}

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            return _cache_test_route(1.0)

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    monkeypatch.setenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", "2.5")

    route_service.preload_route_assets(graph_path)
    route_service.plan_routes(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            include_baseline=False,
        )
    )

    assert constructor_limits == [2.5, 2.5]


def test_clear_route_caches_releases_planner_graph_references() -> None:
    retained_graph = object()
    ScenicRoutePlanner._ELIGIBLE_REACHABILITY_SHARED_CACHE[
        (retained_graph, "stamp", "start", "goal", True)
    ] = False
    ScenicRoutePlanner._FASTEST_PATH_SHARED_GRAPH = retained_graph

    route_service.clear_route_caches()

    assert not ScenicRoutePlanner._ELIGIBLE_REACHABILITY_SHARED_CACHE
    assert ScenicRoutePlanner._FASTEST_PATH_SHARED_GRAPH is None


def test_plan_routes_caches_graph_and_tile_parsing_and_preserves_response_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    tile_path = tmp_path / "tiles.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(tile_path, 8.0)
    route_service.clear_route_caches()

    graph_loads = 0
    tile_loads = 0
    original_graph_loader = route_service._load_graph
    original_tile_loader = route_service.load_tile_scores

    def counted_graph_loader(path: Path, **kwargs: object) -> object:
        nonlocal graph_loads
        graph_loads += 1
        return original_graph_loader(path, **kwargs)

    def counted_tile_loader(path: Path, **kwargs: object) -> object:
        nonlocal tile_loads
        tile_loads += 1
        return original_tile_loader(path, **kwargs)

    monkeypatch.setattr(route_service, "_load_graph", counted_graph_loader)
    monkeypatch.setattr(route_service, "load_tile_scores", counted_tile_loader)

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            self.graph = graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            edge = next(iter(self.graph.edges.values()))
            return _cache_test_route(float(edge.scenic_score))

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    request = route_service.RouteRequest(
        graph_geojson=str(graph_path),
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        include_baseline=False,
        tile_scores_json=str(tile_path),
    )

    first = route_service.plan_routes(request)
    second = route_service.plan_routes(request)

    assert graph_loads == 1
    assert tile_loads == 1
    assert first["routes"] == second["routes"]
    assert first["geojson"] == second["geojson"]
    assert set(first) == {
        "request",
        "diagnostics",
        "score_mapping",
        "routes",
        "geojson",
    }
    assert second["diagnostics"]["route_response_cache_hit"] is True
    assert second["diagnostics"]["graph_cache_hit"] is False
    assert second["diagnostics"]["tile_score_cache_hit"] is False
    assert second["diagnostics"]["scored_graph_cache_hit"] is False
    assert second["score_mapping"] == first["score_mapping"]
    for response in (first, second):
        metrics = response["routes"][0]["metrics"]
        feature = response["geojson"]["features"][0]
        assert feature["properties"] == {
            "route_kind": "scenic",
            "snapped_start": metrics["snapped_start"],
            "snapped_end": metrics["snapped_end"],
        }
        assert feature["geometry"] is metrics["geometry"]
        assert feature["properties"]["snapped_start"] is metrics["snapped_start"]
        assert feature["properties"]["snapped_end"] is metrics["snapped_end"]
        assert response["diagnostics"]["planning_elapsed_ms"] >= 0.0
    for field in (
        "graph_load_elapsed_ms",
        "tile_score_load_elapsed_ms",
        "score_application_elapsed_ms",
    ):
        assert second["diagnostics"][field] == 0.0
    cached_response = next(iter(route_service._ROUTE_RESPONSE_CACHE.values()))
    assert not any(
        name.endswith(("_cache_hit", "_elapsed_ms"))
        for name in cached_response["diagnostics"]
    )


@pytest.mark.parametrize("include_baseline", [False, True])
def test_plan_routes_cache_isolates_mutations_and_preserves_geometry_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_baseline: bool,
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    route_service.clear_route_caches()
    planning_calls = 0

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            del graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            nonlocal planning_calls
            del kwargs
            planning_calls += 1
            return _cache_test_route(8.0)

        def find_fastest_route(self, **kwargs: object) -> Route:
            del kwargs
            return _cache_test_route(1.0)

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    request = route_service.RouteRequest(
        graph_geojson=str(graph_path),
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        include_baseline=include_baseline,
    )
    first = route_service.plan_routes(request)
    expected = deepcopy(first)

    def mutate(response: dict[str, object]) -> None:
        for row, feature in zip(response["routes"], response["geojson"]["features"]):
            metrics = row["metrics"]
            assert metrics["geometry"] is feature["geometry"]
            assert metrics["snapped_start"] is feature["properties"]["snapped_start"]
            assert metrics["snapped_end"] is feature["properties"]["snapped_end"]
            feature["geometry"]["coordinates"][0][0] = 0.0
            metrics["segment_identity"][0]["scenic_score"] = -1.0
            metrics["edge_ids"].append("caller-mutation")
            metrics["score_provenance"]["matched_edges"] = -1
            feature["properties"]["snapped_start"][0] = 0.0
            metrics["snapped_end"][0] = 0.0
            feature["properties"]["route_kind"] = "caller-mutation"
        response["request"]["start"][0] = 0.0
        response["score_mapping"]["matched_edges"] = -1
        response["diagnostics"]["graph_nodes"] = -1
        response["diagnostics"]["planning_elapsed_ms"] = -1.0
        response["diagnostics"]["route_response_cache_hit"] = False

    mutate(first)
    second = route_service.plan_routes(request)
    assert second["routes"] == expected["routes"]
    assert second["geojson"] == expected["geojson"]
    assert second["request"] == expected["request"]
    assert second["score_mapping"] == expected["score_mapping"]
    assert second["diagnostics"]["graph_nodes"] == expected["diagnostics"]["graph_nodes"]
    assert second["diagnostics"]["planning_elapsed_ms"] >= 0.0
    assert second["diagnostics"]["route_response_cache_hit"] is True
    assert (
        second["routes"][0]["metrics"]["geometry"]
        is not first["routes"][0]["metrics"]["geometry"]
    )
    mutate(second)
    third = route_service.plan_routes(request)
    assert third["routes"] == expected["routes"]
    assert third["geojson"] == expected["geojson"]
    assert third["request"] == expected["request"]
    assert third["score_mapping"] == expected["score_mapping"]
    assert third["diagnostics"]["graph_nodes"] == expected["diagnostics"]["graph_nodes"]
    assert third["diagnostics"]["planning_elapsed_ms"] >= 0.0
    assert third["diagnostics"]["route_response_cache_hit"] is True
    for row, feature in zip(third["routes"], third["geojson"]["features"]):
        assert row["metrics"]["geometry"] is feature["geometry"]
        assert row["metrics"]["snapped_start"] is feature["properties"]["snapped_start"]
        assert row["metrics"]["snapped_end"] is feature["properties"]["snapped_end"]
    assert planning_calls == 1


def test_plan_routes_invalidates_graph_and_tile_caches_when_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    tile_path = tmp_path / "tiles.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(tile_path, 8.0)
    route_service.clear_route_caches()

    scores: list[float] = []

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            self.graph = graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            edge = next(iter(self.graph.edges.values()))
            scores.append(float(edge.scenic_score))
            return _cache_test_route(float(edge.scenic_score))

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    request = route_service.RouteRequest(
        graph_geojson=str(graph_path),
        start=(42.0, -72.0),
        end=(42.1, -72.0),
        include_baseline=False,
        tile_scores_json=str(tile_path),
    )

    first = route_service.plan_routes(request)
    _write_cache_graph(graph_path, scenic_score=2.0)
    second = route_service.plan_routes(request)
    _write_cache_tiles(tile_path, 4.0)
    third = route_service.plan_routes(request)

    responses = (first, second, third)
    assert scores == [8.0, 8.0, 4.0]
    assert [
        response["routes"][0]["metrics"]["average_scenic_score"]
        for response in responses
    ] == [8.0, 8.0, 4.0]
    assert [response["score_mapping"]["matched_edges"] for response in responses] == [
        1,
        1,
        1,
    ]


def test_plan_routes_tile_score_overlay_does_not_leak_between_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    first_tiles = tmp_path / "tiles-a.json"
    second_tiles = tmp_path / "tiles-b.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(first_tiles, 9.0)
    _write_cache_tiles(second_tiles, 3.0)
    route_service.clear_route_caches()

    scores: list[float] = []

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            self.graph = graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            edge = next(iter(self.graph.edges.values()))
            scores.append(float(edge.scenic_score))
            return _cache_test_route(float(edge.scenic_score))

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    base_request = {
        "graph_geojson": str(graph_path),
        "start": (42.0, -72.0),
        "end": (42.1, -72.0),
        "include_baseline": False,
    }

    route_service.plan_routes(
        route_service.RouteRequest(**base_request, tile_scores_json=str(first_tiles))
    )
    route_service.plan_routes(route_service.RouteRequest(**base_request))
    route_service.plan_routes(
        route_service.RouteRequest(**base_request, tile_scores_json=str(second_tiles))
    )

    canonical = route_service._load_cached_graph(graph_path)[0]
    assert scores == [9.0, 1.0, 3.0]
    assert next(iter(canonical.edges.values())).scenic_score == pytest.approx(1.0)


def test_preload_route_assets_populates_cache_before_first_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    tile_path = tmp_path / "tiles.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(tile_path, 8.0)
    route_service.clear_route_caches()

    preload = route_service.preload_route_assets(
        graph_path,
        tile_path,
        None,
        1.0,
    )
    assert preload["graph_cache_hit"] is False
    assert preload["tile_score_cache_hit"] is False
    assert preload["scored_graph_cache_hit"] is False
    assert preload["edge_projection_index"]["state"] == "missing"
    assert preload["score_mapping"]["matched_ratio"] == pytest.approx(1.0)

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            self.graph = graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            edge = next(iter(self.graph.edges.values()))
            return _cache_test_route(float(edge.scenic_score))

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    result = route_service.plan_routes(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            include_baseline=False,
            tile_scores_json=str(tile_path),
            tile_score_fallback=1.0,
        )
    )
    diagnostics = result["diagnostics"]
    assert diagnostics["graph_cache_hit"] is True
    assert diagnostics["tile_score_cache_hit"] is True
    assert diagnostics["scored_graph_cache_hit"] is True
    assert result["routes"][0]["metrics"]["average_scenic_score"] == pytest.approx(8.0)


def test_diagnose_reuses_exclusive_scored_graph_after_control_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    tile_path = tmp_path / "tiles.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(tile_path, 8.0)
    route_service.clear_route_caches()

    preload = route_service.preload_route_assets(
        graph_path,
        tile_path,
        None,
        None,
        exclusive_scoring=True,
    )
    assert preload["score_mapping"]["matched_ratio"] == pytest.approx(1.0)

    def fail_raw_load(_path: Path, **_kwargs: object) -> object:
        pytest.fail("diagnostics reloaded the raw graph")

    monkeypatch.setattr(route_service, "_load_graph", fail_raw_load)
    diagnostics = route_service.diagnose_route_request(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            include_baseline=False,
            tile_scores_json=str(tile_path),
        )
    )
    assert diagnostics["graph_edges"] == 1


def test_scored_graph_default_clone_and_exclusive_source_modes(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    tile = lat_lon_to_tile(42.05, -72.0, 14)
    score_map = {(14, tile[0], tile[1]): 8.0}
    graph_key = (
        route_service._resolved_path_key(graph_path),
        route_service._file_signature(graph_path),
    )
    tile_key = ("tiles", (1, 2, 3, 4, 5))

    route_service.clear_route_caches()
    raw_graph = route_service._load_graph(graph_path)
    cloned_graph, *_ = route_service._get_scored_graph(
        raw_graph,
        graph_key=graph_key,
        tile_key=tile_key,
        score_map=score_map,
        zoom=14,
        fallback=None,
    )
    assert cloned_graph is not raw_graph
    assert next(iter(raw_graph.edges.values())).scenic_score == pytest.approx(1.0)
    assert next(iter(cloned_graph.edges.values())).scenic_score == pytest.approx(8.0)

    route_service.clear_route_caches()
    exclusive_graph = route_service._load_graph(graph_path)
    scored_graph, *_ = route_service._get_scored_graph(
        exclusive_graph,
        graph_key=graph_key,
        tile_key=tile_key,
        score_map=score_map,
        zoom=14,
        fallback=None,
        exclusive_source=True,
    )
    assert scored_graph is exclusive_graph
    assert next(iter(exclusive_graph.edges.values())).scenic_score == pytest.approx(8.0)
    route_service.clear_route_caches()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "invalid_target",
    ["matched_report", "unused_report", "fallback", "unmatched_fallback"],
)
def test_apply_tile_scores_rejects_nonfinite_inputs_before_mutation(
    tmp_path: Path, score: float, invalid_target: str
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    graph = route_service._load_graph(graph_path)
    tile = lat_lon_to_tile(42.05, -72.0, 14)
    tile_key = (14, tile[0], tile[1])
    score_map = {tile_key: 8.0}
    fallback = None
    if invalid_target == "matched_report":
        score_map[tile_key] = score
    elif invalid_target == "unused_report":
        score_map[(14, tile[0] + 1, tile[1])] = score
    else:
        fallback = score
        if invalid_target == "unmatched_fallback":
            score_map.clear()

    with pytest.raises(ValueError):
        route_service.apply_tile_scores_to_graph(
            graph, score_map, zoom=14, fallback=fallback
        )

    assert next(iter(graph.edges.values())).scenic_score == pytest.approx(1.0)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "invalid_target",
    ["matched_report", "unused_report", "fallback", "unmatched_fallback"],
)
@pytest.mark.parametrize("mode", ["clone", "exclusive", "cached"])
def test_scored_graph_rejects_nonfinite_inputs_before_cache_or_source_changes(
    tmp_path: Path, score: float, invalid_target: str, mode: str
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    graph = route_service._load_graph(graph_path)
    graph_key = (
        route_service._resolved_path_key(graph_path),
        route_service._file_signature(graph_path),
    )
    report_key = ("tiles", (1, 2, 3, 4, 5))
    tile = lat_lon_to_tile(42.05, -72.0, 14)
    tile_key = (14, tile[0], tile[1])
    score_map = {tile_key: 8.0}
    route_service.clear_route_caches()
    try:
        cached_graph = None
        if mode == "cached":
            cached_graph, *_ = route_service._get_scored_graph(
                graph,
                graph_key=graph_key,
                tile_key=report_key,
                score_map=score_map,
                zoom=14,
                fallback=None,
            )
        graph_cache = dict(route_service._GRAPH_CACHE)
        scored_cache = dict(route_service._SCORED_GRAPH_CACHE)
        active_variants = dict(route_service._ACTIVE_GRAPH_VARIANT_KEYS)
        fallback = None
        if invalid_target == "matched_report":
            score_map[tile_key] = score
        elif invalid_target == "unused_report":
            score_map[(14, tile[0] + 1, tile[1])] = score
        else:
            fallback = score
            if invalid_target == "unmatched_fallback":
                score_map.clear()

        with pytest.raises(ValueError):
            route_service._get_scored_graph(
                graph,
                graph_key=graph_key,
                tile_key=report_key,
                score_map=score_map,
                zoom=14,
                fallback=fallback,
                exclusive_source=mode == "exclusive",
            )

        assert next(iter(graph.edges.values())).scenic_score == pytest.approx(1.0)
        assert dict(route_service._GRAPH_CACHE) == graph_cache
        assert dict(route_service._SCORED_GRAPH_CACHE) == scored_cache
        assert dict(route_service._ACTIVE_GRAPH_VARIANT_KEYS) == active_variants
        if cached_graph is not None:
            assert next(iter(cached_graph.edges.values())).scenic_score == pytest.approx(
                8.0
            )
    finally:
        route_service.clear_route_caches()


def test_plan_routes_accepts_snap_limit_boundary_and_rejects_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    snap_values = {
        "start_snap_km": 1.0,
        "end_snap_km": 0.25,
        "start_all_road_snap_km": None,
        "end_all_road_snap_km": None,
        "start_node_id": "n0",
        "end_node_id": "n1",
    }
    monkeypatch.setattr(
        route_service,
        "_endpoint_snap_diagnostics",
        lambda graph, request, **kwargs: dict(snap_values),
    )

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            del graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            return _cache_test_route(1.0)

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    accepted = route_service.plan_routes(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            include_baseline=False,
            max_snap_distance_km=1.0,
        )
    )
    assert accepted["diagnostics"]["start_snap_km"] == pytest.approx(1.0)

    over_limit = dict(snap_values)
    over_limit["start_snap_km"] = 1.000001
    over_limit["start_all_road_snap_km"] = 1.000001
    monkeypatch.setattr(
        route_service,
        "_endpoint_snap_diagnostics",
        lambda graph, request, **kwargs: dict(over_limit),
    )
    route_service.clear_route_response_cache()
    with pytest.raises(route_service.RouteCoverageError) as caught:
        route_service.plan_routes(
            route_service.RouteRequest(
                graph_geojson=str(graph_path),
                start=(42.0, -72.0),
                end=(42.1, -72.0),
                include_baseline=False,
                max_snap_distance_km=1.0,
            )
        )
    assert caught.value.endpoint == "start"
    assert caught.value.snap_distance_km == pytest.approx(1.000001)
    assert caught.value.max_snap_distance_km == pytest.approx(1.0)


def test_plan_routes_skips_strict_avoidance_when_only_all_road_snap_is_near(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    monkeypatch.setattr(
        route_service,
        "_endpoint_snap_diagnostics",
        lambda graph, request, **kwargs: {
            "start_snap_km": 2.0,
            "end_snap_km": 2.0,
            "start_all_road_snap_km": 0.5,
            "end_all_road_snap_km": 0.5,
            "start_node_id": "n0",
            "end_node_id": "n1",
        },
    )

    class FakePlanner:
        calls: list[tuple[str, bool]] = []

        def __init__(self, *, graph: object) -> None:
            del graph

        def find_fastest_route(self, **kwargs: object) -> Route:
            self.calls.append(("fastest", bool(kwargs["avoid_highways"])))
            return _cache_test_route(1.0)

        def find_scenic_route(self, **kwargs: object) -> Route:
            self.calls.append(("scenic", bool(kwargs["avoid_highways"])))
            assert kwargs["avoid_highways"] is False
            assert kwargs["highway_preference"] == pytest.approx(2.0)
            result = _cache_test_route(5.0)
            result.fastest_duration_minutes = 12.0
            result.duration_cap_minutes = 21.6
            result.objective_value = 0.5
            return result

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    result = route_service.plan_routes(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            avoid_highways=True,
            include_baseline=False,
            max_snap_distance_km=1.0,
        )
    )

    diagnostics = result["diagnostics"]
    assert FakePlanner.calls == [("fastest", False), ("scenic", False)]
    assert diagnostics["start_all_road_snap_km"] == pytest.approx(0.5)
    assert diagnostics["end_all_road_snap_km"] == pytest.approx(0.5)
    assert diagnostics["avoid_highways_applied"] is False
    assert diagnostics["highway_avoidance_fallback"] is True
    assert diagnostics["highway_avoidance_mode"] == "best_effort_fallback"
    assert (
        diagnostics["highway_avoidance_fallback_reason"] == "strict_snap_outside_limit"
    )


def test_concurrent_report_builds_publish_isolated_native_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    first_tiles = tmp_path / "tiles-a.json"
    second_tiles = tmp_path / "tiles-b.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(first_tiles, 9.0)
    _write_cache_tiles(second_tiles, 3.0)
    route_service.clear_route_caches()

    original_apply = route_service._apply_tile_scores_to_graph_native
    first_builder_started = Event()
    second_builder_started = Event()
    release_first_builder = Event()
    builder_graphs: list[object] = []
    builder_lock = Lock()

    def gated_apply(graph: object, *args: object, **kwargs: object) -> tuple[int, int]:
        with builder_lock:
            build_index = len(builder_graphs)
            builder_graphs.append(graph)
        if build_index == 0:
            first_builder_started.set()
            if not release_first_builder.wait(timeout=5.0):
                raise AssertionError("timed out waiting to release first builder")
        else:
            second_builder_started.set()
        return original_apply(graph, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        route_service, "_apply_tile_scores_to_graph_native", gated_apply
    )

    planned_graphs: list[object] = []

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            with builder_lock:
                planned_graphs.append(graph)
            self.graph = graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            del kwargs
            edge = next(iter(self.graph.edges.values()))
            return _cache_test_route(float(edge.scenic_score))

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    base_request = {
        "graph_geojson": str(graph_path),
        "start": (42.0, -72.0),
        "end": (42.1, -72.0),
        "include_baseline": False,
    }
    first_request = route_service.RouteRequest(
        **base_request, tile_scores_json=str(first_tiles)
    )
    second_request = route_service.RouteRequest(
        **base_request, tile_scores_json=str(second_tiles)
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(route_service.plan_routes, first_request)
        assert first_builder_started.wait(timeout=5.0)
        second_future = pool.submit(route_service.plan_routes, second_request)
        # The first builder owns the cache lock through clone, scoring, and
        # publication, so the second report cannot enter construction yet.
        try:
            assert not second_builder_started.is_set()
        finally:
            release_first_builder.set()
        first_result = first_future.result(timeout=5.0)
        second_result = second_future.result(timeout=5.0)

    assert second_builder_started.is_set()
    assert len(builder_graphs) == 2
    first_variant, second_variant = builder_graphs
    assert first_variant is not second_variant
    assert float(
        next(iter(first_variant.edges.values())).scenic_score
    ) == pytest.approx(9.0)
    assert float(
        next(iter(second_variant.edges.values())).scenic_score
    ) == pytest.approx(3.0)
    assert float(
        next(iter(first_variant.edges.values())).scenic_score
    ) == pytest.approx(9.0)
    assert first_result["routes"][0]["metrics"][
        "average_scenic_score"
    ] == pytest.approx(9.0)
    assert second_result["routes"][0]["metrics"][
        "average_scenic_score"
    ] == pytest.approx(3.0)
    assert len(planned_graphs) == 2
    assert planned_graphs[0] is not planned_graphs[1]

    canonical = route_service._load_cached_graph(graph_path)[0]
    assert float(next(iter(canonical.edges.values())).scenic_score) == pytest.approx(
        1.0
    )


def test_deadline_seconds_from_env_default_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCENIC_ROUTE_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", raising=False)
    route_service.clear_route_caches()
    assert route_service._deadline_seconds_from_env() == 20.0
    assert route_service.validate_route_configuration() is None

    with pytest.raises(route_service.RouteConfigurationError, match="finite"):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("SCENIC_ROUTE_DEADLINE_SECONDS", "not-a-number")
            route_service._deadline_seconds_from_env()

    for bad in ("-1", "inf", "nan"):
        with pytest.raises(route_service.RouteConfigurationError, match="finite"):
            with pytest.MonkeyPatch().context() as mp:
                mp.setenv("SCENIC_ROUTE_DEADLINE_SECONDS", bad)
                route_service._deadline_seconds_from_env()


class _UnexpectedPlanner:
    def __init__(self, *, graph: object) -> None:
        raise AssertionError("planner should not be instantiated")


def test_plan_routes_raises_routing_timeout_for_already_expired_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    monkeypatch.setattr(route_service, "ScenicRoutePlanner", _UnexpectedPlanner)

    expired = RoutingDeadline.after(0.0)
    with pytest.raises(RoutingTimeout):
        route_service.plan_routes(
            route_service.RouteRequest(
                graph_geojson=str(graph_path),
                start=(42.0, -72.0),
                end=(42.1, -72.0),
                include_baseline=False,
            ),
            deadline=expired,
        )


def test_deadline_propagated_by_identity_to_planner_and_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    seen_deadlines: list[RoutingDeadline | None] = []
    captured_highway_deadline: list[RoutingDeadline | None] = []
    original_highway_count = route_service._route_highway_count

    def counting_highway_count(
        route: Route, *, deadline: RoutingDeadline | None = None
    ) -> int:
        captured_highway_deadline.append(deadline)
        return original_highway_count(route, deadline=deadline)

    monkeypatch.setattr(route_service, "_route_highway_count", counting_highway_count)

    deadline = RoutingDeadline.after(60.0)

    class FakePlanner:
        def __init__(self, *, graph: object) -> None:
            self.graph = graph

        def find_scenic_route(self, **kwargs: object) -> Route:
            seen_deadlines.append(kwargs.get("deadline"))
            edge = next(iter(self.graph.edges.values()))
            return _cache_test_route(float(edge.scenic_score))

        def find_fastest_route(self, **kwargs: object) -> Route:
            seen_deadlines.append(kwargs.get("deadline"))
            edge = next(iter(self.graph.edges.values()))
            return _cache_test_route(float(edge.scenic_score))

    monkeypatch.setattr(route_service, "ScenicRoutePlanner", FakePlanner)
    result = route_service.plan_routes(
        route_service.RouteRequest(
            graph_geojson=str(graph_path),
            start=(42.0, -72.0),
            end=(42.1, -72.0),
            include_baseline=True,
        ),
        deadline=deadline,
    )

    assert len(result["routes"]) == 2
    assert all(d is deadline for d in seen_deadlines)
    assert all(d is deadline for d in captured_highway_deadline)
    assert len(captured_highway_deadline) == 3


def test_routing_cancelled_during_scoring_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    tile_path = tmp_path / "tiles.json"
    _write_cache_graph(graph_path)
    _write_cache_tiles(tile_path, 8.0)
    route_service.clear_route_caches()

    def exploding_apply(*args: object, **kwargs: object) -> None:
        raise RoutingCancelled("cancelled during scoring")

    monkeypatch.setattr(
        route_service, "_apply_tile_scores_to_graph_native", exploding_apply
    )
    monkeypatch.setattr(route_service, "ScenicRoutePlanner", _UnexpectedPlanner)

    with pytest.raises(RoutingCancelled, match="cancelled during scoring"):
        route_service.plan_routes(
            route_service.RouteRequest(
                graph_geojson=str(graph_path),
                start=(42.0, -72.0),
                end=(42.1, -72.0),
                include_baseline=False,
                tile_scores_json=str(tile_path),
            ),
            deadline=RoutingDeadline.after(60.0),
        )


def test_routing_timeout_during_diagnostics_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)

    def timeout_diagnostics(*args: object, **kwargs: object) -> None:
        raise RoutingTimeout("deadline in diagnostics")

    monkeypatch.setattr(
        route_service, "_endpoint_snap_diagnostics", timeout_diagnostics
    )
    monkeypatch.setattr(route_service, "ScenicRoutePlanner", _UnexpectedPlanner)

    with pytest.raises(RoutingTimeout, match="deadline in diagnostics"):
        route_service.plan_routes(
            route_service.RouteRequest(
                graph_geojson=str(graph_path),
                start=(42.0, -72.0),
                end=(42.1, -72.0),
                include_baseline=False,
            ),
            deadline=RoutingDeadline.after(60.0),
        )


class _CountingDeadline:
    """A test-only deadline that raises after a configurable number of checks."""

    def __init__(self, raise_after: int, exception: Exception) -> None:
        self._raise_after = raise_after
        self._exception = exception
        self.checks = 0

    def check(self) -> None:
        self.checks += 1
        if self.checks >= self._raise_after:
            raise self._exception


def test_clone_graph_for_scoring_checks_periodically() -> None:
    graph = RoadGraph()
    for i in range(3_000):
        graph.add_node(Node(id=f"n{i}", lat=42.0 + i * 1e-6, lon=-72.0 + i * 1e-6))
        if i > 0:
            graph.add_edge(
                Edge(
                    id=f"e{i}",
                    start_node_id=f"n{i - 1}",
                    end_node_id=f"n{i}",
                    distance_km=1.0,
                    scenic_score=5.0,
                )
            )

    deadline = _CountingDeadline(2, RoutingCancelled("clone cancelled"))
    with pytest.raises(RoutingCancelled, match="clone cancelled"):
        route_service._clone_graph_for_scoring(graph, deadline=deadline)
    assert deadline.checks >= 1


def test_load_cached_graph_checks_after_load_before_publication(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.geojson"
    _write_cache_graph(graph_path)
    route_service.clear_route_caches()

    deadline = _CountingDeadline(2, RoutingCancelled("after load"))
    with pytest.raises(RoutingCancelled, match="after load"):
        route_service._load_cached_graph(graph_path, deadline=deadline)
    assert len(route_service._GRAPH_CACHE) == 0


def test_route_to_feature_checks_during_waypoint_materialization() -> None:
    waypoints = [(42.0 + i * 1e-6, -72.0 + i * 1e-6) for i in range(3_000)]
    route = Route(
        waypoints=waypoints,
        total_distance_km=10.0,
        estimated_duration_minutes=15.0,
        average_scenic_score=6.0,
        edge_ids=[],
        traversal_ids=[],
        segments=[],
    )

    deadline = _CountingDeadline(2, RoutingTimeout("waypoint timeout"))
    with pytest.raises(RoutingTimeout, match="waypoint timeout"):
        route_service.route_to_feature(route, "scenic", deadline=deadline)
    assert deadline.checks >= 1


def _sqlite_cache_graph(path: Path, edge_id: str) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="B", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id=edge_id,
            start_node_id="A",
            end_node_id="B",
            distance_km=11.0,
            scenic_score=5.0,
        )
    )
    graph.save(path)


def test_sqlite_graph_swapped_during_row_reading_is_reloaded_not_poisoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from scenicdriver.route_planner import graph as graph_module

    graph_path = tmp_path / "road_graph.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    _sqlite_cache_graph(graph_path, "old")
    _sqlite_cache_graph(replacement, "new")
    route_service.clear_route_caches()
    original_edges = graph_module._iter_sqlite_edges
    swaps = 0

    def swapping_edges(*args, **kwargs):
        nonlocal swaps
        for index, row in enumerate(original_edges(*args, **kwargs)):
            if index == 0 and swaps == 0:
                swaps += 1
                os.replace(replacement, graph_path)
            yield row

    monkeypatch.setattr(graph_module, "_iter_sqlite_edges", swapping_edges)
    graph, path_key, signature, hit = route_service._load_cached_graph(graph_path)

    assert swaps == 1
    assert hit is False
    assert list(graph.edges) == ["new"]
    assert signature == route_service._file_signature(graph_path)
    assert list(route_service._GRAPH_CACHE) == [(path_key, signature)]
    assert list(route_service._GRAPH_CACHE[(path_key, signature)].edges) == ["new"]


def test_graph_swapped_after_every_read_raises_and_caches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = tmp_path / "graph.geojson"
    staging = tmp_path / "staging.geojson"
    _write_cache_graph(graph_path, scenic_score=1.0)
    route_service.clear_route_caches()
    original_loader = route_service._load_graph
    loads = 0

    def swapping_loader(path: Path, **kwargs: object) -> object:
        nonlocal loads
        loads += 1
        graph = original_loader(path, **kwargs)
        _write_cache_graph(staging, scenic_score=float(loads + 1))
        staging.replace(graph_path)
        return graph

    monkeypatch.setattr(route_service, "_load_graph", swapping_loader)
    with pytest.raises(route_service.RouteAssetChangedError, match="changed while loading"):
        route_service._load_cached_graph(graph_path)
    assert loads == 2
    assert not route_service._GRAPH_CACHE


def test_tile_report_swapped_during_read_is_reloaded_not_poisoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    tile_path = tmp_path / "tiles.json"
    staging = tmp_path / "staging.json"
    _write_cache_tiles(tile_path, 2.0)
    _write_cache_tiles(staging, 9.0)
    route_service.clear_route_caches()
    original_loader = route_service.load_tile_scores
    swaps = 0

    def swapping_loader(path: Path, **kwargs: object) -> object:
        nonlocal swaps
        result = original_loader(path, **kwargs)
        if swaps == 0:
            swaps += 1
            staging.replace(tile_path)
        return result

    monkeypatch.setattr(route_service, "load_tile_scores", swapping_loader)
    (
        score_map,
        _zoom,
        report_sha256,
        path_key,
        signature,
        hit,
    ) = route_service._load_cached_tile_scores(tile_path)

    assert swaps == 1
    assert hit is False
    assert set(score_map.values()) == {9.0}
    assert report_sha256 == hashlib.sha256(tile_path.read_bytes()).hexdigest()
    assert signature == route_service._file_signature(tile_path)
    assert list(route_service._TILE_SCORE_CACHE) == [(path_key, signature)]
    cached_map, _cached_zoom, cached_sha = route_service._TILE_SCORE_CACHE[
        (path_key, signature)
    ]
    assert set(cached_map.values()) == {9.0}
    assert cached_sha == report_sha256


def test_tile_report_swapped_after_every_read_raises_and_caches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile_path = tmp_path / "tiles.json"
    staging = tmp_path / "staging.json"
    _write_cache_tiles(tile_path, 2.0)
    route_service.clear_route_caches()
    original_loader = route_service.load_tile_scores
    loads = 0

    def swapping_loader(path: Path, **kwargs: object) -> object:
        nonlocal loads
        loads += 1
        result = original_loader(path, **kwargs)
        _write_cache_tiles(staging, float(loads + 2))
        staging.replace(tile_path)
        return result

    monkeypatch.setattr(route_service, "load_tile_scores", swapping_loader)
    with pytest.raises(route_service.RouteAssetChangedError, match="changed while loading"):
        route_service._load_cached_tile_scores(tile_path)
    assert loads == 2
    assert not route_service._TILE_SCORE_CACHE
