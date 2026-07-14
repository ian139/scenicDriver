from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
import pytest

from src.route_planner import service as route_service
from src.route_planner.planner import Route, RouteSegment, ScenicRoutePlanner


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_promote_regression_model_smoke(tmp_path: Path) -> None:
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    registry_json = tmp_path / "model_registry.json"
    candidate_ckpt = tmp_path / "candidate.pt"
    candidate_ckpt.write_bytes(b"placeholder")

    candidate_metrics.write_text(json.dumps({"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 100}), encoding="utf-8")
    baseline_metrics.write_text(json.dumps({"corr": 0.9, "mae": 0.25, "rmse": 0.35, "samples": 100}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/modeling/promote_regression_model.py"),
        "--candidate-metrics",
        str(candidate_metrics),
        "--baseline-metrics",
        str(baseline_metrics),
        "--candidate-checkpoint",
        str(candidate_ckpt),
        "--registry-json",
        str(registry_json),
        "--run-name",
        "smoke_vx",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    assert registry["active"] is not None
    assert registry["active"]["run_name"] == "smoke_vx"
    assert registry["active"]["checkpoint"] == str(candidate_ckpt)


def test_promote_regression_model_from_benchmark_comparison(tmp_path: Path) -> None:
    active_comparison = tmp_path / "active_compare.json"
    control_comparison = tmp_path / "control_compare.json"
    registry_json = tmp_path / "model_registry.json"
    candidate_ckpt = tmp_path / "candidate.pt"
    candidate_ckpt.write_bytes(b"placeholder")

    comparison_payload = {
        "baseline": {"corr": 0.9, "mae": 0.25, "rmse": 0.35, "samples": 20},
        "candidate": {"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 20},
        "deltas": {"corr": 0.05, "mae": -0.05, "rmse": -0.05},
    }
    active_comparison.write_text(json.dumps(comparison_payload), encoding="utf-8")
    control_comparison.write_text(json.dumps(comparison_payload), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/modeling/promote_regression_model.py"),
        "--benchmark-comparison",
        str(active_comparison),
        "--required-control-comparison",
        str(control_comparison),
        "--candidate-checkpoint",
        str(candidate_ckpt),
        "--registry-json",
        str(registry_json),
        "--run-name",
        "benchmark_vx",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    assert registry["active"] is not None
    assert registry["active"]["run_name"] == "benchmark_vx"
    assert registry["active"]["metrics"] == {"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 20}
    assert registry["active"]["source_metrics"] == str(active_comparison)


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


def test_plan_routes_uses_requested_filter_for_baseline_and_reports_constraints(
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
            return route(12.0, 10.0, 8.0)

        def find_fastest_route(self, **kwargs: object) -> Route:
            self.calls.append(("fastest", bool(kwargs["avoid_highways"])))
            return route(10.0, 8.0, 4.0)

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

    assert FakePlanner.calls == [("scenic", True), ("fastest", True)]
    assert diagnostics["requested_max_detour_factor"] == pytest.approx(1.5)
    assert diagnostics["applied_max_detour_factor"] == pytest.approx(1.5)
    assert diagnostics["scenic_fastest_duration_ratio"] == pytest.approx(1.2)
    assert diagnostics["scenic_fastest_distance_ratio"] == pytest.approx(1.25)
    assert diagnostics["avoid_highways_applied"] is True
    assert diagnostics["score_mapping_coverage"] == pytest.approx(
        result["score_mapping"]["matched_ratio"]
    )
    assert diagnostics["score_mapping_coverage"] == pytest.approx(0.0)
    assert diagnostics["requested_scenic_weight"] == pytest.approx(0.8)
    assert diagnostics["applied_scenic_weight"] == pytest.approx(0.8)
    scenic_metrics = result["routes"][0]["metrics"]
    baseline_metrics = result["routes"][1]["metrics"]
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenic_weight", float("nan")),
        ("scenic_weight", float("inf")),
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
            scenic_weight=0.5,
            max_detour_factor=1.5,
        )
    )
    diagnostics = result["diagnostics"]
    assert diagnostics["scenic_fastest_duration_ratio"] == pytest.approx(1.2)
    assert result["routes"][0]["metrics"]["actual_duration_ratio"] == pytest.approx(
        1.2
    )
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


def _write_cache_tiles(path: Path, score: float) -> None:
    x, y = route_service.lat_lon_to_tile(42.05, -72.0, 14)
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

    def counted_graph_loader(path: Path) -> object:
        nonlocal graph_loads
        graph_loads += 1
        return original_graph_loader(path)

    def counted_tile_loader(path: Path) -> object:
        nonlocal tile_loads
        tile_loads += 1
        return original_tile_loader(path)

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
    assert set(first) == {"request", "diagnostics", "score_mapping", "routes", "geojson"}
    assert second["diagnostics"]["graph_cache_hit"] is True
    assert second["diagnostics"]["tile_score_cache_hit"] is True
    assert second["diagnostics"]["scored_graph_cache_hit"] is True
    assert second["score_mapping"] == first["score_mapping"]


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
    assert [
        response["score_mapping"]["matched_edges"]
        for response in responses
    ] == [1, 1, 1]


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
    assert float(next(iter(first_variant.edges.values())).scenic_score) == pytest.approx(
        9.0
    )
    assert float(next(iter(second_variant.edges.values())).scenic_score) == pytest.approx(
        3.0
    )
    assert float(next(iter(first_variant.edges.values())).scenic_score) == pytest.approx(
        9.0
    )
    assert first_result["routes"][0]["metrics"]["average_scenic_score"] == pytest.approx(
        9.0
    )
    assert second_result["routes"][0]["metrics"][
        "average_scenic_score"
    ] == pytest.approx(3.0)
    assert len(planned_graphs) == 2
    assert planned_graphs[0] is not planned_graphs[1]

    canonical = route_service._load_cached_graph(graph_path)[0]
    assert float(next(iter(canonical.edges.values())).scenic_score) == pytest.approx(1.0)
