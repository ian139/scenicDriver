from __future__ import annotations

import io
import json
from pathlib import Path
import struct

from PIL import Image
import pytest
from fastapi.testclient import TestClient

import src.app_api.main as app_api

from src.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from src.app_api.main import create_app


@pytest.fixture(autouse=True)
def _stub_named_test_run_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "test-run-report.json"
    report.write_text('{"tiles": [], "summary": {}}', encoding="utf-8")
    original = app_api._run_report_path
    monkeypatch.setattr(
        app_api,
        "_run_report_path",
        lambda run_name: report if run_name == "test-run" else original(run_name),
    )


client = TestClient(create_app())


def test_healthz() -> None:
    resp = client.get("/v1/healthz")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert "regions_available" in payload

def test_scalar_docs() -> None:
    resp = client.get("/scalar")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "ScenicDrive API Scalar Reference" in resp.text
    assert "/openapi.json" in resp.text

def test_root_advertises_scalar_docs() -> None:
    payload = client.get("/").json()
    assert payload["scalar_docs"] == "/scalar"


def test_healthz_redacts_preload_asset_paths(tmp_path, monkeypatch) -> None:
    graph_path = tmp_path / "graph.json"
    report_path = tmp_path / "report.json"
    graph_path.write_text("graph", encoding="utf-8")
    report_path.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        app_api,
        "_configured_regions",
        lambda: [
            {
                "region": "public-test",
                "graph_path": graph_path,
                "tile_scores_path": report_path,
                "run_name": "public-run",
            }
        ],
    )
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "public-test")
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "required")
    monkeypatch.setattr(
        app_api,
        "preload_route_assets",
        lambda *args, **kwargs: {
            "graph_path": str(graph_path),
            "tile_scores_path": str(report_path),
            "graph_cache_hit": True,
            "tile_score_cache_hit": True,
            "scored_graph_cache_hit": False,
            "score_mapping": {
                "source": str(report_path),
                "matched_edges": 3,
                "fallback_edges": 1,
                "total_edges": 4,
                "matched_ratio": 0.75,
                "report_signature": "report-sig",
                "graph_signature": "graph-sig",
                "normalization": "linear-v1",
                "score_run": "public-run",
            },
            "preload_elapsed_ms": 1.5,
        },
    )
    with TestClient(create_app()) as test_client:
        payload = test_client.get("/v1/healthz").json()
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    row = payload["route_preload"]["regions"][0]
    assert row["coverage"]["matched_ratio"] == 0.75
    assert row["signatures"]["graph_signature"] == "graph-sig"
    assert row["timings"]["preload_elapsed_ms"] == 1.5
    assert "graph_path" not in row
    assert "tile_scores_path" not in row


def test_route_compare_missing_graph_does_not_leak_path(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api,
        "_region_to_graph",
        lambda region: (_ for _ in ()).throw(
            FileNotFoundError("/private/secret/road_graph.json")
        ),
    )
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": True,
        },
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "/private/secret" not in detail
    assert detail == "Route assets are unavailable for region 'philadelphia'"
def test_regions_list() -> None:
    resp = client.get("/v1/regions")
    assert resp.status_code == 200
    payload = resp.json()
    assert "regions" in payload
    assert isinstance(payload["regions"], list)


def test_region_graph_fallback_and_discovery_support_sqlite(
    tmp_path, monkeypatch
) -> None:
    graph_dir = tmp_path / "sqlite_region"
    graph_dir.mkdir()
    sqlite_graph = graph_dir / "road_graph.sqlite3"
    sqlite_graph.write_bytes(b"sqlite")
    monkeypatch.setattr(app_api, "ROAD_GRAPHS_DIR", tmp_path)
    monkeypatch.setattr(app_api, "_configured_regions", lambda: [])
    monkeypatch.setattr(app_api, "_default_region_key", lambda: None)

    assert app_api._region_to_graph("sqlite_region") == sqlite_graph
    discovered = app_api._list_regions()
    assert discovered == [
        {
            "region": "sqlite_region",
            "display_name": "sqlite_region",
            "graph_exists": True,
            "latest_run_name": None,
            "bbox": None,
            "is_default": False,
            "source": "discovered",
        }
    ]

def test_training_results_projects_active_record(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "model_registry.json"
    registry.write_text(
        """
        {
          "active": {
            "run_name": "remote-v6",
            "checkpoint": "models/remote-v6.pt",
            "metrics": {"corr": 0.885, "mae": 0.220, "rmse": 0.425, "samples": 750},
            "updated_at": "2026-07-04T05:00:10+00:00",
            "private_history": ["must not leak"]
          },
          "history": ["must not leak"]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(app_api, "MODEL_REGISTRY_PATH", registry)

    response = TestClient(create_app()).get("/v1/training-results")

    assert response.status_code == 200
    assert response.json() == {
        "run_name": "remote-v6",
        "checkpoint": "models/remote-v6.pt",
        "metrics": {"corr": 0.885, "mae": 0.220, "rmse": 0.425, "samples": 750},
        "updated_at": "2026-07-04T05:00:10+00:00",
    }


def test_training_results_returns_404_without_registry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_api, "MODEL_REGISTRY_PATH", tmp_path / "missing.json")

    response = TestClient(create_app()).get("/v1/training-results")

    assert response.status_code == 404
    assert response.json() == {"detail": "Active training result is unavailable"}

def test_heatmap_uses_report_score_range(tmp_path, monkeypatch) -> None:
    run_name = "new-england-test"
    report_dir = tmp_path / run_name / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": {"min": 2.0, "max": 6.0, "total_tiles": 1},
                "tiles": [{"z": 14, "x": 5000, "y": 5900, "scenic_score": 4.0}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_api, "RUNS_DIR", tmp_path)

    response = TestClient(create_app()).get(
        "/v1/heatmap",
        params={"region": "new_england_north", "run_name": run_name},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tile_zoom"] == 14
    assert payload["normalization"] == {
        "min": 2.0,
        "max": 6.0,
        "source": "report_score_range",
    }
    assert payload["geojson_tiles"]["features"][0]["properties"] == {
        "score": 4.0,
        "score_norm": 0.5,
    }

def test_heatmap_image_preserves_z14_tile_grid(tmp_path, monkeypatch) -> None:
    run_name = "new-england-test"
    report_dir = tmp_path / run_name / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": {"min": 2.0, "max": 6.0},
                "tiles": [
                    {"z": 14, "x": 5000, "y": 5900, "scenic_score": 2.0},
                    {"z": 14, "x": 5001, "y": 5900, "scenic_score": 6.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_api, "RUNS_DIR", tmp_path)

    response = TestClient(create_app()).get(
        "/v1/heatmap-image",
        params={"region": "new_england_north", "run_name": run_name},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-scenic-tile-zoom"] == "14"
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (2, 1)
    assert image.getpixel((0, 0)) != image.getpixel((1, 0))

    score_response = TestClient(create_app()).get(
        "/v1/heatmap-scores.bin",
        params={"region": "new_england_north", "run_name": run_name},
    )
    assert score_response.status_code == 200
    assert score_response.headers["x-scenic-tile-zoom"] == "14"
    assert score_response.headers["x-scenic-min-x"] == "5000"
    assert score_response.headers["x-scenic-min-y"] == "5900"
    assert score_response.headers["x-scenic-grid-width"] == "2"
    assert score_response.headers["x-scenic-grid-height"] == "1"
    assert struct.unpack("<2f", score_response.content) == (2.0, 6.0)


def test_search_suggest_and_retrieve_contract(monkeypatch) -> None:
    class FakeResponse:
        ok = True

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        assert params["session_token"] == "session-1"
        assert timeout == 15
        if url.endswith("/suggest"):
            assert params["q"] == "Bangor"
            return FakeResponse(
                {
                    "suggestions": [
                        {
                            "mapbox_id": "place.1",
                            "name": "Bangor",
                            "full_address": "Bangor, Maine",
                            "feature_type": "place",
                        }
                    ]
                }
            )
        assert url.endswith("/retrieve/place.1")
        return FakeResponse(
            {
                "features": [
                    {
                        "geometry": {"coordinates": [-68.771305, 44.801616]},
                        "properties": {
                            "name": "Bangor",
                            "full_address": "Bangor, Maine",
                        },
                    }
                ]
            }
        )

    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(app_api.requests, "get", fake_get)
    test_client = TestClient(create_app())

    suggest = test_client.get(
        "/v1/search/suggest",
        params={"q": "Bangor", "session_token": "session-1"},
    )
    assert suggest.status_code == 200
    assert suggest.json()["suggestions"][0]["mapbox_id"] == "place.1"

    retrieve = test_client.get(
        "/v1/search/retrieve",
        params={"mapbox_id": "place.1", "session_token": "session-1"},
    )
    assert retrieve.status_code == 200
    assert retrieve.json()["result"] == {
        "name": "Bangor",
        "full_address": "Bangor, Maine",
        "lat": 44.801616,
        "lon": -68.771305,
    }


def test_validated_route_projects_canonical_artifacts(tmp_path, monkeypatch) -> None:
    run_name = "new-england-test"
    report_dir = tmp_path / "runs" / run_name / "report"
    report_dir.mkdir(parents=True)
    route_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"route_kind": route_kind},
                "geometry": {"type": "LineString", "coordinates": [[-73, 44], [-69, 45]]},
            }
            for route_kind in ("scenic", "baseline")
        ],
    }
    (report_dir / "route.geojson").write_text(json.dumps(route_geojson), encoding="utf-8")
    (report_dir / "route_metrics.json").write_text(
        json.dumps(
            {
                "request": {
                    "start": [44.4, -73.2],
                    "end": [44.8, -68.8],
                    "scenic_weight": 0.8,
                    "max_detour_factor": 1.8,
                    "graph_geojson": "private/graph.json",
                },
                "score_mapping": {
                    "enabled": True,
                    "source": "private/report.json",
                    "zoom": 14,
                    "matched_edges": 10,
                    "total_edges": 10,
                    "matched_ratio": 1.0,
                },
                "scenic": {"route_kind": "scenic", "total_distance_km": 461.2},
                "baseline": {"route_kind": "baseline", "total_distance_km": 471.0},
            }
        ),
        encoding="utf-8",
    )
    regions_path = tmp_path / "regions.json"
    regions_path.write_text(
        json.dumps(
            {
                "default_region": "new_england_north",
                "regions": [
                    {"region": "new_england_north", "run_name": run_name}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_api, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(app_api, "APP_REGIONS_PATH", regions_path)

    response = TestClient(create_app()).get(
        "/v1/validated-route", params={"region": "new_england_north"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["geojson"] == route_geojson
    assert payload["routes"]["scenic"]["total_distance_km"] == 461.2
    assert payload["score_mapping"] == {
        "enabled": True,
        "zoom": 14,
        "matched_edges": 10,
        "total_edges": 10,
        "matched_ratio": 1.0,
    }
    assert "source" not in payload["score_mapping"]
    assert "graph_geojson" not in payload["request"]


def test_contrib_session_and_label_flow() -> None:
    start = client.post(
        "/v1/contrib/session/start",
        json={"contributor_id": "tester_api", "display_name": "Tester API", "region": "philadelphia"},
    )
    assert start.status_code == 200
    s_payload = start.json()
    assert "profile" in s_payload
    tasks = s_payload.get("tasks", [])
    if not tasks:
        return

    task = tasks[0]
    label = client.post(
        "/v1/contrib/labels",
        json={
            "contributor_id": "tester_api",
            "task_id": task["task_id"],
            "image_path": task["image_path"],
            "scenic_human": 6.5,
            "confidence": "high",
            "skip": False,
            "notes": "api smoke test",
            "region": "philadelphia",
        },
    )
    assert label.status_code == 200
    l_payload = label.json()
    assert l_payload["saved"] is True

    profile = client.get("/v1/contrib/profile", params={"contributor_id": "tester_api"})
    assert profile.status_code == 200
    assert "profile" in profile.json()


def test_route_compare_contract() -> None:
    resp = client.post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": True,
        },
    )
    # In environments missing local graph/report assets this may return 404.
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        payload = resp.json()
        assert "routes" in payload
        assert "score_mapping" in payload
        assert "geojson" in payload


def _strict_route_metrics(
    *,
    distance_km: float,
    duration_min: float,
    raw_score: float,
    edge_id: str,
    route_kind: str = "scenic",
    cap: float = 1.8,
) -> dict[str, object]:
    return {
        "route_kind": route_kind,
        "segments": 1,
        "edge_ids": [edge_id],
        "segment_identity": [
            {
                "edge_id": edge_id,
                "start": [40.03, -75.22],
                "end": [40.065, -75.19],
            }
        ],
        "total_distance_km": distance_km,
        "average_scenic_score": raw_score,
        "raw_scenic_score": raw_score,
        "normalized_scenic_score": raw_score / 10.0,
        "estimated_duration_minutes": duration_min,
        "objective": raw_score,
        "objective_value": raw_score,
        "objective_components": {"objective_value": raw_score},
        "requested_scenic_weight": 0.8,
        "applied_scenic_weight": 0.8,
        "requested_max_detour_factor": cap,
        "applied_max_detour_factor": cap,
        "actual_duration_ratio": duration_min / 16.0,
        "exactness_status": "exact",
        "certified_upper_bound": None,
        "optimality_gap": 0.0,
        "highway_count": 0,
        "score_coverage": 1.0,
        "score_run": [[edge_id, raw_score]],
        "zero_improvement_reason": None,
        "requested_start": [40.03, -75.22],
        "requested_end": [40.065, -75.19],
        "snapped_start": [40.03, -75.22],
        "snapped_end": [40.065, -75.19],
        "no_route_reason": None,
    }

def _strict_geojson(*metrics: dict[str, object]) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "properties": {
                    "route_kind": route_metrics["route_kind"],
                    "segment_identity": route_metrics["segment_identity"],
                    "requested_start": route_metrics["requested_start"],
                    "requested_end": route_metrics["requested_end"],
                    "snapped_start": route_metrics["snapped_start"],
                    "snapped_end": route_metrics["snapped_end"],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-75.22, 40.03],
                        [-75.19, 40.065],
                    ],
                },
            }
            for route_metrics in metrics
        ],
    }


def _strict_diagnostics() -> dict[str, object]:
    return {
        "planner": "ok",
        "requested_scenic_weight": 0.8,
        "applied_scenic_weight": 0.8,
        "requested_max_detour_factor": 1.8,
        "applied_max_detour_factor": 1.8,
        "scenic_fastest_duration_ratio": 1.25,
        "optimization_mode": "exact",
        "optimization_status": "optimal",
        "optimality_gap": 0.0,
        "certified_upper_bound": None,
        "scenic_score_delta_absolute": 0.6,
        "scenic_score_delta_relative": 3.0,
        "same_route": False,
        "no_better_route_reason": None,
        "avoid_highways_applied": True,
        "score_mapping_coverage": 1.0,
        "planning_elapsed_ms": 0.1,
    }


def _strict_score_mapping() -> dict[str, object]:
    return {
        "version": "test",
        "report_signature": "report-test",
        "graph_signature": "graph-test",
        "normalization": "linear-v1",
        "fallback_edges": 0,
        "score_run": "test-run",
    }


def test_route_compare_success_plans_once_without_diagnosis(monkeypatch) -> None:
    graph_calls: list[str] = []
    run_calls: list[str] = []
    plan_calls: list[object] = []
    diagnosis_calls: list[object] = []

    def fake_graph(region: str):
        graph_calls.append(region)
        return Path("/tmp/fake-road-graph.geojson")

    def fake_run(region: str):
        run_calls.append(region)
        return "test-run"

    scenic_metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    baseline_metrics = _strict_route_metrics(
        distance_km=8.0,
        duration_min=16.0,
        raw_score=0.2,
        edge_id="baseline-edge",
        route_kind="baseline",
    )

    def fake_plan(request, **kwargs):
        plan_calls.append(request)
        return {
            "routes": [
                {"route_kind": "scenic", "metrics": scenic_metrics},
                {"route_kind": "baseline", "metrics": baseline_metrics},
            ],
            "diagnostics": _strict_diagnostics(),
            "score_mapping": _strict_score_mapping(),
            "geojson": _strict_geojson(scenic_metrics, baseline_metrics),
        }

    def fake_diagnose(request, **kwargs):
        diagnosis_calls.append(request)
        return {"planner": "diagnosed"}

    monkeypatch.setattr(app_api, "_region_to_graph", fake_graph)
    monkeypatch.setattr(app_api, "_latest_run_for_region", fake_run)
    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    monkeypatch.setattr(app_api, "diagnose_route_request", fake_diagnose)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": True,
            "include_baseline": True,
        },
    )

    assert response.status_code == 200
    assert graph_calls == ["philadelphia"]
    assert run_calls == ["philadelphia"]
    assert len(plan_calls) == 1
    assert diagnosis_calls == []

    payload = response.json()
    assert set(payload) == {
        "request",
        "run_name",
        "diagnostics",
        "routes",
        "deltas",
        "score_mapping",
        "geojson",
    }
    assert payload["diagnostics"] == _strict_diagnostics()
    assert payload["request"]["max_detour_factor"] == 1.8
    assert payload["request"]["avoid_highways"] is True
    assert payload["routes"] == {"scenic": scenic_metrics, "baseline": baseline_metrics}
    assert payload["deltas"] == {
        "distance_km": 2.0,
        "duration_min": 4.0,
        "scenic_score": 0.6,
        "scenic_score_absolute": 0.6,
        "scenic_score_relative": 3.0,
        "normalized_scenic_score": 0.06,
    }
    assert payload["score_mapping"] == _strict_score_mapping()
    assert payload["geojson"] == _strict_geojson(scenic_metrics, baseline_metrics)


def _strict_geometry_request(
    *, include_baseline: bool = False
) -> app_api.RouteRequest:
    return app_api.RouteRequest(
        graph_geojson="test-graph",
        start=(40.03, -75.22),
        end=(40.065, -75.19),
        include_baseline=include_baseline,
    )


def test_route_geometry_rejects_missing_or_wrong_geometry() -> None:
    metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    geojson = _strict_geojson(metrics)
    geojson["features"][0]["geometry"] = {
        "type": "Point",
        "coordinates": [-75.22, 40.03],
    }

    with pytest.raises(app_api.HTTPException) as raised:
        app_api._validate_route_geometry(
            geojson,
            request=_strict_geometry_request(),
            routes={"scenic": metrics},
        )

    assert raised.value.status_code == 502
    assert "LineString" in str(raised.value.detail)


def test_route_geometry_rejects_duplicate_features_and_identity_mismatch() -> None:
    metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    duplicate = _strict_geojson(metrics, metrics)
    with pytest.raises(app_api.HTTPException, match="duplicate"):
        app_api._validate_route_geometry(
            duplicate,
            request=_strict_geometry_request(),
            routes={"scenic": metrics},
        )

    mismatched = _strict_geojson(metrics)
    mismatched["features"][0]["properties"]["segment_identity"] = [
        {
            "edge_id": "other-edge",
            "start": [40.03, -75.22],
            "end": [40.065, -75.19],
        }
    ]
    with pytest.raises(app_api.HTTPException, match="does not match"):
        app_api._validate_route_geometry(
            mismatched,
            request=_strict_geometry_request(),
            routes={"scenic": metrics},
        )


def test_route_geometry_rejects_discontinuous_segments_and_wrong_endpoint() -> None:
    metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    metrics["segment_identity"] = [
        {
            "edge_id": "first",
            "start": [40.03, -75.22],
            "end": [40.04, -75.21],
        },
        {
            "edge_id": "second",
            "start": [40.05, -75.20],
            "end": [40.065, -75.19],
        },
    ]
    geojson = _strict_geojson(metrics)
    with pytest.raises(app_api.HTTPException, match="not continuous"):
        app_api._validate_route_geometry(
            geojson,
            request=_strict_geometry_request(),
            routes={"scenic": metrics},
        )

    endpoint_metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    endpoint_geojson = _strict_geojson(endpoint_metrics)
    endpoint_geojson["features"][0]["geometry"]["coordinates"][-1] = [
        -75.18,
        40.065,
    ]
    with pytest.raises(app_api.HTTPException, match="snapped_end"):
        app_api._validate_route_geometry(
            endpoint_geojson,
            request=_strict_geometry_request(),
            routes={"scenic": endpoint_metrics},
        )
    


def test_route_geometry_rejects_requested_metadata_mismatch() -> None:
    metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    geojson = _strict_geojson(metrics)
    geojson["features"][0]["properties"]["requested_start"] = [40.04, -75.22]

    with pytest.raises(app_api.HTTPException) as raised:
        app_api._validate_route_geometry(
            geojson,
            request=_strict_geometry_request(),
            routes={"scenic": metrics},
        )

    assert raised.value.status_code == 502
    assert raised.value.detail == {
        "error": "invalid_route_service_response",
        "message": "scenic requested_start does not match the route request",
    }


def test_route_geometry_accepts_zero_edge_with_distinct_requested_points() -> None:
    request = app_api.RouteRequest(
        graph_geojson="test-graph",
        start=(40.03, -75.22),
        end=(40.065, -75.19),
        include_baseline=False,
    )
    properties = {
        "route_kind": "scenic",
        "requested_start": [40.03, -75.22],
        "requested_end": [40.065, -75.19],
        "snapped_start": [40.04, -75.21],
        "snapped_end": [40.04, -75.21],
        "segment_identity": [],
    }
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-75.21, 40.04], [-75.21, 40.04]],
                },
            }
        ],
    }

    assert app_api._validate_route_geometry(
        geojson,
        request=request,
        routes={"scenic": {"segment_identity": []}},
    )["scenic"] == geojson["features"][0]
    
def test_route_compare_rejects_without_relaxing_detour_cap(monkeypatch) -> None:
    plan_calls: list[object] = []
    diagnosis_calls: list[object] = []
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        plan_calls.append(request)
        raise ValueError("detour cap is too tight")

    def fake_diagnose(request, **kwargs):
        diagnosis_calls.append(request)
        return {"graph_nodes": 4, "graph_edges": 3}

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    monkeypatch.setattr(app_api, "diagnose_route_request", fake_diagnose)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.4,
            "avoid_highways": True,
            "include_baseline": True,
        },
    )

    assert response.status_code == 422
    assert len(plan_calls) == 1
    request = plan_calls[0]
    assert request.max_detour_factor == 1.4
    assert request.avoid_highways is True
    assert diagnosis_calls == [request]
    detail = response.json()["detail"]
    assert detail["diagnostics"] == {"graph_nodes": 4, "graph_edges": 3}
    assert detail["message"] == "No route satisfies the avoid-highways constraint."
    assert detail["hint"] == (
        "Turn off Avoid highways, choose different points, or increase max detour."
    )


def test_route_compare_reports_endpoint_coverage_error(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api,
        "_region_to_graph",
        lambda region: Path("/tmp/fake-road-graph.sqlite3"),
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")
    monkeypatch.setattr(
        app_api,
        "_app_region",
        lambda region: {"region": region, "max_route_snap_km": 1.0},
    )
    plan_calls: list[object] = []

    def fake_plan(request, **kwargs):
        plan_calls.append(request)
        raise app_api.RouteCoverageError("start", 1.234, 1.0)

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": True,
        },
    )

    assert response.status_code == 422
    assert len(plan_calls) == 1
    assert plan_calls[0].max_snap_distance_km == pytest.approx(1.0)
    assert response.json()["detail"] == {
        "error": "route_endpoint_outside_coverage",
        "message": "The start point is too far from the supported road network.",
        "hint": "Choose a point within the selected region's route coverage.",
        "endpoint": "start",
        "snap_distance_km": 1.234,
        "max_snap_distance_km": 1.0,
    }

def test_route_compare_reports_invalid_route_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        raise app_api.RouteConfigurationError("invalid frontier budget")

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.4,
            "avoid_highways": True,
            "include_baseline": True,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error": "route_configuration_invalid",
        "message": "Route planning configuration is invalid.",
    }


def test_route_compare_rejects_incomplete_service_response(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")
    scenic_metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    scenic_metrics.pop("score_run")

    monkeypatch.setattr(
        app_api,
        "plan_routes",
        lambda request, **kwargs: {
            "routes": [{"route_kind": "scenic", "metrics": scenic_metrics}],
            "diagnostics": _strict_diagnostics(),
            "score_mapping": _strict_score_mapping(),
        },
    )
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 502
    assert response.json()["detail"]["missing_metrics"] == ["score_run"]


def test_route_compare_request_fixes_scenic_weight_at_point_eight() -> None:
    from pydantic import ValidationError

    request = app_api.RouteCompareRequest(
        start={"lat": 40.03, "lon": -75.22},
        end={"lat": 40.065, "lon": -75.19},
    )
    assert request.scenic_weight == pytest.approx(0.8)
    with pytest.raises(ValidationError):
        app_api.RouteCompareRequest(
            start={"lat": 40.03, "lon": -75.22},
            end={"lat": 40.065, "lon": -75.19},
            scenic_weight=0.7,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenic_weight", float("nan")),
        ("scenic_weight", float("inf")),
        ("max_detour_factor", float("nan")),
        ("max_detour_factor", float("inf")),
    ],
)
def test_route_compare_request_rejects_nonfinite_controls(field, value) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        app_api.RouteCompareRequest(
            start={"lat": 40.03, "lon": -75.22},
            end={"lat": 40.065, "lon": -75.19},
            **{field: value},
        )

@pytest.mark.parametrize(
    ("start", "end"),
    [
        ({"lat": 91.0, "lon": 0.0}, {"lat": 40.0, "lon": 0.0}),
        ({"lat": -91.0, "lon": 0.0}, {"lat": 40.0, "lon": 0.0}),
        ({"lat": 40.0, "lon": 181.0}, {"lat": 40.0, "lon": 0.0}),
        ({"lat": 40.0, "lon": -181.0}, {"lat": 40.0, "lon": 0.0}),
    ],
)
def test_route_compare_request_rejects_invalid_coordinates(start, end) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        app_api.RouteCompareRequest(start=start, end=end)


def test_route_compare_rejects_inconsistent_certified_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/private/graph.json")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")
    scenic_metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    scenic_metrics.update(
        {
            "exactness_status": "approximate_certified",
            "certified_upper_bound": 1.0,
            "optimality_gap": 0.5,
        }
    )
    diagnostics = _strict_diagnostics()
    diagnostics.update(
        {
            "optimization_status": "approximate_certified",
            "certified_upper_bound": 1.0,
            "optimality_gap": 0.5,
        }
    )
    monkeypatch.setattr(
        app_api,
        "plan_routes",
        lambda request, **kwargs: {
            "routes": [{"route_kind": "scenic", "metrics": scenic_metrics}],
            "diagnostics": diagnostics,
            "score_mapping": _strict_score_mapping(),
        },
    )
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 502
    assert "certified gap" in response.json()["detail"]["message"]


def test_route_compare_rejects_per_route_weight_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/private/graph.json")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")
    scenic_metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    scenic_metrics["applied_scenic_weight"] = 0.7
    monkeypatch.setattr(
        app_api,
        "plan_routes",
        lambda request, **kwargs: {
            "routes": [{"route_kind": "scenic", "metrics": scenic_metrics}],
            "diagnostics": _strict_diagnostics(),
            "score_mapping": _strict_score_mapping(),
        },
    )
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 502
    assert "requested settings" in response.json()["detail"]["message"]


def test_route_compare_redacts_private_asset_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/private/graph.json")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")
    scenic_metrics = _strict_route_metrics(
        distance_km=10.0,
        duration_min=20.0,
        raw_score=0.8,
        edge_id="scenic-edge",
    )
    scenic_metrics["score_provenance"] = {"source": "/private/report.json"}
    score_mapping = _strict_score_mapping()
    score_mapping["source"] = "/private/report.json"
    diagnostics = _strict_diagnostics()
    diagnostics["score_mapping_coverage"] = 0.37
    geojson = _strict_geojson(scenic_metrics)
    geojson["features"][0]["properties"]["score_provenance"] = {
        "source": "/private/report.json"
    }
    monkeypatch.setattr(
        app_api,
        "plan_routes",
        lambda request, **kwargs: {
            "routes": [{"route_kind": "scenic", "metrics": scenic_metrics}],
            "diagnostics": diagnostics,
            "score_mapping": score_mapping,
            "geojson": geojson,
        },
    )
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 40.03, "lon": -75.22},
            "end": {"lat": 40.065, "lon": -75.19},
            "scenic_weight": 0.8,
            "region": "philadelphia",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostics"]["score_mapping_coverage"] == 0.37
    assert set(payload["request"]) == {
        "region",
        "run_name",
        "start",
        "end",
        "scenic_weight",
        "avoid_highways",
        "max_detour_factor",
        "include_baseline",
    }
    assert "source" not in payload["score_mapping"]
    assert "source" not in payload["routes"]["scenic"]["score_provenance"]
    assert (
        "source"
        not in payload["geojson"]["features"][0]["properties"]["score_provenance"]
    )
def test_invalid_frontier_configuration_fails_startup_even_when_preload_off(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "off")
    monkeypatch.setenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", "nan")

    with pytest.raises(
        app_api.RouteConfigurationError, match="SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS"
    ):
        with TestClient(create_app()):
            pass


def test_new_england_viewer_renders_diagnostics_contract() -> None:
    html = Path("apps/new_england_north/index.html").read_text(encoding="utf-8")
    viewer = Path("apps/new_england_north/viewer.js").read_text(encoding="utf-8")
    for element_id in (
        "requestedScenicWeight",
        "appliedScenicWeight",
        "requestedDetourCap",
        "appliedDetourCap",
        "actualDurationRatio",
        "optimizationMode",
        "optimizationStatus",
        "optimalityGap",
        "certifiedUpperBound",
        "sameRoute",
        "noBetterRouteReason",
    ):
        assert f'id="{element_id}"' in html
    assert "humanizeReason" in viewer
    assert "approximation_did_not_find_scenic_improvement" in viewer
    assert "X-Route-Request-ID" in viewer
    assert "X-Route-Request-Fingerprint" in viewer
    assert "Tile/report score coverage" in viewer
    assert "addressRequestSequence" in viewer
    assert "isCurrentAddressRequest" in viewer
    assert 'id="scenicWeight"' not in html
    assert "scenic_weight: DEFAULTS.scenicWeight" in viewer
    assert 'url.searchParams.set("scenic_weight"' not in viewer


def test_route_preload_lifespan_calls_once_and_skips_missing_optional(
    tmp_path, monkeypatch
) -> None:
    default_graph = tmp_path / "default.graph.json"
    default_report = tmp_path / "default.report.json"
    default_graph.write_text("valid", encoding="utf-8")
    default_report.write_text("valid", encoding="utf-8")
    optional_graph = tmp_path / "optional.graph.json"
    optional_report = tmp_path / "optional.report.json"
    rows = [
        {
            "region": "optional",
            "graph_path": optional_graph,
            "tile_scores_path": optional_report,
            "run_name": "optional-run",
        },
        {
            "region": "default",
            "graph_path": default_graph,
            "tile_scores_path": default_report,
            "run_name": "default-run",
        },
    ]
    monkeypatch.setattr(app_api, "_configured_regions", lambda: rows)
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "default")
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "required")
    preload_calls: list[tuple[Path, Path, object, object, bool]] = []

    def fake_preload(graph_path, tile_path, zoom, fallback, *, exclusive_scoring):
        preload_calls.append(
            (graph_path, tile_path, zoom, fallback, exclusive_scoring)
        )
        return {
            "graph_cache_hit": False,
            "tile_score_cache_hit": False,
            "scored_graph_cache_hit": False,
        }

    monkeypatch.setattr(app_api, "preload_route_assets", fake_preload)
    with TestClient(create_app()) as test_client:
        health = test_client.get("/v1/healthz")
        assert health.status_code == 200
        payload = health.json()["route_preload"]

    assert preload_calls == [(default_graph, default_report, None, None, True)]
    assert payload["loaded_regions"] == ["default"]
    assert payload["skipped_regions"] == ["optional"]
    assert payload["regions"][0]["region"] == "optional"
    assert payload["regions"][1]["region"] == "default"


def test_route_preload_lifespan_fails_for_corrupt_default(
    tmp_path, monkeypatch
) -> None:
    graph_path = tmp_path / "default.graph.json"
    report_path = tmp_path / "default.report.json"
    graph_path.write_text("present", encoding="utf-8")
    report_path.write_text("present", encoding="utf-8")
    monkeypatch.setattr(
        app_api,
        "_configured_regions",
        lambda: [
            {
                "region": "default",
                "graph_path": graph_path,
                "tile_scores_path": report_path,
                "run_name": "default-run",
            }
        ],
    )
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "default")
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "required")

    def corrupt_preload(*args, **kwargs):
        raise ValueError("corrupt graph")

    monkeypatch.setattr(app_api, "preload_route_assets", corrupt_preload)
    with pytest.raises(RuntimeError, match="default.*preload failed"):
        with TestClient(create_app()):
            pass


def test_route_preload_default_mode_skips_missing_assets(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "region": "optional",
            "graph_path": tmp_path / "optional.graph.json",
            "tile_scores_path": tmp_path / "optional.report.json",
            "run_name": "optional-run",
        },
        {
            "region": "default",
            "graph_path": tmp_path / "default.graph.json",
            "tile_scores_path": tmp_path / "default.report.json",
            "run_name": "default-run",
        },
    ]
    monkeypatch.setattr(app_api, "_configured_regions", lambda: rows)
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "default")
    monkeypatch.delenv("SCENIC_ROUTE_PRELOAD", raising=False)
    preload_calls: list[object] = []
    monkeypatch.setattr(
        app_api,
        "preload_route_assets",
        lambda *args, **kwargs: preload_calls.append((args, kwargs)),
    )

    with TestClient(create_app()) as test_client:
        payload = test_client.get("/v1/healthz").json()["route_preload"]

    assert payload["mode"] == "best_effort"
    assert payload["enabled"] is True
    assert payload["loaded_regions"] == []
    assert payload["skipped_regions"] == ["optional", "default"]
    assert [row["status"] for row in payload["regions"]] == ["skipped", "skipped"]
    assert preload_calls == []


def test_route_preload_off_skips_all_assets(tmp_path, monkeypatch) -> None:
    graph_path = tmp_path / "default.graph.json"
    report_path = tmp_path / "default.report.json"
    graph_path.write_text("present", encoding="utf-8")
    report_path.write_text("present", encoding="utf-8")
    monkeypatch.setattr(
        app_api,
        "_configured_regions",
        lambda: [
            {
                "region": "default",
                "graph_path": graph_path,
                "tile_scores_path": report_path,
                "run_name": "default-run",
            }
        ],
    )
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "default")
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "off")
    preload_calls: list[object] = []
    monkeypatch.setattr(
        app_api,
        "preload_route_assets",
        lambda *args, **kwargs: preload_calls.append((args, kwargs)),
    )

    with TestClient(create_app()) as test_client:
        payload = test_client.get("/v1/healthz").json()["route_preload"]

    assert payload == {
        "enabled": False,
        "mode": "off",
        "default_region": "default",
        "regions": [],
        "loaded_regions": [],
        "skipped_regions": [],
    }
    assert preload_calls == []

def test_route_supervisor_starts_after_preload_and_closes_on_lifespan_exit(
    monkeypatch,
) -> None:
    diagnostics = {
        "enabled": True,
        "mode": "required",
        "loaded_regions": ["default"],
        "regions": [],
    }
    monkeypatch.setattr(app_api, "validate_route_configuration", lambda: None)
    monkeypatch.setattr(app_api, "_route_preload_mode", lambda: "required")
    monkeypatch.setattr(
        app_api, "_preload_configured_route_assets", lambda mode: diagnostics
    )
    starts: list[dict[str, object]] = []
    closed: list[object] = []

    class FakeSupervisor:
        @classmethod
        def start(cls, **kwargs):
            starts.append(kwargs)
            return cls()

        def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr(app_api, "PreloadedRouteSupervisor", FakeSupervisor)
    app = create_app()
    with TestClient(app):
        assert app.state.route_supervisor is not None

    assert len(starts) == 1
    assert starts[0]["preload_marker"] is diagnostics
    assert starts[0]["default_deadline_seconds"] is None
    assert starts[0]["default_grace_seconds"] == 0.5
    assert len(closed) == 1
    assert app.state.route_supervisor is None


def test_route_supervisor_not_started_when_preload_is_off(monkeypatch) -> None:
    monkeypatch.setattr(app_api, "validate_route_configuration", lambda: None)
    monkeypatch.setattr(app_api, "_route_preload_mode", lambda: "off")

    class FailingSupervisor:
        @classmethod
        def start(cls, **kwargs):
            raise AssertionError("supervisor must not start with preload off")

    monkeypatch.setattr(app_api, "PreloadedRouteSupervisor", FailingSupervisor)
    with TestClient(create_app()):
        pass


def test_route_preload_required_fails_for_missing_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        app_api,
        "_configured_regions",
        lambda: [
            {
                "region": "default",
                "graph_path": tmp_path / "default.graph.json",
                "tile_scores_path": tmp_path / "default.report.json",
                "run_name": "default-run",
            }
        ],
    )
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "default")
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "required")

    with pytest.raises(RuntimeError, match="default.*missing"):
        with TestClient(create_app()):
            pass


def test_route_compare_maps_routing_timeout_to_504(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        raise RoutingTimeout("deadline exceeded")

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 504
    assert response.json()["detail"]["error"] == "routing_deadline_exceeded"


def test_route_compare_maps_routing_cancelled_to_non_no_route(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        raise RoutingCancelled("cancelled")

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code != 422
    assert response.json()["detail"]["error"] == "routing_cancelled"


def test_route_compare_passes_one_deadline_to_plan_routes_and_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    captured: list[tuple[object, RoutingDeadline | None]] = []
    diagnosis_captured: list[tuple[object, RoutingDeadline | None]] = []

    def fake_plan(request, **kwargs):
        captured.append((request, kwargs.get("deadline")))
        raise ValueError("no route")

    def fake_diagnose(request, **kwargs):
        diagnosis_captured.append((request, kwargs.get("deadline")))
        return {"graph_nodes": 4, "graph_edges": 3}

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    monkeypatch.setattr(app_api, "diagnose_route_request", fake_diagnose)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )

    assert response.status_code == 422
    assert len(captured) == 1
    request, deadline = captured[0]
    assert isinstance(deadline, RoutingDeadline)
    assert len(diagnosis_captured) == 1
    assert diagnosis_captured[0][1] is deadline


def test_route_compare_fallback_diagnose_maps_routing_timeout_to_504(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        raise ValueError("no route")

    def fake_diagnose(request, **kwargs):
        raise RoutingTimeout("deadline exceeded during fallback diagnostics")

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    monkeypatch.setattr(app_api, "diagnose_route_request", fake_diagnose)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 504
    assert response.json()["detail"]["error"] == "routing_deadline_exceeded"


def test_route_compare_fallback_diagnose_maps_routing_cancelled_to_non_no_route(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        raise ValueError("no route")

    def fake_diagnose(request, **kwargs):
        raise RoutingCancelled("cancelled during fallback diagnostics")

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    monkeypatch.setattr(app_api, "diagnose_route_request", fake_diagnose)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "scenic_weight": 0.8,
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code != 422
    assert response.json()["detail"]["error"] == "routing_cancelled"


def _route_compare_payload() -> dict[str, object]:
    return {
        "start": {"lat": 44.4, "lon": -70.2},
        "end": {"lat": 44.5, "lon": -70.1},
        "scenic_weight": 0.8,
        "region": "new_england_north",
        "max_detour_factor": 1.8,
        "avoid_highways": False,
        "include_baseline": False,
    }


def _stub_route_compare_assets(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api,
        "_region_to_graph",
        lambda region: Path("/tmp/fake-road-graph.geojson"),
    )
    monkeypatch.setattr(
        app_api, "_latest_run_for_region", lambda region: "test-run"
    )


def test_route_compare_supervisor_receives_parent_deadline_without_in_process_plan(
    monkeypatch,
) -> None:
    _stub_route_compare_assets(monkeypatch)
    plan_calls: list[object] = []
    run_calls: list[tuple[object, object, RoutingDeadline]] = []
    validation_deadlines: list[RoutingDeadline] = []

    def fail_plan(request, **kwargs):
        plan_calls.append(request)
        raise AssertionError("in-process planning is not expected")

    def fake_validate(result, *, request, deadline):
        validation_deadlines.append(deadline)
        return (
            {
                "scenic_score_delta_absolute": 0.0,
                "scenic_score_delta_relative": 0.0,
            },
            {
                "scenic": {
                    "total_distance_km": 1.0,
                    "estimated_duration_minutes": 2.0,
                    "normalized_scenic_score": 0.5,
                }
            },
        )

    class FakeSupervisor:
        def run_job(self, func, route_request, *, deadline):
            run_calls.append((func, route_request, deadline))
            return {"score_mapping": {}, "geojson": {}}

    monkeypatch.setattr(app_api, "plan_routes", fail_plan)
    monkeypatch.setattr(app_api, "_validate_route_contract", fake_validate)
    app = create_app()
    app.state.route_supervisor = FakeSupervisor()
    response = TestClient(app).post("/v1/route/compare", json=_route_compare_payload())

    assert response.status_code == 200
    assert plan_calls == []
    assert len(run_calls) == 1
    worker, route_request, deadline = run_calls[0]
    assert worker is app_api._plan_routes_worker
    assert isinstance(route_request, app_api.RouteRequest)
    assert isinstance(deadline, RoutingDeadline)
    assert validation_deadlines == [deadline]


def test_route_compare_supervisor_routing_timeout_maps_to_504(monkeypatch) -> None:
    _stub_route_compare_assets(monkeypatch)

    class FakeSupervisor:
        def run_job(self, *args, **kwargs):
            raise RoutingTimeout("deadline exceeded in supervisor")

    monkeypatch.setattr(
        app_api,
        "plan_routes",
        lambda *args, **kwargs: pytest.fail("in-process planning is not expected"),
    )
    app = create_app()
    app.state.route_supervisor = FakeSupervisor()
    response = TestClient(app).post("/v1/route/compare", json=_route_compare_payload())

    assert response.status_code == 504
    assert response.json()["detail"]["error"] == "routing_deadline_exceeded"


def test_route_compare_supervisor_error_maps_to_structured_503(monkeypatch) -> None:
    _stub_route_compare_assets(monkeypatch)

    class FakeSupervisor:
        def run_job(self, *args, **kwargs):
            raise app_api.SupervisorError("private supervisor failure")

    app = create_app()
    app.state.route_supervisor = FakeSupervisor()
    response = TestClient(app).post("/v1/route/compare", json=_route_compare_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "error": "routing_supervisor_unavailable",
        "message": "The routing service is temporarily unavailable.",
    }