from __future__ import annotations

import json
import io
import struct

from PIL import Image

from fastapi.testclient import TestClient

import src.app_api.main as app_api

from src.app_api.main import create_app


client = TestClient(create_app())


def test_healthz() -> None:
    resp = client.get("/v1/healthz")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert "regions_available" in payload


def test_regions_list() -> None:
    resp = client.get("/v1/regions")
    assert resp.status_code == 200
    payload = resp.json()
    assert "regions" in payload
    assert isinstance(payload["regions"], list)

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
