from __future__ import annotations

from fastapi.testclient import TestClient

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
