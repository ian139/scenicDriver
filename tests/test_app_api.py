from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import struct
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import threading
import time
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
import sys

import pandas as pd

from PIL import Image
import pytest
from fastapi.testclient import TestClient

from scenicdriver.app_api.contrib_repo import ContribConfig, ContribRepo

# The exported app retains this repository, so retain its root for the module's
# lifetime too. Restore main's by-value constructor binding after the import.
_import_directory = TemporaryDirectory(prefix="scenic-api-tests-")
with patch(
    "scenicdriver.app_api.contrib_repo.ContribRepo",
    return_value=ContribRepo(ContribConfig(root_dir=Path(_import_directory.name))),
):
    import scenicdriver.app_api.main as app_api
app_api.ContribRepo = ContribRepo

from scenicdriver.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from scenicdriver.app_api.main import create_app


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


class _InlineTestSupervisor:
    """Keep API contract tests in-process; isolation has separate coverage."""

    def run_job(self, function, *args, deadline, **kwargs):
        return function(deadline, *args, **kwargs)

    def health(self):
        return {"alive": True, "closed": False}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _route_runtime(monkeypatch):
    original = app_api.create_app

    def test_app():
        app = original()
        app.state.route_supervisor = _InlineTestSupervisor()
        return app

    monkeypatch.setattr(app_api, "create_app", test_app)
    monkeypatch.setattr(sys.modules[__name__], "create_app", test_app)


@pytest.fixture(autouse=True)
def contrib_repo(tmp_path, monkeypatch, _route_runtime):
    source = tmp_path / "source.csv"
    seed = tmp_path / "seed.csv"
    rows = [{"image_path": f"philadelphia/{i}.jpg", "scenic_human": 6.5} for i in range(40)]
    pd.DataFrame(rows).to_csv(source, index=False)
    pd.DataFrame(rows).to_csv(seed, index=False)
    repo = ContribRepo(ContribConfig(tmp_path / "contrib", seed, source))
    monkeypatch.setattr(app_api, "ContribRepo", lambda: repo)
    global client
    client = TestClient(create_app())
    try:
        yield repo
    finally:
        client.close()
        del client


def test_healthz() -> None:
    resp = client.get("/v1/healthz")
    assert resp.status_code == 503
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["liveness"] == {"ok": True}
    assert payload["routing"]["ready"] is False
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
    _write_graph_readiness(graph_path)
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
    monkeypatch.setattr(
        app_api.PreloadedRouteSupervisor,
        "start",
        lambda **_kwargs: _InlineTestSupervisor(),
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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": True,
        },
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "/private/secret" not in detail
    assert detail == "Route assets are unavailable for region 'new_england_north'"


def test_regions_list() -> None:
    resp = client.get("/v1/regions")
    assert resp.status_code == 200
    payload = resp.json()
    assert "regions" in payload
    assert isinstance(payload["regions"], list)


def test_northeast_expanded_region_configuration_and_discovery() -> None:
    resp = client.get("/v1/regions")
    assert resp.status_code == 200
    payload = resp.json()
    assert "regions" in payload
    regions = payload["regions"]
    region_map = {r["region"]: r for r in regions}

    # Stale new_england_expanded key must no longer be present
    assert "new_england_expanded" not in region_map

    # Truthful northeast_expanded region must be present
    assert "northeast_expanded" in region_map
    ne_region = region_map["northeast_expanded"]
    assert ne_region["display_name"] == "Northeast Expanded (Prompt Two exp02)"
    assert (
        ne_region["latest_run_name"] == "prompt_two_candidate_exp02_expanded_20260810"
    )
    assert ne_region["is_default"] is False
    assert ne_region["source"] == "config"
    assert ne_region["graph_exists"] is (
        app_api.PROJECT_ROOT
        / "data/processed/road_graphs/northeast_expanded_scored_tiles_v1/road_graph.compact.json"
    ).is_file()
    if not ne_region["graph_exists"]:
        assert ne_region["route_planning_enabled"] is False

    desc = ne_region["description"]
    assert (
        desc
        == "Expanded scenic coverage and cross-region routing across New England North and the western scored extent."
    )

    item = app_api._app_region("northeast_expanded")
    assert item is not None
    assert item["route_planning"] is True
    assert item["route_deadline_seconds"] == 120
    assert (
        item["graph"]
        == "data/processed/road_graphs/northeast_expanded_scored_tiles_v1/road_graph.compact.json"
    )
    assert item["run_name"] == "prompt_two_candidate_exp02_expanded_20260810"

    # The configured bounds match the full Stage One candidate-pool extent.
    assert ne_region["bbox"] == {
        "min_lat": 37.99616267972812,
        "min_lon": -83.56201171875,
        "max_lat": 47.50235895196859,
        "max_lon": -66.77490234375,
    }

    # Check map center and zoom for full extent visualization
    assert ne_region["map"]["center"]["lat"] == 42.749261
    assert ne_region["map"]["center"]["lon"] == -75.168457
    assert ne_region["map"]["zoom"] == 5.5

    # Verify canonical default region remains unchanged and is default
    default_region = region_map.get("new_england_north")
    assert default_region is not None
    assert default_region["is_default"] is True


def test_northeast_expanded_report_availability_semantics() -> None:
    # Ensure northeast_expanded is configured with expected report path
    item = app_api._app_region("northeast_expanded")
    assert item is not None
    assert item["run_name"] == "prompt_two_candidate_exp02_expanded_20260810"
    assert (
        item["tile_scores_path"]
        == "data/processed/heuristic_runs/prompt_two_candidate_exp02_expanded_20260810/report/report.json"
    )

    # Validate preload diagnostics: optional region handling
    diag = app_api._preload_configured_route_assets(mode="best_effort")
    assert (
        "northeast_expanded" in diag["skipped_regions"]
        or "northeast_expanded" in diag["loaded_regions"]
    )
    assert (
        "new_england_north" in diag["loaded_regions"]
        or "new_england_north" in diag["skipped_regions"]
    )


def test_preload_configured_route_assets_skips_disabled_route_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_graph = tmp_path / "road_graph.json"
    fake_graph.write_text("{}", encoding="utf-8")
    fake_report = tmp_path / "report.json"
    fake_report.write_text("{}", encoding="utf-8")

    disabled_item = {
        "region": "disabled_region",
        "graph_path": str(fake_graph),
        "tile_scores_path": str(fake_report),
        "route_planning": False,
    }

    monkeypatch.setattr(app_api, "_configured_regions", lambda: [disabled_item])
    monkeypatch.setattr(app_api, "_default_region_key", lambda: None)

    preload_called = False

    def _fail_preload(*args: Any, **kwargs: Any) -> Any:
        nonlocal preload_called
        preload_called = True
        raise AssertionError("preload_route_assets should not be called for disabled region")

    monkeypatch.setattr(app_api, "preload_route_assets", _fail_preload)

    diag = app_api._preload_configured_route_assets(mode="best_effort")

    assert preload_called is False
    assert "disabled_region" in diag["skipped_regions"]
    assert "disabled_region" not in diag["loaded_regions"]
    region_diags = [r for r in diag["regions"] if r["region"] == "disabled_region"]
    assert len(region_diags) == 1
    assert region_diags[0]["status"] == "skipped"
    assert region_diags[0]["reason"] == "route planning disabled"


def _write_graph_readiness(graph: Path) -> Path:
    import hashlib

    gate = graph.parent / "run.json"
    gate.write_text(json.dumps({
        "output_graph": str(graph),
        "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "graph_size_bytes": graph.stat().st_size,
        "counts": {"nodes": 2, "edges": 1},
    }), encoding="utf-8")
    return gate


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("gate_state", [
    "missing", "malformed", "empty", "bad_hash", "wrong_graph", "stale_gate",
])
def test_unready_regions_do_not_list_route_compare_or_preload(
    tmp_path, monkeypatch, configured, gate_state
) -> None:
    graph_dir = tmp_path / "test_region"
    graph_dir.mkdir()
    graph = graph_dir / "road_graph.json"
    graph.write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text('{"tiles": [], "summary": {}}', encoding="utf-8")
    item = {
        "region": "test_region", "graph_path": str(graph),
        "tile_scores_path": str(report), "run_name": "test-run",
    }
    monkeypatch.setattr(app_api, "ROAD_GRAPHS_DIR", tmp_path)
    monkeypatch.setattr(app_api, "_configured_regions", lambda: [item] if configured else [])
    monkeypatch.setattr(app_api, "_default_region_key", lambda: None)
    gate = graph_dir / "run.json"
    if gate_state == "malformed":
        gate.write_text("{", encoding="utf-8")
    elif gate_state == "empty":
        gate.write_text("{}", encoding="utf-8")
    elif gate_state in {"bad_hash", "wrong_graph"}:
        _write_graph_readiness(graph)
        record = json.loads(gate.read_text())
        record["graph_sha256" if gate_state == "bad_hash" else "output_graph"] = (
            "0" * 64 if gate_state == "bad_hash" else str(tmp_path / "other.json")
        )
        gate.write_text(json.dumps(record), encoding="utf-8")
    elif gate_state == "stale_gate":
        _write_graph_readiness(graph)
        assert app_api._list_regions()[0]["route_planning_enabled"] is True
        graph.write_text("[]", encoding="utf-8")  # Same size; the old digest must not be reused.
    rows = app_api._list_regions()
    assert rows[0]["graph_exists"] is True
    assert rows[0]["route_planning_enabled"] is False
    response = client.post("/v1/route/compare", json={
        "region": "test_region",
        "start": {"lat": 42, "lon": -72}, "end": {"lat": 42.01, "lon": -72},
    })
    assert response.status_code == 404
    assert response.json()["detail"] == "Route assets are unavailable for region 'test_region'"
    if configured:
        diagnostics = app_api._preload_configured_route_assets(mode="best_effort")
        assert diagnostics["loaded_regions"] == []
        assert diagnostics["skipped_regions"] == ["test_region"]
        monkeypatch.setattr(app_api, "_default_region_key", lambda: "test_region")
        with pytest.raises(RuntimeError, match="readiness"):
            app_api._preload_configured_route_assets(mode="required")


@pytest.mark.parametrize("configured", [False, True])
def test_region_readiness_is_withdrawn_during_graph_swap(tmp_path, monkeypatch, configured) -> None:
    graph_dir = tmp_path / "test_region"
    graph_dir.mkdir()
    graph = graph_dir / "road_graph.json"
    graph.write_text("{}", encoding="utf-8")
    item = {"region": "test_region", "graph_path": str(graph)}
    monkeypatch.setattr(app_api, "ROAD_GRAPHS_DIR", tmp_path)
    monkeypatch.setattr(app_api, "_configured_regions", lambda: [item] if configured else [])
    monkeypatch.setattr(app_api, "_default_region_key", lambda: None)
    gate = _write_graph_readiness(graph)
    assert app_api._list_regions()[0]["route_planning_enabled"] is True
    assert app_api._region_to_graph("test_region") == graph
    gate.unlink()
    graph.write_text('{"new": true}', encoding="utf-8")
    assert app_api._list_regions()[0]["route_planning_enabled"] is False
    with pytest.raises(FileNotFoundError, match="readiness"):
        app_api._region_to_graph("test_region")
    _write_graph_readiness(graph)
    assert app_api._list_regions()[0]["route_planning_enabled"] is True
    assert app_api._region_to_graph("test_region") == graph

def test_region_graph_fallback_and_discovery_support_sqlite(
    tmp_path, monkeypatch
) -> None:
    graph_dir = tmp_path / "sqlite_region"
    graph_dir.mkdir()
    sqlite_graph = graph_dir / "road_graph.sqlite3"
    sqlite_graph.write_bytes(b"sqlite")
    _write_graph_readiness(sqlite_graph)
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
            "route_planning_enabled": True,
            "latest_run_name": None,
            "bbox": None,
            "is_default": False,
            "source": "discovered",
        }
    ]


def test_latest_run_for_region_uses_exact_key_without_legacy_aliases(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "MassWhites_z14_learned"
    run_dir.mkdir()
    (tmp_path / "masswhites-other").mkdir()
    monkeypatch.setattr(app_api, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app_api, "_app_region", lambda region: None)

    assert app_api._latest_run_for_region("  MASSWHITES  ") == run_dir.name
    assert app_api._latest_run_for_region("mass") is None
    assert app_api._latest_run_for_region("masswhites_wide") is None
    assert app_api._latest_run_for_region("pittsfield") is None


def test_latest_run_for_region_preserves_configured_run_name(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api,
        "_app_region",
        lambda region: {"region": region, "run_name": "canonical-run"},
    )

    assert app_api._latest_run_for_region("pittsfield") == "canonical-run"


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
    import math
    run_name = "new-england-test"
    report_dir = tmp_path / run_name / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": {"min": 2.0, "max": 6.0},
                "tiles": [
                    {"z": 14, "x": 5000, "y": 5900, "scenic_score": 2.0},
                    {"z": 14, "x": 5002, "y": 5900, "scenic_score": 6.0},
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
    assert image.size == (3, 1)
    assert image.getpixel((0, 0)) != image.getpixel((2, 0))
    assert image.getpixel((1, 0))[3] == 0

    score_response = TestClient(create_app()).get(
        "/v1/heatmap-scores.bin",
        params={"region": "new_england_north", "run_name": run_name},
    )
    assert score_response.status_code == 200
    assert score_response.headers["x-scenic-tile-zoom"] == "14"
    assert score_response.headers["x-scenic-min-x"] == "5000"
    assert score_response.headers["x-scenic-min-y"] == "5900"
    assert score_response.headers["x-scenic-grid-width"] == "3"
    assert score_response.headers["x-scenic-grid-height"] == "1"
    low, absent, high = struct.unpack("<3f", score_response.content)
    assert (low, high) == (2.0, 6.0)
    assert math.isnan(absent)


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
                        "geometry": {"type": "Point", "coordinates": [-68.771305, 44.801616]},
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
                "properties": {
                    "route_kind": route_kind,
                    "segment_identity": [{"edge_id": "rich-artifact-segment"}],
                    "score_provenance": {"source": "/private/report.json"},
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-73, 44], [-69, 45]],
                },
            }
            for route_kind in ("scenic", "baseline")
        ],
    }
    (report_dir / "route.geojson").write_text(
        json.dumps(route_geojson), encoding="utf-8"
    )
    (report_dir / "route_metrics.json").write_text(
        json.dumps(
            {
                "request": {
                    "start": [44.4, -73.2],
                    "end": [44.8, -68.8],
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
                "scenic": {
                    "route_kind": "scenic", "total_distance_km": 461.2,
                    "estimated_duration_minutes": 400, "raw_scenic_score": 7,
                    "normalized_scenic_score": 0.7,
                },
                "baseline": {
                    "route_kind": "baseline", "total_distance_km": 471.0,
                    "estimated_duration_minutes": 390, "raw_scenic_score": 6,
                    "normalized_scenic_score": 0.6,
                },
            }
        ),
        encoding="utf-8",
    )
    regions_path = tmp_path / "regions.json"
    regions_path.write_text(
        json.dumps(
            {
                "default_region": "new_england_north",
                "regions": [{"region": "new_england_north", "run_name": run_name}],
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
    for feature in payload["geojson"]["features"]:
        kind = feature["properties"]["route_kind"]
        route = payload["routes"][kind]
        assert feature["geometry"] == route["geometry"]
        assert route["snapped_start"] == [44, -73]
        assert route["snapped_end"] == [45, -69]
        assert feature["properties"] == {
            "route_kind": kind, "snapped_start": [44, -73], "snapped_end": [45, -69],
        }
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


def _new_contributor(display_name: str = "Tester") -> tuple[str, dict[str, str]]:
    response = client.post("/v1/contrib/profile", json={"display_name": display_name})
    assert response.status_code == 201
    body = response.json()
    return body["profile"]["contributor_id"], {"X-Contributor-Token": body["contributor_token"]}


def _contrib_payload(repo, contributor_id="tester", **changes):
    task = repo.next_tasks(contributor_id, "philadelphia", count=1)[0]
    return {"contributor_id": contributor_id, "task_id": task["task_id"],
            "image_path": task["image_path"], "scenic_human": 6.5, **changes}


def _label_body(payload):
    # Identity travels only in X-Contributor-Token; the body never names it.
    return {key: value for key, value in payload.items() if key != "contributor_id"}


def _contrib_csv(repo, path):
    repo.export_csvs()
    return pd.read_csv(path)


def _contrib_snapshot(repo):
    with repo._transaction() as connection:
        return tuple(connection.iterdump())


def test_contrib_token_holder_flow_end_to_end(contrib_repo, monkeypatch) -> None:
    created = client.post("/v1/contrib/profile", json={"display_name": "Tester API"})
    assert created.status_code == 201
    token = created.json()["contributor_token"]
    contributor_id = created.json()["profile"]["contributor_id"]
    assert created.json()["profile"]["display_name"] == "Tester API"
    assert "token_sha256" not in created.json()["profile"]
    # Only the hash is persisted; the plaintext is returned exactly once.
    stored = str(_contrib_snapshot(contrib_repo))
    assert token not in stored
    assert hashlib.sha256(token.encode()).hexdigest() in stored
    headers = {"X-Contributor-Token": token}

    start = client.post("/v1/contrib/session/start", json={"region": "philadelphia"},
                        headers=headers)
    assert start.status_code == 200
    assert start.json()["profile"]["contributor_id"] == contributor_id
    assert start.json()["tasks"]
    more = client.get("/v1/contrib/tasks/next", params={"region": "philadelphia", "count": 2},
                      headers=headers)
    assert more.status_code == 200
    task = more.json()["tasks"][0]
    assert task["task_id"] == start.json()["tasks"][0]["task_id"]

    label = client.post("/v1/contrib/labels", headers=headers, json={
        "task_id": task["task_id"], "image_path": task["image_path"], "scenic_human": 6.5,
        "confidence": "high", "skip": False, "notes": "api smoke test",
        "region": "philadelphia",
    })
    assert label.status_code == 200
    assert label.json()["saved"] is True
    assert label.json()["record"]["contributor_id"] == contributor_id

    renamed = client.patch("/v1/contrib/profile", json={"display_name": "Renamed"},
                           headers=headers)
    assert renamed.status_code == 200
    assert renamed.json()["profile"]["display_name"] == "Renamed"

    monkeypatch.setenv("CONTRIB_ADMIN_TOKEN", "admin-secret")
    review = client.post("/v1/admin/contrib/review/run", headers={"X-Admin-Token": "admin-secret"})
    assert review.status_code == 200
    assert review.json()["promoted"] == 1

    board = client.get("/v1/contrib/leaderboard").json()["leaderboard"]
    assert [(row["contributor_id"], row["display_name"], row["credits"]) for row in board] == [
        (contributor_id, "Renamed", 1.5)
    ]
    assert all("token_sha256" not in row and "agreement_count" not in row for row in board)
    profile = client.get("/v1/contrib/profile", params={"contributor_id": contributor_id})
    assert profile.status_code == 200
    assert profile.json()["profile"] == board[0]


def test_contrib_profile_get_is_read_only(contrib_repo) -> None:
    before = _contrib_snapshot(contrib_repo)
    response = client.get("/v1/contrib/profile", params={"contributor_id": "unknown"})
    assert response.status_code == 404
    assert _contrib_snapshot(contrib_repo) == before


def test_contrib_impersonation_with_known_id_is_rejected(contrib_repo) -> None:
    victim_id, victim_headers = _new_contributor("Victim")
    victim_task = client.get("/v1/contrib/tasks/next", params={"region": "philadelphia", "count": 1},
                             headers=victim_headers).json()["tasks"][0]
    # The victim's ID is public; knowing it must not grant any mutation.
    assert client.get("/v1/contrib/leaderboard").json()["leaderboard"][0]["contributor_id"] == victim_id
    forged_label = {"contributor_id": victim_id, "task_id": victim_task["task_id"],
                    "image_path": victim_task["image_path"], "scenic_human": 6.5}
    before = _contrib_snapshot(contrib_repo)
    for headers in ({}, {"X-Contributor-Token": victim_id}, {"X-Contributor-Token": "forged"}):
        assert client.post("/v1/contrib/labels", json=forged_label, headers=headers).status_code == 401
        assert client.post("/v1/contrib/labels", json=_label_body(forged_label),
                           headers=headers).status_code == 401
        assert client.post("/v1/contrib/session/start", json={"contributor_id": victim_id},
                           headers=headers).status_code == 401
        assert client.get("/v1/contrib/tasks/next", params={"contributor_id": victim_id},
                          headers=headers).status_code == 401
        assert client.patch("/v1/contrib/profile", json={"display_name": "Owned"},
                            headers=headers).status_code == 401
    assert _contrib_snapshot(contrib_repo) == before

    # A valid token cannot act for someone else: body identity is not accepted,
    # and the victim's task was never issued to the attacker.
    _, attacker_headers = _new_contributor("Attacker")
    assert client.post("/v1/contrib/labels", json=forged_label,
                       headers=attacker_headers).status_code == 422
    assert client.post("/v1/contrib/labels", json=_label_body(forged_label),
                       headers=attacker_headers).status_code == 422
    assert _contrib_csv(contrib_repo, contrib_repo.pending_path).empty
    assert contrib_repo.get_profile(victim_id)["display_name"] == "Victim"


def test_contrib_legacy_profile_without_token_cannot_mutate(contrib_repo) -> None:
    # Legacy CSVs predate token_sha256; their rows stay readable but immutable.
    pd.DataFrame([{
        "contributor_id": "legacy", "display_name": "Legacy", "credits": 3.0,
        "approved_labels": 2, "pending_labels": 0, "agreement_score": 0.9,
        "agreement_count": 2, "created_at": "2026-01-01", "updated_at": "2026-01-01",
    }]).to_csv(contrib_repo.profiles_path, index=False)
    # The legacy snapshot is installed before the first database initialization.
    contrib_repo.db_path.unlink()
    repo = ContribRepo(contrib_repo.cfg)
    profile = client.get("/v1/contrib/profile", params={"contributor_id": "legacy"})
    assert profile.status_code == 200
    assert profile.json()["profile"]["display_name"] == "Legacy"
    assert "token_sha256" not in profile.json()["profile"]
    empty_hash = hashlib.sha256(b"").hexdigest()
    for token in ("", "legacy", empty_hash):
        assert repo.authenticate(token) is None
        headers = {"X-Contributor-Token": token}
        assert client.get("/v1/contrib/tasks/next", headers=headers).status_code == 401
        assert client.patch("/v1/contrib/profile", json={"display_name": "x"},
                            headers=headers).status_code == 401
    assert _contrib_csv(repo, repo.tasks_path).empty


def test_contrib_concurrent_submissions_lose_nothing(contrib_repo):
    repo = contrib_repo
    tasks = repo.next_tasks("parallel", "philadelphia", count=25)
    # Independent instances share SQLite's transactional write serialization.
    repos = [ContribRepo(repo.cfg) for _ in tasks]
    with ThreadPoolExecutor(max_workers=10) as workers:
        list(workers.map(
            lambda item: item[0].submit_label({
                "contributor_id": "parallel", "task_id": item[1]["task_id"],
                "image_path": item[1]["image_path"], "scenic_human": 6.5,
            }), zip(repos, tasks)))
    pending = _contrib_csv(repo, repo.pending_path)
    assert set(pending["task_id"]) == {task["task_id"] for task in tasks}
    assert repo.get_profile("parallel")["pending_labels"] == 25


@pytest.mark.parametrize("marker", ["", "NA", "N/A", "nan", "NaN", "null", "None"])
@pytest.mark.parametrize("column", ["class_id", "scenic_score"])
def test_contrib_optional_numeric_metadata_is_missing(contrib_repo, marker, column):
    pd.DataFrame([{
        "image_path": "philadelphia/NA.jpg", "class_id": "",
        "class_name": "NA", "scenic_score": "", column: marker,
    }]).to_csv(contrib_repo.cfg.labels_source_csv, index=False)
    _, headers = _new_contributor()
    response = client.post("/v1/contrib/session/start", json={"region": "philadelphia"},
                           headers=headers)
    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["class_id"] is None
    assert task["scenic_score"] is None
    assert task["class_name"] == "NA"
    assert task["image_path"] == "philadelphia/NA.jpg"


def test_contrib_float_form_class_ids_beside_blanks(contrib_repo):
    pd.DataFrame([
        {"image_path": "philadelphia/1.jpg", "class_id": "1.0", "scenic_score": "6.5"},
        {"image_path": "philadelphia/2.jpg", "class_id": "", "scenic_score": ""},
    ]).to_csv(contrib_repo.cfg.labels_source_csv, index=False)
    _, headers = _new_contributor()
    response = client.post("/v1/contrib/session/start", json={"region": "philadelphia"},
                           headers=headers)
    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert [(task["class_id"], task["scenic_score"]) for task in tasks] == [(1, 6.5), (None, None)]


@pytest.mark.parametrize("score", ["inf", "-inf", "Infinity", "-Infinity", "1e309"])
def test_contrib_nonfinite_task_scores_are_rejected(contrib_repo, score):
    pd.DataFrame([{
        "image_path": "philadelphia/bad.jpg", "scenic_score": score,
    }]).to_csv(contrib_repo.cfg.labels_source_csv, index=False)
    _, headers = _new_contributor()
    response = client.post("/v1/contrib/session/start", json={"region": "philadelphia"},
                           headers=headers)
    assert response.status_code == 422
    assert _contrib_csv(contrib_repo, contrib_repo.tasks_path).empty


@pytest.mark.parametrize("class_id", ["1.5", "inf", "unknown"])
def test_contrib_invalid_class_ids_are_rejected(contrib_repo, class_id):
    pd.DataFrame([{
        "image_path": "philadelphia/bad.jpg", "class_id": class_id,
    }]).to_csv(contrib_repo.cfg.labels_source_csv, index=False)
    _, headers = _new_contributor()
    response = client.post("/v1/contrib/session/start", json={"region": "philadelphia"},
                           headers=headers)
    assert response.status_code == 422
    assert _contrib_csv(contrib_repo, contrib_repo.tasks_path).empty


def _exercise_unpatched_apps(root):
    # Spawned processes import this module without running its pytest fixtures.
    cfg = ContribConfig(root_dir=Path(root))
    with patch("scenicdriver.app_api.contrib_repo.ContribConfig", return_value=cfg):
        fresh = create_app()
    results = []
    for app, display_name in [(app_api.app, "imported"), (fresh, "fresh")]:
        with TestClient(app) as test_client:
            created = test_client.post("/v1/contrib/profile", json={"display_name": display_name})
            contributor_id = created.json()["profile"]["contributor_id"]
            response = test_client.get("/v1/contrib/profile", params={"contributor_id": contributor_id})
            results.append((created.status_code, response.status_code,
                             response.json()["profile"]["display_name"]))
    return results, (cfg.root_dir / "contributors.sqlite3").exists()


def test_contrib_imported_app_and_unpatched_factory_remain_usable(tmp_path):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as workers:
        results, persisted = workers.submit(_exercise_unpatched_apps, tmp_path / "factory").result(timeout=60)
    assert results == [(201, 200, "imported"), (201, 200, "fresh")]
    assert persisted


@pytest.mark.parametrize("contributor_id", ["007", "NA", "nan", "1e5"])
def test_contrib_identifier_round_trip(contrib_repo, contributor_id, monkeypatch):
    import scenicdriver.app_api.contrib_repo as storage
    repo = contrib_repo
    monkeypatch.setattr(storage.uuid, "uuid4", lambda: contributor_id)
    profile, token = repo.create_profile("Round trip")
    assert profile["contributor_id"] == contributor_id
    assert repo.authenticate(token) == contributor_id
    payload = _contrib_payload(repo, contributor_id)
    assert payload["task_id"] == contributor_id
    repo = ContribRepo(repo.cfg)
    saved = repo.submit_label(payload)
    assert saved["task_id"] == contributor_id
    assert repo.get_profile(contributor_id)["contributor_id"] == contributor_id
    assert repo.run_qa_promotion()["promoted"] == 1
    assert repo.get_profile(contributor_id)["credits"] == 1.5
    assert len(_contrib_csv(repo, repo.profiles_path)) == 1


@pytest.mark.parametrize("score", [None, 6.5])
def test_contrib_skip_never_approved_or_credited(contrib_repo, score):
    contributor_id, headers = _new_contributor()
    payload = _contrib_payload(contrib_repo, contributor_id, skip=True)
    if score is None:
        payload.pop("scenic_human")
    response = client.post("/v1/contrib/labels", json=_label_body(payload), headers=headers)
    assert response.status_code == 200
    result = contrib_repo.run_qa_promotion()
    assert result["promoted"] == 0
    assert result["remaining_pending"] == 0
    profile = contrib_repo.get_profile(contributor_id)
    assert profile["credits"] == profile["approved_labels"] == profile["pending_labels"] == 0
    assert _contrib_csv(contrib_repo, contrib_repo.approved_path).empty
    events = _contrib_csv(contrib_repo, contrib_repo.events_path)
    assert json.loads(events.iloc[0]["metadata"])["status"] == "skipped"


def test_contrib_first_approved_agreement_initializes_score(contrib_repo):
    contributor_id, headers = _new_contributor()
    start = client.post("/v1/contrib/session/start", json={"region": "philadelphia"},
                        headers=headers)
    assert start.status_code == 200
    initial_profile = start.json()["profile"]
    assert initial_profile["agreement_score"] == 0
    assert "agreement_count" not in initial_profile
    assert "token_sha256" not in initial_profile
    assert _contrib_csv(contrib_repo, contrib_repo.profiles_path).iloc[0]["agreement_count"] == 0
    payload = _label_body(_contrib_payload(contrib_repo, contributor_id))
    response = client.post("/v1/contrib/labels", json=payload, headers=headers)
    assert response.status_code == 200
    assert set(response.json()["profile"]) == set(initial_profile)
    assert contrib_repo.run_qa_promotion()["promoted"] == 1
    profile = client.get("/v1/contrib/profile", params={"contributor_id": contributor_id})
    assert profile.status_code == 200
    assert profile.json()["profile"]["agreement_score"] == 1.0
    assert set(profile.json()["profile"]) == set(initial_profile)
    assert _contrib_csv(contrib_repo, contrib_repo.profiles_path).iloc[0]["agreement_count"] == 1


@pytest.mark.parametrize("same_batch", [False, True])
def test_contrib_agreement_mean_persists_and_reaches_leaderboard(contrib_repo, same_batch):
    repo = contrib_repo
    repo.submit_label(_contrib_payload(repo))
    if not same_batch:
        assert repo.run_qa_promotion()["promoted"] == 1
        repo = ContribRepo(repo.cfg)
    repo.submit_label(_contrib_payload(repo, scenic_human=1.5))
    assert repo.run_qa_promotion(min_agreement=0.5)["promoted"] == (2 if same_batch else 1)
    reopened = ContribRepo(repo.cfg)
    profile = reopened.get_profile("tester")
    assert profile["agreement_score"] == 0.75
    assert profile["approved_labels"] == 2
    assert _contrib_csv(repo, repo.profiles_path).iloc[0]["agreement_count"] == 2
    response = client.get("/v1/contrib/leaderboard")
    assert response.status_code == 200
    assert response.json()["leaderboard"] == [profile]
    # A third approval must weight the existing two labels, not just the mean.
    reopened.submit_label(_contrib_payload(reopened))
    assert reopened.run_qa_promotion()["promoted"] == 1
    assert reopened.get_profile("tester")["agreement_score"] == pytest.approx(5 / 6)
    assert _contrib_csv(repo, repo.profiles_path).iloc[0]["agreement_count"] == 3


@pytest.mark.parametrize("missing_count", ["column", "value"])
def test_contrib_legacy_agreement_restarts_at_next_approval(contrib_repo, missing_count):
    repo = contrib_repo
    pd.DataFrame([{
        "contributor_id": "tester", "display_name": "tester", "credits": 0.0,
        "approved_labels": 0, "pending_labels": 0, "agreement_score": 0.0,
        "agreement_count": 0, "created_at": "2026-01-01", "updated_at": "2026-01-01",
    }]).to_csv(repo.profiles_path, index=False)
    profiles = pd.read_csv(repo.profiles_path)
    profiles["agreement_score"] = 0.875
    profiles["approved_labels"] = 8
    if missing_count == "column":
        profiles = profiles.drop(columns="agreement_count")
    else:
        profiles["agreement_count"] = ""
    profiles.to_csv(repo.profiles_path, index=False)
    repo.db_path.unlink()
    repo = ContribRepo(repo.cfg)
    assert repo.get_profile("tester")["agreement_score"] == 0.875
    repo.submit_label(_contrib_payload(repo))
    assert repo.run_qa_promotion()["promoted"] == 1
    assert repo.get_profile("tester")["agreement_score"] == 1.0
    assert repo.get_profile("tester")["approved_labels"] == 9
    assert _contrib_csv(repo, repo.profiles_path).iloc[0]["agreement_count"] == 1


def test_contrib_replay_foreign_and_mismatched_tasks_rejected(contrib_repo):
    owner_id, owner_headers = _new_contributor("Owner")
    stranger_id, stranger_headers = _new_contributor("Stranger")
    payload = _label_body(_contrib_payload(contrib_repo, owner_id))
    assert client.post("/v1/contrib/labels", json=payload,
                       headers=stranger_headers).status_code == 422
    for changes in ({"contributor_id": owner_id}, {"task_id": "unknown"},
                    {"image_path": "philadelphia/other.jpg"}, {"task_id": " "}):
        assert client.post("/v1/contrib/labels", json={**payload, **changes},
                           headers=owner_headers).status_code == 422
    assert client.post("/v1/contrib/labels", json=payload, headers=owner_headers).status_code == 200
    assert client.post("/v1/contrib/labels", json=payload, headers=owner_headers).status_code == 422
    assert contrib_repo.run_qa_promotion()["promoted"] == 1
    assert client.post("/v1/contrib/labels", json=payload, headers=owner_headers).status_code == 422
    assert contrib_repo.get_profile(owner_id)["credits"] == 1.5
    assert contrib_repo.get_profile(stranger_id)["credits"] == 0
    # Different contributors may annotate the same image independently.
    other = _label_body(_contrib_payload(contrib_repo, stranger_id))
    assert other["image_path"] == payload["image_path"]
    assert client.post("/v1/contrib/labels", json=other, headers=stranger_headers).status_code == 200


def test_contrib_concurrent_replay_earns_one_credit(contrib_repo):
    payload = _contrib_payload(contrib_repo)

    def submit(_):
        try:
            contrib_repo.submit_label(payload)
            return "saved"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(submit, range(8)))
    assert results.count("saved") == 1
    assert results.count("rejected") == 7
    assert contrib_repo.run_qa_promotion()["promoted"] == 1
    assert contrib_repo.get_profile("tester")["credits"] == 1.5


def _submit_process_batch(cfg, payloads):
    repo = ContribRepo(cfg)
    _submission_barrier.wait(timeout=30)
    results = []
    for payload in payloads:
        try:
            repo.submit_label(payload)
            results.append(("saved", payload["task_id"]))
        except ValueError:
            results.append(("rejected", payload["task_id"]))
    return results


def _init_process_barrier(barrier):
    global _submission_barrier
    _submission_barrier = barrier


def test_contrib_multiprocess_submissions_and_replay(contrib_repo):
    repo = contrib_repo
    tasks = repo.next_tasks("processes", "philadelphia", count=20)
    payloads = [{"contributor_id": "processes", "task_id": task["task_id"],
                 "image_path": task["image_path"], "scenic_human": 6.5} for task in tasks]
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    with ProcessPoolExecutor(max_workers=4, mp_context=context,
                             initializer=_init_process_barrier, initargs=(barrier,)) as workers:
        futures = [workers.submit(_submit_process_batch, repo.cfg, payloads[i::4]) for i in range(4)]
        saved = [row for future in futures for row in future.result(timeout=60)]
        assert set(saved) == {("saved", task["task_id"]) for task in tasks}
        pending = _contrib_csv(repo, repo.pending_path)
        assert len(pending) == 20
        assert set(pending["task_id"]) == {task["task_id"] for task in tasks}
        assert repo.get_profile("processes")["pending_labels"] == 20

        replay = _contrib_payload(repo, "replay")
        futures = [workers.submit(_submit_process_batch, repo.cfg, [replay]) for _ in range(4)]
        results = [row for future in futures for row in future.result(timeout=60)]
        assert results.count(("saved", replay["task_id"])) == 1
        assert results.count(("rejected", replay["task_id"])) == 3
    pending = _contrib_csv(repo, repo.pending_path)
    expected = {task["task_id"] for task in tasks} | {replay["task_id"]}
    assert len(pending) == 21
    assert set(pending["task_id"]) == expected
    assert repo.run_qa_promotion()["promoted"] == 21
    assert repo.run_qa_promotion()["promoted"] == 0
    assert repo.get_profile("replay")["credits"] == 1.5
    assert repo.get_profile("replay")["approved_labels"] == 1
    assert repo.get_profile("processes")["credits"] == 30
    approved = _contrib_csv(repo, repo.approved_path)
    assert len(approved) == 21
    assert set(approved["task_id"]) == expected
    events = _contrib_csv(repo, repo.events_path)
    assert len(events) == 21
    assert set(events["contributor_id"]) == {"processes", "replay"}




def _contrib_fault_connection(repo, monkeypatch, *, fail_commit=False, fail_rollback=False):
    import sqlite3
    original_connect = repo._connect

    class FaultConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if not fail_commit and sql.startswith("INSERT INTO events"):
                raise OSError("injected event persistence failure")
            return super().execute(sql, parameters)

        def commit(self):
            if fail_commit:
                raise OSError("injected transaction commit failure")
            return super().commit()

        def rollback(self):
            if fail_rollback:
                raise OSError("interrupted rollback")
            return super().rollback()

    def connect():
        connection = sqlite3.connect(repo.db_path, isolation_level=None, factory=FaultConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    monkeypatch.setattr(repo, "_connect", connect)
    return original_connect


def test_contrib_mid_promotion_failure_rolls_back(contrib_repo, monkeypatch):
    repo = contrib_repo
    repo.submit_label(_contrib_payload(repo))
    before = _contrib_snapshot(repo)
    original_connect = _contrib_fault_connection(repo, monkeypatch)
    with pytest.raises(OSError, match="injected"):
        repo.run_qa_promotion()
    monkeypatch.setattr(repo, "_connect", original_connect)
    assert _contrib_snapshot(repo) == before
    assert repo.run_qa_promotion()["promoted"] == 1
    assert repo.get_profile("tester")["credits"] == 1.5


def test_contrib_promotion_commit_failure_changes_nothing(contrib_repo, monkeypatch):
    repo = contrib_repo
    repo.submit_label(_contrib_payload(repo))
    before = _contrib_snapshot(repo)
    original_connect = _contrib_fault_connection(repo, monkeypatch, fail_commit=True)
    with pytest.raises(OSError, match="commit"):
        repo.run_qa_promotion()
    monkeypatch.setattr(repo, "_connect", original_connect)
    assert _contrib_snapshot(repo) == before


def test_contrib_failed_rollback_recovers_on_reopen(contrib_repo, monkeypatch):
    repo = contrib_repo
    repo.submit_label(_contrib_payload(repo))
    before = _contrib_snapshot(repo)
    _contrib_fault_connection(repo, monkeypatch, fail_rollback=True)
    with pytest.raises(OSError, match="rollback"):
        repo.run_qa_promotion()
    reopened = ContribRepo(repo.cfg)
    assert _contrib_snapshot(reopened) == before
    assert reopened.run_qa_promotion()["promoted"] == 1
    assert reopened.get_profile("tester")["credits"] == 1.5


def test_contrib_admin_authorization(monkeypatch):
    monkeypatch.setenv("CONTRIB_ADMIN_TOKEN", "test-admin-secret")
    endpoint = "/v1/admin/contrib/review/run"
    assert client.post(endpoint).status_code == 401
    assert client.post(endpoint, headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.post(endpoint, headers={"X-Admin-Token": "test-admin-secret"}).status_code == 200
    monkeypatch.delenv("CONTRIB_ADMIN_TOKEN")
    response = client.post(endpoint, headers={"X-Admin-Token": "test-admin-secret"})
    assert response.status_code == 503
    assert "configured" in response.json()["detail"].lower()


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_contrib_blank_identifier_rejected(contrib_repo, blank):
    assert client.get("/v1/contrib/profile", params={"contributor_id": blank}).status_code == 422
    assert client.post("/v1/contrib/profile", json={"display_name": blank}).status_code == 422
    contributor_id, headers = _new_contributor()
    assert client.patch("/v1/contrib/profile", json={"display_name": blank},
                        headers=headers).status_code == 422
    payload = _label_body(_contrib_payload(contrib_repo, contributor_id))
    assert client.post("/v1/contrib/labels", json={**payload, "task_id": blank},
                       headers=headers).status_code == 422


def test_contrib_score_required_unless_skipping(contrib_repo):
    contributor_id, headers = _new_contributor()
    payload = _label_body(_contrib_payload(contrib_repo, contributor_id))
    payload.pop("scenic_human")
    assert client.post("/v1/contrib/labels", json=payload, headers=headers).status_code == 422


def test_contrib_legacy_csv_migration_is_once_and_enforces_bindings(contrib_repo, tmp_path):
    import sqlite3
    repo = contrib_repo
    profile, token = repo.create_profile("Migrated")
    contributor_id = profile["contributor_id"]
    approved = _contrib_payload(repo, contributor_id)
    repo.submit_label(approved)
    assert repo.run_qa_promotion()["promoted"] == 1
    pending = _contrib_payload(repo, contributor_id, scenic_human=1.5)
    repo.submit_label(pending)
    issued = _contrib_payload(repo, contributor_id)
    expected_profile = repo.get_profile(contributor_id)
    exports = repo.export_csvs()
    root = tmp_path / "legacy"
    root.mkdir()
    for path in exports.values():
        (root / path.name).write_bytes(path.read_bytes())
    migrated = ContribRepo(ContribConfig(root, repo.cfg.approved_seed_csv, repo.cfg.labels_source_csv))
    assert migrated.get_profile(contributor_id) == expected_profile
    assert migrated.authenticate(token) == contributor_id
    with migrated._transaction() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT count(*) FROM annotations WHERE status='pending'").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM annotations WHERE status='approved'").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    for payload in (approved, pending):
        with pytest.raises(ValueError, match="already"):
            migrated.submit_label(payload)
    with migrated._transaction(write=True) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO issued_tasks VALUES (?, ?, ?, 'issued')",
                               ("duplicate", contributor_id, issued["image_path"]))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO annotations SELECT * FROM annotations LIMIT 1")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("""INSERT INTO annotations
                (task_id, contributor_id, image_path, region, confidence, skip, notes, status, timestamp, qa_reason)
                VALUES (?, 'foreign', ?, '', '', 1, '', 'pending', '', '')""",
                (issued["task_id"], issued["image_path"]))
    # Neither corrupting snapshots nor deleting them changes the live store.
    for path in root.glob("*.csv"):
        path.write_text("not a table", encoding="utf-8")
    reopened = ContribRepo(migrated.cfg)
    assert reopened.get_profile(contributor_id) == expected_profile
    assert reopened.authenticate(token) == contributor_id
    for path in root.glob("*.csv"):
        path.unlink()
    assert reopened.run_qa_promotion(min_agreement=0.5)["promoted"] == 1
    assert reopened.get_profile(contributor_id)["agreement_score"] == 0.75
    assert reopened.submit_label(issued)["status"] == "pending"
    assert reopened.export_csvs()["approved"].exists()


def test_contrib_interleaved_wal_readers_and_row_writes(contrib_repo, monkeypatch):
    repo = contrib_repo
    other = ContribRepo(repo.cfg)
    first, token = repo.create_profile("First")
    second, _ = other.create_profile("Second")
    a = _contrib_payload(repo, first["contributor_id"])
    b = _contrib_payload(other, second["contributor_id"])
    repo.submit_label(a)
    assert repo.run_qa_promotion()["promoted"] == 1  # Prime both external input indexes.
    next_a = _contrib_payload(repo, first["contributor_id"])
    snapshots = repo.export_csvs()
    before = {path: path.read_bytes() for path in snapshots.values()}

    def forbidden(*args, **kwargs):
        raise AssertionError("live operation reparsed or rewrote a whole CSV")

    monkeypatch.setattr(ContribRepo, "_read_csv", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    with repo._transaction() as reader:
        old_credits = reader.execute("SELECT credits FROM profiles WHERE contributor_id=?",
                                     (first["contributor_id"],)).fetchone()[0]
        other.submit_label(b)
        repo.submit_label(next_a)
        assert other.run_qa_promotion()["promoted"] == 2
        # A WAL reader remains on its original snapshot while writers commit.
        assert reader.execute("SELECT credits FROM profiles WHERE contributor_id=?",
                              (first["contributor_id"],)).fetchone()[0] == old_credits
    assert repo.authenticate(token) == first["contributor_id"]
    assert repo.get_profile(first["contributor_id"])["credits"] == 3
    assert other.get_profile(second["contributor_id"])["credits"] == 1.5
    assert {path: path.read_bytes() for path in before} == before
    with repo._transaction() as connection:
        for query, values in [
            ("SELECT * FROM profiles WHERE contributor_id=?", (first["contributor_id"],)),
            ("SELECT * FROM issued_tasks WHERE contributor_id=? AND image_path=?", (first["contributor_id"], a["image_path"])),
            ("SELECT * FROM annotations WHERE contributor_id=? AND image_path=?", (first["contributor_id"], a["image_path"])),
            ("SELECT * FROM source_images WHERE image_path=?", (a["image_path"],)),
            ("SELECT * FROM annotations WHERE status=?", ("pending",)),
        ]:
            plan = " ".join(row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + query, values))
            assert "SEARCH" in plan and "INDEX" in plan


def _interrupt_contrib_transaction(cfg):
    import os
    repo = ContribRepo(cfg)
    with repo._transaction(write=True) as connection:
        connection.execute("UPDATE profiles SET credits=999")
        connection.execute("UPDATE annotations SET status='approved'")
        os._exit(73)


def test_contrib_process_interruption_rolls_back_on_reopen(contrib_repo):
    repo = contrib_repo
    repo.submit_label(_contrib_payload(repo))
    before = _contrib_snapshot(repo)
    process = multiprocessing.get_context("spawn").Process(target=_interrupt_contrib_transaction, args=(repo.cfg,))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 73
    reopened = ContribRepo(repo.cfg)
    assert _contrib_snapshot(reopened) == before
    assert reopened.run_qa_promotion()["promoted"] == 1


def test_contrib_migration_failure_can_restart(contrib_repo):
    import sqlite3
    repo = contrib_repo
    profile, token = repo.create_profile("Migration")
    repo.export_csvs()
    profiles = pd.read_csv(repo.profiles_path)
    pd.concat([profiles, profiles]).to_csv(repo.profiles_path, index=False)
    repo.db_path.unlink()
    with pytest.raises(sqlite3.IntegrityError):
        ContribRepo(repo.cfg)
    profiles.to_csv(repo.profiles_path, index=False)
    reopened = ContribRepo(repo.cfg)
    assert reopened.get_profile(profile["contributor_id"]) == profile
    assert reopened.authenticate(token) == profile["contributor_id"]


@pytest.mark.parametrize("deadline,override,status", [(30, None, 200), (0, None, 504), (30, "0", 504)])
def test_route_compare_contract(tmp_path, monkeypatch, deadline, override, status) -> None:
    from scenicdriver.data_pipeline.web_mercator import lat_lon_to_tile
    graph = tmp_path / "graph.geojson"
    graph.write_text(json.dumps({
        "type": "FeatureCollection", "features": [{
            "type": "Feature",
            "properties": {"id": "edge", "road_type": "secondary", "one_way": True},
            "geometry": {"type": "LineString", "coordinates": [[-72, 42], [-72, 42.01]]},
        }],
    }), encoding="utf-8")
    x, y = lat_lon_to_tile(42.005, -72, 13)
    report = tmp_path / "configured-report.json"
    report.write_text(json.dumps({"tiles": [
        {"z": 13, "x": x, "y": y, "scenic_score": 6.5},
    ]}), encoding="utf-8")
    monkeypatch.setattr(app_api, "_app_region", lambda region: {
        "region": region, "route_planning": True, "run_name": "configured",
        "graph": str(graph), "tile_scores_json": str(report),
        "tile_score_zoom": 13, "route_deadline_seconds": deadline,
    })
    monkeypatch.setattr(app_api, "_region_to_graph", lambda region: graph)
    if override is None:
        monkeypatch.delenv("SCENIC_ROUTE_DEADLINE_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SCENIC_ROUTE_DEADLINE_SECONDS", override)
    response = client.post("/v1/route/compare", json={
        "start": {"lat": 42, "lon": -72}, "end": {"lat": 42.01, "lon": -72},
        "region": "enabled-smoke",
    })
    assert response.status_code == status
    if status == 200:
        payload = response.json()
        assert set(payload["routes"]) == {"scenic", "baseline"}
        assert payload["routes"]["scenic"]["raw_scenic_score"] == pytest.approx(6.5)
        assert payload["score_mapping"]["zoom"] == 13
    else:
        assert response.json()["detail"]["error"] == "routing_deadline_exceeded"


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
        "geometry": {
            "type": "LineString",
            "coordinates": [[-75.22, 40.03], [-75.19, 40.065]],
        },
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
                "type": "Feature",
                "properties": {
                    # Only map rendering and endpoint-summary metadata belongs here.
                    "route_kind": route_metrics["route_kind"],
                    "snapped_start": route_metrics["snapped_start"],
                    "snapped_end": route_metrics["snapped_end"],
                },
                "geometry": route_metrics["geometry"],
            }
            for route_metrics in metrics
        ],
    }


def _strict_diagnostics() -> dict[str, object]:
    return {
        "planner": "ok",
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
        "highway_avoidance_fallback": False,
        "highway_avoidance_fallback_reason": None,
        "highway_avoidance_mode": "strict",
        "highway_preference": 0.0,
        "highway_avoidance_cost": 0.0,
        "hard_highway_count": 0,
        "detour_reference_duration_minutes": 16.0,
        "duration_cap_minutes": 28.8,
        "duration_cap_satisfied": True,
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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": True,
            "include_baseline": True,
        },
    )

    assert response.status_code == 200
    assert graph_calls == ["new_england_north"]
    assert run_calls == ["new_england_north"]
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


def test_route_compare_serialization_cache_skips_validation_and_ipc(tmp_path, monkeypatch) -> None:
    graph = tmp_path / "graph.geojson"
    graph.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_api, "_app_region", lambda region: None)
    monkeypatch.setattr(app_api, "_region_to_graph", lambda region: graph)
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")
    metrics = _strict_route_metrics(
        distance_km=10.0, duration_min=20.0, raw_score=0.8, edge_id="scenic-edge",
    )
    calls = {"plan": 0, "validate": 0, "project": 0}
    original_validate = app_api._validate_route_contract
    original_project = app_api._project_route_response

    def plan(request, **kwargs):
        calls["plan"] += 1
        diagnostics = _strict_diagnostics()
        diagnostics.update({
            "route_response_cache_hit": False, "graph_cache_hit": True,
            "tile_score_cache_hit": True, "scored_graph_cache_hit": True,
            "graph_load_elapsed_ms": 19.0, "tile_score_load_elapsed_ms": 13.0,
            "score_application_elapsed_ms": 7.0, "planning_elapsed_ms": 12345.0,
        })
        return {
            "routes": [{"route_kind": "scenic", "metrics": metrics}],
            "diagnostics": diagnostics, "score_mapping": _strict_score_mapping(),
            "geojson": _strict_geojson(metrics),
        }

    def validate(*args, **kwargs):
        calls["validate"] += 1
        return original_validate(*args, **kwargs)

    def project(*args, **kwargs):
        calls["project"] += 1
        return original_project(*args, **kwargs)

    monkeypatch.setattr(app_api, "plan_routes", plan)
    monkeypatch.setattr(app_api, "_validate_route_contract", validate)
    monkeypatch.setattr(app_api, "_project_route_response", project)
    body = {
        "start": {"lat": 40.03, "lon": -75.22},
        "end": {"lat": 40.065, "lon": -75.19},
        "region": "cache-region", "max_detour_factor": 1.8,
        "avoid_highways": False, "include_baseline": False,
    }
    first = client.post("/v1/route/compare", json=body)
    second = client.post("/v1/route/compare", json=body)
    assert first.status_code == second.status_code == 200
    assert calls == {"plan": 1, "validate": 1, "project": 1}
    first_payload, second_payload = first.json(), second.json()
    assert first_payload["diagnostics"]["route_response_cache_hit"] is False
    diagnostics = second_payload.pop("diagnostics")
    first_payload.pop("diagnostics")
    assert second_payload == first_payload
    assert diagnostics["route_response_cache_hit"] is True
    assert diagnostics["planning_elapsed_ms"] != 12345.0
    assert diagnostics["graph_load_elapsed_ms"] == 0.0
    assert diagnostics["tile_score_load_elapsed_ms"] == 0.0
    assert diagnostics["score_application_elapsed_ms"] == 0.0
    assert not any(diagnostics[key] for key in ("graph_cache_hit", "tile_score_cache_hit", "scored_graph_cache_hit"))
    assert set(second_payload["geojson"]["features"][0]["properties"]) == {
        "route_kind", "snapped_start", "snapped_end",
    }
    # Each identity-changing input invalidates the public serialization, not
    # only the planner's internal response cache.
    body["avoid_highways"] = True
    assert client.post("/v1/route/compare", json=body).status_code == 200
    graph.write_text('{"changed": true}', encoding="utf-8")
    assert client.post("/v1/route/compare", json=body).status_code == 200
    report = app_api._run_report_path("test-run")
    report.write_text('{"tiles": [], "summary": {"changed": true}}', encoding="utf-8")
    assert client.post("/v1/route/compare", json=body).status_code == 200
    monkeypatch.setenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", "0.25")
    assert client.post("/v1/route/compare", json=body).status_code == 200
    assert calls == {"plan": 5, "validate": 5, "project": 5}


def test_route_projection_redacts_shared_containers_once_without_mutation() -> None:
    metrics = _strict_route_metrics(
        distance_km=10.0, duration_min=20.0, raw_score=0.8, edge_id="scenic-edge",
    )
    shared = {"source": "/private/report.json", "nested": [{"score": 0.8}]}
    metrics["score_provenance"] = shared
    mapping = _strict_score_mapping()
    mapping["provenance"] = shared
    result = {
        "routes": [{"route_kind": "scenic", "metrics": metrics}],
        "diagnostics": _strict_diagnostics(), "score_mapping": mapping,
        "geojson": _strict_geojson(metrics),
    }
    payload = app_api.RouteCompareRequest(
        start={"lat": 40.03, "lon": -75.22}, end={"lat": 40.065, "lon": -75.19},
        region="test", include_baseline=False,
    )
    public = app_api._project_route_response(
        result, payload=payload, run_name="test-run", request=_strict_geometry_request(),
        deadline=RoutingDeadline.after(30),
    )
    route = public["routes"]["scenic"]
    feature = public["geojson"]["features"][0]
    assert route["geometry"] is feature["geometry"]
    assert route["snapped_start"] is feature["properties"]["snapped_start"]
    assert route["score_provenance"] is public["score_mapping"]["provenance"]
    assert route["score_provenance"] == {"nested": [{"score": 0.8}]}
    assert route["score_provenance"] is not shared
    assert shared["source"] == "/private/report.json"


def _strict_geometry_request(*, include_baseline: bool = False) -> app_api.RouteRequest:
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
    metrics["geometry"] = {
        "type": "Point",
        "coordinates": [-75.22, 40.03],
    }
    geojson = _strict_geojson(metrics)

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
    mismatched["features"][0]["geometry"] = {
        "type": "LineString", "coordinates": [[-75.22, 40.03], [-75.18, 40.065]],
    }
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
    metrics["requested_start"] = [40.04, -75.22]

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
                "properties": {key: properties[key] for key in ("route_kind", "snapped_start", "snapped_end")},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-75.21, 40.04], [-75.21, 40.04]],
                },
            }
        ],
    }
    properties["geometry"] = geojson["features"][0]["geometry"]

    assert (
        app_api._validate_route_geometry(
            geojson,
            request=request,
            routes={"scenic": properties},
        )["scenic"]
        == geojson["features"][0]
    )


def test_route_compare_rejects_without_relaxing_detour_cap(monkeypatch) -> None:
    plan_calls: list[object] = []
    diagnosis_calls: list[object] = []
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        plan_calls.append(request)
        # The planner's real duration-cap failure reason string (see
        # src/scenicdriver/route_planner/planner.py, matched verbatim by service.py).
        raise ValueError("No route satisfies the requested duration cap.")

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
    assert detail["reason"] == "No route satisfies the requested duration cap."
    assert detail["message"] == (
        "No route satisfies the requested controls, including best-effort "
        "highway avoidance."
    )
    assert detail["hint"] == (
        "Choose different points or increase max detour; highway use is "
        "already allowed when required."
    )


def test_route_compare_omits_reason_for_unrecognized_planner_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        app_api, "_region_to_graph", lambda region: Path("/tmp/fake-road-graph.geojson")
    )
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")

    def fake_plan(request, **kwargs):
        raise ValueError("internal asset problem")

    def fake_diagnose(request, **kwargs):
        return {"graph_nodes": 4, "graph_edges": 3}

    monkeypatch.setattr(app_api, "plan_routes", fake_plan)
    monkeypatch.setattr(app_api, "diagnose_route_request", fake_diagnose)

    response = TestClient(create_app()).post(
        "/v1/route/compare",
        json={
            "start": {"lat": 44.4, "lon": -70.2},
            "end": {"lat": 44.5, "lon": -70.1},
            "region": "new_england_north",
            "max_detour_factor": 1.4,
            "avoid_highways": False,
            "include_baseline": True,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "no_route_found"
    # An unrecognized ValueError is not a structured cause: no `reason`, so
    # clients fall back to `message` and internal failure text never leaks.
    assert "reason" not in detail
    assert detail["message"] == "No route satisfies the requested controls."


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
        "reason": "The start point is too far from the supported road network.",
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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 502
    assert response.json()["detail"]["missing_metrics"] == ["score_run"]


def test_route_compare_request_has_no_unapplied_weight() -> None:
    request = app_api.RouteCompareRequest(
        start={"lat": 40.03, "lon": -75.22},
        end={"lat": 40.065, "lon": -75.19},
    )
    assert not any("weight" in key for key in request.model_dump())
    assert not any("weight" in key for key in request.model_json_schema()["properties"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 502
    assert "certified gap" in response.json()["detail"]["message"]


def test_route_compare_rejects_per_route_detour_mismatch(monkeypatch) -> None:
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
    scenic_metrics["applied_max_detour_factor"] = 1.7
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
            "region": "new_england_north",
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
            "region": "new_england_north",
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
        "avoid_highways",
        "max_detour_factor",
        "include_baseline",
    }
    assert "source" not in payload["score_mapping"]
    assert "source" not in payload["routes"]["scenic"]["score_provenance"]
    assert set(payload["geojson"]["features"][0]["properties"]) == {
        "route_kind", "snapped_start", "snapped_end",
    }


def test_invalid_frontier_configuration_fails_startup_even_when_preload_off(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "off")
    monkeypatch.setenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", "nan")

    with pytest.raises(
        app_api.RouteConfigurationError,
        match="SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS",
    ):
        with TestClient(create_app()):
            pass




def test_route_preload_lifespan_calls_once_and_skips_missing_optional(
    tmp_path, monkeypatch
) -> None:
    default_graph = tmp_path / "default.graph.json"
    default_report = tmp_path / "default.report.json"
    default_graph.write_text("valid", encoding="utf-8")
    default_report.write_text("valid", encoding="utf-8")
    _write_graph_readiness(default_graph)
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
        preload_calls.append((graph_path, tile_path, zoom, fallback, exclusive_scoring))
        return {
            "graph_cache_hit": False,
            "tile_score_cache_hit": False,
            "scored_graph_cache_hit": False,
        }

    monkeypatch.setattr(app_api, "preload_route_assets", fake_preload)
    monkeypatch.setattr(
        app_api.PreloadedRouteSupervisor,
        "start",
        lambda **_kwargs: _InlineTestSupervisor(),
    )
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
    _write_graph_readiness(graph_path)
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


@pytest.mark.parametrize("mode", ["off", "best_effort"])
def test_route_supervisor_starts_without_preloaded_assets(tmp_path, monkeypatch, mode) -> None:
    from scenicdriver.data_pipeline.web_mercator import lat_lon_to_tile

    graph, report = tmp_path / "missing.geojson", tmp_path / "scores.json"
    monkeypatch.setattr(app_api, "validate_route_configuration", lambda: None)
    monkeypatch.setattr(app_api, "_route_preload_mode", lambda: mode)
    monkeypatch.setattr(app_api, "_configured_regions", lambda: [{
        "region": "cold", "graph_path": graph, "run_name": "missing",
        "tile_scores_path": report, "tile_score_zoom": 13,
    }])
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "cold")
    app = create_app()
    with TestClient(app) as test_client:
        assert app.state.route_supervisor is not None
        assert app.state.route_supervisor.health()["alive"] is True
        health = test_client.get("/v1/healthz")
        assert health.status_code == 503
        assert health.json()["routing"]["assets_ready"] is False
        assert health.json()["routing"]["execution_mode"] == "hard_stop_supervisor"
        # Publish assets after the supervisor fork: its first planning child
        # must load them lazily, without any parent preload cache.
        graph.write_text(json.dumps({
            "type": "FeatureCollection", "features": [{
                "type": "Feature",
                "properties": {"id": "edge", "road_type": "secondary", "one_way": True},
                "geometry": {"type": "LineString", "coordinates": [[-72, 42], [-72, 42.01]]},
            }],
        }), encoding="utf-8")
        x, y = lat_lon_to_tile(42.005, -72, 13)
        report.write_text(json.dumps({"tiles": [
            {"z": 13, "x": x, "y": y, "scenic_score": 6.5},
        ]}), encoding="utf-8")
        _write_graph_readiness(graph)
        route = test_client.post("/v1/route/compare", json={
            "start": {"lat": 42, "lon": -72},
            "end": {"lat": 42.01, "lon": -72}, "region": "cold",
        })
        assert route.status_code == 200
        assert route.json()["routes"]["scenic"]["raw_scenic_score"] == pytest.approx(6.5)
        assert test_client.get("/v1/healthz").json()["routing"]["ready"] is True


def test_route_preload_required_fails_for_missing_default(
    tmp_path, monkeypatch
) -> None:
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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 504
    assert response.json()["detail"] == {
        "error": "routing_deadline_exceeded",
        "message": "The route planning deadline was exceeded.",
        "reason": "deadline exceeded",
    }


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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code != 422
    assert response.json()["detail"] == {
        "error": "routing_cancelled",
        "message": "The route planning request was cancelled.",
        "reason": "cancelled",
    }


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


def test_route_compare_fallback_diagnose_maps_routing_timeout_to_504(
    monkeypatch,
) -> None:
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
            "region": "new_england_north",
            "max_detour_factor": 1.8,
            "avoid_highways": False,
            "include_baseline": False,
        },
    )
    assert response.status_code == 504
    assert response.json()["detail"]["error"] == "routing_deadline_exceeded"


def test_route_compare_fallback_diagnose_maps_routing_cancelled_to_non_no_route(
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "test-run")


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
                    "route_kind": "scenic",
                    "geometry": {"type": "LineString", "coordinates": [[-70.2, 44.4], [-70.1, 44.5]]},
                    "snapped_start": [44.4, -70.2],
                    "snapped_end": [44.5, -70.1],
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


@pytest.mark.parametrize("params", [
    {"min_overlap": 0}, {"min_overlap": -1}, {"min_agreement": -0.1},
    {"min_agreement": 1.1}, {"min_agreement": "nan"}, {"min_agreement": "inf"},
])
def test_review_rejects_invalid_thresholds(monkeypatch, params):
    monkeypatch.setenv("CONTRIB_ADMIN_TOKEN", "secret")
    response = client.post("/v1/admin/contrib/review/run", params=params,
                           headers={"X-Admin-Token": "secret"})
    assert response.status_code == 422


def test_review_counts_seed_annotations_before_averaging(contrib_repo):
    payload = _contrib_payload(contrib_repo)
    pd.DataFrame([
        {"image_path": payload["image_path"], "scenic_human": 6},
        {"image_path": payload["image_path"], "scenic_human": 7},
    ]).to_csv(contrib_repo.cfg.approved_seed_csv, index=False)
    contrib_repo.submit_label(payload)
    assert contrib_repo.run_qa_promotion(min_overlap=3)["remaining_pending"] == 1
    assert contrib_repo.run_qa_promotion(min_overlap=2)["promoted"] == 1
    assert contrib_repo.get_profile("tester")["pending_labels"] == 0


def test_contrib_region_does_not_fall_back_or_trust_label_region(contrib_repo):
    contributor_id, headers = _new_contributor()
    response = client.get("/v1/contrib/tasks/next", params={"region": "unsupported"},
                          headers=headers)
    assert response.status_code == 200
    assert response.json()["tasks"] == []
    payload = _label_body(_contrib_payload(contrib_repo, contributor_id, region="forged"))
    response = client.post("/v1/contrib/labels", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["record"]["region"] == "philadelphia"


@pytest.mark.parametrize("path,params", [
    ("/v1/heatmap", {"region": "test", "max_points": 0}),
    ("/v1/heatmap", {"region": "test", "max_tiles": 100001}),
    ("/v1/heatmap", {"region": "test", "tile_offset": -1}),
    ("/v1/contrib/tasks/next", {"count": 0}),
    ("/v1/contrib/leaderboard", {"limit": -1}),
])
def test_api_rejects_invalid_batch_sizes(path, params):
    _, headers = _new_contributor()
    assert client.get(path, params=params, headers=headers).status_code == 422


@pytest.mark.parametrize("artifact", [
    [], {"tiles": [None], "summary": {"min": 0, "max": 10}},
    {"tiles": [], "summary": {"min": 0, "max": 10}},
    {"tiles": [{"x": 1, "y": 1, "z": 14, "scenic_score": float("nan")}],
     "summary": {"min": 0, "max": 10}},
    {"tiles": [{"x": 1, "y": 1, "z": 14, "scenic_score": 5}],
     "summary": {"min": 0, "max": float("inf")}},
    {"tiles": [{"x": -1, "y": 1, "z": 14, "scenic_score": 5}],
     "summary": {"min": 0, "max": 10}},
])
def test_heatmap_artifact_validation_is_shared(tmp_path, monkeypatch, artifact):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(app_api, "_run_report_path", lambda run: report)
    for endpoint in ("heatmap", "heatmap-image", "heatmap-scores.bin"):
        response = client.get(f"/v1/{endpoint}", params={"region": "test", "run_name": "test"})
        assert response.status_code == 502
        assert "message" in response.json()["detail"]


@pytest.mark.parametrize("failure,status", [
    (app_api.requests.Timeout("timeout"), 504),
    (app_api.requests.ConnectionError("offline"), 502),
    (ValueError("invalid JSON"), 502),
])
def test_search_provider_failures_are_controlled(monkeypatch, failure, status):
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "secret")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(app_api.requests, "get", fail)
    for endpoint, params in [
        ("suggest", {"q": "main", "session_token": "test"}),
        ("retrieve", {"mapbox_id": "id", "session_token": "test"}),
    ]:
        assert client.get(f"/v1/search/{endpoint}", params=params).status_code == status


@pytest.mark.parametrize("field,size", [
    ("task_id", 129), ("image_path", 2049), ("notes", 4097), ("confidence", 33),
])
def test_contrib_rejects_oversized_text(contrib_repo, field, size):
    contributor_id, headers = _new_contributor()
    payload = _label_body(_contrib_payload(contrib_repo, contributor_id))
    assert client.post("/v1/contrib/labels", headers=headers, json={
        **payload, field: "x" * size,
    }).status_code == 422


def test_contrib_rejects_oversized_display_name():
    assert client.post("/v1/contrib/profile", json={"display_name": "x" * 129}).status_code == 422


def test_api_limits_request_body():
    assert client.post("/v1/contrib/session/start",
                       content=b"x" * 65537).status_code == 413


@pytest.mark.parametrize("artifact", [
    {"features": [{"geometry": None, "properties": {}}]},
    {"features": [{"geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": []}]},
    {"features": "invalid"}, [],
])
def test_search_rejects_malformed_nested_result(monkeypatch, artifact):
    class Response:
        ok = True
        def json(self):
            return artifact
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "secret")
    monkeypatch.setattr(app_api.requests, "get", lambda *args, **kwargs: Response())
    assert client.get("/v1/search/retrieve", params={
        "mapbox_id": "id", "session_token": "test",
    }).status_code == 502


def test_validated_route_run_selection_and_complete_contract(tmp_path, monkeypatch):
    run = "requested"
    report = tmp_path / run / "report"
    report.mkdir(parents=True)
    routes = {
        kind: _strict_route_metrics(distance_km=10, duration_min=16,
                                    raw_score=6, edge_id=kind, route_kind=kind)
        for kind in ("scenic", "baseline")
    }
    geojson = _strict_geojson(*routes.values())
    metrics = {
        **routes, "score_mapping": {}, "request": {
            "start": [40.03, -75.22], "end": [40.065, -75.19],
            "max_detour_factor": 1.8,
        },
    }
    route_file = report / "route.geojson"
    metrics_file = report / "route_metrics.json"
    route_file.write_text(json.dumps(geojson), encoding="utf-8")
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(app_api, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app_api, "_latest_run_for_region", lambda region: "other")
    params = {"region": "test", "run_name": run}
    response = client.get("/v1/validated-route", params=params)
    assert response.status_code == 200
    assert response.json()["run_name"] == run
    malformed = [
        {**geojson, "features": geojson["features"] + [geojson["features"][0]]},
        {**geojson, "features": [
            {**geojson["features"][0], "geometry": {"type": "LineString", "coordinates": [[0, 0], [float("nan"), 1]]}},
            geojson["features"][1],
        ]},
        {**geojson, "features": [
            {**geojson["features"][0], "geometry": {"type": "Point", "coordinates": [0, 0]}},
            geojson["features"][1],
        ]},
    ]
    for artifact in malformed:
        route_file.write_text(json.dumps(artifact), encoding="utf-8")
        assert client.get("/v1/validated-route", params=params).status_code == 502
    route_file.write_text(json.dumps(geojson), encoding="utf-8")
    del metrics["scenic"]["estimated_duration_minutes"]
    metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
    assert client.get("/v1/validated-route", params=params).status_code == 502


@pytest.mark.parametrize("error", [json.JSONDecodeError("bad artifact", "x", 0), ValueError("invalid asset")])
def test_route_asset_errors_are_not_no_route(monkeypatch, error):
    _stub_route_compare_assets(monkeypatch)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(app_api, "plan_routes", fail)
    monkeypatch.setattr(app_api, "diagnose_route_request", fail)
    response = client.post("/v1/route/compare", json=_route_compare_payload())
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "route_assets_invalid"


def test_http_disconnect_cancels_supervised_route(monkeypatch):
    import asyncio
    from scenicdriver.route_planner.supervisor import PreloadedRouteSupervisor
    _stub_route_compare_assets(monkeypatch)
    def wait_for_cancel(request, *, deadline):
        while True:
            deadline.check()
            time.sleep(0.01)
    monkeypatch.setattr(app_api, "plan_routes", wait_for_cancel)
    app = create_app()
    supervisor = PreloadedRouteSupervisor.start(default_deadline_seconds=5, default_grace_seconds=0.05)
    app.state.route_supervisor = supervisor
    messages = []
    async def exercise():
        delivered = False
        disconnect_at = time.monotonic() + 0.1
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": json.dumps(_route_compare_payload()).encode(), "more_body": False}
            if time.monotonic() < disconnect_at:
                await asyncio.sleep(1)
            return {"type": "http.disconnect"}
        async def send(message):
            messages.append(message)
        await asyncio.wait_for(app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/v1/route/compare",
            "raw_path": b"/v1/route/compare", "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1), "server": ("test", 80),
        }, receive, send), timeout=5)
    try:
        asyncio.run(exercise())
    finally:
        supervisor.close()
    assert messages[0]["status"] == 499
    assert json.loads(messages[1]["body"])["detail"]["error"] == "routing_cancelled"


@pytest.mark.parametrize("path,source_region,expected", [
    ("images/satellite/philadelphia/z14/0.jpg", None, "philadelphia"),
    ("images/satellite/z14/philadelphia.jpg", None, "unknown"),
    ("images/satellite/z14/0.jpg", "philadelphia", "philadelphia"),
])
def test_contrib_region_comes_from_image_provenance(contrib_repo, path, source_region, expected):
    image = {"image_path": path}
    if source_region:
        image["region"] = source_region
    pd.DataFrame([image]).to_csv(contrib_repo.cfg.labels_source_csv, index=False)
    contributor_id, headers = _new_contributor()
    task = contrib_repo.next_tasks(contributor_id, "philadelphia", 1)[0]
    response = client.post("/v1/contrib/labels", headers=headers, json={
        "task_id": task["task_id"], "image_path": task["image_path"],
        "scenic_human": 6, "region": "forged",
    })
    assert response.status_code == 200
    assert response.json()["record"]["region"] == expected


def test_search_is_rate_limited_per_client_after_burst(monkeypatch):
    monkeypatch.setenv("SCENIC_SEARCH_BURST", "2")
    monkeypatch.setenv("SCENIC_SEARCH_RATE_PER_MIN", "1")
    test_client = TestClient(create_app())
    # Blank queries never reach the provider but still spend the client's budget.
    for _ in range(2):
        response = test_client.get("/v1/search/suggest", params={"q": " ", "session_token": "s"})
        assert response.status_code == 200
    for endpoint, params in [
        ("suggest", {"q": " ", "session_token": "s"}),
        ("retrieve", {"mapbox_id": "id", "session_token": "s"}),
    ]:
        response = test_client.get(f"/v1/search/{endpoint}", params=params)
        assert response.status_code == 429
        assert response.json()["detail"]["error"] == "search_rate_limited"
        assert 1 <= int(response.headers["retry-after"]) <= 60


def test_search_rate_limiter_refills_and_tracks_clients_separately(monkeypatch):
    limiter = app_api._ClientRateLimiter(rate_per_minute=60, burst=1)
    now = [100.0]
    monkeypatch.setattr(app_api.time, "monotonic", lambda: now[0])
    assert limiter.acquire("a") == 0
    assert limiter.acquire("a") == pytest.approx(1.0)
    assert limiter.acquire("b") == 0
    now[0] += 1.0
    assert limiter.acquire("a") == 0


def test_route_compare_rejects_beyond_queue_cap_with_503(monkeypatch):
    monkeypatch.setenv("SCENIC_ROUTE_MAX_PENDING", "1")
    _stub_route_compare_assets(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    class BlockingSupervisor:
        def run_job(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=10)
            raise RoutingTimeout("deadline exceeded")

    app = create_app()
    app.state.route_supervisor = BlockingSupervisor()
    with ThreadPoolExecutor(max_workers=1) as workers:
        first = workers.submit(TestClient(app).post, "/v1/route/compare",
                               json=_route_compare_payload())
        assert entered.wait(timeout=10)
        rejected = TestClient(app).post("/v1/route/compare", json=_route_compare_payload())
        release.set()
        assert first.result(timeout=10).status_code == 504
    assert rejected.status_code == 503
    assert rejected.json()["detail"] == {
        "error": "routing_capacity_exceeded",
        "message": "The routing service is busy; retry shortly.",
    }
    # The finished request released its slot.
    entered.clear()
    assert TestClient(app).post("/v1/route/compare", json=_route_compare_payload()).status_code == 504


def test_cors_allows_only_configured_origins(monkeypatch):
    monkeypatch.setenv("SCENIC_ALLOWED_ORIGINS", "https://scenic.example, https://beta.example")
    test_client = TestClient(create_app())
    allowed = test_client.get("/", headers={"Origin": "https://beta.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://beta.example"
    exposed = {name.strip() for name in allowed.headers["access-control-expose-headers"].split(",")}
    assert exposed == {"X-Scenic-Tile-Zoom", "X-Scenic-Min-X", "X-Scenic-Min-Y",
                       "X-Scenic-Grid-Width", "X-Scenic-Grid-Height"}
    denied = test_client.get("/", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers
    preflight = test_client.options("/v1/contrib/labels", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in preflight.headers


def test_cors_defaults_to_local_viewer_origins(monkeypatch):
    monkeypatch.delenv("SCENIC_ALLOWED_ORIGINS", raising=False)
    test_client = TestClient(create_app())
    for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        response = test_client.get("/", headers={"Origin": origin})
        assert response.headers["access-control-allow-origin"] == origin
    response = test_client.get("/", headers={"Origin": "https://scenic.example"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("region,run_name", [
    ("new_england_north", "other-run"),
    ("new_england_north", "philadelphia_z16_learned_v5"),
    ("unconfigured_region", "test-run"),
])
def test_route_compare_rejects_unreleased_score_runs(monkeypatch, region, run_name):
    _stub_route_compare_assets(monkeypatch)
    monkeypatch.setattr(app_api, "plan_routes",
                        lambda *args, **kwargs: pytest.fail("planning must not start"))
    response = client.post("/v1/route/compare",
                           json={**_route_compare_payload(), "region": region, "run_name": run_name})
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "score_run_not_released"


def test_route_compare_accepts_the_released_score_run(monkeypatch):
    _stub_route_compare_assets(monkeypatch)
    monkeypatch.setattr(app_api, "_app_region",
                        lambda region: {"region": region, "run_name": "test-run"})

    def timeout(*args, **kwargs):
        raise RoutingTimeout("deadline exceeded")

    monkeypatch.setattr(app_api, "plan_routes", timeout)
    response = client.post("/v1/route/compare",
                           json={**_route_compare_payload(), "run_name": "test-run"})
    assert response.status_code == 504


def test_route_admission_hashes_off_event_loop(tmp_path, monkeypatch):
    import asyncio

    graph = tmp_path / "graph.json"
    graph.write_text("graph", encoding="utf-8")
    _write_graph_readiness(graph)
    configured = {"region": "test", "graph_path": graph, "run_name": "test-run"}
    loop_threads, hash_threads = [], []
    original_hash = app_api.hashlib.file_digest

    def region_config(region):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            loop_threads.append(threading.get_ident())
        return configured

    def hasher(*args, **kwargs):
        hash_threads.append(threading.get_ident())
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(app_api, "_app_region", region_config)
    monkeypatch.setattr(app_api.hashlib, "file_digest", hasher)
    monkeypatch.setattr(app_api, "plan_routes", lambda *args, **kwargs: (_ for _ in ()).throw(RoutingTimeout("planned")))
    response = client.post("/v1/route/compare", json={**_route_compare_payload(), "region": "test"})
    assert response.status_code == 504
    assert hash_threads and loop_threads
    assert set(hash_threads).isdisjoint(loop_threads)


def test_route_validation_and_projection_run_off_loop(monkeypatch):
    import asyncio

    _stub_route_compare_assets(monkeypatch)
    calls = []

    def validate(result, *, request, deadline):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        calls.append(("validate", deadline))
        return {"scenic_score_delta_absolute": 0.0}, {
            "scenic": {
                "route_kind": "scenic",
                "geometry": {"type": "LineString", "coordinates": [[-70.2, 44.4], [-70.1, 44.5]]},
                "snapped_start": [44.4, -70.2], "snapped_end": [44.5, -70.1],
            },
        }

    original_redact = app_api._redact_private_response

    def redact(value):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        calls.append(("redact", None))
        return original_redact(value)

    monkeypatch.setattr(app_api, "_validate_route_contract", validate)
    monkeypatch.setattr(app_api, "_redact_private_response", redact)
    monkeypatch.setattr(app_api, "plan_routes", lambda *args, **kwargs: {"score_mapping": {}, "geojson": {}})
    response = client.post("/v1/route/compare", json=_route_compare_payload())
    assert response.status_code == 200
    assert calls[0][0] == "validate"
    assert len(calls) == 2
    assert calls[1] == ("redact", None)


def test_route_deadline_includes_artifact_admission(monkeypatch):
    _stub_route_compare_assets(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def resolve(region):
        entered.set()
        assert release.wait(timeout=10)
        return Path("/tmp/unused.json")

    monkeypatch.setattr(app_api, "_region_to_graph", resolve)
    monkeypatch.setattr(app_api, "_route_deadline_seconds", lambda configured: 0.05)
    monkeypatch.setattr(app_api, "plan_routes", lambda *args, **kwargs: pytest.fail("expired admission must not plan"))
    try:
        response = client.post("/v1/route/compare", json=_route_compare_payload())
        assert entered.is_set()
        assert not release.is_set()
        assert response.status_code == 504
    finally:
        release.set()


def test_heatmap_png_and_extents_cached_by_full_file_identity(tmp_path, monkeypatch):
    import os

    report = tmp_path / "report.json"
    payload = {
        "tiles": [
            {"x": 1, "y": 1, "z": 4, "scenic_score": 2.0},
            {"x": 2, "y": 2, "z": 4, "scenic_score": 4.0},
        ],
        "summary": {"min": 0.0, "max": 5.0},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(app_api, "_run_report_path", lambda run: report)
    calls = []
    original_new = Image.new

    def new_image(*args, **kwargs):
        calls.append(args)
        return original_new(*args, **kwargs)

    monkeypatch.setattr(Image, "new", new_image)
    params = {"region": "test", "run_name": "cache-test"}
    first = client.get("/v1/heatmap-image", params=params)
    second = client.get("/v1/heatmap-image", params=params)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.headers == second.headers
    assert len(calls) == 1

    class NoFullScan(list):
        def __iter__(self):
            raise AssertionError("limited JSON page must reuse cached extents")

    _, _, parsed = app_api._heatmap_artifact("test", "cache-test")
    parsed["tiles"] = NoFullScan(parsed["tiles"])
    page = client.get("/v1/heatmap", params={**params, "max_points": 1, "max_tiles": 1, "tile_offset": 0})
    assert page.status_code == 200
    assert page.json()["total_tiles"] == 2

    old_stat = report.stat()
    payload["tiles"][1]["x"] = 3
    report.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(report, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    changed = client.get("/v1/heatmap-image", params=params)
    assert changed.status_code == 200
    assert changed.content != first.content
    assert len(calls) == 2
    assert Image.open(io.BytesIO(changed.content)).size == (3, 2)


def test_health_reports_dead_supervisor_despite_successful_preload(tmp_path, monkeypatch):
    graph, report = tmp_path / "graph.json", tmp_path / "report.json"
    graph.write_text("graph", encoding="utf-8")
    report.write_text("report", encoding="utf-8")
    _write_graph_readiness(graph)
    monkeypatch.setattr(app_api, "_configured_regions", lambda: [{
        "region": "test", "graph_path": graph, "run_name": "run",
        "tile_scores_path": report,
    }])
    monkeypatch.setattr(app_api, "_default_region_key", lambda: "test")

    class Supervisor(_InlineTestSupervisor):
        alive = True

        def health(self):
            return {"alive": self.alive, "closed": False}

    supervisor = Supervisor()
    app = create_app()
    app.state.route_supervisor = supervisor
    app.state.route_preload_diagnostics = {"loaded_regions": ["test"], "regions": []}
    test_client = TestClient(app)
    ready = test_client.get("/v1/healthz")
    assert ready.status_code == 200
    assert ready.json()["routing"]["ready"] is True
    supervisor.alive = False
    dead = test_client.get("/v1/healthz")
    assert dead.status_code == 503
    assert dead.json()["ok"] is True
    assert dead.json()["liveness"] == {"ok": True}
    assert dead.json()["routing"] == {
        "ready": False, "assets_ready": True, "supervisor_alive": False,
        "execution_mode": "unavailable", "ready_regions": ["test"],
    }


def test_routing_fails_closed_without_supervisor(monkeypatch):
    _stub_route_compare_assets(monkeypatch)
    app = create_app()
    app.state.route_supervisor = None
    monkeypatch.setattr(app_api, "plan_routes", lambda *args, **kwargs: pytest.fail("parent must not plan"))
    response = TestClient(app).post("/v1/route/compare", json=_route_compare_payload())
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "routing_supervisor_unavailable"
