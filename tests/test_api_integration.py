"""HTTP integration lane with admitted compact assets and real process isolation.

Only filesystem timing is injected in the race/concurrency tests; admission,
loaders, planner, supervisor IPC, response projection and lifespan stay real.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from scenicdriver.app_api.contrib_repo import ContribConfig, ContribRepo
from scenicdriver.data_pipeline.web_mercator import lat_lon_to_tile
from scenicdriver.route_planner import service as route_service
from scenicdriver.route_planner.graph import (
    CompactRoadGraph,
    Edge,
    Node,
    RoadGraph,
    write_compact_graph,
)
from scenicdriver.route_planner.supervisor import PreloadedRouteSupervisor

pytestmark = pytest.mark.skipif(os.name != "posix", reason="real supervisor requires POSIX fork")


def _worker_cache_snapshot(deadline):
    """Inspect the persistent planning child, not the parent's empty caches."""
    deadline.check()
    return {
        "pid": os.getpid(),
        "graphs": len(route_service._GRAPH_CACHE),
        "scored_graphs": len(route_service._SCORED_GRAPH_CACHE),
        "responses": len(route_service._ROUTE_RESPONSE_CACHE),
    }


@dataclass
class _Runtime:
    api: object
    manifest: Path
    root: Path
    payload: dict

    @contextmanager
    def open(self):
        app = self.api.create_app()
        with TestClient(app) as client:
            supervisor = app.state.route_supervisor
            assert isinstance(supervisor, PreloadedRouteSupervisor)
            assert supervisor.health()["alive"] is True
            yield client, supervisor
        assert supervisor.health()["alive"] is False
        assert app.state.route_supervisor is None


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    repo = ContribRepo(ContribConfig(root_dir=tmp_path / "contrib"))
    # Protect the exported app's import-time repository just as the API unit
    # tests do, then restore the module's by-value constructor binding.
    with patch("scenicdriver.app_api.contrib_repo.ContribRepo", return_value=repo):
        api = importlib.import_module("scenicdriver.app_api.main")
    api.ContribRepo = ContribRepo
    monkeypatch.setattr(api, "ContribRepo", lambda: repo)
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api, "APP_REGIONS_PATH", tmp_path / "app_regions.json")
    monkeypatch.setattr(api, "ROAD_GRAPHS_DIR", tmp_path / "graphs")
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("SCENIC_ROUTE_PRELOAD", "off")
    monkeypatch.setenv("SCENIC_ROUTE_MAX_PENDING", "1")
    monkeypatch.setenv("SCENIC_ROUTE_DEADLINE_SECONDS", "10")
    monkeypatch.delenv("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS", raising=False)

    graph_dir = tmp_path / "graphs" / "integration"
    graph_dir.mkdir(parents=True)
    graph = RoadGraph()
    graph.add_node(Node("start", 42.0, -72.0))
    graph.add_node(Node("end", 42.001, -72.0))
    # Exactly 15 minutes. A hidden 1 km/h speed floor would halve it.
    graph.add_edge(Edge(
        "slow", "start", "end", distance_km=0.125,
        scenic_score=5.0, road_type="residential", speed_limit_kmh=0.5,
    ))
    sqlite = graph_dir / "road_graph.sqlite3"
    manifest = graph_dir / "road_graph.compact.json"
    graph.save(sqlite)
    publication = write_compact_graph(sqlite, manifest)
    (graph_dir / "run.json").write_text(json.dumps({
        "output_graph": str(sqlite.relative_to(tmp_path)),
        "graph_sha256": publication["source_sha256"],
        "graph_size_bytes": publication["source_size_bytes"],
        "runtime_graph": str(manifest.relative_to(tmp_path)),
        "compact_manifest_sha256": publication["manifest_sha256"],
    }), encoding="utf-8")
    report = tmp_path / "runs" / "integration-run" / "report" / "report.json"
    report.parent.mkdir(parents=True)
    x, y = lat_lon_to_tile(42.0005, -72.0, 14)
    report.write_text(json.dumps({
        "tiles": [{"z": 14, "x": x, "y": y, "scenic_score": 5.0}],
        "summary": {"min": 0.0, "max": 10.0},
    }), encoding="utf-8")
    # Provision the content-addressed pair without warming service caches.
    with CompactRoadGraph.load(manifest) as compact:
        assert compact.activate_report_scores(
            {(14, x, y): 5.0}, zoom=14, fallback=None,
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            normalization=route_service._NORMALIZATION_VERSION,
        ) == (1, 1)
        sidecar_manifest, sidecar_payload = compact._score_sidecar_paths(
            hashlib.sha256(report.read_bytes()).hexdigest(), 14, None,
            route_service._NORMALIZATION_VERSION,
        )
        assert sidecar_manifest.is_file() and sidecar_payload.is_file()
    (tmp_path / "app_regions.json").write_text(json.dumps({
        "default_region": "integration",
        "regions": [{
            "region": "integration", "graph": str(manifest.relative_to(tmp_path)),
            "run_name": "integration-run", "tile_score_zoom": 14,
            "route_planning": True, "max_route_snap_km": 1,
        }],
    }), encoding="utf-8")
    route_service.clear_route_caches()
    api._parse_heatmap_artifact.cache_clear()
    api._load_heatmap_png.cache_clear()
    try:
        yield _Runtime(api, manifest, tmp_path, {
            "region": "integration",
            "start": {"lat": 42.0, "lon": -72.0},
            "end": {"lat": 42.001, "lon": -72.0},
            "include_baseline": True, "max_detour_factor": 1.8,
            "avoid_highways": False,
        })
    finally:
        route_service.clear_route_caches()
        api._parse_heatmap_artifact.cache_clear()
        api._load_heatmap_png.cache_clear()
        api._readiness_file_digest.cache_clear()


def test_cold_lifespan_starts_supervisor_and_first_http_request_loads_assets(runtime):
    with runtime.open() as (client, supervisor):
        assert client.app.state.route_preload_diagnostics["mode"] == "off"
        assert client.app.state.route_preload_diagnostics["loaded_regions"] == []
        cold = supervisor.run_job(_worker_cache_snapshot)
        assert cold["graphs"] == cold["scored_graphs"] == cold["responses"] == 0
        assert client.get("/v1/healthz").status_code == 200
        first = client.post("/v1/route/compare", json=runtime.payload)
        assert first.status_code == 200, first.text
        assert set(first.json()["routes"]) == {"scenic", "baseline"}
        assert first.json()["score_mapping"]["matched_edges"] == 1
        warm = supervisor.run_job(_worker_cache_snapshot)
        assert warm["pid"] == cold["pid"] != os.getpid()
        assert warm["graphs"] == warm["scored_graphs"] == warm["responses"] == 1
        assert not route_service._GRAPH_CACHE  # planning never ran in the API process
        assert client.post("/v1/route/compare", json=runtime.payload).status_code == 200


def test_atomic_swap_during_real_load_returns_asset_error_without_stale_cache(runtime, monkeypatch):
    original_load = route_service._load_graph
    enabled = runtime.root / "swap-enabled"
    swaps = runtime.root / "swap-count"
    enabled.touch()

    def swap_after_load(path, **kwargs):
        graph = original_load(path, **kwargs)
        if enabled.exists():
            # Keep valid bytes/publication hashes, change the inode after each
            # actual load so both bounded attempts detect a new generation.
            replacement = path.with_name("replacement.json")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            with swaps.open("a", encoding="utf-8") as stream:
                stream.write("swapped\n")
        return graph

    monkeypatch.setattr(route_service, "_load_graph", swap_after_load)
    with runtime.open() as (client, supervisor):
        response = client.post("/v1/route/compare", json=runtime.payload)
        assert response.status_code == 500, response.text
        assert response.json()["detail"]["error"] == "route_assets_invalid"
        assert swaps.read_text(encoding="utf-8").splitlines() == ["swapped", "swapped"]
        rejected = supervisor.run_job(_worker_cache_snapshot)
        assert rejected["graphs"] == rejected["scored_graphs"] == rejected["responses"] == 0
        enabled.unlink()
        recovered = client.post("/v1/route/compare", json=runtime.payload)
        assert recovered.status_code == 200, recovered.text
        admitted = supervisor.run_job(_worker_cache_snapshot)
        assert admitted["pid"] == rejected["pid"]
        assert admitted["graphs"] == admitted["scored_graphs"] == admitted["responses"] == 1


def test_dead_supervisor_preserves_liveness_but_fails_routing_readiness(runtime):
    with runtime.open() as (client, supervisor):
        assert client.get("/v1/healthz").status_code == 200
        # Kill the real process, not a stand-in health() implementation.
        supervisor._process.terminate()
        supervisor._process.join(timeout=5)
        assert supervisor.health()["alive"] is False
        response = client.get("/v1/healthz")
        assert response.status_code == 503
        payload = response.json()
        assert payload["ok"] is True
        assert payload["liveness"] == {"ok": True}
        assert payload["routing"]["assets_ready"] is True
        assert payload["routing"]["ready"] is False
        assert payload["routing"]["supervisor_alive"] is False
        unavailable = client.post("/v1/route/compare", json=runtime.payload)
        assert unavailable.status_code == 503
        assert unavailable.json()["detail"]["error"] == "routing_supervisor_unavailable"


def test_concurrent_http_requests_exceed_route_admission_capacity(runtime, monkeypatch):
    original_load = route_service._load_graph
    entered, release = runtime.root / "load-entered", runtime.root / "load-release"

    def held_load(path, **kwargs):
        entered.touch()
        stop_at = time.monotonic() + 5
        while not release.exists():
            if kwargs.get("check_cancelled"):
                kwargs["check_cancelled"]()
            if time.monotonic() >= stop_at:
                raise TimeoutError("test did not release the real asset loader")
            time.sleep(0.01)
        return original_load(path, **kwargs)

    monkeypatch.setattr(route_service, "_load_graph", held_load)
    with runtime.open() as (client, _supervisor), ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/v1/route/compare", json=runtime.payload)
        try:
            stop_at = time.monotonic() + 5
            while not entered.exists() and time.monotonic() < stop_at:
                time.sleep(0.01)
            assert entered.exists(), "first HTTP request never entered worker asset loading"
            saturated = client.post("/v1/route/compare", json=runtime.payload)
            assert saturated.status_code == 503, saturated.text
            assert saturated.json()["detail"]["error"] == "routing_capacity_exceeded"
            assert not first.done()
        finally:
            release.touch()
        completed = first.result(timeout=10)
        assert completed.status_code == 200, completed.text
        assert client.post("/v1/route/compare", json=runtime.payload).status_code == 200


def test_sub_one_kmh_duration_is_exact_through_http(runtime):
    with runtime.open() as (client, _supervisor):
        response = client.post("/v1/route/compare", json=runtime.payload)
        assert response.status_code == 200, response.text
        for route in response.json()["routes"].values():
            assert route["total_distance_km"] == 0.125
            assert route["estimated_duration_minutes"] == 15.0


def test_identical_heatmap_http_requests_reuse_rendered_png(runtime, monkeypatch):
    renders = []
    original_new = Image.new

    def counted_new(*args, **kwargs):
        renders.append(args)
        return original_new(*args, **kwargs)

    monkeypatch.setattr(Image, "new", counted_new)
    with runtime.open() as (client, _supervisor):
        params = {"region": "integration"}
        first = client.get("/v1/heatmap-image", params=params)
        second = client.get("/v1/heatmap-image", params=params)
        assert first.status_code == second.status_code == 200
        assert first.headers["content-type"] == "image/png"
        assert first.content == second.content
        assert len(renders) == 1
        assert runtime.api._load_heatmap_png.cache_info().hits == 1
        with Image.open(io.BytesIO(second.content)) as image:
            assert image.size == (1, 1)
