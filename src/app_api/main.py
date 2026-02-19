from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.route_planner.service import RouteRequest, plan_routes

from .contrib_repo import ContribRepo
from .schemas import (
    ContributorLabelRequest,
    ContributorSessionStartRequest,
    RouteCompareRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROAD_GRAPHS_DIR = PROJECT_ROOT / "data/processed/road_graphs"
RUNS_DIR = PROJECT_ROOT / "data/processed/heuristic_runs"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "data/processed/regression/model_registry.json"


def _region_to_graph(region: str) -> Path:
    key = region.strip().lower()
    candidates = [
        ROAD_GRAPHS_DIR / f"{key}_core/road_graph.geojson",
        ROAD_GRAPHS_DIR / key / "road_graph.geojson",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No road graph found for region '{region}'")


def _latest_run_for_region(region: str) -> str | None:
    key = region.strip().lower()
    if not RUNS_DIR.exists():
        return None
    matches = [d for d in RUNS_DIR.iterdir() if d.is_dir() and key in d.name.lower()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].name


def _list_regions() -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if not ROAD_GRAPHS_DIR.exists():
        return regions
    for d in sorted([x for x in ROAD_GRAPHS_DIR.iterdir() if x.is_dir()]):
        graph = d / "road_graph.geojson"
        if not graph.exists():
            continue
        region = d.name.replace("_core", "")
        latest_run = _latest_run_for_region(region)
        regions.append(
            {
                "region": region,
                "graph_geojson": str(graph),
                "latest_run_name": latest_run,
                "report_json": str(RUNS_DIR / latest_run / "report/report.json") if latest_run else None,
            }
        )
    return regions


def create_app() -> FastAPI:
    app = FastAPI(title="ScenicDrive API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    repo = ContribRepo()

    @app.get("/v1/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "model_registry_exists": MODEL_REGISTRY_PATH.exists(),
            "regions_available": len(_list_regions()),
        }

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "ScenicDrive API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/v1/healthz",
            "regions": "/v1/regions",
        }

    @app.get("/v1/regions")
    def regions() -> dict[str, Any]:
        return {"regions": _list_regions()}

    @app.post("/v1/route/compare")
    def route_compare(payload: RouteCompareRequest) -> dict[str, Any]:
        try:
            graph_path = _region_to_graph(payload.region)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        run_name = payload.run_name or _latest_run_for_region(payload.region)
        if not run_name:
            raise HTTPException(status_code=404, detail=f"No run report available for region '{payload.region}'")
        report_json = RUNS_DIR / run_name / "report/report.json"

        req = RouteRequest(
            graph_geojson=str(graph_path),
            start=(payload.start.lat, payload.start.lon),
            end=(payload.end.lat, payload.end.lon),
            scenic_weight=float(payload.scenic_weight),
            avoid_highways=bool(payload.avoid_highways),
            max_detour_factor=float(payload.max_detour_factor),
            include_baseline=bool(payload.include_baseline),
            tile_scores_json=str(report_json) if report_json.exists() else None,
        )
        result = plan_routes(req)
        routes = {r["route_kind"]: r["metrics"] for r in result.get("routes", [])}
        scenic = routes.get("scenic")
        baseline = routes.get("baseline")
        deltas = None
        if scenic and baseline:
            deltas = {
                "distance_km": float(scenic["total_distance_km"]) - float(baseline["total_distance_km"]),
                "duration_min": float(scenic["estimated_duration_minutes"]) - float(baseline["estimated_duration_minutes"]),
                "scenic_score": float(scenic["average_scenic_score"]) - float(baseline["average_scenic_score"]),
            }
        return {
            "request": req.to_dict(),
            "run_name": run_name,
            "routes": routes,
            "deltas": deltas,
            "score_mapping": result.get("score_mapping", {}),
            "geojson": result.get("geojson", {}),
        }

    @app.post("/v1/contrib/session/start")
    def contrib_session_start(payload: ContributorSessionStartRequest) -> dict[str, Any]:
        profile = repo.upsert_profile(payload.contributor_id, payload.display_name)
        tasks = repo.next_tasks(payload.contributor_id, payload.region, count=25)
        return {"profile": profile, "tasks": tasks, "region": payload.region}

    @app.get("/v1/contrib/tasks/next")
    def contrib_next_tasks(contributor_id: str, region: str = "pittsfield", count: int = 25) -> dict[str, Any]:
        profile = repo.upsert_profile(contributor_id)
        tasks = repo.next_tasks(contributor_id, region, count=count)
        return {"profile": profile, "tasks": tasks, "region": region}

    @app.post("/v1/contrib/labels")
    def contrib_submit_label(payload: ContributorLabelRequest) -> dict[str, Any]:
        repo.upsert_profile(payload.contributor_id)
        rec = repo.submit_label(payload.model_dump())
        profile = repo.get_profile(payload.contributor_id)
        return {"saved": True, "record": rec, "profile": profile, "credits_delta": 0.0}

    @app.get("/v1/contrib/profile")
    def contrib_profile(contributor_id: str) -> dict[str, Any]:
        return {"profile": repo.get_profile(contributor_id)}

    @app.get("/v1/contrib/leaderboard")
    def contrib_leaderboard(limit: int = 25) -> dict[str, Any]:
        return {"leaderboard": repo.leaderboard(limit=limit)}

    @app.post("/v1/admin/contrib/review/run")
    def contrib_review_run(min_overlap: int = 1, min_agreement: float = 0.65) -> dict[str, Any]:
        return repo.run_qa_promotion(min_overlap=min_overlap, min_agreement=min_agreement)

    return app


app = create_app()
