from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

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
POINT_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$")


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
        bbox = None
        run_json = d / "run.json"
        if run_json.exists():
            try:
                payload = __import__("json").loads(run_json.read_text(encoding="utf-8"))
                bbox = payload.get("bbox")
            except Exception:
                bbox = None
        regions.append(
            {
                "region": region,
                "graph_geojson": str(graph),
                "latest_run_name": latest_run,
                "report_json": str(RUNS_DIR / latest_run / "report/report.json") if latest_run else None,
                "bbox": bbox,
            }
        )
    return regions


def _canonical_point(lat: float, lon: float) -> dict[str, Any]:
    lat_f = float(lat)
    lon_f = float(lon)
    return {
        "lat": lat_f,
        "lon": lon_f,
        "latlon": f"{lat_f:.6f},{lon_f:.6f}",
        "wkt": f"POINT({lon_f:.6f} {lat_f:.6f})",
    }


def _parse_coordinate_query(query: str) -> dict[str, Any] | None:
    m = POINT_RE.match(query)
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    pt = _canonical_point(lat, lon)
    return {
        "label": pt["latlon"],
        "match_type": "parsed_point",
        **pt,
    }


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

    @app.get("/v1/geocode")
    def geocode(q: str, region: str | None = None) -> dict[str, Any]:
        query = q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must be non-empty")

        parsed = _parse_coordinate_query(query)
        if parsed is not None:
            return {"provider": "parser", "results": [parsed]}

        region_bbox = None
        if region:
            region_key = region.strip().lower()
            for item in _list_regions():
                if str(item.get("region", "")).lower() == region_key:
                    region_bbox = item.get("bbox")
                    break

        mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if mapbox_token:
            url = "https://api.mapbox.com/geocoding/v5/mapbox.places/" + requests.utils.quote(query) + ".json"
            params = {"access_token": mapbox_token, "limit": 5}
            if region_bbox:
                params["bbox"] = (
                    f"{region_bbox['min_lon']},{region_bbox['min_lat']},"
                    f"{region_bbox['max_lon']},{region_bbox['max_lat']}"
                )
            resp = requests.get(
                url,
                params=params,
                timeout=15,
            )
            if resp.ok:
                payload = resp.json()
                features = payload.get("features", [])
                items = []
                for feat in features:
                    center = feat.get("center") or [None, None]
                    lon, lat = center[0], center[1]
                    if lat is None or lon is None:
                        continue
                    items.append(
                        {
                            "label": feat.get("place_name"),
                            "match_type": "mapbox",
                            **_canonical_point(float(lat), float(lon)),
                        }
                    )
                return {"provider": "mapbox", "results": items}

        # Fallback provider: OpenStreetMap Nominatim
        nom_params: dict[str, Any] = {"q": query, "format": "json", "limit": 5}
        if region_bbox:
            nom_params["viewbox"] = (
                f"{region_bbox['min_lon']},{region_bbox['max_lat']},"
                f"{region_bbox['max_lon']},{region_bbox['min_lat']}"
            )
            nom_params["bounded"] = 1
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=nom_params,
            headers={"User-Agent": "scenicdrive-api/0.1.0"},
            timeout=15,
        )
        if not resp.ok:
            raise HTTPException(status_code=502, detail=f"Geocoding request failed: {resp.status_code}")
        rows = resp.json()
        results = []
        for row in rows:
            if "lat" not in row or "lon" not in row:
                continue
            results.append(
                {
                    "label": row.get("display_name", query),
                    "match_type": "nominatim",
                    **_canonical_point(float(row["lat"]), float(row["lon"])),
                }
            )
        return {"provider": "nominatim", "results": results}

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
        try:
            result = plan_routes(req)
        except ValueError as exc:
            # Most common case: points geocode outside the selected graph/connected component.
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "no_route_found",
                    "message": str(exc),
                    "hint": f"Try different points in region '{payload.region}' or use /v1/geocode with region bias.",
                },
            ) from exc
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
