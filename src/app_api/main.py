from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from urllib.parse import quote

from src.route_planner.service import RouteRequest, diagnose_route_request, plan_routes

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
APP_REGIONS_PATH = PROJECT_ROOT / "config/app_regions.json"
POINT_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$")


def _load_app_region_config() -> dict[str, Any]:
    if not APP_REGIONS_PATH.exists():
        return {"default_region": None, "regions": []}
    return json.loads(APP_REGIONS_PATH.read_text(encoding="utf-8"))


def _configured_regions() -> list[dict[str, Any]]:
    payload = _load_app_region_config()
    rows = payload.get("regions", [])
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("region"):
            continue
        item = dict(row)
        graph = item.get("graph")
        if graph:
            item["graph_path"] = PROJECT_ROOT / str(graph)
        out.append(item)
    return out


def _app_region(region: str) -> dict[str, Any] | None:
    key = region.strip().lower()
    for item in _configured_regions():
        if str(item.get("region", "")).strip().lower() == key:
            return item
    return None


def _default_region_key() -> str | None:
    default_region = _load_app_region_config().get("default_region")
    return str(default_region).strip().lower() if default_region else None


def _region_to_graph(region: str) -> Path:
    configured = _app_region(region)
    if configured and configured.get("graph_path"):
        graph_path = Path(configured["graph_path"])
        if graph_path.exists():
            return graph_path
        raise FileNotFoundError(
            f"Configured road graph for region '{region}' is missing: {graph_path}"
        )

    key = region.strip().lower()
    candidates = [
        ROAD_GRAPHS_DIR / f"{key}_core/road_graph.geojson",
        ROAD_GRAPHS_DIR / f"{key}_core/road_graph.json",
        ROAD_GRAPHS_DIR / key / "road_graph.geojson",
        ROAD_GRAPHS_DIR / key / "road_graph.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No road graph found for region '{region}'")


def _latest_run_for_region(region: str) -> str | None:
    configured = _app_region(region)
    if configured and configured.get("run_name"):
        return str(configured["run_name"])

    key = region.strip().lower()
    aliases: dict[str, list[str]] = {
        # Pittsfield runs were historically named with "masswhites".
        "pittsfield": ["pittsfield", "masswhites"],
    }
    terms = aliases.get(key, [key])
    # Common derived graph names (e.g., *_wide, *_core) should still resolve
    # to the underlying run namespace.
    if key.endswith("_wide"):
        terms.append(key[: -len("_wide")])
    if key.endswith("_core"):
        terms.append(key[: -len("_core")])
    parts = key.split("_")
    if parts:
        terms.append(parts[0])
    # Deduplicate terms while preserving order.
    dedup_terms: list[str] = []
    seen_terms: set[str] = set()
    for t in terms:
        if not t:
            continue
        if t in seen_terms:
            continue
        seen_terms.add(t)
        dedup_terms.append(t)
    terms = dedup_terms
    if not RUNS_DIR.exists():
        return None
    matches = [
        d
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and any(term in d.name.lower() for term in terms)
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].name


def _list_regions() -> list[dict[str, Any]]:
    def _compute_geojson_bbox(path: Path) -> dict[str, float] | None:
        try:
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        feats = payload.get("features", [])
        min_lat = 90.0
        min_lon = 180.0
        max_lat = -90.0
        max_lon = -180.0
        seen = 0
        for feat in feats:
            geom = feat.get("geometry") or {}
            if geom.get("type") != "LineString":
                continue
            for coord in geom.get("coordinates") or []:
                if not isinstance(coord, list) or len(coord) < 2:
                    continue
                lon = float(coord[0])
                lat = float(coord[1])
                min_lat = min(min_lat, lat)
                min_lon = min(min_lon, lon)
                max_lat = max(max_lat, lat)
                max_lon = max(max_lon, lon)
                seen += 1
        if seen == 0:
            return None
        return {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon,
        }

    regions: list[dict[str, Any]] = []
    configured_keys: set[str] = set()
    configured_graphs: set[Path] = set()
    default_region = _default_region_key()
    for item in _configured_regions():
        region = str(item.get("region"))
        configured_keys.add(region.lower())
        graph = Path(item["graph_path"]) if item.get("graph_path") else None
        if graph:
            configured_graphs.add(graph.resolve())
        latest_run = str(item["run_name"]) if item.get("run_name") else _latest_run_for_region(region)
        regions.append(
            {
                "region": region,
                "display_name": item.get("display_name", region),
                "description": item.get("description"),
                "graph_geojson": str(graph) if graph else None,
                "graph_exists": bool(graph and graph.exists()),
                "latest_run_name": latest_run,
                "report_json": str(RUNS_DIR / latest_run / "report/report.json")
                if latest_run
                else None,
                "bbox": item.get("bbox"),
                "map": item.get("map"),
                "model_checkpoint": item.get("model_checkpoint"),
                "is_default": region.lower() == default_region,
                "source": "config",
            }
        )

    if not ROAD_GRAPHS_DIR.exists():
        return regions
    for d in sorted([x for x in ROAD_GRAPHS_DIR.iterdir() if x.is_dir()]):
        graph_geojson = d / "road_graph.geojson"
        graph_json = d / "road_graph.json"
        graph = graph_geojson if graph_geojson.exists() else graph_json
        if not graph.exists():
            continue
        region = d.name.replace("_core", "")
        if region.lower() in configured_keys:
            continue
        if graph.resolve() in configured_graphs:
            continue
        latest_run = _latest_run_for_region(region)
        bbox = None
        run_json = d / "run.json"
        if run_json.exists():
            try:
                payload = __import__("json").loads(run_json.read_text(encoding="utf-8"))
                bbox = payload.get("bbox")
            except Exception:
                bbox = None
        if bbox is None:
            bbox = _compute_geojson_bbox(graph)
        regions.append(
            {
                "region": region,
                "display_name": region,
                "graph_geojson": str(graph),
                "graph_exists": True,
                "latest_run_name": latest_run,
                "report_json": str(RUNS_DIR / latest_run / "report/report.json")
                if latest_run
                else None,
                "bbox": bbox,
                "is_default": region.lower() == default_region,
                "source": "discovered",
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


def _expand_geocode_query_variants(query: str, region: str | None) -> list[str]:
    base = query.strip()
    if not base:
        return []
    variants: list[str] = [base]
    q = base.lower()

    # Common local shorthand expansions.
    rewrites = [
        ("south philly", "south philadelphia"),
        ("philly", "philadelphia"),
        (" center city ", " center city philadelphia "),
    ]
    for src, dst in rewrites:
        if src in q:
            variants.append(re.sub(re.escape(src), dst, base, flags=re.IGNORECASE))

    if region:
        variants.append(f"{base}, {region}")
        variants.append(f"{base} in {region}")
    if "usa" not in q and "united states" not in q:
        variants.append(f"{base}, USA")

    # Add a POI hint form for brand-like queries.
    if any(
        tok in q
        for tok in ["target", "walmart", "costco", "airport", "station", "mall"]
    ):
        variants.append(f"{base} near {region}" if region else f"{base} near me")
        for tok in _intent_tokens(base):
            variants.append(tok)
            if region:
                variants.append(f"{tok} {region}")
                variants.append(f"{tok} near {region}")

    deduped: list[str] = []
    seen: set[str] = set()
    for v in variants:
        k = v.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(v)
    return deduped


def _tokenize_query(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _score_candidate(query: str, label: str) -> float:
    q_tokens = _tokenize_query(query)
    if not q_tokens:
        return 0.0
    label_l = label.lower()
    matched = sum(1 for tok in q_tokens if tok in label_l)
    score = matched / len(q_tokens)
    if "target" in q_tokens and "target" in label_l:
        score += 0.5
    if "airport" in q_tokens and "airport" in label_l:
        score += 0.4
    if "philadelphia" in q_tokens and "philadelphia" in label_l:
        score += 0.25
    return score


def _intent_tokens(query: str) -> list[str]:
    q = query.lower()
    tokens: list[str] = []
    for tok in ["target", "walmart", "costco", "airport", "station", "mall"]:
        if tok in q:
            tokens.append(tok)
    return tokens


def _filter_by_intent_tokens(
    query: str, items: list[dict[str, Any]], *, strict: bool
) -> list[dict[str, Any]]:
    intents = _intent_tokens(query)
    if not intents:
        return items
    filtered = []
    for row in items:
        label = str(row.get("label", "")).lower()
        if all(tok in label for tok in intents):
            filtered.append(row)
    if filtered:
        return filtered
    return [] if strict else items


def _in_bbox(lat: float, lon: float, bbox: dict[str, Any] | None) -> bool:
    if not bbox:
        return True
    return float(bbox["min_lat"]) <= float(lat) <= float(bbox["max_lat"]) and float(
        bbox["min_lon"]
    ) <= float(lon) <= float(bbox["max_lon"])


def _normalize_bbox(bbox: Any) -> dict[str, float] | None:
    if not isinstance(bbox, dict):
        return None
    required = ["min_lat", "min_lon", "max_lat", "max_lon"]
    if not all(k in bbox for k in required):
        return None
    try:
        return {
            "min_lat": float(bbox["min_lat"]),
            "min_lon": float(bbox["min_lon"]),
            "max_lat": float(bbox["max_lat"]),
            "max_lon": float(bbox["max_lon"]),
        }
    except (TypeError, ValueError):
        return None


def _tile_to_center_latlon(x: int, y: int, z: int) -> tuple[float, float]:
    n = 2.0**z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = __import__("math").atan(
        __import__("math").sinh(__import__("math").pi * (1.0 - 2.0 * ((y + 0.5) / n)))
    )
    lat = __import__("math").degrees(lat_rad)
    return lat, lon


def _tile_to_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    n = 2.0**z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n_rad = __import__("math").atan(
        __import__("math").sinh(__import__("math").pi * (1.0 - 2.0 * (y / n)))
    )
    lat_s_rad = __import__("math").atan(
        __import__("math").sinh(__import__("math").pi * (1.0 - 2.0 * ((y + 1) / n)))
    )
    lat_n = __import__("math").degrees(lat_n_rad)
    lat_s = __import__("math").degrees(lat_s_rad)
    return lat_s, lon_w, lat_n, lon_e


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
            "default_region": _default_region_key(),
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
                    region_bbox = _normalize_bbox(item.get("bbox"))
                    break

        mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if not mapbox_token:
            raise HTTPException(
                status_code=500, detail="MAPBOX_ACCESS_TOKEN is not configured"
            )

        url = (
            "https://api.mapbox.com/geocoding/v5/mapbox.places/"
            + quote(query)
            + ".json"
        )
        params: dict[str, Any] = {
            "access_token": mapbox_token,
            "limit": 5,
            "autocomplete": "true",
            "country": "us",
            "language": "en",
            "types": "poi,address,place,locality,neighborhood",
        }
        if region_bbox:
            params["bbox"] = (
                f"{region_bbox['min_lon']},{region_bbox['min_lat']},"
                f"{region_bbox['max_lon']},{region_bbox['max_lat']}"
            )

        resp = requests.get(url, params=params, timeout=15)
        if not resp.ok:
            return {"provider": "mapbox", "results": []}
        payload = resp.json()
        features = payload.get("features", [])
        if not features and region_bbox:
            params.pop("bbox", None)
            resp = requests.get(url, params=params, timeout=15)
            if resp.ok:
                payload = resp.json()
                features = payload.get("features", [])
        results = []
        for feat in features:
            center = feat.get("center") or [None, None]
            lon, lat = center[0], center[1]
            if lat is None or lon is None:
                continue
            results.append(
                {
                    "label": feat.get("place_name", query),
                    "match_type": "mapbox_geocoding",
                    **_canonical_point(float(lat), float(lon)),
                }
            )
        return {"provider": "mapbox", "results": results}

    @app.post("/v1/route/compare")
    def route_compare(payload: RouteCompareRequest) -> dict[str, Any]:
        try:
            graph_path = _region_to_graph(payload.region)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        run_name = payload.run_name or _latest_run_for_region(payload.region)
        if not run_name:
            raise HTTPException(
                status_code=404,
                detail=f"No run report available for region '{payload.region}'",
            )
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
            tile_score_fallback=1.0,
        )
        diagnostics: dict[str, Any] = {}
        try:
            diagnostics = diagnose_route_request(req)
        except Exception:
            diagnostics = {}
        try:
            result = plan_routes(req)
            diagnostics = dict(result.get("diagnostics", {}))
        except ValueError as exc:
            # Most common case: points geocode outside the selected graph/connected component
            # or max detour cap is too tight. Retry with a higher detour cap once.
            retry_cap = max(float(payload.max_detour_factor), 2.2)
            if retry_cap > float(payload.max_detour_factor):
                retry_req = RouteRequest(
                    graph_geojson=str(graph_path),
                    start=(payload.start.lat, payload.start.lon),
                    end=(payload.end.lat, payload.end.lon),
                    scenic_weight=float(payload.scenic_weight),
                    avoid_highways=bool(payload.avoid_highways),
                    max_detour_factor=retry_cap,
                    include_baseline=bool(payload.include_baseline),
                    tile_scores_json=str(report_json) if report_json.exists() else None,
                    tile_score_fallback=1.0,
                )
                try:
                    result = plan_routes(retry_req)
                    diagnostics = dict(result.get("diagnostics", {}))
                    result["retry_used"] = True
                    result["retry_max_detour_factor"] = retry_cap
                except ValueError:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "no_route_found",
                            "message": str(exc),
                            "hint": f"Try different points in region '{payload.region}' or increase max detour.",
                            "diagnostics": diagnostics,
                        },
                    ) from exc
            else:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "no_route_found",
                        "message": str(exc),
                        "hint": f"Try different points in region '{payload.region}' or increase max detour.",
                        "diagnostics": diagnostics,
                    },
                ) from exc
        routes = {r["route_kind"]: r["metrics"] for r in result.get("routes", [])}
        scenic = routes.get("scenic")
        baseline = routes.get("baseline")
        deltas = None
        if scenic and baseline:
            deltas = {
                "distance_km": float(scenic["total_distance_km"])
                - float(baseline["total_distance_km"]),
                "duration_min": float(scenic["estimated_duration_minutes"])
                - float(baseline["estimated_duration_minutes"]),
                "scenic_score": float(scenic["average_scenic_score"])
                - float(baseline["average_scenic_score"]),
            }
        return {
            "request": req.to_dict(),
            "run_name": run_name,
            "diagnostics": diagnostics,
            "routes": routes,
            "deltas": deltas,
            "score_mapping": result.get("score_mapping", {}),
            "geojson": result.get("geojson", {}),
        }

    @app.get("/v1/heatmap")
    def heatmap(
        region: str,
        run_name: str | None = None,
        max_points: int = 3000,
        max_tiles: int = 12000,
    ) -> dict[str, Any]:
        selected_run = run_name or _latest_run_for_region(region)
        if not selected_run:
            raise HTTPException(
                status_code=404, detail=f"No run report available for region '{region}'"
            )
        report_json = RUNS_DIR / selected_run / "report/report.json"
        if not report_json.exists():
            raise HTTPException(
                status_code=404, detail=f"Report not found for run '{selected_run}'"
            )
        payload = __import__("json").loads(report_json.read_text(encoding="utf-8"))
        tiles = payload.get("tiles", [])
        z_counts: dict[int, int] = {}
        for t in tiles:
            z_v = t.get("z")
            if z_v is None:
                continue
            z_i = int(z_v)
            z_counts[z_i] = z_counts.get(z_i, 0) + 1
        tile_zoom = max(z_counts.items(), key=lambda kv: kv[1])[0] if z_counts else None
        xs = [int(t["x"]) for t in tiles if t.get("x") is not None]
        ys = [int(t["y"]) for t in tiles if t.get("y") is not None]
        min_x = min(xs) if xs else None
        max_x = max(xs) if xs else None
        min_y = min(ys) if ys else None
        max_y = max(ys) if ys else None

        # Build robust normalization stats from interior tiles (ignore border ring).
        interior_scores: list[float] = []
        for tile in tiles:
            x = tile.get("x")
            y = tile.get("y")
            scenic = tile.get("scenic_score")
            if x is None or y is None or scenic is None:
                continue
            x_i = int(x)
            y_i = int(y)
            if (
                min_x is not None
                and max_x is not None
                and min_y is not None
                and max_y is not None
                and (x_i == min_x or x_i == max_x or y_i == min_y or y_i == max_y)
            ):
                continue
            interior_scores.append(float(scenic))

        # Absolute scale for map coloring: scenic score is defined on [0, 10].
        norm_min = 0.0
        norm_max = 10.0

        feats = []
        tile_feats = []
        point_limit = max(1, int(max_points))
        if len(tiles) <= point_limit:
            point_tiles = tiles
        else:
            # Deterministic spread sampling across the full run extent (avoid
            # taking the first N tiles, which can cluster spatially).
            step = len(tiles) / float(point_limit)
            point_tiles = [tiles[int(i * step)] for i in range(point_limit)]

        tile_limit = max(1, int(max_tiles))
        if len(tiles) <= tile_limit:
            polygon_tiles = tiles
        else:
            step = len(tiles) / float(tile_limit)
            polygon_tiles = [tiles[int(i * step)] for i in range(tile_limit)]

        for tile in point_tiles:
            x = tile.get("x")
            y = tile.get("y")
            z = tile.get("z")
            scenic = tile.get("scenic_score")
            if x is None or y is None or z is None or scenic is None:
                continue
            lat, lon = _tile_to_center_latlon(int(x), int(y), int(z))
            scenic_f = float(scenic)
            score_norm = (scenic_f - norm_min) / (norm_max - norm_min)
            score_norm = max(0.0, min(1.0, score_norm))
            lat_s, lon_w, lat_n, lon_e = _tile_to_bounds(int(x), int(y), int(z))
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"score": scenic_f, "score_norm": score_norm},
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                }
            )

        for tile in polygon_tiles:
            x = tile.get("x")
            y = tile.get("y")
            z = tile.get("z")
            scenic = tile.get("scenic_score")
            if x is None or y is None or z is None or scenic is None:
                continue
            scenic_f = float(scenic)
            score_norm = (scenic_f - norm_min) / (norm_max - norm_min)
            score_norm = max(0.0, min(1.0, score_norm))
            lat_s, lon_w, lat_n, lon_e = _tile_to_bounds(int(x), int(y), int(z))
            tile_feats.append(
                {
                    "type": "Feature",
                    "properties": {"score": scenic_f, "score_norm": score_norm},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [lon_w, lat_s],
                                [lon_e, lat_s],
                                [lon_e, lat_n],
                                [lon_w, lat_n],
                                [lon_w, lat_s],
                            ]
                        ],
                    },
                }
            )
        return {
            "region": region,
            "run_name": selected_run,
            "tile_zoom": tile_zoom,
            "normalization": {
                "min": norm_min,
                "max": norm_max,
                "source": "absolute_0_10",
            },
            "geojson": {"type": "FeatureCollection", "features": feats},
            "geojson_tiles": {"type": "FeatureCollection", "features": tile_feats},
        }

    @app.post("/v1/contrib/session/start")
    def contrib_session_start(
        payload: ContributorSessionStartRequest,
    ) -> dict[str, Any]:
        profile = repo.upsert_profile(payload.contributor_id, payload.display_name)
        tasks = repo.next_tasks(payload.contributor_id, payload.region, count=25)
        return {"profile": profile, "tasks": tasks, "region": payload.region}

    @app.get("/v1/contrib/tasks/next")
    def contrib_next_tasks(
        contributor_id: str, region: str = "pittsfield", count: int = 25
    ) -> dict[str, Any]:
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
    def contrib_review_run(
        min_overlap: int = 1, min_agreement: float = 0.65
    ) -> dict[str, Any]:
        return repo.run_qa_promotion(
            min_overlap=min_overlap, min_agreement=min_agreement
        )

    return app


app = create_app()
