from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NoReturn
from array import array
from functools import lru_cache
import io
import json
import logging
import math
import re
import sys
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from scalar_fastapi import get_scalar_api_reference
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from urllib.parse import quote

from src.route_planner.service import (
    RouteConfigurationError,
    RouteRequest,
    diagnose_route_request,
    plan_routes,
    preload_route_assets,
    validate_route_configuration,
)

from .contrib_repo import ContribRepo
from .schemas import (
    ContributorLabelRequest,
    ContributorSessionStartRequest,
    RouteCompareRequest,
)

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROAD_GRAPHS_DIR = PROJECT_ROOT / "data/processed/road_graphs"
RUNS_DIR = PROJECT_ROOT / "data/processed/heuristic_runs"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "data/processed/regression/model_registry.json"
APP_REGIONS_PATH = PROJECT_ROOT / "config/app_regions.json"
_LOGGER = logging.getLogger(__name__)

_SAFE_REGION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SAFE_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _safe_asset_name(value: str, *, kind: str) -> str:
    pattern = _SAFE_REGION_RE if kind == "region" else _SAFE_RUN_RE
    if not pattern.fullmatch(value):
        raise HTTPException(status_code=422, detail=f"Invalid {kind}")
    return value

def _run_report_path(run_name: str) -> Path:
    safe_run = _safe_asset_name(run_name, kind="run_name")
    return RUNS_DIR / safe_run / "report/report.json"


def _load_app_region_config() -> dict[str, Any]:
    if not APP_REGIONS_PATH.exists():
        return {"default_region": None, "regions": []}
    return json.loads(APP_REGIONS_PATH.read_text(encoding="utf-8"))

def _load_active_training_result() -> dict[str, Any]:
    try:
        payload = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=404, detail="Active training result is unavailable"
        ) from None

    active = payload.get("active") if isinstance(payload, dict) else None
    if not isinstance(active, dict):
        raise HTTPException(
            status_code=404, detail="Active training result is unavailable"
        )

    run_name = active.get("run_name")
    checkpoint = active.get("checkpoint")
    metrics = active.get("metrics")
    updated_at = active.get("updated_at")
    metric_values = (
        metrics.get("corr"),
        metrics.get("mae"),
        metrics.get("rmse"),
    ) if isinstance(metrics, dict) else ()
    valid_metrics = (
        len(metric_values) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in metric_values
        )
        and isinstance(metrics.get("samples"), int)
        and not isinstance(metrics.get("samples"), bool)
    )
    if (
        not isinstance(run_name, str)
        or not run_name
        or not isinstance(checkpoint, str)
        or not checkpoint
        or not valid_metrics
        or not isinstance(updated_at, str)
        or not updated_at
    ):
        raise HTTPException(
            status_code=404, detail="Active training result is unavailable"
        )

    return {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "metrics": {
            "corr": metrics["corr"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "samples": metrics["samples"],
        },
        "updated_at": updated_at,
    }


@lru_cache(maxsize=4)
def _load_tile_score_grid(
    report_path: str, modified_ns: int
) -> tuple[bytes, int, int, int, int, int]:
    del modified_ns
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    tiles = payload["tiles"]
    zooms = {int(tile["z"]) for tile in tiles}
    if len(zooms) != 1:
        raise ValueError("mixed tile zooms")
    xs = [int(tile["x"]) for tile in tiles]
    ys = [int(tile["y"]) for tile in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    scores = array("f", [math.nan]) * (width * height)
    for tile in tiles:
        x = int(tile["x"])
        y = int(tile["y"])
        scores[(y - min_y) * width + (x - min_x)] = float(
            tile["scenic_score"]
        )
    if sys.byteorder != "little":
        scores.byteswap()
    return scores.tobytes(), next(iter(zooms)), min_x, min_y, width, height


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

_ROUTE_PRELOAD_MODES = frozenset({"off", "best_effort", "required"})
_DEFAULT_ROUTE_PRELOAD_MODE = "best_effort"
_LEGACY_ROUTE_PRELOAD_MODES = {
    "0": "off",
    "false": "off",
    "no": "off",
    "off": "off",
    "1": "required",
    "true": "required",
    "yes": "required",
    "on": "required",
}


def _route_preload_mode() -> str:
    """Return the explicit startup route preload mode.

    A fresh checkout defaults to best effort so ignored graph/report assets are
    not required just to start the API.  Legacy boolean values retain their
    former strict-enabled/disabled behavior where possible.
    """

    raw_value = os.getenv("SCENIC_ROUTE_PRELOAD")
    if raw_value is None or not raw_value.strip():
        return _DEFAULT_ROUTE_PRELOAD_MODE

    value = raw_value.strip().lower().replace("-", "_")
    if value in _ROUTE_PRELOAD_MODES:
        return value
    if value in _LEGACY_ROUTE_PRELOAD_MODES:
        return _LEGACY_ROUTE_PRELOAD_MODES[value]

    _LOGGER.warning(
        "Unknown SCENIC_ROUTE_PRELOAD=%r; defaulting to %s",
        raw_value,
        _DEFAULT_ROUTE_PRELOAD_MODE,
    )
    return _DEFAULT_ROUTE_PRELOAD_MODE


def _route_preload_enabled() -> bool:
    """Return whether startup route materialization is enabled.

    Kept as a compatibility helper for callers that only need the old
    boolean view; startup itself uses :func:`_route_preload_mode`.
    """

    return _route_preload_mode() != "off"


def _preload_configured_route_assets(
    mode: str = _DEFAULT_ROUTE_PRELOAD_MODE,
) -> dict[str, Any]:
    """Preload configured graph/report pairs, retaining the default in cache."""

    if mode not in _ROUTE_PRELOAD_MODES or mode == "off":
        raise ValueError(f"Unsupported route preload mode: {mode!r}")

    default_region = _default_region_key()
    configured = list(_configured_regions())
    configured.sort(
        key=lambda item: (
            str(item.get("region", "")).strip().lower() == default_region,
        )
    )
    if default_region and not any(
        str(item.get("region", "")).strip().lower() == default_region
        for item in configured
    ):
        if mode == "required":
            raise RuntimeError(
                f"Configured default region '{default_region}' has no region entry"
            )
        _LOGGER.warning(
            "Skipping route preload: configured default region '%s' has no "
            "region entry",
            default_region,
        )

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "default_region": default_region,
        "regions": [],
    }
    for item in configured:
        region = str(item.get("region", "")).strip()
        region_key = region.lower()
        is_default = bool(default_region and region_key == default_region)
        graph_value = item.get("graph_path") or item.get("graph")
        graph_path = Path(graph_value) if graph_value else None
        run_name = (
            str(item["run_name"])
            if item.get("run_name")
            else _latest_run_for_region(region)
        )
        tile_value = (
            item.get("tile_scores_json")
            or item.get("tile_scores_path")
            or item.get("report")
            or item.get("report_json")
        )
        tile_path = (
            Path(tile_value)
            if tile_value
            else RUNS_DIR / run_name / "report/report.json"
            if run_name
            else None
        )
        region_diag: dict[str, Any] = {
            "region": region,
            "default": is_default,
            "graph_path": str(graph_path) if graph_path else None,
            "tile_scores_path": str(tile_path) if tile_path else None,
        }
        missing: list[str] = []
        if graph_path is None or not graph_path.exists():
            missing.append("graph")
        if tile_path is None or not tile_path.exists():
            missing.append("report")
        if missing:
            reason = f"missing {', '.join(missing)} asset(s)"
            if is_default and mode == "required":
                raise RuntimeError(
                    f"Configured default region '{region}' {reason}"
                )
            region_diag.update({"status": "skipped", "reason": reason})
            diagnostics["regions"].append(region_diag)
            _LOGGER.warning("Skipping route preload for %s: %s", region, reason)
            continue
        try:
            preload_result = preload_route_assets(
                graph_path,
                tile_path,
                item.get("tile_score_zoom"),
                item.get("tile_score_fallback", 1.0),
            )
            if not isinstance(preload_result, dict):
                preload_result = {}
        except Exception as exc:
            if is_default and mode == "required":
                raise RuntimeError(
                    f"Configured default region '{region}' preload failed"
                ) from exc
            reason = f"{type(exc).__name__}: {exc}"
            region_diag.update({"status": "skipped", "reason": reason})
            diagnostics["regions"].append(region_diag)
            _LOGGER.warning(
                "Skipping route preload for %s%s: %s",
                "default " if is_default else "optional ",
                region,
                reason,
            )
            continue
        region_diag.update({"status": "loaded", "preload": preload_result})
        diagnostics["regions"].append(region_diag)
        _LOGGER.info(
            "Preloaded route assets for %s (graph_cache_hit=%s tile_cache_hit=%s "
            "scored_cache_hit=%s)",
            region,
            preload_result.get("graph_cache_hit"),
            preload_result.get("tile_score_cache_hit"),
            preload_result.get("scored_graph_cache_hit"),
        )

    diagnostics["loaded_regions"] = [
        row["region"]
        for row in diagnostics["regions"]
        if row.get("status") == "loaded"
    ]
    diagnostics["skipped_regions"] = [
        row["region"]
        for row in diagnostics["regions"]
        if row.get("status") == "skipped"
    ]
    _LOGGER.info(
        "Route preload complete: loaded=%d skipped=%d default=%s",
        len(diagnostics["loaded_regions"]),
        len(diagnostics["skipped_regions"]),
        default_region,
    )
    return diagnostics


@asynccontextmanager
async def _api_lifespan(app: FastAPI):
    validate_route_configuration()
    mode = _route_preload_mode()
    if mode != "off":
        preload_diagnostics = _preload_configured_route_assets(mode)
    else:
        preload_diagnostics = {
            "enabled": False,
            "mode": mode,
            "default_region": _default_region_key(),
            "regions": [],
            "loaded_regions": [],
            "skipped_regions": [],
        }
        _LOGGER.warning(
            "Route asset preload disabled by SCENIC_ROUTE_PRELOAD=%s",
            mode,
        )
    app.state.route_preload_diagnostics = preload_diagnostics
    yield


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
                "graph_exists": bool(graph and graph.exists()),
                "latest_run_name": latest_run,
                "bbox": item.get("bbox"),
                "map": item.get("map"),
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
                "graph_exists": True,
                "latest_run_name": latest_run,
                "bbox": bbox,
                "is_default": region.lower() == default_region,
                "source": "discovered",
            }
        )
    return regions


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




_REQUIRED_SCORE_MAPPING_FIELDS = (
    "report_signature",
    "graph_signature",
    "normalization",
    "fallback_edges",
    "score_run",
)
_REQUIRED_COMPARISON_DIAGNOSTICS = (
    "requested_scenic_weight",
    "applied_scenic_weight",
    "requested_max_detour_factor",
    "applied_max_detour_factor",
    "scenic_fastest_duration_ratio",
    "score_mapping_coverage",
    "optimization_mode",
    "optimization_status",
    "optimality_gap",
    "certified_upper_bound",
    "scenic_score_delta_absolute",
    "scenic_score_delta_relative",
    "same_route",
    "no_better_route_reason",
)

_REQUIRED_ROUTE_METRICS = (
    "edge_ids",
    "segment_identity",
    "raw_scenic_score",
    "normalized_scenic_score",
    "objective",
    "objective_value",
    "total_distance_km",
    "estimated_duration_minutes",
    "requested_scenic_weight",
    "applied_scenic_weight",
    "requested_max_detour_factor",
    "applied_max_detour_factor",
    "actual_duration_ratio",
    "certified_upper_bound",
    "exactness_status",
    "optimality_gap",
    "highway_count",
    "score_coverage",
    "score_run",
    "zero_improvement_reason",
    "no_route_reason",
    "objective_components",
)


_PRIVATE_RESPONSE_KEYS = frozenset(
    {"source", "graph_geojson", "tile_scores_json", "tile_score_fallback"}
)


def _redact_private_response(value: Any) -> Any:
    """Remove filesystem-bearing metadata from public route responses."""

    if isinstance(value, dict):
        return {
            key: _redact_private_response(item)
            for key, item in value.items()
            if key not in _PRIVATE_RESPONSE_KEYS
        }
    if isinstance(value, list):
        return [_redact_private_response(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_private_response(item) for item in value]
    return value


def _public_route_request(
    payload: RouteCompareRequest,
    *,
    run_name: str,
) -> dict[str, Any]:
    """Project only user-facing route controls; never expose asset paths."""

    return {
        "region": payload.region,
        "run_name": run_name,
        "start": {"lat": payload.start.lat, "lon": payload.start.lon},
        "end": {"lat": payload.end.lat, "lon": payload.end.lon},
        "scenic_weight": payload.scenic_weight,
        "avoid_highways": payload.avoid_highways,
        "max_detour_factor": payload.max_detour_factor,
        "include_baseline": payload.include_baseline,
    }


def _public_preload_diagnostics(value: Any) -> dict[str, Any]:
    """Project health preload data without exposing asset paths or errors."""

    if not isinstance(value, dict):
        return {
            "enabled": False,
            "mode": "off",
            "default_region": _default_region_key(),
            "regions": [],
            "loaded_regions": [],
            "skipped_regions": [],
        }
    public: dict[str, Any] = {
        key: value.get(key)
        for key in (
            "enabled",
            "mode",
            "default_region",
            "loaded_regions",
            "skipped_regions",
        )
        if key in value
    }
    public_regions: list[dict[str, Any]] = []
    for row in value.get("regions", []):
        if not isinstance(row, dict):
            continue
        item: dict[str, Any] = {
            key: row[key]
            for key in ("region", "default", "status")
            if key in row
        }
        preload = row.get("preload")
        if isinstance(preload, dict):
            cache_keys = (
                "graph_cache_hit",
                "tile_score_cache_hit",
                "scored_graph_cache_hit",
            )
            cache = {key: preload[key] for key in cache_keys if key in preload}
            mapping = preload.get("score_mapping")
            if isinstance(mapping, dict):
                coverage = {
                    key: mapping[key]
                    for key in (
                        "enabled",
                        "zoom",
                        "matched_edges",
                        "fallback_edges",
                        "total_edges",
                        "matched_ratio",
                    )
                    if key in mapping
                }
                signatures = {
                    key: mapping[key]
                    for key in (
                        "report_signature",
                        "graph_signature",
                        "normalization",
                        "score_run",
                    )
                    if key in mapping
                }
                if coverage:
                    item["coverage"] = coverage
                if signatures:
                    item["signatures"] = signatures
            timings = {
                key: preload[key]
                for key in (
                    "preload_elapsed_ms",
                    "graph_load_elapsed_ms",
                    "tile_score_load_elapsed_ms",
                    "score_application_elapsed_ms",
                )
                if key in preload
            }
            if cache:
                item["cache"] = cache
            if timings:
                item["timings"] = timings
        public_regions.append(item)
    public["regions"] = public_regions
    return public
def _validate_certified_bound(
    metrics: dict[str, Any],
    *,
    label: str,
) -> None:
    """Validate a certified upper-bound/gap pair without claiming exactness."""

    status = str(metrics.get("exactness_status") or metrics.get("optimization_status") or "").lower()
    exact = status in {"exact", "optimal"}
    approximate = status.startswith("approximate") or status == "certified"
    if not exact and not approximate:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "message": f"{label} result has unknown exactness status",
            },
        )
    objective = metrics.get("objective_value")
    upper_bound = metrics.get("certified_upper_bound")
    gap = metrics.get("optimality_gap")
    if exact:
        valid_objective = (
            not isinstance(objective, bool)
            and isinstance(objective, (int, float))
            and math.isfinite(float(objective))
        )
        valid_gap = gap is None or (
            not isinstance(gap, bool)
            and isinstance(gap, (int, float))
            and math.isfinite(float(gap))
            and abs(float(gap)) <= 1e-9
        )
        valid_bound = upper_bound is None or (
            not isinstance(upper_bound, bool)
            and isinstance(upper_bound, (int, float))
            and math.isfinite(float(upper_bound))
            and valid_objective
            and abs(float(upper_bound) - float(objective)) <= 1e-9
        )
        if not valid_objective or not valid_gap or not valid_bound:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "message": f"{label} exact result has inconsistent certification data",
                },
            )
        return
    if upper_bound is None:
        if approximate:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "message": f"{label} approximate result has no certified upper bound",
                },
            )
        if gap is not None and (
            isinstance(gap, bool)
            or not isinstance(gap, (int, float))
            or not math.isfinite(float(gap))
            or abs(float(gap)) > 1e-9
        ):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "message": f"{label} exact result has a nonzero optimality gap",
                },
            )
        return
    if (
        isinstance(objective, bool)
        or isinstance(upper_bound, bool)
        or isinstance(gap, bool)
        or not isinstance(objective, (int, float))
        or not isinstance(upper_bound, (int, float))
        or not isinstance(gap, (int, float))
        or not math.isfinite(float(objective))
        or not math.isfinite(float(upper_bound))
        or not math.isfinite(float(gap))
    ):
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "message": f"{label} certified bound is not finite numeric data",
            },
        )
    tolerance = max(1e-9, abs(float(upper_bound)) * 1e-6)
    if float(upper_bound) + tolerance < float(objective):
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "message": f"{label} certified upper bound is below objective",
            },
        )
    if abs((float(upper_bound) - float(objective)) - float(gap)) > tolerance:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "message": f"{label} certified gap does not match upper bound minus objective",
            },
        )




def _invalid_route_geometry(message: str) -> NoReturn:
    raise HTTPException(
        status_code=502,
        detail={
            "error": "invalid_route_service_response",
            "message": message,
        },
    )


def _route_coordinate(
    value: Any,
    *,
    label: str,
    latitude_first: bool = False,
) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        _invalid_route_geometry(f"{label} must be a numeric coordinate pair")
    try:
        first = float(value[0])
        second = float(value[1])
    except (TypeError, ValueError, OverflowError):
        _invalid_route_geometry(f"{label} must be a numeric coordinate pair")
    latitude, longitude = (
        (first, second) if latitude_first else (second, first)
    )
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        _invalid_route_geometry(f"{label} contains invalid coordinates")
    return longitude, latitude


def _route_coordinates_match(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return all(
        math.isclose(
            first[index],
            second[index],
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for index in range(2)
    )


def _dedupe_route_coordinates(
    coordinates: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for coordinate in coordinates:
        if not result or not _route_coordinates_match(coordinate, result[-1]):
            result.append(coordinate)
    return result


def _validate_route_geometry(
    geojson: Any,
    *,
    request: RouteRequest,
    routes: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(geojson, dict)
        or geojson.get("type") != "FeatureCollection"
        or not isinstance(geojson.get("features"), list)
    ):
        _invalid_route_geometry("Route service returned invalid route GeoJSON")
    features = geojson["features"]
    expected_kinds = {"scenic"} | (
        {"baseline"} if request.include_baseline else set()
    )
    validated: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            _invalid_route_geometry(f"route feature {index} is malformed")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            _invalid_route_geometry(
                f"route feature {index} has no properties"
            )
        route_kind = properties.get("route_kind")
        if not isinstance(route_kind, str) or route_kind not in expected_kinds:
            _invalid_route_geometry(
                f"route feature {index} has an unexpected route kind"
            )
        if route_kind in validated:
            _invalid_route_geometry(
                f"route GeoJSON contains duplicate {route_kind} features"
            )
        geometry = feature.get("geometry")
        if (
            not isinstance(geometry, dict)
            or geometry.get("type") != "LineString"
            or not isinstance(geometry.get("coordinates"), list)
        ):
            _invalid_route_geometry(
                f"{route_kind} route geometry is not a LineString"
            )
        raw_coordinates = geometry["coordinates"]
        if len(raw_coordinates) < 2:
            _invalid_route_geometry(
                f"{route_kind} route geometry has fewer than two coordinates"
            )
        coordinates = _dedupe_route_coordinates(
            [
                _route_coordinate(
                    coordinate,
                    label=f"{route_kind} geometry coordinate {coordinate_index}",
                )
                for coordinate_index, coordinate in enumerate(raw_coordinates)
            ]
        )
        rows = properties.get("segment_identity")
        if not isinstance(rows, list):
            _invalid_route_geometry(
                f"{route_kind} route has no segment identity rows"
            )
        expected_metrics = routes.get(route_kind)
        if (
            not isinstance(expected_metrics, dict)
            or rows != expected_metrics.get("segment_identity")
        ):
            _invalid_route_geometry(
                f"{route_kind} segment identity does not match route metrics"
            )
        segment_coordinates: list[tuple[float, float]] = []
        previous_end: tuple[float, float] | None = None
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                _invalid_route_geometry(
                    f"{route_kind} segment row {row_index} is malformed"
                )
            segment_start = _route_coordinate(
                row.get("start"),
                label=f"{route_kind} segment row {row_index} start",
                latitude_first=True,
            )
            segment_end = _route_coordinate(
                row.get("end"),
                label=f"{route_kind} segment row {row_index} end",
                latitude_first=True,
            )
            if (
                previous_end is not None
                and not _route_coordinates_match(previous_end, segment_start)
            ):
                _invalid_route_geometry(
                    f"{route_kind} segment rows are not continuous"
                )
            if not segment_coordinates:
                segment_coordinates.append(segment_start)
            segment_coordinates.append(segment_end)
            previous_end = segment_end
        if segment_coordinates:
            expected_coordinates = _dedupe_route_coordinates(segment_coordinates)
            if len(coordinates) != len(expected_coordinates) or any(
                not _route_coordinates_match(actual, expected)
                for actual, expected in zip(coordinates, expected_coordinates)
            ):
                _invalid_route_geometry(
                    f"{route_kind} route geometry omits or reorders scored segments"
                )
        elif len(coordinates) < 2:
            _invalid_route_geometry(
                f"{route_kind} zero-edge route geometry is not renderable"
            )
        validated[route_kind] = feature
    if set(validated) != expected_kinds:
        missing = sorted(expected_kinds - set(validated))
        _invalid_route_geometry(
            f"Route GeoJSON is missing route features: {', '.join(missing)}"
        )
    return validated


def _validate_route_contract(
    result: dict[str, Any],
    *,
    request: RouteRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the single service response schema before exposing it."""

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="Route service returned an invalid response",
        )
    score_mapping = result.get("score_mapping")
    if not isinstance(score_mapping, dict):
        raise HTTPException(
            status_code=502,
            detail="Route service returned no score mapping metadata",
        )
    missing_mapping = [
        key for key in _REQUIRED_SCORE_MAPPING_FIELDS if key not in score_mapping
    ]
    if missing_mapping:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "missing_score_mapping": missing_mapping,
            },
        )
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise HTTPException(
            status_code=502,
            detail="Route service returned no comparison diagnostics",
        )
    missing_diagnostics = [
        key for key in _REQUIRED_COMPARISON_DIAGNOSTICS if key not in diagnostics
    ]
    if missing_diagnostics:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "missing_diagnostics": missing_diagnostics,
            },
        )

    requested_weight = diagnostics["requested_scenic_weight"]
    applied_weight = diagnostics["applied_scenic_weight"]
    requested = diagnostics["requested_max_detour_factor"]
    applied = diagnostics["applied_max_detour_factor"]
    if (
        isinstance(requested_weight, bool)
        or isinstance(applied_weight, bool)
        or not isinstance(requested_weight, (int, float))
        or not isinstance(applied_weight, (int, float))
        or not math.isfinite(float(requested_weight))
        or not math.isfinite(float(applied_weight))
        or float(requested_weight) != float(request.scenic_weight)
        or float(applied_weight) != float(request.scenic_weight)
        or isinstance(requested, bool)
        or isinstance(applied, bool)
        or not isinstance(requested, (int, float))
        or not isinstance(applied, (int, float))
        or not math.isfinite(float(requested))
        or not math.isfinite(float(applied))
        or float(requested) != float(request.max_detour_factor)
        or float(applied) != float(request.max_detour_factor)
    ):
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "message": "Route service changed the requested settings",
            },
        )

    rows = result.get("routes")
    if not isinstance(rows, list):
        raise HTTPException(
            status_code=502,
            detail="Route service returned no routes",
        )
    routes: dict[str, Any] = {}
    expected_route_kinds = {"scenic"} | (
        {"baseline"} if request.include_baseline else set()
    )
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("route_kind"), str):
            raise HTTPException(
                status_code=502,
                detail="Route service returned an invalid route row",
            )
        route_kind = row["route_kind"]
        if route_kind not in expected_route_kinds:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "message": f"Route service returned unexpected {route_kind} route",
                },
            )
        if route_kind in routes:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "message": f"Route service returned duplicate {route_kind} routes",
                },
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise HTTPException(
                status_code=502,
                detail="Route service returned invalid route metrics",
            )
        missing_metrics = [
            key for key in _REQUIRED_ROUTE_METRICS if key not in metrics
        ]
        if missing_metrics:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "route_kind": route_kind,
                    "missing_metrics": missing_metrics,
                },
            )
        metric_requested_weight = metrics["requested_scenic_weight"]
        metric_applied_weight = metrics["applied_scenic_weight"]
        metric_requested = metrics["requested_max_detour_factor"]
        metric_applied = metrics["applied_max_detour_factor"]
        if (
            isinstance(metric_requested_weight, bool)
            or isinstance(metric_applied_weight, bool)
            or not isinstance(metric_requested_weight, (int, float))
            or not isinstance(metric_applied_weight, (int, float))
            or not math.isfinite(float(metric_requested_weight))
            or not math.isfinite(float(metric_applied_weight))
            or float(metric_requested_weight) != float(request.scenic_weight)
            or float(metric_applied_weight) != float(request.scenic_weight)
            or isinstance(metric_requested, bool)
            or isinstance(metric_applied, bool)
            or not isinstance(metric_requested, (int, float))
            or not isinstance(metric_applied, (int, float))
            or not math.isfinite(float(metric_requested))
            or not math.isfinite(float(metric_applied))
            or float(metric_requested) != float(request.max_detour_factor)
            or float(metric_applied) != float(request.max_detour_factor)
        ):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "route_kind": row["route_kind"],
                    "message": "Route service changed the requested settings",
                },
            )
        for numeric_key in (
            "objective",
            "objective_value",
            "total_distance_km",
            "estimated_duration_minutes",
            "raw_scenic_score",
            "normalized_scenic_score",
        ):
            numeric_value = metrics[numeric_key]
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not math.isfinite(float(numeric_value))
            ):
                raise HTTPException(
                    status_code=502,
                    detail=f"Route service returned invalid {numeric_key}",
                )
        if float(metrics["objective"]) != float(metrics["objective_value"]):
            raise HTTPException(
                status_code=502,
                detail="Route service returned inconsistent objective fields",
            )
        components = metrics["objective_components"]
        component_value = (
            components.get("objective_value")
            if isinstance(components, dict)
            else None
        )
        if (
            isinstance(component_value, bool)
            or not isinstance(component_value, (int, float))
            or not math.isfinite(float(component_value))
            or not math.isclose(
                float(component_value),
                float(metrics["objective_value"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise HTTPException(
                status_code=502,
                detail="Route service returned invalid objective components",
            )
        _validate_certified_bound(metrics, label=f"{row['route_kind']} route")
        routes[row["route_kind"]] = dict(metrics)
    if "scenic" not in routes:
        raise HTTPException(
            status_code=502,
            detail="Route service returned no scenic route",
        )
    if request.include_baseline and "baseline" not in routes:
        raise HTTPException(
            status_code=502,
            detail="Route service returned no baseline route",
        )
    geojson = result.get("geojson")
    _validate_route_geometry(geojson, request=request, routes=routes)
    _validate_certified_bound(
        {
            "objective_value": routes["scenic"]["objective_value"],
            "certified_upper_bound": diagnostics["certified_upper_bound"],
            "optimality_gap": diagnostics["optimality_gap"],
            "optimization_status": diagnostics["optimization_status"],
        },
        label="comparison",
    )
    return dict(diagnostics), routes


def create_app() -> FastAPI:
    app = FastAPI(
        title="ScenicDrive API",
        version="0.1.0",
        lifespan=_api_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Scenic-Tile-Zoom",
            "X-Scenic-Min-X",
            "X-Scenic-Min-Y",
            "X-Scenic-Grid-Width",
            "X-Scenic-Grid-Height",
        ],
    )

    @app.get("/scalar", include_in_schema=False)
    def scalar_docs() -> Any:
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=f"{app.title} Scalar Reference",
        )

    repo = ContribRepo()
    @app.get("/v1/healthz")
    def healthz() -> dict[str, Any]:
        preload_mode = _route_preload_mode()
        preload_diagnostics = _public_preload_diagnostics(
            getattr(
                app.state,
                "route_preload_diagnostics",
                {
                    "enabled": preload_mode != "off",
                    "mode": preload_mode,
                    "default_region": _default_region_key(),
                    "regions": [],
                    "loaded_regions": [],
                    "skipped_regions": [],
                },
            )
        )
        return {
            "ok": True,
            "default_region": _default_region_key(),
            "model_registry_exists": MODEL_REGISTRY_PATH.exists(),
            "regions_available": len(_list_regions()),
            "route_preload": preload_diagnostics,
        }

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "ScenicDrive API",
            "version": "0.1.0",
            "docs": "/docs",
            "scalar_docs": "/scalar",
            "health": "/v1/healthz",
            "regions": "/v1/regions",
        }

    @app.get("/v1/regions")
    def regions() -> dict[str, Any]:
        return {"regions": _list_regions()}

    @app.get("/v1/training-results")
    def training_results() -> dict[str, Any]:
        return _load_active_training_result()


    @app.get("/v1/search/suggest")
    def search_suggest(
        q: str, session_token: str, region: str = "new_england_north"
    ) -> dict[str, Any]:
        query = q.strip()
        if not query:
            return {"suggestions": []}
        token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if not token:
            raise HTTPException(
                status_code=503, detail="Address search is unavailable"
            )
        params: dict[str, Any] = {
            "q": query,
            "access_token": token,
            "session_token": session_token,
            "limit": 5,
            "country": "us",
            "language": "en",
            "types": "address,street,place,locality,neighborhood,poi",
        }
        configured = _app_region(region)
        bbox = _normalize_bbox(configured.get("bbox")) if configured else None
        if bbox:
            params["bbox"] = (
                f"{bbox['min_lon']},{bbox['min_lat']},"
                f"{bbox['max_lon']},{bbox['max_lat']}"
            )
        response = requests.get(
            "https://api.mapbox.com/search/searchbox/v1/suggest",
            params=params,
            timeout=15,
        )
        if not response.ok:
            raise HTTPException(
                status_code=502, detail="Address search provider failed"
            )
        rows = response.json().get("suggestions", [])
        return {
            "suggestions": [
                {
                    "mapbox_id": row.get("mapbox_id"),
                    "name": row.get("name"),
                    "full_address": row.get("full_address")
                    or row.get("place_formatted")
                    or row.get("name"),
                    "feature_type": row.get("feature_type"),
                }
                for row in rows
                if isinstance(row, dict)
                and isinstance(row.get("mapbox_id"), str)
                and isinstance(row.get("name"), str)
            ]
        }

    @app.get("/v1/search/retrieve")
    def search_retrieve(mapbox_id: str, session_token: str) -> dict[str, Any]:
        token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if not token:
            raise HTTPException(
                status_code=503, detail="Address search is unavailable"
            )
        response = requests.get(
            "https://api.mapbox.com/search/searchbox/v1/retrieve/"
            + quote(mapbox_id, safe=""),
            params={
                "access_token": token,
                "session_token": session_token,
            },
            timeout=15,
        )
        if not response.ok:
            raise HTTPException(
                status_code=502, detail="Address search provider failed"
            )
        features = response.json().get("features", [])
        feature = features[0] if features else None
        coordinates = (
            feature.get("geometry", {}).get("coordinates")
            if isinstance(feature, dict)
            else None
        )
        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        if (
            not isinstance(coordinates, list)
            or len(coordinates) < 2
            or not all(isinstance(value, (int, float)) for value in coordinates[:2])
        ):
            raise HTTPException(status_code=404, detail="Address result is unavailable")
        lat = float(coordinates[1])
        lon = float(coordinates[0])
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not -90.0 <= lat <= 90.0
            or not -180.0 <= lon <= 180.0
        ):
            raise HTTPException(
                status_code=502,
                detail="Search provider returned invalid coordinates",
            )
        return {
            "result": {
                "name": properties.get("name"),
                "full_address": properties.get("full_address")
                or properties.get("place_formatted")
                or properties.get("name"),
                "lat": lat,
                "lon": lon,
            }
        }

    @app.post("/v1/route/compare")
    def route_compare(payload: RouteCompareRequest) -> dict[str, Any]:
        _safe_asset_name(payload.region, kind="region")
        try:
            graph_path = _region_to_graph(payload.region)
        except FileNotFoundError as exc:
            _LOGGER.warning("Route graph unavailable for region %s: %s", payload.region, exc)
            raise HTTPException(
                status_code=404,
                detail=f"Route assets are unavailable for region '{payload.region}'",
            ) from None

        run_name = payload.run_name or _latest_run_for_region(payload.region)
        if not run_name:
            raise HTTPException(
                status_code=404,
                detail=f"No run report available for region '{payload.region}'",
            )
        run_name = _safe_asset_name(run_name, kind="run_name")
        report_json = _run_report_path(run_name)
        if not report_json.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"No run report available for region '{payload.region}'",
            )

        req = RouteRequest(
            graph_geojson=str(graph_path),
            start=(payload.start.lat, payload.start.lon),
            end=(payload.end.lat, payload.end.lon),
            scenic_weight=payload.scenic_weight,
            avoid_highways=payload.avoid_highways,
            max_detour_factor=payload.max_detour_factor,
            include_baseline=payload.include_baseline,
            tile_scores_json=str(report_json),
            tile_score_fallback=None,
        )
        try:
            result = plan_routes(req)
            diagnostics, routes = _validate_route_contract(result, request=req)
        except RouteConfigurationError as exc:
            _LOGGER.error("Invalid route configuration: %s", exc)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "route_configuration_invalid",
                    "message": "Route planning configuration is invalid.",
                },
            ) from exc
        except ValueError as exc:
            try:
                diagnostics = diagnose_route_request(req)
            except Exception:
                diagnostics = {}
            if req.avoid_highways:
                message = "No route satisfies the avoid-highways constraint."
                hint = (
                    "Turn off Avoid highways, choose different points, "
                    "or increase max detour."
                )
            else:
                message = "No route satisfies the requested controls."
                hint = (
                    f"Try different points in region '{payload.region}' "
                    "or increase max detour."
                )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "no_route_found",
                    "message": message,
                    "hint": hint,
                    "diagnostics": diagnostics,
                },
            ) from exc

        scenic = routes["scenic"]
        baseline = routes.get("baseline")
        deltas = None
        if baseline is not None:
            deltas = {
                "distance_km": float(scenic["total_distance_km"])
                - float(baseline["total_distance_km"]),
                "duration_min": float(scenic["estimated_duration_minutes"])
                - float(baseline["estimated_duration_minutes"]),
                "scenic_score": diagnostics["scenic_score_delta_absolute"],
                "scenic_score_absolute": diagnostics[
                    "scenic_score_delta_absolute"
                ],
                "scenic_score_relative": diagnostics[
                    "scenic_score_delta_relative"
                ],
                "normalized_scenic_score": float(
                    scenic["normalized_scenic_score"]
                )
                - float(baseline["normalized_scenic_score"]),
            }
        public_routes = _redact_private_response(routes)
        public_score_mapping = _redact_private_response(
            result.get("score_mapping", {})
        )
        public_geojson = _redact_private_response(result.get("geojson", {}))
        return {
            "request": _public_route_request(payload, run_name=run_name),
            "run_name": run_name,
            "diagnostics": diagnostics,
            "routes": public_routes,
            "deltas": deltas,
            "score_mapping": public_score_mapping,
            "geojson": public_geojson,
        }

    @app.get("/v1/validated-route")
    def validated_route(region: str) -> dict[str, Any]:
        configured = _app_region(region)
        run_name = str(configured.get("run_name", "")) if configured else ""
        report_dir = RUNS_DIR / run_name / "report"
        route_path = report_dir / "route.geojson"
        metrics_path = report_dir / "route_metrics.json"
        try:
            route_geojson = json.loads(route_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise HTTPException(
                status_code=404, detail="Validated route artifacts are unavailable"
            ) from None

        features = route_geojson.get("features") if isinstance(route_geojson, dict) else None
        request = metrics.get("request") if isinstance(metrics, dict) else None
        score_mapping = metrics.get("score_mapping") if isinstance(metrics, dict) else None
        scenic = metrics.get("scenic") if isinstance(metrics, dict) else None
        baseline = metrics.get("baseline") if isinstance(metrics, dict) else None
        valid_route_features = (
            isinstance(route_geojson, dict)
            and route_geojson.get("type") == "FeatureCollection"
            and isinstance(features, list)
            and all(isinstance(feature, dict) for feature in features)
            and {
                feature.get("properties", {}).get("route_kind")
                for feature in features
                if isinstance(feature.get("properties"), dict)
            }
            == {"scenic", "baseline"}
        )
        if (
            not valid_route_features
            or not isinstance(request, dict)
            or not isinstance(score_mapping, dict)
            or not isinstance(scenic, dict)
            or not isinstance(baseline, dict)
        ):
            raise HTTPException(
                status_code=404, detail="Validated route artifacts are unavailable"
            )

        public_mapping_fields = (
            "enabled",
            "zoom",
            "matched_edges",
            "total_edges",
            "matched_ratio",
            "report_signature",
            "graph_signature",
            "score_run",
        )
        return {
            "region": region,
            "run_name": run_name,
            "request": {
                "start": request.get("start"),
                "end": request.get("end"),
                "scenic_weight": request.get("scenic_weight"),
                "max_detour_factor": request.get("max_detour_factor"),
            },
            "score_mapping": {
                key: score_mapping.get(key)
                for key in public_mapping_fields
                if score_mapping.get(key) is not None
            },
            "routes": _redact_private_response(
                {"scenic": scenic, "baseline": baseline}
            ),
            "geojson": _redact_private_response(route_geojson),
        }

    @app.get("/v1/heatmap")
    def heatmap(
        region: str,
        run_name: str | None = None,
        max_points: int = 3000,
        max_tiles: int = 12000,
        tile_offset: int | None = None,
    ) -> dict[str, Any]:
        _safe_asset_name(region, kind="region")
        selected_run = run_name or _latest_run_for_region(region)
        if not selected_run:
            raise HTTPException(
                status_code=404, detail=f"No run report available for region '{region}'"
            )
        selected_run = _safe_asset_name(selected_run, kind="run_name")
        report_json = _run_report_path(selected_run)
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

        summary = payload.get("summary", {})
        scores = [
            float(tile["scenic_score"])
            for tile in tiles
            if tile.get("scenic_score") is not None
        ]
        try:
            norm_min = float(summary["min"])
            norm_max = float(summary["max"])
        except (KeyError, TypeError, ValueError):
            norm_min = min(scores) if scores else 0.0
            norm_max = max(scores) if scores else 10.0
        if not math.isfinite(norm_min) or not math.isfinite(norm_max) or norm_max <= norm_min:
            raise HTTPException(status_code=404, detail=f"Invalid report for run '{selected_run}'")

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
        if tile_offset is not None:
            offset = max(0, int(tile_offset))
            polygon_tiles = tiles[offset : offset + tile_limit]
        elif len(tiles) <= tile_limit:
            offset = 0
            polygon_tiles = tiles
        else:
            offset = 0
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
            "total_tiles": len(tiles),
            "tile_offset": offset,
            "bounds": {
                "min_lon": _tile_to_bounds(min_x, min_y, tile_zoom)[1],
                "min_lat": _tile_to_bounds(max_x, max_y, tile_zoom)[0],
                "max_lon": _tile_to_bounds(max_x, max_y, tile_zoom)[3],
                "max_lat": _tile_to_bounds(min_x, min_y, tile_zoom)[2],
            } if None not in (min_x, min_y, max_x, max_y, tile_zoom) else None,
            "normalization": {
                "min": norm_min,
                "max": norm_max,
                "source": "report_score_range",
            },
            "summary": {
                key: summary.get(key)
                for key in ("total_tiles", "mean", "median", "std", "min", "max")
            },
            "geojson": {"type": "FeatureCollection", "features": feats},
            "geojson_tiles": {"type": "FeatureCollection", "features": tile_feats},
        }

    @app.get("/v1/heatmap-image")
    def heatmap_image(region: str, run_name: str | None = None) -> Response:
        from PIL import Image

        _safe_asset_name(region, kind="region")
        selected_run = run_name or _latest_run_for_region(region)
        if not selected_run:
            raise HTTPException(status_code=404, detail="No run report available")
        selected_run = _safe_asset_name(selected_run, kind="run_name")
        report_json = _run_report_path(selected_run)
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            tiles = payload["tiles"]
            summary = payload["summary"]
            norm_min = float(summary["min"])
            norm_max = float(summary["max"])
            zooms = {int(tile["z"]) for tile in tiles}
            xs = [int(tile["x"]) for tile in tiles]
            ys = [int(tile["y"]) for tile in tiles]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise HTTPException(
                status_code=404, detail="Learned heatmap artifact is unavailable"
            ) from None
        if len(zooms) != 1 or norm_max <= norm_min:
            raise HTTPException(
                status_code=404, detail="Learned heatmap artifact is unavailable"
            )

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        image = Image.new("RGBA", (max_x - min_x + 1, max_y - min_y + 1))
        pixels = image.load()
        stops = (
            (0.0, (43, 104, 95)),
            (0.35, (135, 166, 93)),
            (0.6, (226, 183, 91)),
            (0.78, (215, 118, 66)),
            (1.0, (145, 54, 51)),
        )

        def score_color(score: float) -> tuple[int, int, int, int]:
            value = max(0.0, min(1.0, (score - norm_min) / (norm_max - norm_min)))
            for index in range(1, len(stops)):
                lower_value, lower_color = stops[index - 1]
                upper_value, upper_color = stops[index]
                if value <= upper_value:
                    ratio = (value - lower_value) / (upper_value - lower_value)
                    return (
                        *(
                            round(lower + (upper - lower) * ratio)
                            for lower, upper in zip(lower_color, upper_color)
                        ),
                        210,
                    )
            return (*stops[-1][1], 210)

        for tile in tiles:
            pixels[int(tile["x"]) - min_x, int(tile["y"]) - min_y] = score_color(
                float(tile["scenic_score"])
            )
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return Response(
            output.getvalue(),
            media_type="image/png",
            headers={
                "X-Scenic-Tile-Zoom": str(next(iter(zooms))),
                "Cache-Control": "public, max-age=3600",
            },
        )

    @app.get("/v1/heatmap-scores.bin")
    def heatmap_scores(region: str, run_name: str | None = None) -> Response:
        _safe_asset_name(region, kind="region")
        selected_run = run_name or _latest_run_for_region(region)
        if not selected_run:
            raise HTTPException(status_code=404, detail="No run report available")
        selected_run = _safe_asset_name(selected_run, kind="run_name")
        report_json = _run_report_path(selected_run)
        try:
            data, zoom, min_x, min_y, width, height = _load_tile_score_grid(
                str(report_json), report_json.stat().st_mtime_ns
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            raise HTTPException(
                status_code=404, detail="Learned heatmap scores are unavailable"
            ) from None
        return Response(
            data,
            media_type="application/octet-stream",
            headers={
                "X-Scenic-Tile-Zoom": str(zoom),
                "X-Scenic-Min-X": str(min_x),
                "X-Scenic-Min-Y": str(min_y),
                "X-Scenic-Grid-Width": str(width),
                "X-Scenic-Grid-Height": str(height),
                "Cache-Control": "public, max-age=3600",
            },
        )

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
