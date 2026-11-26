from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NoReturn
from array import array
from functools import lru_cache
import asyncio
from functools import partial
import threading
import io
import json
import logging
import hashlib
import math
import re
import sys
import time
from dotenv import load_dotenv

from anyio import to_thread
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from scalar_fastapi import get_scalar_api_reference
from fastapi.responses import Response
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from urllib.parse import quote

from scenicdriver.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from scenicdriver.route_planner.service import (
    RouteCoverageError,
    RouteConfigurationError,
    RouteRequest,
    _deadline_seconds_from_env,
    _route_response_cache_key,
    diagnose_route_request,
    plan_routes,
    preload_route_assets,
    validate_route_configuration,
)
from scenicdriver.route_planner.supervisor import (
    PreloadedRouteSupervisor,
    SupervisorError,
)

from .contrib_repo import ContribRepo
from .schemas import (
    ContributorLabelRequest,
    ContributorProfileRequest,
    ContributorSessionStartRequest,
    RouteCompareRequest,
)

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROAD_GRAPHS_DIR = PROJECT_ROOT / "data/processed/road_graphs"
RUNS_DIR = PROJECT_ROOT / "data/processed/heuristic_runs"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "data/processed/regression/model_registry.json"
APP_REGIONS_PATH = PROJECT_ROOT / "config/app_regions.json"

_LOGGER = logging.getLogger(__name__)


def _plan_routes_worker(
    deadline: RoutingDeadline, request: RouteRequest
) -> dict[str, Any]:
    """Run route planning in the persistent hard-stop worker."""
    return plan_routes(request, deadline=deadline)


_SAFE_REGION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SAFE_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _safe_asset_name(value: str, *, kind: str) -> str:
    pattern = _SAFE_REGION_RE if kind == "region" else _SAFE_RUN_RE
    if not pattern.fullmatch(value):
        raise HTTPException(status_code=422, detail=f"Invalid {kind}")
    return value


def _raise_routing_http_exception(exc: BaseException) -> NoReturn:
    """Map routing timeout/cancellation to explicit HTTP responses.

    ``detail["reason"]`` carries the structured cause's own reason string
    verbatim (the messages raised in ``route_planner.cancellation`` and the
    planner search stages) so clients can read the cause without parsing
    ``message`` prose.
    """
    if isinstance(exc, RoutingTimeout):
        _LOGGER.error("Routing deadline exceeded: %s", exc)
        raise HTTPException(
            status_code=504,
            detail={
                "error": "routing_deadline_exceeded",
                "message": "The route planning deadline was exceeded.",
                "reason": str(exc),
            },
        ) from exc
    if isinstance(exc, RoutingCancelled):
        _LOGGER.error("Routing request cancelled: %s", exc)
        raise HTTPException(
            status_code=499,
            detail={
                "error": "routing_cancelled",
                "message": "The route planning request was cancelled.",
                "reason": str(exc),
            },
        ) from exc
    raise exc


# Planner no-route reason strings.  ``src/scenicdriver/route_planner/service.py`` matches
# these verbatim to classify a failed search, so they are the structured cause
# of a 422 ``no_route_found``.  Any other ``ValueError`` is not a recognized
# cause and is reported without a ``reason`` (clients fall back to ``message``).
_PLANNER_NO_ROUTE_REASONS = frozenset({
    "No route found between the given coordinates.",
    "No route satisfies the requested duration cap.",
})


def _run_report_path(run_name: str) -> Path:
    safe_run = _safe_asset_name(run_name, kind="run_name")
    return RUNS_DIR / safe_run / "report/report.json"


def _region_score_report(item: dict[str, Any] | None, run_name: str | None) -> Path | None:
    configured_path = next(
        (item.get(key) for key in ("tile_scores_json", "tile_scores_path", "report", "report_json")
         if item.get(key)),
        None,
    ) if item else None
    return Path(configured_path) if configured_path else _run_report_path(run_name) if run_name else None


def _route_deadline_seconds(item: dict[str, Any] | None) -> float:
    """SCENIC_ROUTE_DEADLINE_SECONDS overrides region route_deadline_seconds."""
    override = os.getenv("SCENIC_ROUTE_DEADLINE_SECONDS", "").strip()
    if override or not item or item.get("route_deadline_seconds") is None:
        return _deadline_seconds_from_env()
    try:
        value = float(item["route_deadline_seconds"])
    except (TypeError, ValueError) as exc:
        raise RouteConfigurationError("Invalid region routing deadline") from exc
    if not math.isfinite(value) or value < 0:
        raise RouteConfigurationError("Invalid region routing deadline")
    return value


async def _run_routing_connected(http_request: Request, routing_deadline: RoutingDeadline, function, *args, **kwargs):
    async def observe_disconnect():
        while True:
            if await http_request.is_disconnected():
                routing_deadline.cancel_event.set()
                return
            await asyncio.sleep(0.05)

    observer = asyncio.create_task(observe_disconnect())
    try:
        routing_deadline.check()
        future = to_thread.run_sync(
            partial(function, *args, **kwargs), abandon_on_cancel=True
        )
        try:
            result = await asyncio.wait_for(
                future, timeout=routing_deadline.remaining_seconds()
            )
        except RoutingTimeout:
            # RoutingTimeout inherits TimeoutError; preserve the planner's
            # structured reason rather than mistaking it for wait_for expiry.
            raise
        except asyncio.TimeoutError:
            routing_deadline.cancel_event.set()
            raise RoutingTimeout("routing request deadline exceeded") from None
        routing_deadline.check()
        return result
    except asyncio.CancelledError:
        routing_deadline.cancel_event.set()
        raise
    finally:
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)


def _search_provider(url: str, params: dict[str, Any], field: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(url, params=params, timeout=15)
        if not response.ok:
            raise ValueError("provider returned an unsuccessful status")
        payload = response.json()
        rows = payload.get(field) if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("invalid provider envelope")
        return rows
    except requests.Timeout:
        raise HTTPException(status_code=504, detail={"message": "Address search provider timed out"}) from None
    except (requests.RequestException, ValueError, TypeError):
        raise HTTPException(status_code=502, detail={"message": "Address search provider failed"}) from None


class _RequestBodyLimit:
    """Reject bodies above 64 KiB before request model parsing, including streams."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > 65536:
                return await JSONResponse(status_code=413, content={"detail": {"message": "Request body exceeds 64 KiB"}})(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        consumed = False
        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


_DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def _allowed_origins() -> list[str]:
    """SCENIC_ALLOWED_ORIGINS is a comma-separated CORS origin allowlist."""
    configured = os.getenv("SCENIC_ALLOWED_ORIGINS", "").strip() or _DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _positive_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


class _ClientRateLimiter:
    """Process-local token bucket per client key (one bucket set per API worker)."""

    _PRUNE_THRESHOLD = 10_000

    def __init__(self, rate_per_minute: float, burst: float) -> None:
        if burst < 1:
            raise ValueError("Rate limiter burst must be at least 1")
        self._rate = rate_per_minute / 60.0
        self._burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, client: str) -> float:
        """Take one token; return 0 when admitted, else seconds until one refills."""
        now = time.monotonic()
        with self._lock:
            tokens, updated = self._buckets.get(client, (self._burst, now))
            tokens = min(self._burst, tokens + (now - updated) * self._rate)
            if tokens < 1:
                self._buckets[client] = (tokens, now)
                return (1 - tokens) / self._rate
            self._buckets[client] = (tokens - 1, now)
            if len(self._buckets) > self._PRUNE_THRESHOLD:
                # Fully refilled buckets carry no state worth keeping.
                self._buckets = {
                    key: (level, at) for key, (level, at) in self._buckets.items()
                    if level + (now - at) * self._rate < self._burst
                }
            return 0.0


class _RouteAdmission:
    """Bound route-compare requests that are running or waiting for a worker."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._admitted = 0
        self._lock = threading.Lock()

    def try_enter(self) -> bool:
        with self._lock:
            if self._admitted >= self._limit:
                return False
            self._admitted += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._admitted -= 1


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
        (
            metrics.get("corr"),
            metrics.get("mae"),
            metrics.get("rmse"),
        )
        if isinstance(metrics, dict)
        else ()
    )
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
def _parse_heatmap_artifact(
    report_path: str, identity: tuple[int, int, int, int, int]
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if _readiness_file_identity(Path(report_path)) != identity:
            raise HTTPException(status_code=502, detail={"message": "Learned heatmap artifact changed while reading"})
    except OSError:
        raise HTTPException(status_code=404, detail={"message": "Learned heatmap artifact is unavailable"}) from None
    except (UnicodeError, json.JSONDecodeError):
        raise HTTPException(status_code=502, detail={"message": "Learned heatmap artifact is invalid"}) from None
    try:
        if not isinstance(payload, dict):
            raise ValueError("invalid report")
        tiles = payload["tiles"]
        summary = payload.get("summary", {})
        if not isinstance(tiles, list) or not tiles or not isinstance(summary, dict):
            raise ValueError("invalid tiles or summary")
        for tile in tiles:
            if not isinstance(tile, dict):
                raise ValueError("invalid tile")
            x, y, z = (tile[key] for key in ("x", "y", "z"))
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y, z)):
                raise ValueError("tile coordinates must be integers")
            if not 0 <= z <= 30 or not 0 <= x < 2 ** z or not 0 <= y < 2 ** z:
                raise ValueError("tile coordinates out of bounds")
            score = tile["scenic_score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or abs(score) > 3.402823466e38:
                raise ValueError("invalid tile score")
        if len({tile["z"] for tile in tiles}) != 1:
            raise ValueError("mixed zooms")
        for value in summary.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("invalid summary")
        scores = [float(tile["scenic_score"]) for tile in tiles]
        low = float(summary.get("min", min(scores)))
        high = float(summary.get("max", max(scores)))
        if not math.isfinite(low) or not math.isfinite(high) or high <= low or not math.isfinite(high - low):
            raise ValueError("invalid normalization")
        min_x, max_x = min(tile["x"] for tile in tiles), max(tile["x"] for tile in tiles)
        min_y, max_y = min(tile["y"] for tile in tiles), max(tile["y"] for tile in tiles)
        width, height = max_x - min_x + 1, max_y - min_y + 1
        if width * height > 16_777_216:
            raise ValueError("heatmap grid is too large")
        return {
            **payload,
            "summary": {**summary, "min": low, "max": high},
            "_grid_extent": (tiles[0]["z"], min_x, min_y, width, height),
            "_report_identity": identity,
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=502, detail={"message": "Learned heatmap artifact is invalid"}) from None


def _heatmap_artifact(region: str, run_name: str | None) -> tuple[str, Path, dict[str, Any]]:
    _safe_asset_name(region, kind="region")
    selected_run = run_name or _latest_run_for_region(region)
    if not selected_run:
        raise HTTPException(status_code=404, detail={"message": "No run report available"})
    selected_run = _safe_asset_name(selected_run, kind="run_name")
    report = _run_report_path(selected_run)
    try:
        identity = _readiness_file_identity(report)
    except OSError:
        raise HTTPException(status_code=404, detail={"message": "Learned heatmap artifact is unavailable"}) from None
    return selected_run, report, _parse_heatmap_artifact(str(report), identity)


@lru_cache(maxsize=4)
def _load_tile_score_grid(
    report_path: str, identity: tuple[int, int, int, int, int]
) -> tuple[bytes, int, int, int, int, int]:
    payload = _parse_heatmap_artifact(report_path, identity)
    tiles = payload["tiles"]
    zoom, min_x, min_y, width, height = payload["_grid_extent"]
    scores = array("f", [math.nan]) * (width * height)
    for tile in tiles:
        x = int(tile["x"])
        y = int(tile["y"])
        scores[(y - min_y) * width + (x - min_x)] = float(tile["scenic_score"])
    if sys.byteorder != "little":
        scores.byteswap()
    return scores.tobytes(), zoom, min_x, min_y, width, height


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


def _readiness_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=16)
def _readiness_file_digest(path: Path, identity: tuple[int, int, int, int, int]) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if _readiness_file_identity(path) != identity:
        raise ValueError("Graph changed while validating readiness")
    return digest


def _graph_readiness(graph: Path | None) -> dict[str, Any] | None:
    """Validate the builder's publication gate, never bare graph existence.

    Digests are reused only for unchanged files; the gate is reread on every
    call so withdrawing it immediately disables even preloaded regions.
    """
    if graph is None:
        return None
    gate = graph.parent / "run.json"
    try:
        gate_identity = _readiness_file_identity(gate)
        record = json.loads(gate.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        output = record.get("output_graph")
        digest = record.get("graph_sha256")
        size = record.get("graph_size_bytes")
        if (
            not isinstance(output, str) or not output
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int or size < 0
        ):
            return None
        source = Path(output)
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        source = source.resolve()
        source_identity = _readiness_file_identity(source)
        if source_identity[2] != size or _readiness_file_digest(source, source_identity) != digest:
            return None
        graph_identity = source_identity
        if graph.resolve() != source:
            compact = record.get("runtime_graph") or record.get("compact_graph")
            compact_digest = record.get("compact_manifest_sha256")
            if not isinstance(compact, str) or not isinstance(compact_digest, str):
                return None
            compact_path = Path(compact)
            if not compact_path.is_absolute():
                compact_path = PROJECT_ROOT / compact_path
            if compact_path.resolve() != graph.resolve():
                return None
            graph_identity = _readiness_file_identity(graph)
            if _readiness_file_digest(graph, graph_identity) != compact_digest:
                return None
        if (
            _readiness_file_identity(source) != source_identity
            or _readiness_file_identity(graph) != graph_identity
            or _readiness_file_identity(gate) != gate_identity
        ):
            return None
        return record
    except (OSError, ValueError, TypeError, UnicodeError):
        return None


def _region_to_graph(region: str) -> Path:
    # Check-then-open window (inherent, bounded): the readiness gate above is
    # validated here and the planner opens the graph later.  The builder
    # withdraws run.json before swapping graph content, and each file is
    # replaced by an atomic per-file swap, so a request whose gate check
    # preceded the withdrawal may still open post-swap content — never torn
    # content — for the remainder of that one request.  The gate is reread on
    # every call and never cached across requests, so the window is strictly
    # request-scoped and no request outlives a withdrawn publication.  Per-call
    # gate semantics hold; a lock redesign is deliberately out of scope.
    configured = _app_region(region)
    if configured and configured.get("graph_path"):
        graph_path = Path(configured["graph_path"])
        if graph_path.exists() and _graph_readiness(graph_path) is not None:
            return graph_path
        raise FileNotFoundError(
            f"Configured road graph readiness for region '{region}' is unavailable: {graph_path}"
        )

    key = region.strip().lower()
    candidates = [
        ROAD_GRAPHS_DIR / f"{key}_core/road_graph.geojson",
        ROAD_GRAPHS_DIR / f"{key}_core/road_graph.json",
        ROAD_GRAPHS_DIR / f"{key}_core/road_graph.sqlite3",
        ROAD_GRAPHS_DIR / key / "road_graph.geojson",
        ROAD_GRAPHS_DIR / key / "road_graph.json",
        ROAD_GRAPHS_DIR / key / "road_graph.sqlite3",
    ]
    for c in candidates:
        if c.exists() and _graph_readiness(c) is not None:
            return c
    raise FileNotFoundError(f"No road graph readiness found for region '{region}'")


def _latest_run_for_region(region: str) -> str | None:
    configured = _app_region(region)
    if configured and configured.get("run_name"):
        return str(configured["run_name"])

    key = region.strip().lower()
    if not key or not RUNS_DIR.exists():
        return None
    matches = [
        d
        for d in RUNS_DIR.iterdir()
        if d.is_dir()
        and (d.name.lower() == key or d.name.lower().startswith(f"{key}_"))
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].name


_ROUTE_PRELOAD_MODES = frozenset({"off", "best_effort", "required"})
_DEFAULT_ROUTE_PRELOAD_MODE = "best_effort"


def _route_preload_mode() -> str:
    """Return the explicit startup route preload mode.

    A fresh checkout defaults to best effort so ignored graph/report assets are
    not required just to start the API.
    """

    raw_value = os.getenv("SCENIC_ROUTE_PRELOAD")
    if raw_value is None or not raw_value.strip():
        return _DEFAULT_ROUTE_PRELOAD_MODE

    value = raw_value.strip().lower().replace("-", "_")
    if value in _ROUTE_PRELOAD_MODES:
        return value

    _LOGGER.warning(
        "Unknown SCENIC_ROUTE_PRELOAD=%r; defaulting to %s",
        raw_value,
        _DEFAULT_ROUTE_PRELOAD_MODE,
    )
    return _DEFAULT_ROUTE_PRELOAD_MODE


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
        tile_path = _region_score_report(item, run_name)
        region_diag: dict[str, Any] = {
            "region": region,
            "default": is_default,
            "graph_path": str(graph_path) if graph_path else None,
            "tile_scores_path": str(tile_path) if tile_path else None,
        }
        if item.get("route_planning") is False:
            reason = "route planning disabled"
            if is_default and mode == "required":
                raise RuntimeError(
                    f"Configured default region '{region}' {reason}"
                )
            region_diag.update({"status": "skipped", "reason": reason})
            diagnostics["regions"].append(region_diag)
            _LOGGER.warning("Skipping route preload for %s: %s", region, reason)
            continue
        missing: list[str] = []
        if graph_path is None or not graph_path.exists():
            missing.append("graph")
        if tile_path is None or not tile_path.exists():
            missing.append("report")
        if graph_path is not None and graph_path.exists() and _graph_readiness(graph_path) is None:
            missing.append("readiness metadata")
        if missing:
            reason = f"missing {', '.join(missing)} asset(s)"
            if is_default and mode == "required":
                raise RuntimeError(f"Configured default region '{region}' {reason}")
            region_diag.update({"status": "skipped", "reason": reason})
            diagnostics["regions"].append(region_diag)
            _LOGGER.warning("Skipping route preload for %s: %s", region, reason)
            continue
        try:
            preload_result = preload_route_assets(
                graph_path,
                tile_path,
                item.get("tile_score_zoom"),
                None,
                exclusive_scoring=True,
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
        row["region"] for row in diagnostics["regions"] if row.get("status") == "loaded"
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
    app.state.route_supervisor = None
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
    # Isolation is independent of preload: the persistent worker also supports
    # loading route assets on its first job.
    if any(item.get("route_planning") is not False for item in _configured_regions()):
        try:
            app.state.route_supervisor = PreloadedRouteSupervisor.start(
                preload_marker=preload_diagnostics,
                default_deadline_seconds=None,
                default_grace_seconds=0.5,
            )
        except SupervisorError:
            if mode == "required":
                raise
            _LOGGER.exception(
                "Route hard-stop supervisor unavailable; "
                "routing will fail closed"
            )
    try:
        yield
    finally:
        supervisor = app.state.route_supervisor
        app.state.route_supervisor = None
        if supervisor is not None:
            supervisor.close()


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
        latest_run = (
            str(item["run_name"])
            if item.get("run_name")
            else _latest_run_for_region(region)
        )
        regions.append(
            {
                "region": region,
                "display_name": item.get("display_name", region),
                "description": item.get("description"),
                "graph_exists": bool(graph and graph.exists()),
                "route_planning_enabled": (
                    item.get("route_planning", True) is not False
                    and _graph_readiness(graph) is not None
                ),
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
        graph = next(
            (
                candidate
                for candidate in (
                    d / "road_graph.geojson",
                    d / "road_graph.json",
                    d / "road_graph.sqlite3",
                )
                if candidate.exists()
            ),
            None,
        )
        if graph is None:
            continue
        region = d.name.replace("_core", "")
        if region.lower() in configured_keys:
            continue
        if graph.resolve() in configured_graphs:
            continue
        latest_run = _latest_run_for_region(region)
        readiness = _graph_readiness(graph)
        bbox = readiness.get("bbox") if readiness is not None else None
        if readiness is not None and bbox is None and graph.suffix != ".sqlite3":
            bbox = _compute_geojson_bbox(graph)
        regions.append(
            {
                "region": region,
                "display_name": region,
                "graph_exists": True,
                "route_planning_enabled": readiness is not None,
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
    "avoid_highways_applied",
    "highway_avoidance_fallback",
    "highway_avoidance_fallback_reason",
    "highway_avoidance_mode",
    "highway_preference",
    "highway_avoidance_cost",
    "hard_highway_count",
    "detour_reference_duration_minutes",
    "duration_cap_minutes",
    "duration_cap_satisfied",
)

_REQUIRED_ROUTE_METRICS = (
    "edge_ids",
    "route_kind",
    "geometry",
    "requested_start",
    "requested_end",
    "snapped_start",
    "snapped_end",
    "segment_identity",
    "raw_scenic_score",
    "normalized_scenic_score",
    "objective",
    "objective_value",
    "total_distance_km",
    "estimated_duration_minutes",
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
    """Project shared containers once, preserving aliases without mutating input."""
    memo: dict[int, Any] = {}

    def project(item: Any) -> Any:
        if not isinstance(item, (dict, list, tuple)):
            return item
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if isinstance(item, dict):
            projected: Any = {}
            memo[identity] = projected
            for key, nested in item.items():
                if key not in _PRIVATE_RESPONSE_KEYS:
                    projected[key] = project(nested)
        else:
            projected = []
            memo[identity] = projected
            projected.extend(project(nested) for nested in item)
        return projected

    return project(value)


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
            key: row[key] for key in ("region", "default", "status") if key in row
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

    status = str(
        metrics.get("exactness_status") or metrics.get("optimization_status") or ""
    ).lower()
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
        raise HTTPException(
            status_code=502,
            detail={
                "error": "invalid_route_service_response",
                "message": f"{label} approximate result has no certified upper bound",
            },
        )
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
    latitude, longitude = (first, second) if latitude_first else (second, first)
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


def _validate_route_geometry(
    geojson: Any,
    *,
    request: RouteRequest,
    routes: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate canonical route geometry; map features contain only references."""
    if (
        not isinstance(geojson, dict)
        or geojson.get("type") != "FeatureCollection"
        or not isinstance(geojson.get("features"), list)
    ):
        _invalid_route_geometry("Route service returned invalid route GeoJSON")
    expected_kinds = {"scenic"} | ({"baseline"} if request.include_baseline else set())
    validated: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(geojson["features"]):
        if not isinstance(feature, dict):
            _invalid_route_geometry(f"route feature {index} is malformed")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            _invalid_route_geometry(f"route feature {index} has no properties")
        route_kind = properties.get("route_kind")
        if not isinstance(route_kind, str) or route_kind not in expected_kinds:
            _invalid_route_geometry(f"route feature {index} has an unexpected route kind")
        if route_kind in validated:
            _invalid_route_geometry(f"route GeoJSON contains duplicate {route_kind} features")
        if set(properties) != {"route_kind", "snapped_start", "snapped_end"}:
            _invalid_route_geometry(f"{route_kind} map properties must contain only map endpoint metadata")
        metrics = routes.get(route_kind)
        if not isinstance(metrics, dict):
            _invalid_route_geometry(f"{route_kind} route has no canonical metrics")
        geometry = metrics.get("geometry")
        if (
            not isinstance(geometry, dict)
            or geometry.get("type") != "LineString"
            or not isinstance(geometry.get("coordinates"), list)
        ):
            _invalid_route_geometry(f"{route_kind} route geometry is not a LineString")
        if feature.get("geometry") is not geometry and feature.get("geometry") != geometry:
            _invalid_route_geometry(f"{route_kind} map geometry does not match canonical route")
        coordinates = geometry["coordinates"]
        if len(coordinates) < 2:
            _invalid_route_geometry(f"{route_kind} route geometry has fewer than two coordinates")
        for endpoint, expected in (("start", request.start), ("end", request.end)):
            requested = _route_coordinate(
                metrics.get(f"requested_{endpoint}"),
                label=f"{route_kind} requested_{endpoint}", latitude_first=True,
            )
            if not _route_coordinates_match(requested, (float(expected[1]), float(expected[0]))):
                _invalid_route_geometry(
                    f"{route_kind} requested_{endpoint} does not match the route request"
                )
        snapped_start = _route_coordinate(
            metrics.get("snapped_start"), label=f"{route_kind} snapped_start",
            latitude_first=True,
        )
        snapped_end = _route_coordinate(
            metrics.get("snapped_end"), label=f"{route_kind} snapped_end",
            latitude_first=True,
        )
        for endpoint in ("snapped_start", "snapped_end"):
            if properties[endpoint] != metrics.get(endpoint):
                _invalid_route_geometry(f"{route_kind} map {endpoint} does not match canonical route")
        rows = metrics.get("segment_identity")
        if not isinstance(rows, list):
            _invalid_route_geometry(f"{route_kind} route has no segment identity rows")

        def geometry_positions():
            previous = None
            for coordinate_index, raw in enumerate(coordinates):
                position = _route_coordinate(
                    raw, label=f"{route_kind} geometry coordinate {coordinate_index}",
                )
                if coordinate_index == 0 and not _route_coordinates_match(snapped_start, position):
                    _invalid_route_geometry(f"{route_kind} snapped_start does not match route geometry")
                if coordinate_index == len(coordinates) - 1 and not _route_coordinates_match(snapped_end, position):
                    _invalid_route_geometry(f"{route_kind} snapped_end does not match route geometry")
                if previous is None or not _route_coordinates_match(previous, position):
                    yield position
                    previous = position

        positions = iter(geometry_positions())
        if not rows:
            next(positions)
            if len(coordinates) != 2 or next(positions, None) is not None:
                _invalid_route_geometry(
                    f"{route_kind} zero-edge route geometry must contain exactly two equal positions"
                )
        else:
            previous_end = None
            previous_position = None
            mismatch = False
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    _invalid_route_geometry(f"{route_kind} segment row {row_index} is malformed")
                start = _route_coordinate(
                    row.get("start"), label=f"{route_kind} segment row {row_index} start",
                    latitude_first=True,
                )
                end = _route_coordinate(
                    row.get("end"), label=f"{route_kind} segment row {row_index} end",
                    latitude_first=True,
                )
                if previous_end is not None and not _route_coordinates_match(previous_end, start):
                    _invalid_route_geometry(f"{route_kind} segment rows are not continuous")
                for position in ((start, end) if row_index == 0 else (end,)):
                    if previous_position is None or not _route_coordinates_match(previous_position, position):
                        actual = next(positions, None)
                        if actual is None or not _route_coordinates_match(actual, position):
                            mismatch = True
                        previous_position = position
                previous_end = end
            if next(positions, None) is not None:
                mismatch = True
            # Exhaust the iterator even on mismatches so every coordinate and
            # both endpoints are checked before reporting segment alignment.
            for _ in positions:
                pass
            if mismatch:
                _invalid_route_geometry(
                    f"{route_kind} route geometry omits or reorders scored segments"
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
    deadline: RoutingDeadline | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the single service response schema before exposing it."""

    if deadline is not None:
        deadline.check()
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

    requested = diagnostics["requested_max_detour_factor"]
    applied = diagnostics["applied_max_detour_factor"]
    if (
        isinstance(requested, bool)
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
        missing_metrics = [key for key in _REQUIRED_ROUTE_METRICS if key not in metrics]
        if missing_metrics:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_route_service_response",
                    "route_kind": route_kind,
                    "missing_metrics": missing_metrics,
                },
            )
        if metrics["route_kind"] != route_kind:
            _invalid_route_geometry(f"{route_kind} canonical route kind does not match route row")
        metric_requested = metrics["requested_max_detour_factor"]
        metric_applied = metrics["applied_max_detour_factor"]
        if (
            isinstance(metric_requested, bool)
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
            components.get("objective_value") if isinstance(components, dict) else None
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
        routes[row["route_kind"]] = metrics
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


def _resolve_route_assets(
    payload: RouteCompareRequest, configured: dict[str, Any] | None
) -> tuple[RouteRequest, str]:
    """Resolve and authenticate artifacts outside the event loop."""
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
        raise HTTPException(status_code=404, detail=f"No run report available for region '{payload.region}'")
    run_name = _safe_asset_name(run_name, kind="run_name")
    report_json = _region_score_report(configured, run_name)
    if not report_json.is_file():
        raise HTTPException(status_code=404, detail=f"No run report available for region '{payload.region}'")
    return RouteRequest(
        graph_geojson=str(graph_path),
        start=(payload.start.lat, payload.start.lon),
        end=(payload.end.lat, payload.end.lon),
        avoid_highways=payload.avoid_highways,
        max_detour_factor=payload.max_detour_factor,
        include_baseline=payload.include_baseline,
        tile_scores_json=str(report_json),
        tile_score_zoom=configured.get("tile_score_zoom") if configured else None,
        tile_score_fallback=None,
        max_snap_distance_km=configured.get("max_route_snap_km") if configured else None,
    ), run_name


def _public_route_geojson(routes: dict[str, Any]) -> dict[str, Any]:
    """Map projection shares canonical geometry and endpoint containers."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "route_kind": kind,
                    "snapped_start": metrics["snapped_start"],
                    "snapped_end": metrics["snapped_end"],
                },
                "geometry": metrics["geometry"],
            }
            for kind, metrics in routes.items()
        ],
    }


def _project_route_response(
    result: dict[str, Any], *, payload: RouteCompareRequest, run_name: str,
    request: RouteRequest, deadline: RoutingDeadline,
) -> dict[str, Any]:
    """Validate and redact the complete response within the request budget."""
    diagnostics, routes = _validate_route_contract(result, request=request, deadline=deadline)
    scenic, baseline = routes["scenic"], routes.get("baseline")
    deltas = None
    if baseline is not None:
        deltas = {
            "distance_km": float(scenic["total_distance_km"]) - float(baseline["total_distance_km"]),
            "duration_min": float(scenic["estimated_duration_minutes"]) - float(baseline["estimated_duration_minutes"]),
            "scenic_score": diagnostics["scenic_score_delta_absolute"],
            "scenic_score_absolute": diagnostics["scenic_score_delta_absolute"],
            "scenic_score_relative": diagnostics["scenic_score_delta_relative"],
            "normalized_scenic_score": float(scenic["normalized_scenic_score"]) - float(baseline["normalized_scenic_score"]),
        }
    response = {
        "request": _public_route_request(payload, run_name=run_name),
        "run_name": run_name,
        "diagnostics": diagnostics,
        "routes": routes,
        "deltas": deltas,
        "score_mapping": result.get("score_mapping", {}),
        "geojson": _public_route_geojson(routes),
    }
    response = _redact_private_response(response)
    deadline.check()
    return response


class _PublicRouteCache:
    """Bounded artifact/request cache of validated, redacted JSON bytes."""

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[Any, ...], tuple[bytes, bytes, tuple[str, ...]]] = OrderedDict()
        self._lock = threading.Lock()

    def key(
        self, payload: RouteCompareRequest, request: RouteRequest, run_name: str,
    ) -> tuple[Any, ...] | None:
        graph = Path(request.graph_geojson)
        if not graph.exists():
            return None
        return (payload.region, run_name, _route_response_cache_key(request, graph))

    def lookup(
        self, payload: RouteCompareRequest, request: RouteRequest, run_name: str, *,
        deadline: RoutingDeadline, started_at: float,
    ) -> tuple[tuple[Any, ...] | None, Response | None]:
        key = self.key(payload, request, run_name)
        return key, self.get(key, deadline=deadline, started_at=started_at)

    @staticmethod
    def _json(value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def get(
        self, key: tuple[Any, ...] | None, *, deadline: RoutingDeadline, started_at: float,
    ) -> Response | None:
        deadline.check()
        with self._lock:
            entry = self._entries.get(key) if key is not None else None
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is None:
            return None
        payload_prefix, diagnostics_prefix, dynamic_keys = entry
        dynamic = {
            name: (False if name.endswith("_cache_hit") else 0.0)
            for name in dynamic_keys
        }
        dynamic["route_response_cache_hit"] = True
        dynamic["planning_elapsed_ms"] = (time.perf_counter() - started_at) * 1000.0
        body = payload_prefix + diagnostics_prefix + self._json(dynamic)[1:] + b"}"
        deadline.check()
        return Response(content=body, media_type="application/json")

    def store(
        self, key: tuple[Any, ...] | None, result: dict[str, Any], *,
        payload: RouteCompareRequest, run_name: str, request: RouteRequest,
        deadline: RoutingDeadline,
    ) -> Response:
        response = _project_route_response(
            result, payload=payload, run_name=run_name, request=request, deadline=deadline,
        )
        diagnostics = response.pop("diagnostics")
        dynamic = {
            name: value for name, value in diagnostics.items()
            if name.endswith("_cache_hit") or name.endswith("_elapsed_ms")
        }
        static = {name: value for name, value in diagnostics.items() if name not in dynamic}
        payload_prefix = self._json(response)[:-1] + b',"diagnostics":'
        static_json = self._json(static)
        diagnostics_prefix = static_json[:-1] + (b"," if static else b"")
        body = payload_prefix + (
            diagnostics_prefix + self._json(dynamic)[1:] if dynamic else static_json
        ) + b"}"
        deadline.check()
        if key is not None:
            with self._lock:
                deadline.check()
                self._entries[key] = (payload_prefix, diagnostics_prefix, tuple(dynamic))
                self._entries.move_to_end(key)
                while len(self._entries) > 8:
                    self._entries.popitem(last=False)
        return Response(content=body, media_type="application/json")


@lru_cache(maxsize=4)
def _load_heatmap_png(
    report_path: str, identity: tuple[int, int, int, int, int]
) -> tuple[bytes, int]:
    from PIL import Image

    payload = _parse_heatmap_artifact(report_path, identity)
    zoom, min_x, min_y, width, height = payload["_grid_extent"]
    norm_min, norm_max = payload["summary"]["min"], payload["summary"]["max"]
    image = Image.new("RGBA", (width, height))
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
                    *(round(lower + (upper - lower) * ratio)
                      for lower, upper in zip(lower_color, upper_color)),
                    210,
                )
        return (*stops[-1][1], 210)

    for tile in payload["tiles"]:
        pixels[int(tile["x"]) - min_x, int(tile["y"]) - min_y] = score_color(float(tile["scenic_score"]))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), zoom


def create_app() -> FastAPI:
    app = FastAPI(
        title="ScenicDrive API",
        version="0.1.0",
        description=(
            "Request bodies are limited to 64 KiB. Batch limits must be positive "
            "and within the documented maximum; offsets must be non-negative. "
            "Missing heatmap artifacts return 404; malformed artifacts return 502. "
            "SCENIC_ROUTE_DEADLINE_SECONDS overrides per-region route_deadline_seconds. "
            "Contributor mutations require the X-Contributor-Token issued once by "
            "POST /v1/contrib/profile. Address search is rate limited per client IP "
            "(SCENIC_SEARCH_RATE_PER_MIN, SCENIC_SEARCH_BURST; 429 with Retry-After); "
            "route comparisons beyond SCENIC_ROUTE_MAX_PENDING running or waiting "
            "requests return 503. SCENIC_ALLOWED_ORIGINS lists CORS origins."
            " /v1/healthz preserves ok as process liveness and adds liveness and "
            "routing (ready, assets_ready, supervisor_alive, execution_mode, "
            "ready_regions); HTTP 503 means routing is unavailable. Route "
            "planning always uses hard-stop supervision, even with preload off."
        ),
        lifespan=_api_lifespan,
    )
    app.add_middleware(_RequestBodyLimit)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
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
    search_limiter = _ClientRateLimiter(
        _positive_env("SCENIC_SEARCH_RATE_PER_MIN", 30.0),
        _positive_env("SCENIC_SEARCH_BURST", 30.0),
    )
    route_admission = _RouteAdmission(int(_positive_env("SCENIC_ROUTE_MAX_PENDING", 8)))
    public_route_cache = _PublicRouteCache()

    def admit_search(request: Request) -> None:
        """Per-client-IP limit on the paid geocoding proxy."""
        retry_after = search_limiter.acquire(request.client.host if request.client else "unknown")
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "search_rate_limited",
                    "message": "Too many address searches; retry shortly.",
                },
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

    def contributor_identity(
        token: str | None = Header(default=None, alias="X-Contributor-Token"),
    ) -> str:
        """Derive the acting contributor from its secret token, never the request body."""
        if not token:
            raise HTTPException(status_code=401, detail="Contributor token is required")
        contributor_id = repo.authenticate(token)
        if contributor_id is None:
            raise HTTPException(status_code=401, detail="Invalid contributor token")
        return contributor_id

    @app.get("/v1/healthz")
    def healthz(response: Response) -> dict[str, Any]:
        """Liveness remains ``ok``; routing readiness determines HTTP status."""
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
        supervisor = getattr(app.state, "route_supervisor", None)
        snapshot = supervisor.health() if supervisor is not None else {}
        supervisor_alive = bool(snapshot.get("alive") and not snapshot.get("closed"))
        available = _list_regions()
        ready_regions = []
        for item in available:
            region = item["region"]
            if not item.get("route_planning_enabled") or not item.get("latest_run_name"):
                continue
            report = _region_score_report(_app_region(region), item["latest_run_name"])
            if report.is_file():
                ready_regions.append(region)
        default_region = _default_region_key()
        assets_ready = default_region in ready_regions if default_region else bool(ready_regions)
        routing_ready = supervisor_alive and assets_ready
        if not routing_ready:
            response.status_code = 503
        return {
            "ok": True,
            "default_region": _default_region_key(),
            "model_registry_exists": MODEL_REGISTRY_PATH.exists(),
            "regions_available": len(available),
            "route_preload": preload_diagnostics,
            "liveness": {"ok": True},
            "routing": {
                "ready": routing_ready,
                "assets_ready": assets_ready,
                "supervisor_alive": supervisor_alive,
                "execution_mode": "hard_stop_supervisor" if supervisor_alive else "unavailable",
                "ready_regions": ready_regions,
            },
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

    @app.get("/v1/search/suggest", dependencies=[Depends(admit_search)])
    def search_suggest(
        q: str, session_token: str, region: str = "new_england_north"
    ) -> dict[str, Any]:
        query = q.strip()
        if not query:
            return {"suggestions": []}
        token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if not token:
            raise HTTPException(status_code=503, detail="Address search is unavailable")
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
        rows = _search_provider(
            "https://api.mapbox.com/search/searchbox/v1/suggest", params, "suggestions"
        )
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

    @app.get("/v1/search/retrieve", dependencies=[Depends(admit_search)])
    def search_retrieve(mapbox_id: str, session_token: str) -> dict[str, Any]:
        token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if not token:
            raise HTTPException(status_code=503, detail="Address search is unavailable")
        features = _search_provider(
            "https://api.mapbox.com/search/searchbox/v1/retrieve/" + quote(mapbox_id, safe=""),
            {"access_token": token, "session_token": session_token},
            "features",
        )
        feature = features[0] if features else None
        if feature is None:
            raise HTTPException(status_code=404, detail="Address result is unavailable")
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if (
            not isinstance(geometry, dict) or geometry.get("type") != "Point"
            or not isinstance(properties, dict)
            or not isinstance(coordinates, list) or len(coordinates) != 2
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in coordinates)
            or any(value is not None and not isinstance(value, str) for key, value in properties.items()
                   if key in {"name", "full_address", "place_formatted"})
        ):
            raise HTTPException(status_code=502, detail={"message": "Search provider returned an invalid result"})
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
    async def route_compare(request: Request, payload: RouteCompareRequest) -> Response:
        configured = _app_region(payload.region)
        if configured is not None and configured.get("route_planning") is False:
            raise HTTPException(
                status_code=404,
                detail=f"Route planning is disabled for region '{payload.region}'",
            )
        _safe_asset_name(payload.region, kind="region")
        # Only the configured (released) run may be selected explicitly.
        if payload.run_name is not None and (
            configured is None or configured.get("run_name") != payload.run_name
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "score_run_not_released",
                    "message": (
                        f"Run '{payload.run_name}' is not released for region "
                        f"'{payload.region}'."
                    ),
                },
            )
        if not route_admission.try_enter():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "routing_capacity_exceeded",
                    "message": "The routing service is busy; retry shortly.",
                },
            )
        try:
            started_at = time.perf_counter()
            deadline = RoutingDeadline.after(_route_deadline_seconds(configured), cancel_event=threading.Event())
            req, run_name = await _run_routing_connected(
                request, deadline, _resolve_route_assets, payload, configured
            )
            supervisor = getattr(request.app.state, "route_supervisor", None)
            if supervisor is None:
                raise SupervisorError("routing isolation is unavailable")
            cache_key, cached = await _run_routing_connected(
                request, deadline, public_route_cache.lookup, payload, req, run_name,
                deadline=deadline, started_at=started_at,
            )
            if cached is not None:
                return cached
            result = await _run_routing_connected(request, deadline, supervisor.run_job, _plan_routes_worker, req, deadline=deadline)
            return await _run_routing_connected(
                request, deadline, public_route_cache.store, cache_key, result,
                payload=payload, run_name=run_name, request=req, deadline=deadline,
            )
        except SupervisorError as exc:
            _LOGGER.error("Route planning supervisor unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "routing_supervisor_unavailable",
                    "message": "The routing service is temporarily unavailable.",
                },
            ) from exc
        except RouteConfigurationError as exc:
            _LOGGER.error("Invalid route configuration: %s", exc)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "route_configuration_invalid",
                    "message": "Route planning configuration is invalid.",
                },
            ) from exc
        except RouteCoverageError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "route_endpoint_outside_coverage",
                    "message": (
                        f"The {exc.endpoint} point is too far from the "
                        "supported road network."
                    ),
                    "hint": (
                        "Choose a point within the selected region's route coverage."
                    ),
                    # Structured cause: RouteCoverageError's own reason string.
                    "reason": str(exc),
                    "endpoint": exc.endpoint,
                    "snap_distance_km": float(exc.snap_distance_km),
                    "max_snap_distance_km": float(exc.max_snap_distance_km),
                },
            ) from exc
        except (RoutingTimeout, RoutingCancelled) as exc:
            _raise_routing_http_exception(exc)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=500, detail={"error": "route_assets_invalid", "message": "Route planning assets are invalid."}) from exc
        except ValueError as exc:
            try:
                diagnostics = await _run_routing_connected(request, deadline, diagnose_route_request, req, deadline=deadline)
            except (RoutingTimeout, RoutingCancelled) as routing_exc:
                _raise_routing_http_exception(routing_exc)
            except Exception as asset_exc:
                raise HTTPException(status_code=500, detail={"error": "route_assets_invalid", "message": "Route planning assets are invalid."}) from asset_exc
            if req.avoid_highways:
                message = (
                    "No route satisfies the requested controls, including "
                    "best-effort highway avoidance."
                )
                hint = (
                    "Choose different points or increase max detour; highway "
                    "use is already allowed when required."
                )
            else:
                message = "No route satisfies the requested controls."
                hint = (
                    f"Try different points in region '{payload.region}' "
                    "or increase max detour."
                )
            detail: dict[str, Any] = {
                "error": "no_route_found",
                "message": message,
                "hint": hint,
                "diagnostics": diagnostics,
            }
            reason = str(exc)
            if reason in _PLANNER_NO_ROUTE_REASONS:
                # Structured cause: the planner's own reason string, reused
                # verbatim so clients never parse message prose for it.
                detail["reason"] = reason
            raise HTTPException(status_code=422, detail=detail) from exc
        finally:
            route_admission.leave()


    @app.get("/v1/validated-route")
    def validated_route(region: str, run_name: str | None = None) -> dict[str, Any]:
        _safe_asset_name(region, kind="region")
        run_name = run_name or _latest_run_for_region(region)
        if not run_name:
            raise HTTPException(status_code=404, detail={"message": "Validated route artifacts are unavailable"})
        run_name = _safe_asset_name(run_name, kind="run_name")
        report_dir = RUNS_DIR / run_name / "report"
        route_path = report_dir / "route.geojson"
        metrics_path = report_dir / "route_metrics.json"
        try:
            route_geojson = json.loads(route_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except OSError:
            raise HTTPException(status_code=404, detail={"message": "Validated route artifacts are unavailable"}) from None
        except (UnicodeError, json.JSONDecodeError):
            raise HTTPException(status_code=502, detail={"message": "Validated route artifacts are invalid"}) from None

        features = (
            route_geojson.get("features") if isinstance(route_geojson, dict) else None
        )
        request = metrics.get("request") if isinstance(metrics, dict) else None
        score_mapping = (
            metrics.get("score_mapping") if isinstance(metrics, dict) else None
        )
        scenic = metrics.get("scenic") if isinstance(metrics, dict) else None
        baseline = metrics.get("baseline") if isinstance(metrics, dict) else None
        valid_route_features = (
            isinstance(route_geojson, dict)
            and route_geojson.get("type") == "FeatureCollection"
            and isinstance(features, list)
            and len(features) == 2
            and all(isinstance(feature, dict) for feature in features)
            and all(
                isinstance(feature.get("properties"), dict)
                and isinstance(feature["properties"].get("route_kind"), str)
                for feature in features
            )
            and {feature["properties"]["route_kind"] for feature in features} == {"scenic", "baseline"}
        )
        if (
            not valid_route_features
            or not isinstance(request, dict)
            or not isinstance(score_mapping, dict)
            or not isinstance(scenic, dict)
            or not isinstance(baseline, dict)
        ):
            raise HTTPException(
                status_code=502, detail={"message": "Validated route artifacts are invalid"}
            )
        for feature in features:
            geometry = feature.get("geometry")
            if (
                not isinstance(geometry, dict) or geometry.get("type") != "LineString"
                or not isinstance(geometry.get("coordinates"), list)
                or len(geometry["coordinates"]) < 2
            ):
                _invalid_route_geometry("Saved route requires complete LineString geometry")
            for coordinate in geometry["coordinates"]:
                _route_coordinate(coordinate, label="Saved route")
        for kind, route in (("scenic", scenic), ("baseline", baseline)):
            if route.get("route_kind") != kind:
                _invalid_route_geometry("Saved route metrics have incorrect route kind")
            for key in ("total_distance_km", "estimated_duration_minutes", "raw_scenic_score", "normalized_scenic_score"):
                value = route.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    _invalid_route_geometry(f"Saved route is missing finite {key}")
        for endpoint in ("start", "end"):
            _route_coordinate(request.get(endpoint), label=f"Saved {endpoint}", latitude_first=True)
        value = request.get("max_detour_factor")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            _invalid_route_geometry("Saved request is missing finite max_detour_factor")

        routes = {"scenic": scenic, "baseline": baseline}
        for feature in features:
            route = routes[feature["properties"]["route_kind"]]
            geometry = feature["geometry"]
            route["geometry"] = geometry
            first, last = geometry["coordinates"][0], geometry["coordinates"][-1]
            route["snapped_start"] = [first[1], first[0]]
            route["snapped_end"] = [last[1], last[0]]

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
        return _redact_private_response({
            "region": region,
            "run_name": run_name,
            "request": {
                "start": request.get("start"),
                "end": request.get("end"),
                "max_detour_factor": request.get("max_detour_factor"),
            },
            "score_mapping": {
                key: score_mapping.get(key)
                for key in public_mapping_fields
                if score_mapping.get(key) is not None
            },
            "routes": routes,
            "geojson": _public_route_geojson(routes),
        })

    @app.get("/v1/heatmap")
    def heatmap(
        region: str,
        run_name: str | None = None,
        max_points: int = Query(default=3000, ge=1, le=10000),
        max_tiles: int = Query(default=12000, ge=1, le=100000),
        tile_offset: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        selected_run, report_json, payload = _heatmap_artifact(region, run_name)
        tiles = payload["tiles"]
        tile_zoom, min_x, min_y, width, height = payload["_grid_extent"]
        max_x, max_y = min_x + width - 1, min_y + height - 1

        summary = payload.get("summary", {})
        norm_min = summary["min"]
        norm_max = summary["max"]

        feats = []
        tile_feats = []
        point_limit = max_points
        if len(tiles) <= point_limit:
            point_tiles = tiles
        else:
            # Deterministic spread sampling across the full run extent (avoid
            # taking the first N tiles, which can cluster spatially).
            step = len(tiles) / float(point_limit)
            point_tiles = [tiles[int(i * step)] for i in range(point_limit)]

        tile_limit = max_tiles
        if tile_offset is not None:
            offset = tile_offset
            polygon_tiles = tiles[offset : offset + tile_limit]
        elif len(tiles) <= tile_limit:
            offset = 0
            polygon_tiles = tiles
        else:
            offset = 0
            step = len(tiles) / float(tile_limit)
            polygon_tiles = [tiles[int(i * step)] for i in range(tile_limit)]

        for tile in point_tiles:
            x, y, z = tile["x"], tile["y"], tile["z"]
            scenic = tile["scenic_score"]
            lat, lon = _tile_to_center_latlon(x, y, z)
            scenic_f = float(scenic)
            score_norm = (scenic_f - norm_min) / (norm_max - norm_min)
            score_norm = max(0.0, min(1.0, score_norm))
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"score": scenic_f, "score_norm": score_norm},
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                }
            )

        for tile in polygon_tiles:
            x, y, z = tile["x"], tile["y"], tile["z"]
            scenic = tile["scenic_score"]
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
            }
            if None not in (min_x, min_y, max_x, max_y, tile_zoom)
            else None,
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
        selected_run, report_json, payload = _heatmap_artifact(region, run_name)
        data, zoom = _load_heatmap_png(str(report_json), payload["_report_identity"])
        return Response(
            data,
            media_type="image/png",
            headers={
                "X-Scenic-Tile-Zoom": str(zoom),
                "Cache-Control": "public, max-age=3600",
            },
        )

    @app.get("/v1/heatmap-scores.bin")
    def heatmap_scores(region: str, run_name: str | None = None) -> Response:
        selected_run, report_json, payload = _heatmap_artifact(region, run_name)
        try:
            data, zoom, min_x, min_y, width, height = _load_tile_score_grid(
                str(report_json), payload["_report_identity"]
            )
        except OSError:
            raise HTTPException(status_code=404, detail={"message": "Learned heatmap artifact is unavailable"}) from None
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

    @app.post("/v1/contrib/profile", status_code=201)
    def contrib_create_profile(payload: ContributorProfileRequest) -> dict[str, Any]:
        """Create a contributor; the plaintext token is returned only here, once."""
        try:
            profile, token = repo.create_profile(payload.display_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"profile": profile, "contributor_token": token}

    @app.patch("/v1/contrib/profile")
    def contrib_update_profile(
        payload: ContributorProfileRequest,
        contributor_id: str = Depends(contributor_identity),
    ) -> dict[str, Any]:
        try:
            return {"profile": repo.update_display_name(contributor_id, payload.display_name)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @app.post("/v1/contrib/session/start")
    def contrib_session_start(
        payload: ContributorSessionStartRequest,
        contributor_id: str = Depends(contributor_identity),
    ) -> dict[str, Any]:
        try:
            profile = repo.get_profile(contributor_id)
            tasks = repo.next_tasks(contributor_id, payload.region, count=25)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"profile": profile, "tasks": tasks, "region": payload.region}

    @app.get("/v1/contrib/tasks/next")
    def contrib_next_tasks(
        region: str = "pittsfield",
        count: int = Query(default=25, ge=1, le=100),
        contributor_id: str = Depends(contributor_identity),
    ) -> dict[str, Any]:
        try:
            profile = repo.get_profile(contributor_id)
            tasks = repo.next_tasks(contributor_id, region, count=count)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"profile": profile, "tasks": tasks, "region": region}

    @app.post("/v1/contrib/labels")
    def contrib_submit_label(
        payload: ContributorLabelRequest,
        contributor_id: str = Depends(contributor_identity),
    ) -> dict[str, Any]:
        try:
            rec = repo.submit_label({**payload.model_dump(), "contributor_id": contributor_id})
            profile = repo.get_profile(contributor_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return {"saved": True, "record": rec, "profile": profile, "credits_delta": 0.0}

    @app.get("/v1/contrib/profile")
    def contrib_profile(contributor_id: str) -> dict[str, Any]:
        try:
            profile = repo.get_profile(contributor_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        if profile is None:
            raise HTTPException(status_code=404, detail="Contributor not found")
        return {"profile": profile}

    @app.get("/v1/contrib/leaderboard")
    def contrib_leaderboard(limit: int = Query(default=25, ge=1, le=100)) -> dict[str, Any]:
        return {"leaderboard": repo.leaderboard(limit=limit)}

    @app.post("/v1/admin/contrib/review/run")
    def contrib_review_run(
        request: Request, min_overlap: int = Query(default=1, ge=1),
        min_agreement: float = Query(default=0.65, ge=0, le=1, allow_inf_nan=False),
    ) -> dict[str, Any]:
        """Require X-Admin-Token matching the configured CONTRIB_ADMIN_TOKEN."""
        from secrets import compare_digest

        configured_token = os.environ.get("CONTRIB_ADMIN_TOKEN")
        if not configured_token:
            raise HTTPException(
                status_code=503, detail="Contributor admin token is not configured"
            )
        supplied_token = request.headers.get("X-Admin-Token")
        if not supplied_token:
            raise HTTPException(status_code=401, detail="Admin token is required")
        if not compare_digest(supplied_token.encode(), configured_token.encode()):
            raise HTTPException(status_code=403, detail="Invalid admin token")
        return repo.run_qa_promotion(
            min_overlap=min_overlap, min_agreement=min_agreement
        )

    return app


app = create_app()
