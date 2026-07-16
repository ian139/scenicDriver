from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import MappingProxyType
from typing import Any

from .cost import (
    HIGHWAY_ROAD_TYPES,
    SCENIC_NORMALIZATION_VERSION,
    duration_component,
    normalize_scenic_score,
)
from .graph import RoadGraph
from .planner import (
    Route,
    ScenicRoutePlanner,
    _normalize_search_diagnostics,
)


_DEFAULT_SCENIC_WEIGHT = 0.8
_DEFAULT_AVOID_HIGHWAYS = False
_DEFAULT_MAX_DETOUR_FACTOR = 1.8
_DEFAULT_INCLUDE_BASELINE = True


class RouteConfigurationError(RuntimeError):
    """Raised when deployment-supplied route configuration is invalid."""


def _frontier_time_limit_from_env() -> float | None:
    raw_value = os.environ.get("SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS")
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return ScenicRoutePlanner.validate_frontier_time_limit_seconds(
            raw_value.strip()
        )
    except ValueError as exc:
        raise RouteConfigurationError(
            "SCENIC_ROUTE_FRONTIER_TIME_LIMIT_SECONDS must be finite and between 0 and 60 seconds"
        ) from exc


def validate_route_configuration() -> None:
    """Validate deployment-supplied route settings before serving requests."""
    _frontier_time_limit_from_env()


@dataclass
class RouteRequest:
    graph_geojson: str
    start: tuple[float, float]  # (lat, lon)
    end: tuple[float, float]  # (lat, lon)
    scenic_weight: float = _DEFAULT_SCENIC_WEIGHT
    avoid_highways: bool = _DEFAULT_AVOID_HIGHWAYS
    max_detour_factor: float = _DEFAULT_MAX_DETOUR_FACTOR
    include_baseline: bool = _DEFAULT_INCLUDE_BASELINE
    tile_scores_json: str | None = None
    tile_score_zoom: int | None = None
    tile_score_fallback: float | None = None

    def __post_init__(self) -> None:
        self.start = _parse_point(self.start, field_name="start")
        self.end = _parse_point(self.end, field_name="end")
        self.scenic_weight = float(self.scenic_weight)
        if (
            not math.isfinite(self.scenic_weight)
            or not 0.0 <= self.scenic_weight <= 1.0
        ):
            raise ValueError("scenic_weight must be finite and between 0 and 1")
        self.max_detour_factor = float(self.max_detour_factor)
        if (
            not math.isfinite(self.max_detour_factor)
            or not 1.0 <= self.max_detour_factor <= 3.0
        ):
            raise ValueError("max_detour_factor must be finite and between 1 and 3")
        self.avoid_highways = _parse_bool(
            self.avoid_highways, field_name="avoid_highways"
        )
        self.include_baseline = _parse_bool(
            self.include_baseline, field_name="include_baseline"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteRequest":
        if "graph_geojson" not in payload:
            raise ValueError("Missing required field: graph_geojson")
        if "start" not in payload:
            raise ValueError("Missing required field: start")
        if "end" not in payload:
            raise ValueError("Missing required field: end")

        return cls(
            graph_geojson=str(payload["graph_geojson"]),
            start=_parse_point(payload["start"], field_name="start"),
            end=_parse_point(payload["end"], field_name="end"),
            scenic_weight=float(
                payload.get("scenic_weight", _DEFAULT_SCENIC_WEIGHT)
            ),
            avoid_highways=_parse_bool(
                payload.get("avoid_highways", _DEFAULT_AVOID_HIGHWAYS),
                field_name="avoid_highways",
            ),
            max_detour_factor=float(
                payload.get("max_detour_factor", _DEFAULT_MAX_DETOUR_FACTOR)
            ),
            include_baseline=_parse_bool(
                payload.get("include_baseline", _DEFAULT_INCLUDE_BASELINE),
                field_name="include_baseline",
            ),
            tile_scores_json=str(payload["tile_scores_json"])
            if payload.get("tile_scores_json") is not None
            else None,
            tile_score_zoom=int(payload["tile_score_zoom"])
            if payload.get("tile_score_zoom") is not None
            else None,
            tile_score_fallback=(
                float(payload["tile_score_fallback"])
                if payload.get("tile_score_fallback") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_geojson": self.graph_geojson,
            "start": [float(self.start[0]), float(self.start[1])],
            "end": [float(self.end[0]), float(self.end[1])],
            "scenic_weight": float(self.scenic_weight),
            "avoid_highways": bool(self.avoid_highways),
            "max_detour_factor": float(self.max_detour_factor),
            "include_baseline": bool(self.include_baseline),
            "tile_scores_json": self.tile_scores_json,
            "tile_score_zoom": self.tile_score_zoom,
            "tile_score_fallback": self.tile_score_fallback,
        }


def _parse_bool(value: Any, *, field_name: str) -> bool:
    """Parse API booleans without Python's surprising ``bool('false')``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _parse_point(value: Any, *, field_name: str) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lat, lon = float(value[0]), float(value[1])
    elif isinstance(value, Mapping) and "lat" in value and "lon" in value:
        lat, lon = float(value["lat"]), float(value["lon"])
    else:
        raise ValueError(
            f"Invalid {field_name} format. Expected [lat, lon] or {{lat, lon}}."
        )
    if (
        not math.isfinite(lat)
        or not math.isfinite(lon)
        or not -90.0 <= lat <= 90.0
        or not -180.0 <= lon <= 180.0
    ):
        raise ValueError(f"{field_name} coordinates must be finite and in bounds")
    return lat, lon



def _score_run_identity(path: Path) -> str:
    if path.parent.name.lower() in {"report", "reports"} and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem



_FileSignature = tuple[int, int, int, int, int]
_GraphCacheKey = tuple[str, _FileSignature]
_NORMALIZATION_VERSION = SCENIC_NORMALIZATION_VERSION
_TileCacheValue = tuple[Mapping[tuple[int, int, int], float], int]


def _signature_digest(path_key: str, signature: _FileSignature) -> str:
    payload = f"{path_key}|{signature}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ``_GRAPH_CACHE`` retains either the canonical raw graph or the active
# scored variant, never both.  A scored miss briefly holds the raw graph,
# its private native-edge clone, and (while replacing it) the previous
# scored variant.  Only one graph variant is retained after publication;
# requests already using an evicted variant keep their own reference.
_GRAPH_CACHE: OrderedDict[_GraphCacheKey, RoadGraph] = OrderedDict()
_TILE_SCORE_CACHE: OrderedDict[_GraphCacheKey, _TileCacheValue] = OrderedDict()
_ScoredGraphCacheKey = tuple[
    _GraphCacheKey, _GraphCacheKey, int, float | None, str
]
_SCORED_GRAPH_CACHE: OrderedDict[
    _ScoredGraphCacheKey, RoadGraph
] = OrderedDict()
_ACTIVE_GRAPH_VARIANT_KEY: _ScoredGraphCacheKey | None = None
_CACHE_CAPACITY = 1
_TILE_CACHE_CAPACITY = 1
_SCORED_GRAPH_CACHE_CAPACITY = 1
_CACHE_LOCK = RLock()

def _resolved_path_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _file_signature(path: Path) -> _FileSignature:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def clear_route_caches() -> None:
    """Clear process-local graph, tile, and scored-view caches.

    The service normally invalidates entries from file signatures.  This
    explicit hook keeps tests and long-lived workers able to release retained
    graph memory between independent jobs.
    """

    global _ACTIVE_GRAPH_VARIANT_KEY
    with _CACHE_LOCK:
        _GRAPH_CACHE.clear()
        _TILE_SCORE_CACHE.clear()
        _SCORED_GRAPH_CACHE.clear()
        ScenicRoutePlanner.clear_shared_caches()
        _ACTIVE_GRAPH_VARIANT_KEY = None


def _apply_tile_scores_to_graph_native(
    graph: RoadGraph,
    score_map: Mapping[tuple[int, int, int], float],
    *,
    zoom: int,
    fallback: float | None,
) -> tuple[int, int]:
    """Materialize tile scores directly on one private graph variant.

    Callers build a native-edge clone before entering this helper and publish
    it only after this pass completes.  Published variants are never scored
    again, so requests with different reports cannot observe one another's
    edge mutations.
    """


    matched = 0
    total = 0
    cache_limit = 4096
    midpoint_tiles: OrderedDict[tuple[float, float], tuple[int, int]] = OrderedDict()
    tile_results: OrderedDict[tuple[int, int, int], object] = OrderedDict()
    cache_miss = object()

    for edge in graph.edges.values():
        total += 1
        start = graph.get_node(edge.start_node_id)
        end = graph.get_node(edge.end_node_id)
        midpoint_key = (
            0.5 * (start.lat + end.lat),
            0.5 * (start.lon + end.lon),
        )
        tile = midpoint_tiles.get(midpoint_key)
        if tile is None:
            tile = lat_lon_to_tile(*midpoint_key, zoom)
            if len(midpoint_tiles) >= cache_limit:
                midpoint_tiles.popitem(last=False)
            midpoint_tiles[midpoint_key] = tile
        tile_key = (zoom, tile[0], tile[1])
        tile_score = tile_results.get(tile_key, cache_miss)
        if tile_score is cache_miss:
            tile_score = score_map.get(tile_key)
            if len(tile_results) >= cache_limit:
                tile_results.popitem(last=False)
            tile_results[tile_key] = tile_score

        if tile_score is not None:
            matched += 1
            score = float(min(max(tile_score, 0.0), 10.0))
        elif fallback is not None:
            score = float(min(max(fallback, 0.0), 10.0))
        else:
            continue
        if score != float(edge.scenic_score):
            edge.scenic_score = score
    return matched, total


def _clone_graph_for_scoring(graph: RoadGraph) -> RoadGraph:
    """Copy nodes and native ``Edge`` objects for an isolated score variant."""

    variant = RoadGraph()
    for node in graph.nodes.values():
        variant.add_node(copy(node))
    for edge in graph.edges.values():
        variant.add_edge(copy(edge))
    return variant


def _clear_scored_variant_locked() -> None:
    """Drop the sole scored variant while retaining no stale graph alias."""

    global _ACTIVE_GRAPH_VARIANT_KEY
    if _ACTIVE_GRAPH_VARIANT_KEY is not None:
        _GRAPH_CACHE.pop(_ACTIVE_GRAPH_VARIANT_KEY[0], None)
    _SCORED_GRAPH_CACHE.clear()
    _ACTIVE_GRAPH_VARIANT_KEY = None


def _load_cached_graph(
    path: Path,
    *,
    scored_cache_key: _ScoredGraphCacheKey | None = None,
) -> tuple[RoadGraph, str, _FileSignature, bool]:
    global _ACTIVE_GRAPH_VARIANT_KEY

    path_key = _resolved_path_key(path)
    signature = _file_signature(path)
    cache_key = (path_key, signature)
    with _CACHE_LOCK:
        if scored_cache_key is not None:
            scored = _SCORED_GRAPH_CACHE.get(scored_cache_key)
            if scored is not None and scored_cache_key[0] == cache_key:
                _GRAPH_CACHE[cache_key] = scored
                _GRAPH_CACHE.move_to_end(cache_key)
                _SCORED_GRAPH_CACHE.move_to_end(scored_cache_key)
                while len(_GRAPH_CACHE) > _CACHE_CAPACITY:
                    _GRAPH_CACHE.popitem(last=False)
                _ACTIVE_GRAPH_VARIANT_KEY = scored_cache_key
                return scored, path_key, signature, True

        cached = _GRAPH_CACHE.get(cache_key)
        if cached is not None and _ACTIVE_GRAPH_VARIANT_KEY is None:
            _GRAPH_CACHE.move_to_end(cache_key)
            return cached, path_key, signature, True

        # A native scored variant cannot satisfy an unscored request or a
        # different tile/signature variant.  Release it before reparsing.
        if cached is not None or _ACTIVE_GRAPH_VARIANT_KEY is not None:
            _clear_scored_variant_locked()

        # A changed file supersedes all prior graph objects for this path.
        for stale_key in tuple(_GRAPH_CACHE):
            if stale_key[0] == path_key:
                _GRAPH_CACHE.pop(stale_key, None)
        _clear_scored_variant_locked()
        graph = _load_graph(path)
        final_signature = _file_signature(path)
        final_key = (path_key, final_signature)
        _GRAPH_CACHE[final_key] = graph
        _GRAPH_CACHE.move_to_end(final_key)
        while len(_GRAPH_CACHE) > _CACHE_CAPACITY:
            _GRAPH_CACHE.popitem(last=False)
        return graph, path_key, final_signature, False


def load_tile_scores(path: Path) -> tuple[dict[tuple[int, int, int], float], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    score_map: dict[tuple[int, int, int], float] = {}
    zoom_counts: dict[int, int] = {}
    for tile in payload.get("tiles", []):
        x, y, z, scenic = (
            tile.get("x"),
            tile.get("y"),
            tile.get("z"),
            tile.get("scenic_score"),
        )
        if x is None or y is None or z is None or scenic is None:
            continue
        z_i = int(z)
        score_map[(z_i, int(x), int(y))] = float(scenic)
        zoom_counts[z_i] = zoom_counts.get(z_i, 0) + 1
    if not score_map:
        raise ValueError(f"No tile scores with x/y/z found in: {path}")
    inferred_zoom = max(zoom_counts.items(), key=lambda item: item[1])[0]
    return score_map, inferred_zoom


def _load_cached_tile_scores(
    path: Path,
) -> tuple[Mapping[tuple[int, int, int], float], int, str, _FileSignature, bool]:
    path_key = _resolved_path_key(path)
    signature = _file_signature(path)
    cache_key = (path_key, signature)
    with _CACHE_LOCK:
        cached = _TILE_SCORE_CACHE.get(cache_key)
        if cached is not None:
            _TILE_SCORE_CACHE.move_to_end(cache_key)
            score_map, inferred_zoom = cached
            return score_map, inferred_zoom, path_key, signature, True

        for stale_key in tuple(_TILE_SCORE_CACHE):
            if stale_key[0] == path_key:
                _TILE_SCORE_CACHE.pop(stale_key, None)
        _clear_scored_variant_locked()
        score_map, inferred_zoom = load_tile_scores(path)
        immutable_map = MappingProxyType(dict(score_map))
        final_signature = _file_signature(path)
        final_key = (path_key, final_signature)
        _TILE_SCORE_CACHE[final_key] = (immutable_map, inferred_zoom)
        _TILE_SCORE_CACHE.move_to_end(final_key)
        while len(_TILE_SCORE_CACHE) > _TILE_CACHE_CAPACITY:
            _TILE_SCORE_CACHE.popitem(last=False)
        return immutable_map, inferred_zoom, path_key, final_signature, False


def _get_scored_graph(
    graph: RoadGraph,
    *,
    graph_key: _GraphCacheKey,
    tile_key: _GraphCacheKey,
    score_map: Mapping[tuple[int, int, int], float],
    zoom: int,
    fallback: float | None,
) -> tuple[RoadGraph, int, int, bool]:
    """Atomically get or build one immutable native-edge score variant.

    The cache lock covers the private clone, score materialization, and
    publication.  A same-key caller therefore observes a completed variant
    instead of starting a duplicate build, while a different key receives a
    separate clone and can never mutate a graph already used by another
    request.
    """

    global _ACTIVE_GRAPH_VARIANT_KEY
    cache_key = (
        graph_key,
        tile_key,
        int(zoom),
        fallback,
        _NORMALIZATION_VERSION,
    )
    with _CACHE_LOCK:
        cached = _SCORED_GRAPH_CACHE.get(cache_key)
        if cached is not None:
            _SCORED_GRAPH_CACHE.move_to_end(cache_key)
            _GRAPH_CACHE[graph_key] = cached
            _GRAPH_CACHE.move_to_end(graph_key)
            matched, total, fallback_edges = getattr(
                cached,
                "_route_service_score_mapping",
                (0, len(cached.edges), 0),
            )
            object.__setattr__(
                cached,
                "_route_service_score_mapping",
                (int(matched), int(total), int(fallback_edges)),
            )
            return cached, int(matched), int(total), True

        if (
            _ACTIVE_GRAPH_VARIANT_KEY is not None
            and _ACTIVE_GRAPH_VARIANT_KEY != cache_key
        ):
            _clear_scored_variant_locked()

        # ``graph`` is the canonical raw graph (or a private raw reference
        # fetched before this lock).  Never score it: callers may still be
        # planning against that object while this variant is built.
        scored_graph = _clone_graph_for_scoring(graph)
        matched, total = _apply_tile_scores_to_graph_native(
            scored_graph,
            score_map,
            zoom=int(zoom),
            fallback=fallback,
        )
        fallback_edges = int(total - matched) if fallback is not None else 0
        object.__setattr__(
            scored_graph,
            "_route_service_score_mapping",
            (int(matched), int(total), fallback_edges),
        )
        _GRAPH_CACHE[graph_key] = scored_graph
        _GRAPH_CACHE.move_to_end(graph_key)
        _SCORED_GRAPH_CACHE[cache_key] = scored_graph
        _SCORED_GRAPH_CACHE.move_to_end(cache_key)
        while len(_GRAPH_CACHE) > _CACHE_CAPACITY:
            _GRAPH_CACHE.popitem(last=False)
        while len(_SCORED_GRAPH_CACHE) > _SCORED_GRAPH_CACHE_CAPACITY:
            _SCORED_GRAPH_CACHE.popitem(last=False)
        _ACTIVE_GRAPH_VARIANT_KEY = cache_key
        return scored_graph, matched, total, False


def preload_route_assets(
    graph_path: str | Path,
    tile_scores_path: str | Path | None = None,
    tile_score_zoom: int | None = None,
    tile_score_fallback: float | None = None,
) -> dict[str, Any]:
    """Load and materialize one route graph without planning a route.

    Startup uses this explicit hook to pay the graph, tile, and scored-view
    materialization cost before the API accepts requests.  The returned
    diagnostics describe cache state and score coverage; no synthetic
    endpoints or planner invocation are needed.
    """

    started_at = perf_counter()
    graph_file = Path(graph_path)
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph asset not found: {graph_file}")

    tile_score_cache_hit = False
    scored_graph_cache_hit = False
    score_mapping = {
        "enabled": False,
        "source": None,
        "score_run": None,
        "report_signature": None,
        "graph_signature": None,
        "normalization": _NORMALIZATION_VERSION,
        "zoom": None,
        "matched_edges": 0,
        "fallback_edges": 0,
        "total_edges": 0,
        "matched_ratio": 0.0,
    }
    tile_context: tuple[
        Mapping[tuple[int, int, int], float],
        str,
        _FileSignature,
        int,
        float | None,
    ] | None = None

    if tile_scores_path is not None:
        tile_file = Path(tile_scores_path)
        if not tile_file.exists():
            raise FileNotFoundError(f"Tile score asset not found: {tile_file}")
        (
            score_map,
            inferred_zoom,
            tile_path_key,
            tile_signature,
            tile_score_cache_hit,
        ) = _load_cached_tile_scores(tile_file)
        zoom = (
            int(tile_score_zoom)
            if tile_score_zoom is not None
            else int(inferred_zoom)
        )
        tile_context = (
            score_map,
            tile_path_key,
            tile_signature,
            zoom,
            tile_score_fallback,
        )

    graph_path_key = _resolved_path_key(graph_file)
    graph_signature_hint = _file_signature(graph_file)
    scored_cache_key = (
        (
            (
                graph_path_key,
                graph_signature_hint,
            ),
            tile_context[1:3]
            if tile_context is not None
            else (graph_path_key, graph_signature_hint),
            tile_context[3] if tile_context is not None else 0,
            tile_context[4] if tile_context is not None else None,
            _NORMALIZATION_VERSION,
        )
        if tile_context is not None
        else None
    )
    graph, graph_path_key, graph_signature, graph_cache_hit = _load_cached_graph(
        graph_file,
        scored_cache_key=scored_cache_key,
    )
    if not graph.nodes or not graph.edges:
        raise ValueError(f"Graph asset has no usable nodes/edges: {graph_file}")

    matched = 0
    total = len(graph.edges)
    if tile_context is not None:
        (
            score_map,
            tile_path_key,
            tile_signature,
            zoom,
            fallback,
        ) = tile_context
        (
            graph,
            matched,
            total,
            scored_graph_cache_hit,
        ) = _get_scored_graph(
            graph,
            graph_key=(graph_path_key, graph_signature),
            tile_key=(tile_path_key, tile_signature),
            score_map=score_map,
            zoom=zoom,
            fallback=fallback,
        )
        mapping_meta = getattr(
            graph,
            "_route_service_score_mapping",
            (int(matched), int(total), 0),
        )
        fallback_edges = int(mapping_meta[2])
        score_mapping = {
            "enabled": True,
            "source": str(tile_scores_path),
            "score_run": _score_run_identity(Path(tile_scores_path)),
            "report_signature": _signature_digest(tile_path_key, tile_signature),
            "graph_signature": _signature_digest(graph_path_key, graph_signature),
            "normalization": _NORMALIZATION_VERSION,
            "zoom": int(zoom),
            "matched_edges": int(matched),
            "fallback_edges": fallback_edges,
            "total_edges": int(total),
            "matched_ratio": float(matched / max(total, 1)),
        }

    planner_preload: dict[str, Any] = {}
    frontier_time_limit_seconds = _frontier_time_limit_from_env()
    if frontier_time_limit_seconds is None:
        planner = ScenicRoutePlanner(graph=graph)
    else:
        planner = ScenicRoutePlanner(
            graph=graph,
            frontier_time_limit_seconds=frontier_time_limit_seconds,
        )
    prewarm = getattr(planner, "prewarm_routing_cache", None)
    if callable(prewarm):
        prewarm_result = prewarm()
        if isinstance(prewarm_result, Mapping):
            planner_preload = dict(prewarm_result)

    return {
        "graph_path": str(graph_file),
        "tile_scores_path": str(tile_scores_path)
        if tile_scores_path is not None
        else None,
        "graph_nodes": int(len(graph.nodes)),
        "graph_cache_hit": bool(graph_cache_hit),
        "tile_score_cache_hit": bool(tile_score_cache_hit),
        "scored_graph_cache_hit": bool(scored_graph_cache_hit),
        "score_mapping": score_mapping,
        "planner_preload": planner_preload,
        "preload_elapsed_ms": (perf_counter() - started_at) * 1000.0,
    }

def _normalized_score(raw_score: float) -> float:
    """Use the canonical immutable ``linear-v1`` score normalization."""
    return float(normalize_scenic_score(raw_score))


def _segment_identity(segment: Any, index: int) -> str:
    edge_id = getattr(segment, "edge_id", None)
    if edge_id is None:
        edge_id = getattr(segment, "id", None)
    if edge_id is None:
        return f"segment:{index}"
    if str(edge_id) == "":
        raise ValueError("Route segment edge_id must not be empty")
    return str(edge_id)


def _route_highway_count(route: Route) -> int:
    declared = getattr(route, "highway_count", None)
    if declared is not None:
        return int(declared)
    highway_names = HIGHWAY_ROAD_TYPES
    return sum(
        1
        for segment in route.segments
        if str(getattr(segment, "road_type", "")).lower() in highway_names
    )


def route_to_feature(
    route: Route,
    route_kind: str,
    *,
    objective: Mapping[str, Any] | None = None,
    score_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coords = [[lon, lat] for lat, lon in route.waypoints]
    raw_score = float(route.average_scenic_score)
    normalized_score = float(
        getattr(route, "scenic_score_normalized", _normalized_score(raw_score))
    )
    route_edge_ids = getattr(route, "edge_ids", None)
    identities = list(route_edge_ids) if route_edge_ids else [
        _segment_identity(segment, index)
        for index, segment in enumerate(route.segments)
    ]
    route_traversal_ids = getattr(route, "traversal_ids", None)
    traversal_ids = list(route_traversal_ids) if route_traversal_ids else [
        str(getattr(segment, "traversal_id", identities[index]))
        for index, segment in enumerate(route.segments)
    ]
    segment_rows = [
        {
            "edge_id": identities[index],
            "traversal_id": traversal_ids[index],
            "direction": getattr(segment, "direction", None),
            "start": [float(segment.start[0]), float(segment.start[1])],
            "end": [float(segment.end[0]), float(segment.end[1])],
            "distance_km": float(segment.distance_km),
            "duration_minutes": float(segment.duration_minutes),
            "scenic_score": float(segment.scenic_score),
            "normalized_scenic_score": _normalized_score(segment.scenic_score),
            "road_name": segment.road_name,
            "road_type": segment.road_type,
        }
        for index, segment in enumerate(route.segments)
    ]
    objective_values = dict(objective or {})
    properties: dict[str, Any] = {
        "route_kind": route_kind,
        "segments": len(route.segments),
        "edge_ids": identities,
        "segment_identity": segment_rows,
        "traversal_ids": traversal_ids,
        "total_distance_km": float(route.total_distance_km),
        "average_scenic_score": raw_score,
        "raw_scenic_score": objective_values.get("raw_scenic_score", raw_score),
        "normalized_scenic_score": objective_values.get(
            "normalized_scenic_score", normalized_score
        ),
        "requested_scenic_weight": objective_values.get(
            "requested_scenic_weight", None
        ),
        "applied_scenic_weight": objective_values.get(
            "applied_scenic_weight", None
        ),
        "estimated_duration_minutes": float(route.estimated_duration_minutes),
        "highway_count": _route_highway_count(route),
        "requested_max_detour_factor": objective_values.get(
            "requested_max_detour_factor",
            getattr(route, "requested_max_detour_factor", None),
        ),
        "applied_max_detour_factor": objective_values.get(
            "applied_max_detour_factor",
            getattr(route, "applied_max_detour_factor", None),
        ),
        "actual_duration_ratio": objective_values.get(
            "actual_duration_ratio", getattr(route, "actual_duration_ratio", None)
        ),
        "duration_utility": objective_values.get(
            "duration_utility", getattr(route, "duration_utility", None)
        ),
        "score_coverage": getattr(route, "score_coverage", None),
        "algorithm": getattr(route, "algorithm", None),
        "fastest_duration_minutes": getattr(
            route, "fastest_duration_minutes", None
        ),
        "duration_cap_minutes": getattr(route, "duration_cap_minutes", None),
        "certified_upper_bound": getattr(route, "certified_upper_bound", None),
        "normalization_version": getattr(
            route, "normalization_version", _NORMALIZATION_VERSION
        ),
        "optimization_mode": getattr(
            route, "optimization_mode", "distance_weighted_scenic"
        ),
        "optimization_status": getattr(route, "status", "ok"),
        "exactness_status": getattr(route, "exactness_status", "unknown"),
        "optimality_gap": getattr(route, "optimality_gap", None),
        "status": getattr(route, "status", "ok"),
        "objective_value": objective_values.get(
            "objective_value", getattr(route, "objective_value", None)
        ),
        "objective": objective_values.get(
            "objective_value", getattr(route, "objective_value", None)
        ),
        "scenic_score_delta_absolute": objective_values.get(
            "scenic_score_delta_absolute"
        ),
        "scenic_score_delta_relative": objective_values.get(
            "scenic_score_delta_relative"
        ),
        "same_route": objective_values.get("same_route"),
        "no_better_route_reason": objective_values.get(
            "no_better_route_reason",
            getattr(route, "zero_improvement_reason", None),
        ),
        "zero_improvement_reason": getattr(route, "zero_improvement_reason", None),
        "no_route_reason": getattr(route, "no_route_reason", None),
        "score_run": getattr(route, "score_run", None),
        "search_diagnostics": _normalize_search_diagnostics(
            getattr(route, "search_diagnostics", None)
        ),
    }
    if objective is not None:
        properties["objective_components"] = dict(objective)
    if score_provenance is not None:
        properties["score_provenance"] = dict(score_provenance)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return x, y




def apply_tile_scores_to_graph(
    graph: RoadGraph,
    score_map: Mapping[tuple[int, int, int], float],
    *,
    zoom: int,
    fallback: float | None,
) -> tuple[int, int]:
    """Apply tile scores in-place for direct callers.

    The service request path uses the same native materialization routine on
    its exclusively owned graph variant.  This helper intentionally retains
    the historical mutating behavior for callers that opt into it.
    """

    return _apply_tile_scores_to_graph_native(
        graph,
        score_map,
        zoom=zoom,
        fallback=fallback,
    )


def _load_graph(path: Path) -> RoadGraph:
    # Support both FeatureCollection road graphs (.geojson) and serialized RoadGraph JSONs.
    if path.suffix.lower() == ".geojson":
        return RoadGraph.from_geojson(path)
    return RoadGraph.load(path)


def diagnose_route_request(request: RouteRequest) -> dict[str, Any]:
    graph_path = Path(request.graph_geojson)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph GeoJSON not found: {graph_path}")

    graph = _load_cached_graph(graph_path)[0]

    start_node, start_snap_km = graph.find_nearest_node_with_distance(*request.start)
    end_node, end_snap_km = graph.find_nearest_node_with_distance(*request.end)
    return {
        "graph_nodes": int(len(graph.nodes)),
        "graph_edges": int(len(graph.edges)),
        "start_snap_km": float(start_snap_km),
        "end_snap_km": float(end_snap_km),
        "start_node_id": start_node.id,
        "end_node_id": end_node.id,
    }


def _objective_components(
    request: RouteRequest,
    scenic_route: Route,
    baseline_route: Route | None,
) -> dict[str, Any]:
    scenic_duration = float(scenic_route.estimated_duration_minutes)
    scenic_raw = float(scenic_route.average_scenic_score)
    fastest_duration = (
        float(baseline_route.estimated_duration_minutes)
        if baseline_route is not None
        else float(
            getattr(
                scenic_route,
                "fastest_duration_minutes",
                scenic_duration,
            )
        )
    )
    ratio = (
        scenic_duration / fastest_duration
        if fastest_duration > 0.0
        else (1.0 if scenic_duration == 0.0 else float("inf"))
    )
    kappa = float(request.max_detour_factor)
    duration_utility = float(
        duration_component(scenic_duration, fastest_duration, kappa)
    )
    normalized = _normalized_score(scenic_raw)
    # The route planner already enforces the hard cap and selects the highest
    # normalized scenic score. Keep the response objective aligned with that
    # primary metric instead of reintroducing a duration trade-off here.
    objective = normalized
    baseline_raw = (
        float(baseline_route.average_scenic_score)
        if baseline_route is not None
        else None
    )
    absolute_delta = scenic_raw - baseline_raw if baseline_raw is not None else None
    relative_delta = (
        absolute_delta / abs(baseline_raw)
        if absolute_delta is not None and baseline_raw != 0.0
        else None
    )
    scenic_edges = tuple(
        _segment_identity(segment, i)
        for i, segment in enumerate(scenic_route.segments)
    )
    baseline_edges = (
        tuple(
            _segment_identity(segment, i)
            for i, segment in enumerate(baseline_route.segments)
        )
        if baseline_route is not None
        else None
    )
    same_route = baseline_edges is not None and scenic_edges == baseline_edges
    no_better_reason = None
    certified = bool(getattr(scenic_route, "exact", False)) or (
        getattr(scenic_route, "optimality_gap", None) == 0.0
    )
    if same_route:
        no_better_reason = (
            "same_route"
            if certified
            else "approximation_did_not_find_scenic_improvement"
        )
    elif (
        baseline_route is not None
        and absolute_delta is not None
        and absolute_delta <= 0.0
    ):
        no_better_reason = (
            "no_better_route"
            if certified
            else "approximation_did_not_find_scenic_improvement"
        )
    return {
        "duration_utility": float(duration_utility),
        "scenic_utility": float(normalized),
        "objective_value": float(objective),
        "raw_scenic_score": scenic_raw,
        "normalized_scenic_score": normalized,
        "requested_scenic_weight": float(request.scenic_weight),
        "applied_scenic_weight": float(request.scenic_weight),
        "requested_max_detour_factor": kappa,
        "applied_max_detour_factor": kappa,
        "actual_duration_ratio": ratio,
        "scenic_score_delta_absolute": absolute_delta,
        "scenic_score_delta_relative": relative_delta,
        "same_route": same_route,
        "no_better_route_reason": no_better_reason,
    }


def plan_routes(request: RouteRequest) -> dict[str, Any]:
    started_at = perf_counter()
    graph_path = Path(request.graph_geojson)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph GeoJSON not found: {graph_path}")

    tile_score_cache_hit = False
    scored_graph_cache_hit = False
    tile_score_load_elapsed_ms = 0.0
    score_application_elapsed_ms = 0.0
    score_mapping = {
        "enabled": False,
        "source": None,
        "score_run": None,
        "report_signature": None,
        "graph_signature": None,
        "normalization": _NORMALIZATION_VERSION,
        "zoom": None,
        "matched_edges": 0,
        "fallback_edges": 0,
        "total_edges": 0,
        "matched_ratio": 0.0,
    }

    tile_context: tuple[
        Mapping[tuple[int, int, int], float],
        str,
        _FileSignature,
        int,
        float | None,
    ] | None = None
    if request.tile_scores_json:
        score_path = Path(request.tile_scores_json)
        tile_load_started = perf_counter()
        (
            score_map,
            inferred_zoom,
            tile_path_key,
            tile_signature,
            tile_score_cache_hit,
        ) = _load_cached_tile_scores(score_path)
        tile_score_load_elapsed_ms = (perf_counter() - tile_load_started) * 1000.0
        zoom = (
            request.tile_score_zoom
            if request.tile_score_zoom is not None
            else inferred_zoom
        )
        tile_context = (
            score_map,
            tile_path_key,
            tile_signature,
            int(zoom),
            request.tile_score_fallback,
        )

    graph_load_started = perf_counter()
    graph_path_key = _resolved_path_key(graph_path)
    graph_signature_hint = _file_signature(graph_path)
    scored_cache_key = (
        (
            (graph_path_key, graph_signature_hint),
            tile_context[1:3]
            if tile_context is not None
            else (graph_path_key, graph_signature_hint),
            tile_context[3] if tile_context is not None else 0,
            tile_context[4] if tile_context is not None else None,
            _NORMALIZATION_VERSION,
        )
        if tile_context is not None
        else None
    )
    graph, graph_path_key, graph_signature, graph_cache_hit = _load_cached_graph(
        graph_path,
        scored_cache_key=scored_cache_key,
    )
    graph_load_elapsed_ms = (perf_counter() - graph_load_started) * 1000.0
    score_mapping["graph_signature"] = _signature_digest(
        graph_path_key, graph_signature
    )

    if tile_context is not None:
        score_map, tile_path_key, tile_signature, zoom, fallback = tile_context
        score_application_started = perf_counter()
        (
            graph,
            matched,
            total,
            scored_graph_cache_hit,
        ) = _get_scored_graph(
            graph,
            graph_key=(graph_path_key, graph_signature),
            tile_key=(tile_path_key, tile_signature),
            score_map=score_map,
            zoom=zoom,
            fallback=fallback,
        )
        score_application_elapsed_ms = (
            perf_counter() - score_application_started
        ) * 1000.0
        mapping_meta = getattr(
            graph,
            "_route_service_score_mapping",
            (int(matched), int(total), 0),
        )
        score_mapping = {
            "enabled": True,
            "source": str(request.tile_scores_json),
            "score_run": _score_run_identity(Path(request.tile_scores_json)),
            "report_signature": _signature_digest(tile_path_key, tile_signature),
            "graph_signature": _signature_digest(graph_path_key, graph_signature),
            "normalization": _NORMALIZATION_VERSION,
            "zoom": int(zoom),
            "matched_edges": int(matched),
            "fallback_edges": int(mapping_meta[2]),
            "total_edges": int(total),
            "matched_ratio": float(matched / max(total, 1)),
        }

    start_node, start_snap_km = graph.find_nearest_node_with_distance(*request.start)
    end_node, end_snap_km = graph.find_nearest_node_with_distance(*request.end)
    diagnostics = {
        "graph_nodes": int(len(graph.nodes)),
        "graph_edges": int(len(graph.edges)),
        "start_snap_km": float(start_snap_km),
        "end_snap_km": float(end_snap_km),
        "start_node_id": start_node.id,
        "end_node_id": end_node.id,
        "requested_max_detour_factor": float(request.max_detour_factor),
        "requested_scenic_weight": float(request.scenic_weight),
        "applied_scenic_weight": float(request.scenic_weight),
        "applied_max_detour_factor": float(request.max_detour_factor),
        "avoid_highways_applied": bool(request.avoid_highways),
        "graph_cache_hit": bool(graph_cache_hit),
        "graph_load_elapsed_ms": graph_load_elapsed_ms,
        "score_mapping_coverage": float(score_mapping["matched_ratio"]),
        "score_mapping_fallback_edges": int(score_mapping["fallback_edges"]),
        "score_report_identity": score_mapping["score_run"],
        "score_report_signature": score_mapping["report_signature"],
        "tile_score_cache_hit": bool(tile_score_cache_hit),
        "scored_graph_cache_hit": bool(scored_graph_cache_hit),
        "tile_score_load_elapsed_ms": tile_score_load_elapsed_ms,
        "score_application_elapsed_ms": score_application_elapsed_ms,
    }

    frontier_time_limit_seconds = _frontier_time_limit_from_env()
    if frontier_time_limit_seconds is None:
        planner = ScenicRoutePlanner(graph=graph)
    else:
        planner = ScenicRoutePlanner(
            graph=graph,
            frontier_time_limit_seconds=frontier_time_limit_seconds,
        )
    scenic_route = planner.find_scenic_route(
        start=request.start,
        end=request.end,
        scenic_weight=request.scenic_weight,
        avoid_highways=request.avoid_highways,
        max_detour_factor=request.max_detour_factor,
        scenic_priority=True,
    )

    baseline_route: Route | None = None
    if request.include_baseline:
        # Baseline uses the identical hard highway eligibility filter.
        baseline_route = planner.find_fastest_route(
            start=request.start,
            end=request.end,
            avoid_highways=request.avoid_highways,
        )

    objective = _objective_components(request, scenic_route, baseline_route)
    scenic_feature = route_to_feature(
        scenic_route,
        "scenic",
        objective=objective,
        score_provenance=score_mapping,
    )
    features = [scenic_feature]
    routes = [{"route_kind": "scenic", "metrics": scenic_feature["properties"]}]

    if baseline_route is not None:
        baseline_raw = float(baseline_route.average_scenic_score)
        baseline_normalized = _normalized_score(baseline_raw)
        baseline_objective = {
            "duration_utility": 1.0,
            "scenic_utility": baseline_normalized,
            "objective_value": baseline_normalized,
            "raw_scenic_score": baseline_raw,
            "normalized_scenic_score": baseline_normalized,
            "requested_scenic_weight": float(request.scenic_weight),
            "applied_scenic_weight": float(request.scenic_weight),
            "requested_max_detour_factor": float(request.max_detour_factor),
            "applied_max_detour_factor": float(request.max_detour_factor),
            "actual_duration_ratio": 1.0,
            "scenic_score_delta_absolute": None,
            "scenic_score_delta_relative": None,
            "same_route": None,
            "no_better_route_reason": None,
        }
        baseline_feature = route_to_feature(
            baseline_route,
            "baseline",
            objective=baseline_objective,
            score_provenance=score_mapping,
        )
        features.append(baseline_feature)
        routes.append(
            {"route_kind": "baseline", "metrics": baseline_feature["properties"]}
        )

    fastest_duration = (
        float(baseline_route.estimated_duration_minutes)
        if baseline_route is not None
        else None
    )
    scenic_duration = float(scenic_route.estimated_duration_minutes)
    fastest_distance = (
        float(baseline_route.total_distance_km)
        if baseline_route is not None
        else None
    )
    scenic_distance = float(scenic_route.total_distance_km)
    duration_cap = (
        float(getattr(scenic_route, "duration_cap_minutes", 0.0) or 0.0)
        if baseline_route is not None
        else 0.0
    )
    if baseline_route is not None and duration_cap <= 0.0 and fastest_duration is not None:
        duration_cap = fastest_duration * request.max_detour_factor
    duration_cap_tolerance = 1e-12 * max(1.0, abs(duration_cap))
    diagnostics.update(
        {
            "scenic_fastest_duration_ratio": objective["actual_duration_ratio"],
            "scenic_fastest_distance_ratio": (
                scenic_distance / fastest_distance
                if fastest_distance is not None and fastest_distance > 0.0
                else None
            ),
            "duration_cap_satisfied": (
                fastest_duration is None
                or fastest_duration <= 0.0
                or scenic_duration <= duration_cap + duration_cap_tolerance
            )
            if baseline_route is not None
            else None,
            "optimization_mode": getattr(
                scenic_route, "optimization_mode", "distance_weighted_scenic"
            ),
            "optimization_status": getattr(
                scenic_route, "status", "ok" if baseline_route is not None else "uncertified"
            ),
            "optimality_gap": getattr(scenic_route, "optimality_gap", None),
            "certified_upper_bound": getattr(
                scenic_route, "certified_upper_bound", None
            ),
            "normalized_scenic_score": objective["normalized_scenic_score"],
            "scenic_score_delta_absolute": objective["scenic_score_delta_absolute"],
            "scenic_score_delta_relative": objective["scenic_score_delta_relative"],
            "same_route": objective["same_route"],
            "no_better_route_reason": objective["no_better_route_reason"],
            "hard_highway_count": _route_highway_count(scenic_route),
            "exactness_status": getattr(
                scenic_route, "exactness_status", "unknown"
            ),
        }
    )
    diagnostics["search_diagnostics"] = _normalize_search_diagnostics(
        getattr(scenic_route, "search_diagnostics", None)
    )
    diagnostics["planning_elapsed_ms"] = (perf_counter() - started_at) * 1000.0
    return {
        "request": request.to_dict(),
        "diagnostics": diagnostics,
        "score_mapping": score_mapping,
        "routes": routes,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
