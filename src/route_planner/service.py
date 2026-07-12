from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import MappingProxyType
from typing import Any

from .graph import RoadGraph
from .planner import Route, ScenicRoutePlanner


@dataclass
class RouteRequest:
    graph_geojson: str
    start: tuple[float, float]  # (lat, lon)
    end: tuple[float, float]  # (lat, lon)
    scenic_weight: float = 0.6
    avoid_highways: bool = False
    max_detour_factor: float = 1.8
    include_baseline: bool = True
    tile_scores_json: str | None = None
    tile_score_zoom: int | None = None
    tile_score_fallback: float | None = None

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
            scenic_weight=float(payload.get("scenic_weight", 0.6)),
            avoid_highways=bool(payload.get("avoid_highways", False)),
            max_detour_factor=float(payload.get("max_detour_factor", 1.8)),
            include_baseline=bool(payload.get("include_baseline", True)),
            tile_scores_json=str(payload["tile_scores_json"])
            if payload.get("tile_scores_json")
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


_FileSignature = tuple[int, int, int, int, int]
_GraphCacheKey = tuple[str, _FileSignature]
_TileCacheValue = tuple[Mapping[tuple[int, int, int], float], int]

_CACHE_CAPACITY = 1
_TILE_CACHE_CAPACITY = 1
_SCORED_GRAPH_CACHE_CAPACITY = 1
_CACHE_LOCK = RLock()
# ``_GRAPH_CACHE`` retains either the canonical raw graph or the active
# scored variant, never both.  A scored miss briefly holds the raw graph,
# its private native-edge clone, and (while replacing it) the previous
# scored variant.  Only one graph variant is retained after publication;
# requests already using an evicted variant keep their own reference.
_GRAPH_CACHE: OrderedDict[_GraphCacheKey, RoadGraph] = OrderedDict()
_TILE_SCORE_CACHE: OrderedDict[_GraphCacheKey, _TileCacheValue] = OrderedDict()
_ScoredGraphCacheKey = tuple[
    _GraphCacheKey, _GraphCacheKey, int, float | None
]
_SCORED_GRAPH_CACHE: OrderedDict[
    _ScoredGraphCacheKey, RoadGraph
] = OrderedDict()
_ACTIVE_GRAPH_VARIANT_KEY: _ScoredGraphCacheKey | None = None


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
    cache_key = (graph_key, tile_key, int(zoom), fallback)
    with _CACHE_LOCK:
        cached = _SCORED_GRAPH_CACHE.get(cache_key)
        if cached is not None:
            _SCORED_GRAPH_CACHE.move_to_end(cache_key)
            _GRAPH_CACHE[graph_key] = cached
            _GRAPH_CACHE.move_to_end(graph_key)
            _ACTIVE_GRAPH_VARIANT_KEY = cache_key
            matched, total = getattr(
                cached, "_route_service_score_mapping", (0, len(cached.edges))
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
        object.__setattr__(
            scored_graph,
            "_route_service_score_mapping",
            (matched, total),
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
        "zoom": None,
        "matched_edges": 0,
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
            graph_path_key,
            graph_signature_hint,
        ),
        tile_context[1:3]
        if tile_context is not None
        else (graph_path_key, graph_signature_hint),
        tile_context[3] if tile_context is not None else 0,
        tile_context[4] if tile_context is not None else None,
    ) if tile_context is not None else None
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
        score_mapping = {
            "enabled": True,
            "source": str(tile_scores_path),
            "zoom": int(zoom),
            "matched_edges": int(matched),
            "total_edges": int(total),
            "matched_ratio": float(matched / max(total, 1)),
        }

    planner_preload: dict[str, Any] = {}
    planner = ScenicRoutePlanner(graph=graph)
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
        "graph_edges": int(len(graph.edges)),
        "graph_cache_hit": bool(graph_cache_hit),
        "tile_score_cache_hit": bool(tile_score_cache_hit),
        "scored_graph_cache_hit": bool(scored_graph_cache_hit),
        "score_mapping": score_mapping,
        "planner_preload": planner_preload,
        "preload_elapsed_ms": (perf_counter() - started_at) * 1000.0,
    }


def _parse_point(value: Any, *, field_name: str) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if isinstance(value, dict) and "lat" in value and "lon" in value:
        return (float(value["lat"]), float(value["lon"]))
    raise ValueError(
        f"Invalid {field_name} format. Expected [lat, lon] or {{lat, lon}}."
    )


def route_to_feature(route: Route, route_kind: str) -> dict[str, Any]:
    coords = [[lon, lat] for lat, lon in route.waypoints]
    return {
        "type": "Feature",
        "properties": {
            "route_kind": route_kind,
            "segments": len(route.segments),
            "total_distance_km": route.total_distance_km,
            "average_scenic_score": route.average_scenic_score,
            "estimated_duration_minutes": route.estimated_duration_minutes,
        },
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


def load_tile_scores(path: Path) -> tuple[dict[tuple[int, int, int], float], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tiles = payload.get("tiles", [])
    score_map: dict[tuple[int, int, int], float] = {}
    zoom_counts: dict[int, int] = {}

    for tile in tiles:
        x = tile.get("x")
        y = tile.get("y")
        z = tile.get("z")
        scenic = tile.get("scenic_score")
        if x is None or y is None or z is None or scenic is None:
            continue
        z_i = int(z)
        key = (z_i, int(x), int(y))
        score_map[key] = float(scenic)
        zoom_counts[z_i] = zoom_counts.get(z_i, 0) + 1

    if not score_map:
        raise ValueError(f"No tile scores with x/y/z found in: {path}")

    inferred_zoom = max(zoom_counts.items(), key=lambda kv: kv[1])[0]
    return score_map, inferred_zoom




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
        "zoom": None,
        "matched_edges": 0,
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
        (graph_path_key, graph_signature_hint),
        tile_context[1:3]
        if tile_context is not None
        else (graph_path_key, graph_signature_hint),
        tile_context[3] if tile_context is not None else 0,
        tile_context[4] if tile_context is not None else None,
    ) if tile_context is not None else None
    graph, graph_path_key, graph_signature, graph_cache_hit = _load_cached_graph(
        graph_path,
        scored_cache_key=scored_cache_key,
    )
    graph_load_elapsed_ms = (perf_counter() - graph_load_started) * 1000.0

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
        score_mapping = {
            "enabled": True,
            "source": str(request.tile_scores_json),
            "zoom": int(zoom),
            "matched_edges": int(matched),
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
        "applied_max_detour_factor": float(request.max_detour_factor),
        "avoid_highways_applied": bool(request.avoid_highways),
        "graph_cache_hit": bool(graph_cache_hit),
        "graph_load_elapsed_ms": graph_load_elapsed_ms,
        "score_mapping_coverage": float(score_mapping["matched_ratio"]),
        "tile_score_cache_hit": bool(tile_score_cache_hit),
        "scored_graph_cache_hit": bool(scored_graph_cache_hit),
        "tile_score_load_elapsed_ms": tile_score_load_elapsed_ms,
        "score_application_elapsed_ms": score_application_elapsed_ms,
    }

    planner = ScenicRoutePlanner(graph=graph)
    scenic_route = planner.find_scenic_route(
        start=request.start,
        end=request.end,
        scenic_weight=request.scenic_weight,
        avoid_highways=request.avoid_highways,
        max_detour_factor=request.max_detour_factor,
    )
    features = [route_to_feature(scenic_route, "scenic")]
    routes = [{"route_kind": "scenic", "metrics": features[0]["properties"]}]

    baseline_route: Route | None = None
    if request.include_baseline:
        # The comparison baseline is only meaningful when it uses the same
        # user-selected highway filter as the scenic route.
        baseline_route = planner.find_fastest_route(
            start=request.start,
            end=request.end,
            avoid_highways=request.avoid_highways,
        )
        baseline_feature = route_to_feature(baseline_route, "baseline")
        features.append(baseline_feature)
        routes.append(
            {"route_kind": "baseline", "metrics": baseline_feature["properties"]}
        )

    if baseline_route is not None:
        fastest_duration = float(baseline_route.estimated_duration_minutes)
        scenic_duration = float(scenic_route.estimated_duration_minutes)
        fastest_distance = float(baseline_route.total_distance_km)
        scenic_distance = float(scenic_route.total_distance_km)
        diagnostics["scenic_fastest_duration_ratio"] = (
            scenic_duration / fastest_duration
            if fastest_duration > 0.0
            else None
        )
        diagnostics["scenic_fastest_distance_ratio"] = (
            scenic_distance / fastest_distance
            if fastest_distance > 0.0
            else None
        )
        diagnostics["duration_cap_satisfied"] = (
            fastest_duration <= 0.0
            or scenic_duration <= fastest_duration * request.max_detour_factor
        )
    else:
        diagnostics["scenic_fastest_duration_ratio"] = None
        diagnostics["scenic_fastest_distance_ratio"] = None
        diagnostics["duration_cap_satisfied"] = None

    diagnostics["planning_elapsed_ms"] = (perf_counter() - started_at) * 1000.0
    return {
        "request": request.to_dict(),
        "diagnostics": diagnostics,
        "score_mapping": score_mapping,
        "routes": routes,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
