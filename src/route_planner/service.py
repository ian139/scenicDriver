from __future__ import annotations

from dataclasses import dataclass
import json
import math
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
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
    score_map: dict[tuple[int, int, int], float],
    *,
    zoom: int,
    fallback: float | None,
) -> tuple[int, int]:
    matched = 0
    total = 0

    # Segment-heavy graphs commonly repeat midpoint coordinates and tile keys.
    # Keep both caches bounded so a broad graph cannot turn this into another
    # graph-sized allocation. FIFO eviction makes memory use and access
    # behavior deterministic while retaining insertion order for edges.
    cache_limit = 4096
    midpoint_tiles: OrderedDict[tuple[float, float], tuple[int, int]] = OrderedDict()
    tile_results: OrderedDict[tuple[int, int, int], object] = OrderedDict()
    cache_miss = object()

    for edge in graph.edges.values():
        total += 1
        start = graph.get_node(edge.start_node_id)
        end = graph.get_node(edge.end_node_id)
        mid_lat = 0.5 * (start.lat + end.lat)
        mid_lon = 0.5 * (start.lon + end.lon)

        midpoint_key = (mid_lat, mid_lon)
        tile = midpoint_tiles.get(midpoint_key)
        if tile is None:
            tile = lat_lon_to_tile(mid_lat, mid_lon, zoom)
            if len(midpoint_tiles) >= cache_limit:
                midpoint_tiles.popitem(last=False)
            midpoint_tiles[midpoint_key] = tile
        x, y = tile

        tile_key = (zoom, x, y)
        tile_score = tile_results.get(tile_key, cache_miss)
        if tile_score is cache_miss:
            tile_score = score_map.get(tile_key)
            if len(tile_results) >= cache_limit:
                tile_results.popitem(last=False)
            tile_results[tile_key] = tile_score

        if tile_score is not None:
            edge.scenic_score = float(min(max(tile_score, 0.0), 10.0))
            matched += 1
        elif fallback is not None:
            edge.scenic_score = float(min(max(fallback, 0.0), 10.0))
    return matched, total


def _load_graph(path: Path) -> RoadGraph:
    # Support both FeatureCollection road graphs (.geojson) and serialized RoadGraph JSONs.
    if path.suffix.lower() == ".geojson":
        return RoadGraph.from_geojson(path)
    return RoadGraph.load(path)


def diagnose_route_request(request: RouteRequest) -> dict[str, Any]:
    graph_path = Path(request.graph_geojson)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph GeoJSON not found: {graph_path}")

    graph = _load_graph(graph_path)

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

    graph = _load_graph(graph_path)

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
    }
    score_mapping = {
        "enabled": False,
        "source": None,
        "zoom": None,
        "matched_edges": 0,
        "total_edges": int(len(graph.edges)),
        "matched_ratio": 0.0,
    }

    if request.tile_scores_json:
        score_path = Path(request.tile_scores_json)
        score_map, inferred_zoom = load_tile_scores(score_path)
        zoom = (
            request.tile_score_zoom
            if request.tile_score_zoom is not None
            else inferred_zoom
        )
        matched, total = apply_tile_scores_to_graph(
            graph,
            score_map,
            zoom=zoom,
            fallback=request.tile_score_fallback,
        )
        score_mapping = {
            "enabled": True,
            "source": str(score_path),
            "zoom": int(zoom),
            "matched_edges": int(matched),
            "total_edges": int(total),
            "matched_ratio": float(matched / max(total, 1)),
        }
    diagnostics["score_mapping_coverage"] = float(score_mapping["matched_ratio"])

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
