"""
Plan a scenic route from a GeoJSON LineString graph and emit route GeoJSON.

Example:
  uv run python scripts/route_demo_geojson.py \
    --geojson data/processed/sample_road_graph.geojson \
    --start 42.40 -72.70 \
    --end 42.48 -72.62 \
    --scenic-weight 0.6 \
    --output-geojson data/processed/sample_route.geojson \
    --report-dir data/processed/heuristic_runs/<run_name>/report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import RoadGraph
from src.route_planner.planner import ScenicRoutePlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan scenic route from GeoJSON graph")
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--scenic-weight", type=float, default=0.6)
    parser.add_argument("--avoid-highways", action="store_true")
    parser.add_argument("--max-detour-factor", type=float, default=1.8)
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute a non-scenic baseline route (scenic_weight=0.0) for comparison",
    )
    parser.add_argument("--output-geojson", type=Path, default=Path("data/processed/sample_route.geojson"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="If set, also copy route.geojson into this report directory for overlay",
    )
    parser.add_argument(
        "--tile-scores-json",
        type=Path,
        default=None,
        help="Optional report.json with per-tile scenic scores used to overwrite edge scenic scores.",
    )
    parser.add_argument(
        "--tile-score-zoom",
        type=int,
        default=None,
        help="Tile zoom to use for score lookup (defaults to inferred zoom from tile-scores JSON).",
    )
    parser.add_argument(
        "--tile-score-fallback",
        type=float,
        default=None,
        help="Fallback scenic score for edges with no tile match (default: keep existing edge scenic score).",
    )
    return parser.parse_args()


def _route_feature(route, route_kind: str) -> dict:
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


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y


def _load_tile_scores(path: Path) -> tuple[dict[tuple[int, int, int], float], int]:
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


def _apply_tile_scores_to_graph(
    graph: RoadGraph,
    score_map: dict[tuple[int, int, int], float],
    *,
    zoom: int,
    fallback: float | None,
) -> tuple[int, int]:
    matched = 0
    total = 0
    for edge in graph.edges.values():
        total += 1
        start = graph.get_node(edge.start_node_id)
        end = graph.get_node(edge.end_node_id)
        mid_lat = 0.5 * (start.lat + end.lat)
        mid_lon = 0.5 * (start.lon + end.lon)
        x, y = _lat_lon_to_tile(mid_lat, mid_lon, zoom)
        tile_score = score_map.get((zoom, x, y))
        if tile_score is not None:
            edge.scenic_score = float(min(max(tile_score, 0.0), 10.0))
            matched += 1
        elif fallback is not None:
            edge.scenic_score = float(min(max(fallback, 0.0), 10.0))
    return matched, total


def main() -> None:
    args = parse_args()
    graph = RoadGraph.from_geojson(args.geojson)

    tile_scores_path = args.tile_scores_json
    if tile_scores_path is None and args.report_dir is not None:
        candidate = args.report_dir / "report.json"
        if candidate.exists():
            tile_scores_path = candidate

    if tile_scores_path is not None:
        score_map, inferred_zoom = _load_tile_scores(tile_scores_path)
        zoom = args.tile_score_zoom if args.tile_score_zoom is not None else inferred_zoom
        matched, total = _apply_tile_scores_to_graph(
            graph,
            score_map,
            zoom=zoom,
            fallback=args.tile_score_fallback,
        )
        pct = (100.0 * matched / max(total, 1))
        print(
            f"Applied tile scenic scores from {tile_scores_path} at z{zoom}: "
            f"matched {matched}/{total} edges ({pct:.1f}%)"
        )
    else:
        print("No tile-score map provided; using scenic_score values embedded in graph.")

    planner = ScenicRoutePlanner(graph=graph)
    route = planner.find_scenic_route(
        start=(args.start[0], args.start[1]),
        end=(args.end[0], args.end[1]),
        scenic_weight=args.scenic_weight,
        avoid_highways=args.avoid_highways,
        max_detour_factor=args.max_detour_factor,
    )

    print("[Scenic Route]")
    print(f"Segments: {len(route.segments)}")
    print(f"Total distance (km): {route.total_distance_km:.2f}")
    print(f"Avg scenic score: {route.average_scenic_score:.2f}")
    print(f"Estimated duration (min): {route.estimated_duration_minutes:.1f}")

    features = [_route_feature(route, "scenic")]

    if args.include_baseline:
        baseline = planner.find_scenic_route(
            start=(args.start[0], args.start[1]),
            end=(args.end[0], args.end[1]),
            scenic_weight=0.0,
            avoid_highways=False,
            max_detour_factor=max(args.max_detour_factor, 1.2),
        )
        features.append(_route_feature(baseline, "baseline"))
        print("[Baseline Route]")
        print(f"Segments: {len(baseline.segments)}")
        print(f"Total distance (km): {baseline.total_distance_km:.2f}")
        print(f"Avg scenic score: {baseline.average_scenic_score:.2f}")
        print(f"Estimated duration (min): {baseline.estimated_duration_minutes:.1f}")

    geojson = {"type": "FeatureCollection", "features": features}
    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.output_geojson.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_geojson}")

    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_route = args.report_dir / "route.geojson"
        shutil.copy(args.output_geojson, report_route)
        print(f"Copied overlay to {report_route}")


if __name__ == "__main__":
    main()
