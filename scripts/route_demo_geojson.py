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


def main() -> None:
    args = parse_args()
    graph = RoadGraph.from_geojson(args.geojson)
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
