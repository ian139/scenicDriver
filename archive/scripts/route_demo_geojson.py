"""
Demo scenic route planning from a GeoJSON LineString graph.

Example:
  uv run python scripts/route_demo_geojson.py \
    --geojson data/processed/sample_road_graph.geojson \
    --start 42.40 -72.70 \
    --end 42.48 -72.62 \
    --scenic-weight 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import RoadGraph
from src.route_planner.planner import ScenicRoutePlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a scenic route from GeoJSON graph")
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--scenic-weight", type=float, default=0.5)
    parser.add_argument("--avoid-highways", action="store_true")
    parser.add_argument("--output-geojson", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = RoadGraph.from_geojson(args.geojson)
    planner = ScenicRoutePlanner(graph=graph)
    route = planner.find_scenic_route(
        start=(args.start[0], args.start[1]),
        end=(args.end[0], args.end[1]),
        scenic_weight=args.scenic_weight,
        avoid_highways=args.avoid_highways,
    )
    print(f"Segments: {len(route.segments)}")
    print(f"Total distance (km): {route.total_distance_km:.2f}")
    print(f"Avg scenic score: {route.average_scenic_score:.2f}")
    print(f"Estimated duration (min): {route.estimated_duration_minutes:.1f}")

    if args.output_geojson:
        coords = [[lon, lat] for lat, lon in route.waypoints]
        feature = {
            "type": "Feature",
            "properties": {
                "total_distance_km": route.total_distance_km,
                "average_scenic_score": route.average_scenic_score,
                "estimated_duration_minutes": route.estimated_duration_minutes,
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        }
        geojson = {"type": "FeatureCollection", "features": [feature]}
        args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
        args.output_geojson.write_text(json.dumps(geojson, indent=2))
        print(f"Wrote {args.output_geojson}")


if __name__ == "__main__":
    main()
