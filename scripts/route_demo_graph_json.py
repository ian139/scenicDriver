"""
Demo scenic route planning from a RoadGraph JSON file.

Example:
  uv run python scripts/route_demo_graph_json.py \
    --graph data/processed/amherst_road_graph.json \
    --start 42.34 -72.60 \
    --end 42.42 -72.50 \
    --scenic-weight 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import RoadGraph
from src.route_planner.planner import ScenicRoutePlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a scenic route from RoadGraph JSON")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--scenic-weight", type=float, default=0.5)
    parser.add_argument("--avoid-highways", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = RoadGraph.load(args.graph)
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


if __name__ == "__main__":
    main()
