"""
Build a RoadGraph JSON from OSM data using osmnx.

Example:
  uv sync --extra geo
  uv run python scripts/build_graph_from_osm.py \
    --min-lat 42.2286 --min-lon -72.7250 \
    --max-lat 42.5186 --max-lon -72.3148 \
    --output data/processed/amherst_road_graph.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import RoadGraph, _graph_from_osmnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build road graph from OSM bbox")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", type=str, default="drive")
    parser.add_argument("--max-query-area", type=float, default=None)
    parser.add_argument("--overpass-url", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError("osmnx is required. Run: uv sync --extra geo") from exc

    if args.max_query_area is not None:
        ox.settings.max_query_area_size = float(args.max_query_area)
    if args.overpass_url is not None:
        ox.settings.overpass_url = args.overpass_url
    if args.timeout is not None:
        ox.settings.timeout = int(args.timeout)

    bbox = (args.max_lat, args.min_lat, args.max_lon, args.min_lon)
    try:
        G = ox.graph_from_bbox(
            north=args.max_lat,
            south=args.min_lat,
            east=args.max_lon,
            west=args.min_lon,
            network_type=args.network,
        )
    except TypeError:
        try:
            G = ox.graph_from_bbox(
                bbox,
                network_type=args.network,
            )
        except TypeError:
            # Older osmnx signatures that accept positional args
            G = ox.graph_from_bbox(
                args.max_lat,
                args.min_lat,
                args.max_lon,
                args.min_lon,
                network_type=args.network,
            )

    graph = _graph_from_osmnx(G, scenic_scores={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
