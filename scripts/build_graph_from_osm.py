"""
Build a RoadGraph JSON from OSM data using osmnx.

Example:
  uv sync --extra geo
  uv run python scripts/build_graph_from_osm.py \
    --min-lat 42.35 --min-lon -72.57 \
    --max-lat 42.39 --max-lon -72.52 \
    --output data/processed/amherst_road_graph.json \
    --max-query-area 1e12
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.graph import _graph_from_osmnx


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


def _graph_from_bbox_compat(ox, north: float, south: float, east: float, west: float, network_type: str):
    fn = ox.graph_from_bbox
    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())
    bbox = (north, south, east, west)

    # osmnx>=2 style: graph_from_bbox(bbox=(north, south, east, west), *, network_type=...)
    if "bbox" in params:
        return fn(bbox=bbox, network_type=network_type)

    # osmnx<2 style: graph_from_bbox(north=..., south=..., east=..., west=..., network_type=...)
    if {"north", "south", "east", "west"}.issubset(params):
        return fn(north=north, south=south, east=east, west=west, network_type=network_type)

    # Fallback positional variants
    try:
        return fn(bbox, network_type=network_type)
    except TypeError:
        return fn(north, south, east, west, network_type=network_type)


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

    G = _graph_from_bbox_compat(
        ox=ox,
        north=args.max_lat,
        south=args.min_lat,
        east=args.max_lon,
        west=args.min_lon,
        network_type=args.network,
    )

    graph = _graph_from_osmnx(G, scenic_scores={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph.save(args.output)
    print(f"Wrote {args.output}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")


if __name__ == "__main__":
    main()
