"""
Build a RoadGraph JSON from OSM data using osmnx.

Example:
  uv sync --extra geo
  uv run python scripts/build_graph_from_osm.py \
    --min-lat 42.35 --min-lon -72.57 \
    --max-lat 42.39 --max-lon -72.52 \
    --run-name amherst_core \
    --max-query-area 1e12
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit graph JSON output path. If omitted, a deterministic run folder is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/road_graphs"),
        help="Root folder for deterministic graph-cache runs when --output is omitted.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name. If omitted, a deterministic name is generated from bbox/network.",
    )
    parser.add_argument("--network", type=str, default="drive")
    parser.add_argument("--max-query-area", type=float, default=None)
    parser.add_argument("--overpass-url", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def _graph_from_bbox_compat(ox, north: float, south: float, east: float, west: float, network_type: str):
    fn = ox.graph_from_bbox
    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())
    # osmnx>=2 expects bbox in (left, bottom, right, top) = (west, south, east, north).
    bbox = (west, south, east, north)

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


def _slug_float(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _default_run_name(args: argparse.Namespace) -> str:
    return (
        f"osm_{args.network}_"
        f"{_slug_float(args.min_lat)}_{_slug_float(args.min_lon)}_"
        f"{_slug_float(args.max_lat)}_{_slug_float(args.max_lon)}"
    )


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.output is not None:
        run_name = args.run_name or args.output.stem
        output_path = args.output
        run_dir = output_path.parent
        return output_path, run_dir, run_name

    run_name = args.run_name or _default_run_name(args)
    run_dir = args.output_root / run_name
    output_path = run_dir / "road_graph.json"
    return output_path, run_dir, run_name


def main() -> None:
    args = parse_args()
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError("osmnx is required. Run: uv sync --extra geo") from exc

    if args.min_lat >= args.max_lat:
        raise ValueError("Invalid bbox: --min-lat must be < --max-lat")
    if args.min_lon >= args.max_lon:
        raise ValueError("Invalid bbox: --min-lon must be < --max-lon")

    output_path, run_dir, run_name = _resolve_output_paths(args)

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

    run_dir.mkdir(parents=True, exist_ok=True)
    graph.save(output_path)

    run_record = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_graph": str(output_path),
        "output_dir": str(run_dir),
        "bbox": {
            "min_lat": args.min_lat,
            "min_lon": args.min_lon,
            "max_lat": args.max_lat,
            "max_lon": args.max_lon,
        },
        "network": args.network,
        "osmnx": {
            "version": getattr(ox, "__version__", "unknown"),
            "max_query_area_size": getattr(ox.settings, "max_query_area_size", None),
            "overpass_url": getattr(ox.settings, "overpass_url", None),
            "timeout": getattr(ox.settings, "timeout", None),
        },
        "counts": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
        },
    }
    run_json_path = run_dir / "run.json"
    run_json_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Wrote {run_json_path}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")


if __name__ == "__main__":
    main()
