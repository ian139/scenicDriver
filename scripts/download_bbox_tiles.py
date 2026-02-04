"""
Download Mapbox tiles for a bounding box.

Examples:
  python scripts/download_bbox_tiles.py \
    --min-lat 40.018 --min-lon -75.2284 \
    --max-lat 40.0734 --max-lon -75.185 \
    --zoom 16 --style mapbox.satellite \
    --output data/raw/images/satellite

  python scripts/download_bbox_tiles.py \
    --min-lat 40.018 --min-lon -75.2284 \
    --max-lat 40.0734 --max-lon -75.185 \
    --zoom 16 --style mapbox.terrain-rgb \
    --output data/raw/images/terrain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.mapbox import MapboxTileSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mapbox tiles for bbox")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument("--style", type=str, default="mapbox.satellite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--rate-limit", type=float, default=10.0)
    parser.add_argument("--high-res", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    source = MapboxTileSource(
        cache_dir=output_dir,
        rate_limit=args.rate_limit,
        use_high_res=args.high_res,
        style_id=args.style,
    )

    tiles = list(
        source.get_tiles_for_bbox(
            min_lat=args.min_lat,
            min_lon=args.min_lon,
            max_lat=args.max_lat,
            max_lon=args.max_lon,
            zoom=args.zoom,
            max_tiles=args.max_tiles,
        )
    )

    stats = source.get_stats()
    print(
        f"Saved {len(tiles)} tiles to {output_dir} "
        f"(downloaded={stats.tiles_downloaded}, cached={stats.tiles_cached}, failed={stats.tiles_failed})"
    )


if __name__ == "__main__":
    main()
