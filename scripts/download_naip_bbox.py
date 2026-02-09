"""
Download NAIP tiles for a bounding box using STAC discovery.

Example:
  uv run python scripts/download_naip_bbox.py \
    --min-lat 40.28 --min-lon -105.70 \
    --max-lat 40.35 --max-lon -105.58 \
    --year 2021 \
    --output data/raw/images/naip
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.naip import NAIPDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NAIP tiles for bbox")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--aws-profile", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = NAIPDownloader(cache_dir=output_dir, aws_profile=args.aws_profile)
    tiles = downloader.get_tiles_for_bbox(
        min_lat=args.min_lat,
        min_lon=args.min_lon,
        max_lat=args.max_lat,
        max_lon=args.max_lon,
        year=args.year,
    )

    if args.max_tiles is not None:
        tiles = tiles[: args.max_tiles]

    for tile in tiles:
        downloader.download_tile(tile)

    print(f"Downloaded {len(tiles)} NAIP tiles to {output_dir}")


if __name__ == "__main__":
    main()
