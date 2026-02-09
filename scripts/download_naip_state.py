"""
Download NAIP tiles for a given state/year (S3 listing).

Example:
  uv run python scripts/download_naip_state.py \
    --state CO \
    --year 2021 \
    --output data/raw/images/naip \
    --max-tiles 10
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
    parser = argparse.ArgumentParser(description="Download NAIP tiles for a state")
    parser.add_argument("--state", type=str, required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int, default=10)
    parser.add_argument("--aws-profile", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = NAIPDownloader(cache_dir=output_dir, aws_profile=args.aws_profile)
    tiles_iter = downloader.list_tiles_for_state(args.state, year=args.year)

    downloaded = 0
    for tile in tiles_iter:
        downloader.download_tile(tile)
        downloaded += 1
        if args.max_tiles and downloaded >= args.max_tiles:
            break

    print(f"Downloaded {downloaded} NAIP tiles to {output_dir}")


if __name__ == "__main__":
    main()
