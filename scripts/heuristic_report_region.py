"""
Run heuristic report for a named region subfolder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import types
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.heuristic_report import run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate heuristic report for a region")
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--write-labels", action="store_true", default=False)
    parser.add_argument("--write-raw-labels", action="store_true", default=False)
    parser.add_argument("--no-classifier", action="store_true", default=False)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--open", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = args.region.strip().lower()
    if not region:
        raise ValueError("Region must be a non-empty string.")

    run_name = args.run_name or f"{region}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    sat_dir = f"data/raw/images/satellite/z{args.zoom}/{region}"
    terr_dir = f"data/raw/images/terrain/z{args.zoom}/{region}"

    delegated = types.SimpleNamespace(
        run_name=run_name,
        preview=args.preview,
        max_tiles=args.max_tiles,
        write_labels=args.write_labels,
        write_raw_labels=args.write_raw_labels,
        no_classifier=args.no_classifier,
        device=args.device,
        seed=args.seed,
        open=args.open,
        satellite_dir=sat_dir,
        terrain_dir=terr_dir,
        raw_dir=None,
        s3_only=False,
    )
    run_report(delegated)


if __name__ == "__main__":
    main()
