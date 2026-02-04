"""
Run heuristic report for a named region subfolder.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.heuristic_report import main as _report_main
from scripts.heuristic_report import parse_args as _parse_base_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate heuristic report for a region")
    parser.add_argument("--region", type=str, required=True)
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

    # Delegate to the main report script by rebuilding sys.argv
    run_name = args.run_name or f"{region}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    sat_dir = f"data/raw/images/satellite/z16/{region}"
    terr_dir = f"data/raw/images/terrain/z16/{region}"

    sys.argv = [
        sys.argv[0],
        "--run-name",
        run_name,
        "--satellite-dir",
        sat_dir,
        "--terrain-dir",
        terr_dir,
    ]
    if args.preview:
        sys.argv.append("--preview")
    if args.max_tiles is not None:
        sys.argv.extend(["--max-tiles", str(args.max_tiles)])
    if args.write_labels:
        sys.argv.append("--write-labels")
    if args.write_raw_labels:
        sys.argv.append("--write-raw-labels")
    if args.no_classifier:
        sys.argv.append("--no-classifier")
    if args.device:
        sys.argv.extend(["--device", args.device])
    if args.seed is not None:
        sys.argv.extend(["--seed", str(args.seed)])
    if args.open:
        sys.argv.append("--open")

    _report_main()


if __name__ == "__main__":
    main()
