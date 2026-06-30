"""
Rebuild heuristic report artifacts from an existing labels.csv.

Useful when scoring completed but report writing failed (e.g. disk pressure).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.model import TERRAIN_CLASSES
from src.heuristics.labeler import parse_tile_coords
from src.heuristics.report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild report from labels.csv")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--s3-bucket", type=str, default=None)
    parser.add_argument("--s3-only", action="store_true", default=False)
    parser.add_argument("--skip-thumbs", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    return parser.parse_args()


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _labels_to_tiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        image_path = str(row["image_path"])
        scenic_score = float(row["scenic_score"])

        tile: dict[str, Any] = {
            "image_path": image_path,
            "scenic_score": scenic_score,
        }

        class_id = _safe_int(row["class_id"]) if "class_id" in df.columns else None
        if class_id is not None:
            tile["class_id"] = class_id
            if 0 <= class_id < len(TERRAIN_CLASSES):
                tile["class_name"] = TERRAIN_CLASSES[class_id]

        coords = parse_tile_coords(image_path)
        if coords is not None:
            z, x, y = coords
            tile["z"] = z
            tile["x"] = x
            tile["y"] = y

        tiles.append(tile)
    return tiles


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    labels_csv = args.labels_csv
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv}")

    if args.s3_bucket:
        os.environ["SCENIC_S3_BUCKET"] = args.s3_bucket
    if args.s3_only:
        os.environ["SCENIC_S3_ONLY"] = "1"

    t_read0 = perf_counter()
    df = pd.read_csv(labels_csv)
    t_read1 = perf_counter()
    required = {"image_path", "scenic_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {sorted(missing)}")

    t_tiles0 = perf_counter()
    tiles = _labels_to_tiles(df)
    t_tiles1 = perf_counter()
    run_name = args.run_name or labels_csv.parent.name

    report_dir = args.report_dir or (labels_csv.parent / "report")
    run_dir = report_dir.parent

    run_info = {
        "counts": {
            "paired": len(tiles),
            "satellite_total": len(tiles),
            "terrain_total": len(tiles),
            "missing_pairs": 0,
        },
        "used_classifier": "class_id" in df.columns,
        "device": "n/a",
        "coords_available": all("x" in t and "y" in t and "z" in t for t in tiles),
        "warnings": ["report rebuilt from labels.csv"],
        "seed": None,
        "config": {
            "satellite_dir": "rebuild_from_labels",
            "terrain_dir": "rebuild_from_labels",
            "raw_dir": args.raw_dir,
            "s3_bucket": os.getenv("SCENIC_S3_BUCKET"),
            "s3_only": os.getenv("SCENIC_S3_ONLY", "0") in ("1", "true", "yes"),
            "max_tiles": len(tiles),
            "use_classifier": "class_id" in df.columns,
        },
    }

    t_report0 = perf_counter()
    report_json = build_report(
        tiles=tiles,
        report_dir=report_dir,
        raw_dir=args.raw_dir,
        run_info=run_info,
        include_thumbs=not args.skip_thumbs,
    )
    t_report1 = perf_counter()

    run_record = {
        "run_name": run_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "labels_path": str(labels_csv),
        "raw_labels_path": None,
        "report_dir": str(report_dir),
        "summary": report_json["summary"],
        "histogram": report_json["histogram"],
        "run_info": run_info,
        "config": {
            "rebuild_from_labels": True,
            "max_tiles": len(tiles),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print(f"Report rebuilt at {report_dir}")
    print(f"Run record written to {run_dir / 'run.json'}")
    if args.benchmark:
        total = perf_counter() - t0
        print(f"Benchmark: read_labels_s={t_read1 - t_read0:.2f}")
        print(f"Benchmark: labels_to_tiles_s={t_tiles1 - t_tiles0:.2f}")
        print(f"Benchmark: build_report_s={t_report1 - t_report0:.2f}")
        print(f"Benchmark: total_s={total:.2f}")


if __name__ == "__main__":
    main()
