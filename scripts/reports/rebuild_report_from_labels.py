"""
Rebuild report artifacts from labels.csv and its admitted source run.json.

Labels alone cannot recover provenance. In-place rebuilds leave run.json untouched
and write rebuild.json; a new output directory receives the full source identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scenicdriver.classifier.model import TERRAIN_CLASSES  # noqa: E402
from scenicdriver.heuristics.labeler import parse_tile_coords  # noqa: E402
from scenicdriver.heuristics.report import build_report, recover_report_run_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild report from labels.csv")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--source-run-json", type=Path, default=None,
        help="Admitted source run record (default: run.json beside labels.csv); required for labels-only inputs.",
    )
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
    source_path = args.source_run_json or (labels_csv.parent / "run.json")
    guidance = (
        "Pass --source-run-json <path> to the complete original run.json; "
        "labels.csv alone cannot recover identity. Re-score legacy runs whose identity is unavailable."
    )
    try:
        source_record = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot recover source identity from {source_path}; {guidance}") from exc
    run_info = recover_report_run_info(
        source_record, source=str(source_path), guidance=guidance
    )

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
    run_name = args.run_name or source_record.get("run_name") or labels_csv.parent.name

    report_dir = args.report_dir or (labels_csv.parent / "report")
    run_dir = report_dir.parent

    record_path = run_dir / "run.json"
    if record_path.exists():
        if run_dir.resolve() != source_path.parent.resolve():
            raise ValueError("Output run.json already exists; use a new --report-dir to preserve its provenance.")
        record_path = run_dir / "rebuild.json"
        if record_path.resolve() == source_path.resolve():
            record_path = run_dir / "rebuild-record.json"

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
        **source_record,
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labels_path": str(labels_csv),
        "report_dir": str(report_dir),
        "summary": report_json["summary"],
        "histogram": report_json["histogram"],
        "run_info": run_info,
        "rebuild": {
            "source_run_json": str(source_path.resolve()),
            "include_thumbs": not args.skip_thumbs,
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print(f"Report rebuilt at {report_dir}")
    print(f"Run record written to {record_path}")
    if args.benchmark:
        total = perf_counter() - t0
        print(f"Benchmark: read_labels_s={t_read1 - t_read0:.2f}")
        print(f"Benchmark: labels_to_tiles_s={t_tiles1 - t_tiles0:.2f}")
        print(f"Benchmark: build_report_s={t_report1 - t_report0:.2f}")
        print(f"Benchmark: total_s={total:.2f}")


if __name__ == "__main__":
    main()
