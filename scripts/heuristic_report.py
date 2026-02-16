"""
Run heuristic scoring, write labels/run metadata, and generate a local report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import webbrowser
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heuristics.labeler import HeuristicLabelerConfig, run_heuristic_labeling
from src.heuristics.report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate heuristic labels + report")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--write-labels", action="store_true", default=False)
    parser.add_argument("--write-raw-labels", action="store_true", default=False)
    parser.add_argument("--no-classifier", action="store_true", default=False)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--open", action="store_true", default=False)
    parser.add_argument("--satellite-dir", type=str, default=None)
    parser.add_argument("--terrain-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--s3-only", action="store_true", default=False)
    return parser.parse_args()


def run_report(args: argparse.Namespace) -> Path:
    cfg = HeuristicLabelerConfig()
    if args.satellite_dir:
        cfg.satellite_dir = args.satellite_dir
    if args.terrain_dir:
        cfg.terrain_dir = args.terrain_dir
    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    elif os.getenv("SCENIC_S3_BUCKET"):
        cfg.raw_dir = f"s3://{os.getenv('SCENIC_S3_BUCKET')}/raw"
        if args.s3_only:
            os.environ["SCENIC_S3_ONLY"] = "1"

    run_name = args.run_name or f"heuristic_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if args.preview and args.max_tiles is None:
        max_tiles = min(256, cfg.heuristic_max_tiles)
    else:
        max_tiles = args.max_tiles or cfg.heuristic_max_tiles

    use_classifier = cfg.heuristic_use_classifier and not args.no_classifier
    seed = args.seed if args.seed is not None else cfg.seed

    labels_df, tiles, run_info = run_heuristic_labeling(
        satellite_dir=cfg.satellite_dir,
        terrain_dir=cfg.terrain_dir,
        raw_dir=cfg.raw_dir,
        max_tiles=max_tiles,
        use_classifier=use_classifier,
        classifier_best_ckpt=cfg.classifier_best_ckpt,
        classifier_use_resisc45_stats=cfg.classifier_use_resisc45_stats,
        default_lat=cfg.heuristic_default_lat,
        default_lon=cfg.heuristic_default_lon,
        seed=seed,
        device=args.device,
    )

    run_dir = Path(cfg.processed_dir) / "heuristic_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    labels_path = None
    if not args.preview or args.write_labels:
        labels_path = run_dir / "labels.csv"
        labels_df.to_csv(labels_path, index=False)
    raw_labels_path = None
    if args.write_raw_labels:
        raw_labels_path = Path(cfg.labels_csv)
        raw_labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(raw_labels_path, index=False)

    report_dir = run_dir / "report"
    report_json = build_report(
        tiles=tiles,
        report_dir=report_dir,
        raw_dir=cfg.raw_dir,
        run_info=run_info,
    )

    run_record = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labels_path": str(labels_path) if labels_path else None,
        "raw_labels_path": str(raw_labels_path) if raw_labels_path else None,
        "report_dir": str(report_dir),
        "summary": report_json["summary"],
        "histogram": report_json["histogram"],
        "run_info": run_info,
        "config": {
            "preview": args.preview,
            "max_tiles": max_tiles,
            "use_classifier": use_classifier,
            "device": args.device,
            "seed": seed,
        },
    }

    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    if args.open:
        webbrowser.open((report_dir / "index.html").resolve().as_uri())

    print(f"Report generated at {report_dir}")
    if labels_path:
        print(f"Labels written to {labels_path}")
    if raw_labels_path:
        print(f"Raw labels written to {raw_labels_path}")
    return run_dir


def main() -> None:
    run_report(parse_args())


if __name__ == "__main__":
    main()
