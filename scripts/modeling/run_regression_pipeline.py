"""
Run a promotion-ready regression pipeline by orchestrating existing scripts:
1) Build mixed labels (human-overlap aware aggregation).
2) Export regression dataset features.
3) Train candidate checkpoint.
4) Compare candidate and baseline on the same complete fixed human test benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end regression pipeline")
    parser.add_argument(
        "--heuristic-labels",
        type=Path,
        default=Path("data/processed/regression/labels_masswhites_z14_mixed5000.csv"),
    )
    parser.add_argument("--annotations-csv", type=Path, default=Path("data/raw/labels_human.csv"))
    parser.add_argument("--geographic-splits-csv", type=Path, required=True)
    parser.add_argument("--benchmark-split-csv", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("data/processed/regression/labels_masswhites_z14_mixed5000_v4.csv"),
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=Path("data/processed/regression/features_masswhites_z14_mixed5000_v4_h4.npz"),
    )
    parser.add_argument(
        "--candidate-ckpt",
        type=Path,
        default=Path("models/scenic_regression_baseline_masswhites_z14_mixed5000_v4_weighted_h4.pt"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("data/processed/regression/benchmark_comparison.json"),
    )
    parser.add_argument("--raw-dir", type=str, default="s3://scenicdriver-data/raw")
    parser.add_argument("--classifier-ckpt", type=Path, default=Path("models/classifier/best_model.pt"))
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    return args


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_ckpt.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            py,
            "scripts/modeling/build_mixed_labels.py",
            "--heuristic-labels",
            str(args.heuristic_labels),
            "--annotations-csv",
            str(args.annotations_csv),
            "--output",
            str(args.labels_output),
            "--geographic-splits-csv",
            str(args.geographic_splits_csv),
            "--aggregate",
            args.aggregate,
        ]
    )

    _run(
        [
            py,
            "scripts/modeling/export_regression_dataset.py",
            "--labels-csv",
            str(args.labels_output),
            "--raw-dir",
            args.raw_dir,
            "--output",
            str(args.features_output),
            "--classifier-ckpt",
            str(args.classifier_ckpt),
            "--device",
            args.device,
            "--label-source-column",
            "label_source",
            "--human-weight",
            "4.0",
            "--heuristic-weight",
            "1.0",
            "--default-weight",
            "1.0",
        ]
    )

    _run(
        [
            py,
            "scripts/modeling/train_regression_baseline.py",
            "--dataset",
            str(args.features_output),
            "--labels-csv",
            str(args.labels_output),
            "--benchmark-split-csv",
            str(args.benchmark_split_csv),
            "--classifier-checkpoint",
            str(args.classifier_ckpt),
            "--output",
            str(args.candidate_ckpt),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--val-split",
            str(args.val_split),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--weight-decay",
            str(args.weight_decay),
            "--use-sample-weights",
        ]
    )

    _run(
        [
            py,
            "scripts/modeling/compare_regression_on_benchmark.py",
            "--dataset",
            str(args.features_output),
            "--labels-csv",
            str(args.labels_output),
            "--benchmark-split-csv",
            str(args.benchmark_split_csv),
            "--candidate-checkpoint",
            str(args.candidate_ckpt),
            "--baseline-checkpoint",
            str(args.baseline_checkpoint),
            "--device",
            args.device,
            "--output-json",
            str(args.candidate_metrics),
        ]
    )


if __name__ == "__main__":
    main()
