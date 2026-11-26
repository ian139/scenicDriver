"""
Train baseline scenic regression model from exported .npz features.

Expected input comes from scripts/modeling/export_regression_dataset.py together
with the labels CSV and classifier checkpoint it was exported from. Every
exported row resolves to a region-independent physical z/x/y tile; rows within
the declared buffer of any fixed benchmark val/test tile (or embedded held-out
export row) are excluded, and the remainder is split into train/validation by
physical-block and declared geographic-group components with no adjacent tiles
across the split (scripts/modeling/regression_split.py). The membership is
passed explicitly to the trainer and persisted as a split artifact; a run
manifest binds the checkpoint, split artifact, feature NPZ, labels CSV,
benchmark split CSV, classifier checkpoint and preprocessing identity by
content digest for admission by evaluate_regression_baseline.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scenicdriver.scenic_scorer.regression import resolve_device, train_regression_model  # noqa: E402
from scripts.modeling.regression_split import (  # noqa: E402
    GeographicSplitConfig,
    derive_validation_membership,
    preprocessing_identity,
    training_arrays,
    write_run_manifest,
    write_split_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scenic regression baseline")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to exported .npz dataset")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="Labels CSV the dataset was exported from (physical z/x/y and geographic groups)",
    )
    parser.add_argument("--benchmark-split-csv", type=Path, required=True)
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        required=True,
        help="Classifier checkpoint the dataset features were exported with",
    )
    parser.add_argument(
        "--normalization",
        choices=("imagenet", "resisc45"),
        default=None,
        help="Optional assertion: fail unless the classifier checkpoint's preprocessing "
        "contract uses this normalization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/scenic_regression_baseline.pt"),
        help="Where to save best baseline checkpoint",
    )
    parser.add_argument(
        "--split-artifact",
        type=Path,
        default=None,
        help=(
            "Where to write the training validation-membership split artifact "
            "consumed by evaluate_regression_baseline.py "
            "(default: <output stem>.split.json)"
        ),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Where to write the content-digest run manifest (default: <output stem>.manifest.json)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Physical block edge in tiles; blocks and declared groups never straddle train/val",
    )
    parser.add_argument(
        "--holdout-buffer-tiles",
        type=int,
        default=1,
        help="Tiles around every benchmark held-out tile excluded from training (>= 1)",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--use-sample-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sample_weights array from dataset during optimization",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.holdout_buffer_tiles < 1:
        parser.error("--holdout-buffer-tiles must be >= 1")
    if args.split_artifact is None:
        args.split_artifact = args.output.with_suffix(".split.json")
    if args.run_manifest is None:
        args.run_manifest = args.output.with_suffix(".manifest.json")
    return args


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    preprocessing = preprocessing_identity(
        args.classifier_checkpoint, normalization=args.normalization
    )
    config = GeographicSplitConfig(
        val_split=args.val_split,
        seed=args.seed,
        block_size=args.block_size,
        holdout_buffer_tiles=args.holdout_buffer_tiles,
    )
    membership = derive_validation_membership(
        args.dataset, args.labels_csv, args.benchmark_split_csv, config
    )
    arrays = training_arrays(args.dataset, membership)
    with tempfile.TemporaryDirectory(prefix="scenic-baseline-") as directory:
        training_path = Path(directory) / "training.npz"
        np.savez_compressed(training_path, **arrays)
        best_corr = train_regression_model(
            data_path=training_path,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.lr,
            val_split=args.val_split,
            seed=args.seed,
            device=device,
            weight_decay=args.weight_decay,
            use_sample_weights=args.use_sample_weights,
            train_indices=membership.train_indices,
            val_indices=membership.val_indices,
        )

    write_split_artifact(args.split_artifact, membership)
    write_run_manifest(
        args.run_manifest,
        membership,
        checkpoint=args.output,
        split_artifact=args.split_artifact,
        dataset=args.dataset,
        labels_csv=args.labels_csv,
        benchmark_split_csv=args.benchmark_split_csv,
        classifier_checkpoint=args.classifier_checkpoint,
        preprocessing=preprocessing,
        training={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "use_sample_weights": bool(args.use_sample_weights),
            "device": device,
            "best_val_corr": float(best_corr),
        },
    )

    print(f"Best validation correlation: {best_corr:.4f}")
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved split artifact: {args.split_artifact}")
    print(f"Saved run manifest: {args.run_manifest}")


if __name__ == "__main__":
    main()
