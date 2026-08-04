"""
Train baseline scenic regression model from exported .npz features.

Expected input comes from scripts/modeling/export_regression_dataset.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.regression import train_regression_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scenic regression baseline")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to exported .npz dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/scenic_regression_baseline.pt"),
        help="Where to save best baseline checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--use-sample-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sample_weights array from dataset during optimization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    best_corr = train_regression_model(
        data_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        device=device,
        weight_decay=args.weight_decay,
        use_sample_weights=args.use_sample_weights,
    )

    print(f"Best validation correlation: {best_corr:.4f}")
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
