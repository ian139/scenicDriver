"""CLI for strict stage-two active scenic training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.active_training import (  # noqa: E402
    ActiveTrainingConfig,
    prepare_active_dataset,
    train_active_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train the active scenic regression candidate"
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        required=True,
        help="Validated stage-one stage1_handoff.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Prepared dataset NPZ (defaults under output-dir)",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Fixed geographic split CSV (defaults to handoff artifact)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/active_training")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--sample-weight-scheme",
        choices=["standard", "region_balanced"],
        default="standard",
    )
    parser.add_argument("--loss-function", choices=["mse", "huber"], default="mse")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-sample-weights", action="store_true")
    return parser.parse_args()


def _print_metrics(result: dict[str, Any]) -> None:
    print(f"METRIC state={result.get('state')}")
    print(f"METRIC global_step={result.get('global_step', 0)}")
    metrics = result.get("metrics", {})
    validation = metrics.get("val", {}) if isinstance(metrics, dict) else {}
    if isinstance(validation, dict):
        print(f"METRIC val_rmse={validation.get('rmse', 0.0)}")
        print(f"METRIC val_mae={validation.get('mae', 0.0)}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or (args.output_dir / "active_dataset.npz")
    prepared = prepare_active_dataset(args.handoff, dataset_path)
    split_csv = args.split_csv or Path(prepared["split_path"])
    config = ActiveTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        use_sample_weights=not args.no_sample_weights,
        sample_weight_scheme=args.sample_weight_scheme,
        loss_function=args.loss_function,
    )
    result = train_active_model(
        dataset_path, split_csv, args.output_dir, config, resume=args.resume
    )
    print(
        json.dumps(
            {"prepared": prepared, "training": result}, sort_keys=True, allow_nan=False
        )
    )
    _print_metrics(result)


if __name__ == "__main__":
    main()
