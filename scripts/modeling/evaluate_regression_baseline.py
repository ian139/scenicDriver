"""
Evaluate a trained scenic regression baseline checkpoint on an exported .npz dataset.

Validation membership is not guessed here: it is exactly the membership the
trained model used during training, reproduced from the split artifact written
by train_regression_baseline.py through the shared derivation in
scripts/modeling/regression_split.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.regression import ScenicRegressionModel, ScenicScoreDataset, resolve_device  # noqa: E402
from scripts.modeling.regression_split import load_validation_membership  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scenic regression baseline")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to exported .npz dataset")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained baseline checkpoint")
    parser.add_argument(
        "--split-artifact",
        type=Path,
        required=True,
        help="Training split artifact whose validation membership is evaluated",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional output metrics json path")
    return parser.parse_args()


def _safe_corr(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) <= 1:
        return 0.0
    corr = float(np.corrcoef(preds, targets)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)

    membership = load_validation_membership(args.dataset, args.split_artifact)
    ds = ScenicScoreDataset(args.dataset)
    val_ds = torch.utils.data.Subset(ds, membership.val_dataset_indices.tolist())
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ScenicRegressionModel(
        vit_dim=int(ckpt["vit_dim"]),
        terrain_dim=int(ckpt["terrain_dim"]),
        num_classes=int(ckpt["num_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for vit_emb, terrain, logits, score, _ in val_loader:
            vit_emb = vit_emb.to(device)
            terrain = terrain.to(device)
            logits = logits.to(device)
            pred = model(vit_emb, terrain, logits).squeeze(-1).cpu().numpy()
            target = score.squeeze(-1).cpu().numpy()
            preds.append(pred)
            targets.append(target)

    preds_arr = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    targets_arr = np.concatenate(targets) if targets else np.array([], dtype=np.float32)
    if len(preds_arr) == 0:
        raise ValueError("No validation samples available for evaluation")

    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
    corr = _safe_corr(preds_arr, targets_arr)
    metrics = {
        "samples": int(len(targets_arr)),
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "val_split": membership.val_split,
        "seed": membership.seed,
        "split_artifact": str(args.split_artifact),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
    }

    print(json.dumps(metrics, indent=2))
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
