"""
Compare two regression checkpoints on the deterministic human benchmark split.

Writes a single summary JSON with per-model metrics and deltas on the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.regression import ScenicRegressionModel, ScenicScoreDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regression checkpoints on benchmark split")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--benchmark-split-csv", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/regression/benchmark_compare_v4_vs_v2.json"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_corr(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) <= 1:
        return 0.0
    corr = float(np.corrcoef(preds, targets)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _load_model(ckpt_path: Path, device: str) -> ScenicRegressionModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ScenicRegressionModel(
        vit_dim=int(ckpt["vit_dim"]),
        terrain_dim=int(ckpt["terrain_dim"]),
        num_classes=int(ckpt["num_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _evaluate(model: ScenicRegressionModel, loader: torch.utils.data.DataLoader, device: str) -> dict:
    preds = []
    targets = []
    with torch.no_grad():
        for vit_emb, terrain, logits, score, _ in loader:
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
        raise ValueError("No evaluation samples found for benchmark split")

    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
    corr = _safe_corr(preds_arr, targets_arr)
    return {
        "samples": int(len(targets_arr)),
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
    }


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    split_df = pd.read_csv(args.benchmark_split_csv)
    if "image_path" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError("benchmark split CSV must contain image_path and split columns")
    test_paths = set(
        split_df.loc[split_df["split"].astype(str).str.lower() == "test", "image_path"].astype(str).tolist()
    )
    if not test_paths:
        raise ValueError("No test rows found in benchmark split CSV")

    labels_df = pd.read_csv(args.labels_csv)
    if "image_path" not in labels_df.columns:
        raise ValueError("labels CSV must contain image_path column")
    dataset = ScenicScoreDataset(args.dataset)
    dataset_paths = labels_df["image_path"].astype(str).tolist()
    if len(dataset_paths) != len(dataset):
        raise ValueError(
            "labels CSV row count must match dataset row count. "
            f"labels={len(dataset_paths)} dataset={len(dataset)}"
        )
    test_idx = [i for i, p in enumerate(dataset_paths) if p in test_paths]
    if not test_idx:
        raise ValueError("No benchmark test rows matched dataset image_paths")

    test_ds = torch.utils.data.Subset(dataset, test_idx)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    baseline_model = _load_model(args.baseline_checkpoint, device)
    candidate_model = _load_model(args.candidate_checkpoint, device)
    baseline_metrics = _evaluate(baseline_model, test_loader, device)
    candidate_metrics = _evaluate(candidate_model, test_loader, device)

    summary = {
        "dataset": str(args.dataset),
        "labels_csv": str(args.labels_csv),
        "benchmark_split_csv": str(args.benchmark_split_csv),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "split": "test",
        "device": device,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "deltas": {
            "corr": float(candidate_metrics["corr"]) - float(baseline_metrics["corr"]),
            "mae": float(candidate_metrics["mae"]) - float(baseline_metrics["mae"]),
            "rmse": float(candidate_metrics["rmse"]) - float(baseline_metrics["rmse"]),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
