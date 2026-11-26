"""
Compare two regression checkpoints on the deterministic human benchmark split.

Writes a single summary JSON with per-model metrics and deltas on the test split.
Feature selection uses the NPZ's unique exported image_paths, not label row order.
Every fixed test identity must be covered exactly once and have an explicit
scenic_human_mean or scenic_human target in [0, 10]; weak scenic_score is never
an evaluation target. Both checkpoints receive the same ordered population.
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
for _import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from scenicdriver.scenic_scorer.regression import ScenicRegressionModel, ScenicScoreDataset, resolve_device  # noqa: E402


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
        default=Path("data/processed/regression/benchmark_comparison.json"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def load_benchmark_split(path: Path) -> pd.DataFrame:
    """Read unique exact image identities and their fixed split assignments."""
    frame = pd.read_csv(path)
    if not {"image_path", "split"}.issubset(frame.columns):
        raise ValueError("benchmark split CSV must contain image_path and split columns")
    _validate_image_paths(frame["image_path"].to_numpy(), "benchmark split CSV")
    splits = frame["split"].astype(str).str.strip().str.lower().replace({"validation": "val"})
    if not splits.isin(["train", "val", "test"]).all():
        raise ValueError("benchmark split CSV has invalid split assignments")
    frame["split"] = splits
    return frame.set_index("image_path", drop=False)


def _validate_image_paths(values: np.ndarray, name: str) -> list[str]:
    if values.ndim != 1:
        raise ValueError(f"{name} image_paths must be one-dimensional")
    paths = [p.decode("utf-8") if isinstance(p, bytes) else p for p in values]
    if any(not isinstance(p, str) or not p or p != p.strip() for p in paths):
        raise ValueError(f"{name} requires non-empty exact image_paths")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} contains duplicate image_paths")
    return paths


def load_exported_image_paths(data: np.lib.npyio.NpzFile) -> list[str]:
    """Require the exporter-owned identities, never infer them from label positions."""
    if "image_paths" not in data.files:
        raise ValueError("dataset must contain exported image_paths")
    paths = _validate_image_paths(data["image_paths"], "dataset")
    if len(paths) != len(data["scenic_scores"]):
        raise ValueError("dataset image_paths length does not match scenic_scores")
    return paths


def benchmark_dataset(
    dataset_path: Path, labels_path: Path, split_path: Path
) -> torch.utils.data.Subset:
    """Join a complete unique test population to features and explicit human targets."""
    split = load_benchmark_split(split_path)
    test = split.loc[split["split"] == "test"]
    if test.empty:
        raise ValueError("No test rows found in benchmark split CSV")
    dataset = ScenicScoreDataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        paths = load_exported_image_paths(data)
        if "splits" in data.files:
            embedded = data["splits"]
            if embedded.ndim != 1 or len(embedded) != len(paths):
                raise ValueError("dataset splits must align with image_paths")
            for path, assignment in zip(paths, embedded, strict=True):
                assignment = assignment.decode("utf-8") if isinstance(assignment, bytes) else str(assignment)
                if path in split.index and assignment.strip().lower().replace("validation", "val") != split.at[path, "split"]:
                    raise ValueError(f"benchmark split conflicts with dataset split for {path}")
    positions = {path: i for i, path in enumerate(paths)}
    missing = set(test.index) - set(positions)
    if missing:
        raise ValueError(f"Benchmark has {len(missing)} test paths absent from dataset")
    labels = pd.read_csv(labels_path)
    if "image_path" not in labels.columns:
        raise ValueError("labels CSV must contain image_path column")
    _validate_image_paths(labels["image_path"].to_numpy(), "labels CSV")
    labels = labels.set_index("image_path")
    targets = []
    indices = []
    for path, record in test.iterrows():
        target = None
        for source in (record, labels.loc[path] if path in labels.index else {}):
            for column in ("scenic_human_mean", "scenic_human"):
                value = source.get(column)
                if value is not None and not pd.isna(value):
                    target = float(value)
                    break
            if target is not None:
                break
        if target is None or not np.isfinite(target) or not 0 <= target <= 10:
            raise ValueError(f"Benchmark target for {path} must be finite human score in [0, 10]")
        indices.append(positions[path])
        targets.append(target)
    dataset.scenic_scores[indices] = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    return torch.utils.data.Subset(dataset, indices)


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
    device = resolve_device(args.device)

    test_ds = benchmark_dataset(args.dataset, args.labels_csv, args.benchmark_split_csv)
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
