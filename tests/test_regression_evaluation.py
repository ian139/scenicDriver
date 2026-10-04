"""Validation-membership agreement between baseline regression training and evaluation.

The reported validation metrics MUST be computed on exactly the rows the trained
model validated on during training: the shared, explicit validation-membership
derivation over the filtered training population, persisted by training as a
split artifact and consumed by evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.modeling import evaluate_regression_baseline, train_regression_baseline
from scripts.modeling.regression_split import (
    derive_validation_membership,
    load_validation_membership,
    training_population_mask,
    write_split_artifact,
)
from src.scenic_scorer import regression


def _write_dataset(
    path: Path, n: int, splits: list[str] | None = None
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    arrays: dict[str, np.ndarray] = {
        "image_paths": np.array([f"img_{i:02d}" for i in range(n)], dtype=str),
        "vit_embeddings": rng.uniform(size=(n, 2)).astype(np.float32),
        "terrain_features": rng.uniform(size=(n, 1)).astype(np.float32),
        "class_logits": rng.uniform(size=(n, 3)).astype(np.float32),
        "scenic_scores": rng.uniform(0.0, 10.0, size=(n,)).astype(np.float32),
        "sample_weights": rng.uniform(0.5, 4.0, size=(n,)).astype(np.float32),
    }
    if splits is not None:
        arrays["splits"] = np.array(splits, dtype=str)
    np.savez(path, **arrays)
    return arrays


def _write_benchmark(path: Path, rows: dict[str, str]) -> None:
    lines = ["image_path,split"]
    lines.extend(f"{identity},{split}" for identity, split in rows.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _expected_val_loss(
    arrays: dict[str, np.ndarray], checkpoint: dict, rows: list[int], weighted: bool
) -> float:
    model = regression.ScenicRegressionModel(vit_dim=2, terrain_dim=1, num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        pred = model(
            torch.from_numpy(arrays["vit_embeddings"][rows]),
            torch.from_numpy(arrays["terrain_features"][rows]),
            torch.from_numpy(arrays["class_logits"][rows]),
        ).numpy().reshape(-1)
    error = (pred - arrays["scenic_scores"][rows]) ** 2
    weights = arrays["sample_weights"][rows] if weighted else np.ones_like(error)
    return float((error * weights).sum() / weights.sum())


def test_derive_validation_membership_filters_then_splits(tmp_path: Path) -> None:
    n = 24
    splits = ["train"] * n
    splits[0] = "val"  # embedded held-out, absent from the benchmark
    splits[1] = "test"  # embedded held-out, absent from the benchmark
    splits[2] = "val"
    splits[3] = "test"
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset, n, splits)
    benchmark = tmp_path / "benchmark.csv"
    _write_benchmark(
        benchmark, {"img_02": "val", "img_03": "test", "img_04": "train"}
    )

    membership = derive_validation_membership(
        dataset, benchmark, val_split=0.25, seed=7
    )

    kept = training_population_mask(dataset, benchmark)
    kept_paths = {f"img_{i:02d}" for i in range(n) if kept[i]}
    excluded = {"img_00", "img_01", "img_02", "img_03"}
    all_rows = set(membership.train_paths) | set(membership.val_paths)
    assert all_rows == kept_paths
    assert excluded.isdisjoint(all_rows)
    assert not (set(membership.train_paths) & set(membership.val_paths))
    # same seed/split logic the trainer applies, over the same filtered population
    _, val_idx = regression.validation_split_indices(int(kept.sum()), 0.25, 7)
    assert list(membership.val_indices) == val_idx
    assert list(membership.val_dataset_indices) == [
        int(i) for i in np.flatnonzero(kept)[val_idx]
    ]


def test_trainer_uses_explicit_validation_membership(tmp_path: Path) -> None:
    arrays = _write_dataset(tmp_path / "data.npz", n=20)
    train_idx = list(range(15))
    val_idx = [15, 16, 17, 18, 19]
    output = tmp_path / "model.pt"

    regression.train_regression_model(
        tmp_path / "data.npz",
        output,
        epochs=1,
        batch_size=2,
        device="cpu",
        num_workers=0,
        train_indices=train_idx,
        val_indices=val_idx,
    )

    checkpoint = torch.load(output, weights_only=False)
    expected = _expected_val_loss(arrays, checkpoint, val_idx, weighted=True)
    assert checkpoint["best_val_loss"] == pytest.approx(expected, rel=1e-5)


def test_trainer_rejects_partial_explicit_membership(tmp_path: Path) -> None:
    data_path = tmp_path / "data.npz"
    _write_dataset(data_path, n=20)

    with pytest.raises(ValueError, match="together"):
        regression.train_regression_model(
            data_path,
            tmp_path / "model.pt",
            epochs=1,
            device="cpu",
            num_workers=0,
            train_indices=list(range(15)),
        )


def test_evaluation_reproduces_training_validation_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 24
    splits = ["train"] * n
    splits[0] = "val"  # embedded held-out, absent from the benchmark
    splits[1] = "test"  # embedded held-out, absent from the benchmark
    splits[2] = "val"
    splits[3] = "test"
    dataset = tmp_path / "dataset.npz"
    arrays = _write_dataset(dataset, n, splits)
    benchmark = tmp_path / "benchmark.csv"
    _write_benchmark(
        benchmark, {"img_02": "val", "img_03": "test", "img_04": "train"}
    )
    checkpoint = tmp_path / "checkpoint.pt"
    artifact = tmp_path / "checkpoint.split.json"
    metrics_json = tmp_path / "metrics.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_regression_baseline.py",
            "--dataset",
            str(dataset),
            "--benchmark-split-csv",
            str(benchmark),
            "--output",
            str(checkpoint),
            "--split-artifact",
            str(artifact),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    train_regression_baseline.main()

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    val_paths = payload["val_image_paths"]
    all_rows = set(payload["train_image_paths"]) | set(val_paths)
    held_out = {"img_00", "img_01", "img_02", "img_03"}
    assert held_out.isdisjoint(all_rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_regression_baseline.py",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--split-artifact",
            str(artifact),
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--metrics-json",
            str(metrics_json),
        ],
    )
    evaluate_regression_baseline.main()

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert metrics["samples"] == len(val_paths)
    assert metrics["split_artifact"] == str(artifact)

    # the reported metrics are computed on exactly the persisted validation rows
    row_of = {path: i for i, path in enumerate(arrays["image_paths"].tolist())}
    rows = [row_of[path] for path in val_paths]
    ckpt = torch.load(checkpoint, weights_only=False)
    model = regression.ScenicRegressionModel(vit_dim=2, terrain_dim=1, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        pred = model(
            torch.from_numpy(arrays["vit_embeddings"][rows]),
            torch.from_numpy(arrays["terrain_features"][rows]),
            torch.from_numpy(arrays["class_logits"][rows]),
        ).numpy().reshape(-1).astype(np.float64)
    target = arrays["scenic_scores"][rows].astype(np.float64)
    assert metrics["mae"] == pytest.approx(float(np.mean(np.abs(pred - target))), rel=1e-4)
    assert metrics["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((pred - target) ** 2))), rel=1e-4
    )


def test_split_artifact_round_trip_and_drift_rejected(tmp_path: Path) -> None:
    n = 20
    dataset = tmp_path / "dataset.npz"
    _write_dataset(dataset, n)
    benchmark = tmp_path / "benchmark.csv"
    _write_benchmark(benchmark, {"img_00": "val", "img_01": "test"})
    artifact = tmp_path / "split.json"

    membership = derive_validation_membership(dataset, benchmark, val_split=0.25, seed=3)
    write_split_artifact(
        artifact, membership, dataset_path=dataset, benchmark_split_csv=benchmark
    )

    loaded = load_validation_membership(dataset, artifact)
    assert loaded.val_paths == membership.val_paths
    assert loaded.train_paths == membership.train_paths

    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered["val_image_paths"] = tampered["val_image_paths"][::-1]
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="split artifact"):
        load_validation_membership(dataset, tampered_path)

    other_dataset = tmp_path / "other.npz"
    _write_dataset(other_dataset, n)
    with pytest.raises(ValueError, match="dataset"):
        load_validation_membership(other_dataset, artifact)

    with pytest.raises(FileNotFoundError):
        load_validation_membership(dataset, tmp_path / "missing.json")


def test_evaluation_cli_has_no_split_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_regression_baseline.py",
            "--dataset",
            "dataset.npz",
            "--checkpoint",
            "checkpoint.pt",
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_regression_baseline.parse_args()
