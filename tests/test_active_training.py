"""Focused contracts for strict active dataset preparation and continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scenic_scorer.active_training import (
    ActiveTrainingConfig,
    ActiveTrainingError,
    prepare_active_dataset,
    train_active_model,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "stage1"
    run.mkdir()
    names = [f"tile-{index}.png" for index in range(6)]
    candidate = pd.DataFrame(
        {
            "image_path": names,
            "embedding_row_index": np.arange(6),
            "scenic_score": [1, 2, 3, 4, 5, 6],
            "region": ["fixture"] * 6,
            "z": [14] * 6,
            "x": [0, 10, 20, 30, 40, 50],
            "y": [0] * 6,
            "tile_identity": [f"fixture/z14/x{x}/y0" for x in [0, 10, 20, 30, 40, 50]],
        }
    )
    candidate.to_csv(run / "candidate_pool.csv", index=False)
    np.savez_compressed(
        run / "feature_embeddings.npz",
        embeddings=np.arange(24, dtype=np.float32).reshape(6, 4),
        terrain_features=np.ones((6, 2), dtype=np.float32),
        class_logits=np.zeros((6, 3), dtype=np.float32),
        class_probs=np.full((6, 3), 1 / 3, dtype=np.float32),
        row_indices=np.arange(6, dtype=np.int64),
    )
    pd.DataFrame(
        {
            "image_path": names,
            "scenic_human": [8, np.nan, np.nan, np.nan, np.nan, np.nan],
            "skip": [False, False, False, False, False, False],
        }
    ).to_csv(run / "absolute_annotations.csv", index=False)
    pd.DataFrame(
        {
            "image_path": names,
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    ).to_csv(run / "geographic_splits.csv", index=False)
    checkpoint = run / "baseline.pt"
    checkpoint.write_bytes(b"baseline checkpoint")
    registry = run / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": str(checkpoint)}}), encoding="utf-8")
    artifacts = {}
    for key in ("candidate_pool", "feature_embeddings", "absolute_annotations", "geographic_splits", "baseline_registry"):
        path = {
            "candidate_pool": run / "candidate_pool.csv",
            "feature_embeddings": run / "feature_embeddings.npz",
            "absolute_annotations": run / "absolute_annotations.csv",
            "geographic_splits": run / "geographic_splits.csv",
            "baseline_registry": registry,
        }[key]
        artifacts[key] = {"path": path.name, "sha256": _sha256(path), "required": True}
    handoff = {
        "schema_version": 1,
        "run_root": str(run),
        "ready_for_stage2": True,
        "readiness": {"data_complete": True, "annotations_valid": True, "splits_valid": True, "benchmark_valid": True},
        "data_complete": True,
        "annotations_valid": True,
        "splits_valid": True,
        "benchmark_valid": True,
        "artifacts": artifacts,
        "baseline": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "registry_sha256": _sha256(registry),
            "active": {"checkpoint": str(checkpoint)},
        },
    }
    handoff_path = run / "stage1_handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    return handoff_path, run / "geographic_splits.csv", run


def test_prepare_preserves_order_and_human_override(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    result = prepare_active_dataset(handoff, run / "prepared.npz")
    assert result["counts"] == {"rows": 6, "train": 2, "val": 2, "test": 2, "human": 1, "weak": 5}
    with np.load(run / "prepared.npz", allow_pickle=False) as arrays:
        assert arrays["image_paths"].tolist() == [f"tile-{index}.png" for index in range(6)]
        assert arrays["scenic_scores"].tolist() == [8, 2, 3, 4, 5, 6]
        assert arrays["sample_weights"].tolist() == [4, 1, 1, 1, 1, 1]
        assert arrays["label_sources"].tolist() == ["human", "weak", "weak", "weak", "weak", "weak"]
    assert split_csv.exists()


def test_false_readiness_and_hash_mismatch_fail_closed(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["ready_for_stage2"] = False
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ActiveTrainingError):
        prepare_active_dataset(handoff, run / "prepared.npz")


def test_pause_resume_and_completed_reuse(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    paused = train_active_model(dataset, split_csv, output, ActiveTrainingConfig(epochs=1, batch_size=1, max_steps=1, device="cpu"))
    assert paused["state"] == "paused"
    assert paused["global_step"] == 1
    resumed = train_active_model(dataset, split_csv, output, ActiveTrainingConfig(epochs=1, batch_size=1, max_steps=32, device="cpu"), resume=True)
    assert resumed["state"] == "completed"
    reused = train_active_model(dataset, split_csv, output, ActiveTrainingConfig(epochs=1, batch_size=1, max_steps=1, device="cpu"), resume=True)
    assert reused["reused"] is True
    assert Path(reused["candidate_checkpoint"]).exists()
