"""Focused contracts for strict active dataset preparation and continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import pytest

from src.scenic_scorer.regression import ScenicRegressionModel
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
    run.mkdir(exist_ok=True)
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
    mixed = pd.DataFrame(
        {
            "image_path": names,
            "scenic_score": [8.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label_source": [
                "human_override",
                "heuristic",
                "heuristic",
                "heuristic",
                "heuristic",
                "heuristic",
            ],
            "scenic_human": [8.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    mixed.to_csv(run / "mixed_labels.csv", index=False)
    pd.DataFrame(
        {
            "image_path": names,
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    ).to_csv(run / "geographic_splits.csv", index=False)
    checkpoint = run / "baseline.pt"
    checkpoint.write_bytes(b"baseline checkpoint")
    registry = run / "registry.json"
    registry.write_text(
        json.dumps({"active": {"checkpoint": str(checkpoint)}}), encoding="utf-8"
    )
    artifacts = {}
    for key in (
        "candidate_pool",
        "feature_embeddings",
        "absolute_annotations",
        "mixed_labels",
        "geographic_splits",
        "baseline_registry",
    ):
        path = {
            "candidate_pool": run / "candidate_pool.csv",
            "feature_embeddings": run / "feature_embeddings.npz",
            "absolute_annotations": run / "absolute_annotations.csv",
            "mixed_labels": run / "mixed_labels.csv",
            "geographic_splits": run / "geographic_splits.csv",
            "baseline_registry": registry,
        }[key]
        artifacts[key] = {"path": path.name, "sha256": _sha256(path), "required": True}
    handoff = {
        "schema_version": 1,
        "run_root": str(run),
        "ready_for_stage2": True,
        "readiness": {
            "data_complete": True,
            "annotations_valid": True,
            "splits_valid": True,
            "benchmark_valid": True,
        },
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
    assert result["counts"] == {
        "rows": 6,
        "train": 2,
        "val": 2,
        "test": 2,
        "human": 1,
        "weak": 5,
    }
    with np.load(run / "prepared.npz", allow_pickle=False) as arrays:
        assert arrays["image_paths"].tolist() == [
            f"tile-{index}.png" for index in range(6)
        ]
        assert arrays["scenic_scores"].tolist() == [8, 2, 3, 4, 5, 6]
        assert arrays["sample_weights"].tolist() == [4, 1, 1, 1, 1, 1]
        assert arrays["splits"].tolist() == [
            "train",
            "train",
            "val",
            "val",
            "test",
            "test",
        ]
        assert arrays["label_sources"].tolist() == [
            "human",
            "weak",
            "weak",
            "weak",
            "weak",
            "weak",
        ]
    assert split_csv.exists()


def test_prepare_aggregates_multiple_human_annotations(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    annotations = pd.read_csv(run / "absolute_annotations.csv")
    annotations = pd.concat(
        [
            annotations,
            pd.DataFrame(
                [{"image_path": "tile-0.png", "scenic_human": 4, "skip": False}]
            ),
        ],
        ignore_index=True,
    )
    annotations.to_csv(run / "absolute_annotations.csv", index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["absolute_annotations"]["sha256"] = _sha256(
        run / "absolute_annotations.csv"
    )
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    prepare_active_dataset(handoff, run / "aggregated.npz")

    with np.load(run / "aggregated.npz", allow_pickle=False) as arrays:
        assert arrays["scenic_scores"][0] == pytest.approx(6.0)
        assert arrays["label_sources"][0] == "human"


def test_prepare_uses_hashed_filtered_index_only(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    filtered = run / "filtered_index.csv"
    pd.DataFrame({"candidate_row_index": [0, 2, 4]}).to_csv(filtered, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["filtered_index"] = {
        "path": filtered.name,
        "sha256": _sha256(filtered),
        "required": True,
    }
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = prepare_active_dataset(handoff, run / "filtered.npz")

    assert result["counts"] == {
        "rows": 3,
        "train": 1,
        "val": 1,
        "test": 1,
        "human": 1,
        "weak": 2,
    }
    with np.load(run / "filtered.npz", allow_pickle=False) as arrays:
        assert arrays["image_paths"].tolist() == [
            "tile-0.png",
            "tile-2.png",
            "tile-4.png",
        ]
        assert arrays["vit_embeddings"][:, 0].tolist() == [0, 8, 16]


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
    paused = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=1, batch_size=1, max_steps=1, device="cpu"),
    )
    assert paused["state"] == "paused"
    assert paused["global_step"] == 1
    assert paused["metrics"] == {}
    resumed = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=1, batch_size=1, max_steps=32, device="cpu"),
        resume=True,
    )
    assert resumed["state"] == "completed"
    reused = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=1, batch_size=1, max_steps=1, device="cpu"),
        resume=True,
    )
    assert reused["reused"] is True
    assert Path(reused["candidate_checkpoint"]).exists()


def test_prepare_preserves_historical_human_label_outside_current_batch(
    tmp_path: Path,
) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    names = [f"tile-{index}.png" for index in range(6)]
    mixed = pd.DataFrame(
        {
            "image_path": names,
            "scenic_score": [8.0, 8.5, 3.0, 4.0, 5.0, 6.0],
            "label_source": [
                "human_override",
                "human_override",
                "heuristic",
                "heuristic",
                "heuristic",
                "heuristic",
            ],
            "scenic_human": [8.0, 8.5, np.nan, np.nan, np.nan, np.nan],
        }
    )
    mixed_path = run / "mixed_labels.csv"
    mixed.to_csv(mixed_path, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["mixed_labels"]["sha256"] = _sha256(mixed_path)
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = prepare_active_dataset(handoff, run / "historical_preserved.npz")
    assert result["counts"]["human"] == 2
    assert result["counts"]["weak"] == 4

    with np.load(run / "historical_preserved.npz", allow_pickle=False) as arrays:
        assert arrays["label_sources"].tolist() == [
            "human",
            "human",
            "weak",
            "weak",
            "weak",
            "weak",
        ]
        assert arrays["sample_weights"].tolist() == [4.0, 4.0, 1.0, 1.0, 1.0, 1.0]
        assert arrays["scenic_scores"].tolist() == [8.0, 8.5, 3.0, 4.0, 5.0, 6.0]


def test_prepare_fails_on_malformed_or_hash_mismatched_mixed_labels(
    tmp_path: Path,
) -> None:
    handoff, _, run = _fixture(tmp_path)
    mixed_path = run / "mixed_labels.csv"
    mixed_path.write_text(
        "image_path,scenic_score,label_source\ntile-0.png,9.0,human_override\n",
        encoding="utf-8",
    )
    with pytest.raises(ActiveTrainingError, match="hash mismatch"):
        prepare_active_dataset(handoff, run / "prepared_fail.npz")

    handoff, _, run = _fixture(tmp_path)
    mixed_path = run / "mixed_labels.csv"
    bad_schema = pd.DataFrame(
        {"image_path": [f"tile-{i}.png" for i in range(6)], "scenic_score": [1.0] * 6}
    )
    bad_schema.to_csv(mixed_path, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["mixed_labels"]["sha256"] = _sha256(mixed_path)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ActiveTrainingError, match="missing required columns"):
        prepare_active_dataset(handoff, run / "prepared_fail.npz")

    handoff, _, run = _fixture(tmp_path)
    mixed_path = run / "mixed_labels.csv"
    unknown_df = pd.DataFrame(
        {
            "image_path": [f"tile-{i}.png" for i in range(6)] + ["unknown-tile.png"],
            "scenic_score": [1.0] * 7,
            "label_source": ["heuristic"] * 7,
        }
    )
    unknown_df.to_csv(mixed_path, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["mixed_labels"]["sha256"] = _sha256(mixed_path)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ActiveTrainingError, match="reference unknown image IDs"):
        prepare_active_dataset(handoff, run / "prepared_fail.npz")

    handoff, _, run = _fixture(tmp_path)
    mixed_path = run / "mixed_labels.csv"
    dup_df = pd.DataFrame(
        {
            "image_path": ["tile-0.png", "tile-0.png"]
            + [f"tile-{i}.png" for i in range(2, 6)],
            "scenic_score": [1.0] * 6,
            "label_source": ["heuristic"] * 6,
        }
    )
    dup_df.to_csv(mixed_path, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["mixed_labels"]["sha256"] = _sha256(mixed_path)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ActiveTrainingError, match="duplicate image_path identities"):
        prepare_active_dataset(handoff, run / "prepared_fail.npz")


def test_prepare_fails_on_stale_weak_mixed_score_mutation(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    mixed_path = run / "mixed_labels.csv"
    stale_mixed = pd.DataFrame(
        {
            "image_path": [f"tile-{i}.png" for i in range(6)],
            "scenic_score": [8.0, 9.0, 3.0, 4.0, 5.0, 6.0],
            "label_source": [
                "human_override",
                "heuristic",
                "heuristic",
                "heuristic",
                "heuristic",
                "heuristic",
            ],
            "scenic_human": [8.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    stale_mixed.to_csv(mixed_path, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["mixed_labels"]["sha256"] = _sha256(mixed_path)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ActiveTrainingError,
        match="mixed labels weak scenic_score mismatch against candidate pool",
    ):
        prepare_active_dataset(handoff, run / "prepared_stale.npz")


class DeadlineExceededError(TimeoutError):
    pass


def test_deadline_exceeded_error_propagates_as_resumable_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"

    call_count = 0
    orig_forward = ScenicRegressionModel.forward

    def mock_forward(self: ScenicRegressionModel, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise DeadlineExceededError("orchestrator deadline exceeded")
        return orig_forward(self, *args, **kwargs)

    monkeypatch.setattr(ScenicRegressionModel, "forward", mock_forward)

    paused = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=2, batch_size=2, device="cpu"),
    )
    assert paused["state"] == "paused"
    assert "DeadlineExceededError" in paused["stop_reason"]

    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "paused"

    monkeypatch.undo()
    resumed = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=2, batch_size=2, device="cpu"),
        resume=True,
    )
    assert resumed["state"] == "completed"


def test_resume_validates_summary_hashes_and_cursor_consistency(
    tmp_path: Path,
) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=2, batch_size=2, max_steps=1, device="cpu"),
    )

    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["dataset_sha256"] = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ActiveTrainingError, match="summary input hash mismatch"):
        train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(epochs=2, batch_size=2, max_steps=5, device="cpu"),
            resume=True,
        )

    summary["dataset_sha256"] = _sha256(dataset)
    summary["resume_checkpoint_sha256"] = (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        ActiveTrainingError,
        match="resume checkpoint hash mismatch against training summary",
    ):
        train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(epochs=2, batch_size=2, max_steps=5, device="cpu"),
            resume=True,
        )

    summary["resume_checkpoint_sha256"] = _sha256(output / "resume.pt")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    resume_path = output / "resume.pt"
    ckpt = torch.load(resume_path, weights_only=False)
    ckpt["global_step"] = 999
    torch.save(ckpt, resume_path)
    summary["resume_checkpoint_sha256"] = _sha256(resume_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        ActiveTrainingError,
        match="global_step is inconsistent with epoch and batch cursors",
    ):
        train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(epochs=2, batch_size=2, max_steps=5, device="cpu"),
            resume=True,
        )
