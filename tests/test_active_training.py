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
    _config_hash,
    _restore_rng_state,
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
    assert result["changed_rows"] == 0
    assert result["supplemental_benchmark"] is None
    assert "supplemental_benchmark" not in result["hashes"]
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


def test_prepare_intersects_fixed_split_and_emits_exact_filtered_artifacts(
    tmp_path: Path,
) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    splits = pd.read_csv(split_csv).iloc[[0, 2, 4]].copy()
    splits.to_csv(split_csv, index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["geographic_splits"]["sha256"] = _sha256(split_csv)
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = prepare_active_dataset(handoff, run / "intersected.npz")

    assert result["dropped_without_split"] == 3
    assert result["counts"] == {
        "rows": 3,
        "train": 1,
        "val": 1,
        "test": 1,
        "human": 1,
        "weak": 2,
    }
    filtered_index = pd.read_csv(result["filtered_index_path"])
    filtered_labels = pd.read_csv(result["filtered_labels_path"])
    assert filtered_index["candidate_row_index"].tolist() == [0, 2, 4]
    assert filtered_index["image_path"].tolist() == [
        "tile-0.png",
        "tile-2.png",
        "tile-4.png",
    ]
    assert filtered_labels["scenic_score"].tolist() == [8, 3, 5]
    assert result["filtered_index_sha256"] == _sha256(
        Path(result["filtered_index_path"])
    )
    assert result["filtered_labels_sha256"] == _sha256(
        Path(result["filtered_labels_path"])
    )


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
    assert resumed["evaluated_splits"] == ["train", "val"]
    assert "test" not in resumed["metrics"]
    assert all("test" not in entry["metrics"] for entry in resumed["history"])
    assert resumed["human_sample_weight_multiplier"] == 1.0
    assert resumed["initial_checkpoint"] is None
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


def test_restore_rng_state_normalizes_non_cpu_tensor_and_validates_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=2, batch_size=2, max_steps=1, device=device),
    )

    resume_path = output / "resume.pt"
    ckpt = torch.load(resume_path, weights_only=False)

    if torch.backends.mps.is_available():
        ckpt["rng_state"]["torch"] = ckpt["rng_state"]["torch"].to("mps")
        torch.save(ckpt, resume_path)
        summary_path = output / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["resume_checkpoint_sha256"] = _sha256(resume_path)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        resumed = train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(epochs=2, batch_size=2, max_steps=3, device=device),
            resume=True,
        )
        assert resumed["state"] == "completed"

    passed_to_set_rng_state: torch.Tensor | None = None
    orig_set_rng_state = torch.set_rng_state

    def mock_set_rng_state(tensor: torch.Tensor) -> None:
        nonlocal passed_to_set_rng_state
        passed_to_set_rng_state = tensor
        orig_set_rng_state(tensor)

    monkeypatch.setattr(torch, "set_rng_state", mock_set_rng_state)

    state = dict(ckpt["rng_state"])
    if torch.backends.mps.is_available():
        state["torch"] = state["torch"].to("mps")

    _restore_rng_state(state)
    assert passed_to_set_rng_state is not None
    assert passed_to_set_rng_state.device.type == "cpu"

    bad_dtype_state = dict(ckpt["rng_state"])
    bad_dtype_state["torch"] = torch.tensor([1, 2, 3], dtype=torch.float32)
    with pytest.raises(ActiveTrainingError, match="uint8 ByteTensor"):
        _restore_rng_state(bad_dtype_state)

    bad_type_state = dict(ckpt["rng_state"])
    bad_type_state["torch"] = "not_a_tensor"
    with pytest.raises(ActiveTrainingError, match="uint8 ByteTensor"):
        _restore_rng_state(bad_type_state)


def _benchmark_rows() -> list[dict[str, Any]]:
    return [
        {"image_path": f"tile-{index}.png", "scenic_human_mean": mean, "split": split}
        for index, (mean, split) in enumerate(
            [
                (7.5, "train"),
                (6.25, "train"),
                (9.9, "val"),
                (1.0, "val"),
                (5.5, "test"),
                (2.0, "test"),
            ]
        )
    ]


def _write_benchmark(run: Path, rows: list[dict[str, Any]]) -> Path:
    path = run / "benchmark_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_prepare_merges_supplemental_train_and_preserves_val_test(
    tmp_path: Path,
) -> None:
    handoff, _, run = _fixture(tmp_path)
    benchmark = _write_benchmark(run, _benchmark_rows())
    result = prepare_active_dataset(
        handoff,
        run / "prepared.npz",
        supplemental_benchmark_path=benchmark,
        supplemental_benchmark_sha256=_sha256(benchmark),
    )
    assert result["changed_rows"] == 2
    assert result["counts"] == {
        "rows": 6,
        "train": 2,
        "val": 2,
        "test": 2,
        "human": 2,
        "weak": 4,
    }
    assert result["supplemental_benchmark"] == {
        "path": str(Path(benchmark).resolve()),
        "sha256": _sha256(benchmark),
        "expected_sha256": _sha256(benchmark),
        "rows": 6,
        "train": 2,
        "val": 2,
        "test": 2,
    }
    assert result["hashes"]["supplemental_benchmark"] == _sha256(benchmark)
    with np.load(run / "prepared.npz", allow_pickle=False) as arrays:
        assert arrays["scenic_scores"].tolist() == [7.5, 6.25, 3.0, 4.0, 5.0, 6.0]
        assert arrays["sample_weights"].tolist() == [4.0, 4.0, 1.0, 1.0, 1.0, 1.0]
        assert arrays["label_sources"].tolist() == [
            "human",
            "human",
            "weak",
            "weak",
            "weak",
            "weak",
        ]
        assert arrays["splits"].tolist() == [
            "train",
            "train",
            "val",
            "val",
            "test",
            "test",
        ]


def test_prepare_supplemental_requires_both_path_and_sha(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    benchmark = _write_benchmark(run, _benchmark_rows())
    with pytest.raises(ActiveTrainingError, match="must be provided together"):
        prepare_active_dataset(
            handoff, run / "prepared.npz", supplemental_benchmark_path=benchmark
        )
    with pytest.raises(ActiveTrainingError, match="must be provided together"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_sha256=_sha256(benchmark),
        )


def test_prepare_supplemental_verifies_hash_before_use(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    benchmark = _write_benchmark(run, _benchmark_rows())
    with pytest.raises(ActiveTrainingError, match="supplemental benchmark hash mismatch"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256="0" * 64,
        )
    with pytest.raises(
        ActiveTrainingError, match="must be null or a 64-character hex digest"
    ):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256="not-a-hash",
        )
    with pytest.raises(ActiveTrainingError, match="supplemental benchmark not found"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=run / "missing_benchmark.csv",
            supplemental_benchmark_sha256="0" * 64,
        )


def test_prepare_supplemental_split_mismatch_fails(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    rows[0]["split"] = "test"  # tile-0 is train in the immutable splits
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(
        ActiveTrainingError,
        match="supplemental benchmark split mismatch for tile-0.png: test != train",
    ):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )


def test_prepare_supplemental_unknown_duplicate_and_empty_identities_fail(
    tmp_path: Path,
) -> None:
    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    rows.append(
        {"image_path": "unknown-tile.png", "scenic_human_mean": 5.0, "split": "train"}
    )
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(ActiveTrainingError, match="references unknown image IDs"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )

    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    rows.append(rows[0])
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(ActiveTrainingError, match="duplicate image IDs"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )

    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    rows[0]["image_path"] = ""
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(ActiveTrainingError, match="empty image_path"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )


def test_prepare_supplemental_invalid_human_mean_fails(tmp_path: Path) -> None:
    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    rows[0]["scenic_human_mean"] = 11.0
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(
        ActiveTrainingError,
        match="scenic_human_mean must be in \\[0, 10\\]",
    ):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )

    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    rows[0]["scenic_human_mean"] = np.nan
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(ActiveTrainingError, match="must be finite"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )


def test_prepare_supplemental_requires_columns_and_train_rows(
    tmp_path: Path,
) -> None:
    handoff, _, run = _fixture(tmp_path)
    rows = _benchmark_rows()
    for row in rows:
        row.pop("scenic_human_mean", None)
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(ActiveTrainingError, match="missing required columns"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )

    handoff, _, run = _fixture(tmp_path)
    rows = [row for row in _benchmark_rows() if row["split"] != "train"]
    benchmark = _write_benchmark(run, rows)
    with pytest.raises(ActiveTrainingError, match="must contain train rows"):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )


def test_prepare_supplemental_train_identity_absent_from_prepared_fails(
    tmp_path: Path,
) -> None:
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
    benchmark = _write_benchmark(run, _benchmark_rows())
    with pytest.raises(
        ActiveTrainingError,
        match="supplemental benchmark train identities absent from prepared dataset",
    ):
        prepare_active_dataset(
            handoff,
            run / "prepared.npz",
            supplemental_benchmark_path=benchmark,
            supplemental_benchmark_sha256=_sha256(benchmark),
        )


def test_config_validates_initial_checkpoint_and_multiplier() -> None:
    with pytest.raises(ValueError, match="provided together"):
        ActiveTrainingConfig(initial_checkpoint_path="x.pt").validate()
    with pytest.raises(ValueError, match="provided together"):
        ActiveTrainingConfig(initial_checkpoint_sha256="a" * 64).validate()
    with pytest.raises(ValueError, match="64-character hex digest"):
        ActiveTrainingConfig(
            initial_checkpoint_path="x.pt", initial_checkpoint_sha256="nope"
        ).validate()
    with pytest.raises(ValueError, match="finite and positive"):
        ActiveTrainingConfig(human_sample_weight_multiplier=0).validate()
    with pytest.raises(ValueError, match="finite and positive"):
        ActiveTrainingConfig(human_sample_weight_multiplier=float("nan")).validate()
    with pytest.raises(ValueError, match="must be boolean"):
        ActiveTrainingConfig(evaluate_test_during_training=1).validate()
    valid = ActiveTrainingConfig(
        initial_checkpoint_path="x.pt",
        initial_checkpoint_sha256="a" * 64,
        human_sample_weight_multiplier=4.0,
        evaluate_test_during_training=True,
    )
    valid.validate()
    assert valid.as_dict()["human_sample_weight_multiplier"] == 4.0
    assert valid.as_dict()["initial_checkpoint_sha256"] == "a" * 64


def _write_initial_checkpoint(
    run: Path, architecture: dict[str, int], state: Any
) -> Path:
    path = run / "initial.pt"
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_state": "completed",
            "model_state_dict": state,
            **architecture,
        },
        path,
    )
    return path


def test_training_initializes_from_hash_bound_checkpoint(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    source = ScenicRegressionModel(vit_dim=4, terrain_dim=2, num_classes=3)
    initial = _write_initial_checkpoint(
        run,
        {"vit_dim": 4, "terrain_dim": 2, "num_classes": 3},
        source.state_dict(),
    )
    initial_sha = _sha256(initial)
    output = run / "training"
    config = ActiveTrainingConfig(
        epochs=1,
        batch_size=2,
        device="cpu",
        initial_checkpoint_path=str(initial),
        initial_checkpoint_sha256=initial_sha,
        human_sample_weight_multiplier=4.0,
    )
    summary = train_active_model(dataset, split_csv, output, config)
    assert summary["state"] == "completed"
    assert summary["human_sample_weight_multiplier"] == 4.0
    assert summary["initial_checkpoint"] == {
        "path": str(Path(initial).resolve()),
        "sha256": initial_sha,
        "requested_sha256": initial_sha,
    }
    assert summary["evaluated_splits"] == ["train", "val"]
    assert "test" not in summary["metrics"]
    checkpoint = torch.load(output / "candidate.pt", weights_only=False)
    assert checkpoint["initial_checkpoint_sha256"] == initial_sha
    assert checkpoint["human_sample_weight_multiplier"] == 4.0

    reused = train_active_model(dataset, split_csv, output, config, resume=True)
    assert reused["reused"] is True
    assert reused["initial_checkpoint"]["sha256"] == initial_sha


def test_training_initial_checkpoint_fail_closed(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    source = ScenicRegressionModel(vit_dim=4, terrain_dim=2, num_classes=3)
    state = source.state_dict()

    initial = _write_initial_checkpoint(
        run,
        {"vit_dim": 4, "terrain_dim": 2, "num_classes": 3, "hidden_dim": 256},
        state,
    )
    with pytest.raises(ActiveTrainingError, match="initial checkpoint hash mismatch"):
        train_active_model(
            dataset,
            split_csv,
            run / "bad_hash",
            ActiveTrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                initial_checkpoint_path=str(initial),
                initial_checkpoint_sha256="0" * 64,
            ),
        )
    with pytest.raises(ActiveTrainingError, match="initial checkpoint not found"):
        train_active_model(
            dataset,
            split_csv,
            run / "missing_file",
            ActiveTrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                initial_checkpoint_path=str(run / "absent.pt"),
                initial_checkpoint_sha256="0" * 64,
            ),
        )

    bad_arch = _write_initial_checkpoint(
        run,
        {"vit_dim": 4, "terrain_dim": 2, "num_classes": 3, "hidden_dim": 128},
        state,
    )
    with pytest.raises(
        ActiveTrainingError, match="initial checkpoint architecture is incompatible"
    ):
        train_active_model(
            dataset,
            split_csv,
            run / "bad_arch",
            ActiveTrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                initial_checkpoint_path=str(bad_arch),
                initial_checkpoint_sha256=_sha256(bad_arch),
            ),
        )

    wrong_state = ScenicRegressionModel(vit_dim=8, terrain_dim=5, num_classes=7)
    bad_state = _write_initial_checkpoint(
        run,
        {"vit_dim": 4, "terrain_dim": 2, "num_classes": 3, "hidden_dim": 256},
        wrong_state.state_dict(),
    )
    with pytest.raises(
        ActiveTrainingError, match="initial checkpoint model state is incompatible"
    ):
        train_active_model(
            dataset,
            split_csv,
            run / "bad_state",
            ActiveTrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                initial_checkpoint_path=str(bad_state),
                initial_checkpoint_sha256=_sha256(bad_state),
            ),
        )

    not_a_checkpoint = run / "not_a_checkpoint.pt"
    not_a_checkpoint.write_bytes(b"garbage")
    with pytest.raises(ActiveTrainingError):
        train_active_model(
            dataset,
            split_csv,
            run / "not_ckpt",
            ActiveTrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                initial_checkpoint_path=str(not_a_checkpoint),
                initial_checkpoint_sha256=_sha256(not_a_checkpoint),
            ),
        )


def test_training_multiplier_and_test_metric_suppression(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    summary = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=1, batch_size=2, device="cpu"),
    )
    assert summary["state"] == "completed"
    assert summary["human_sample_weight_multiplier"] == 1.0
    assert summary["evaluated_splits"] == ["train", "val"]
    assert "test" not in summary["metrics"]
    assert all("test" not in entry["metrics"] for entry in summary["history"])
    checkpoint = torch.load(output / "candidate.pt", weights_only=False)
    assert checkpoint["human_sample_weight_multiplier"] == 1.0
    assert checkpoint["initial_checkpoint_sha256"] is None

    output_explicit = run / "training_explicit"
    explicit = train_active_model(
        dataset,
        split_csv,
        output_explicit,
        ActiveTrainingConfig(
            epochs=1,
            batch_size=2,
            device="cpu",
            evaluate_test_during_training=True,
        ),
    )
    assert explicit["evaluated_splits"] == ["train", "val", "test"]
    assert "test" in explicit["metrics"]
    assert all("test" in entry["metrics"] for entry in explicit["history"])


def test_human_only_training_filters_train_to_human_rows(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    summary = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(
            epochs=1, batch_size=1, device="cpu", human_only_training=True
        ),
    )
    assert summary["state"] == "completed"
    assert summary["human_only_training"] is True
    assert summary["train_scope"] == {
        "mode": "human_only",
        "total_train_rows": 2,
        "human_train_rows": 1,
    }
    # Only the single human train row contributes: exactly one batch of one.
    assert summary["counts"]["train"] == 1
    assert summary["counts"]["val"] == 2
    assert summary["counts"]["test"] == 2
    assert summary["global_step"] == 1
    assert summary["metrics"]["train"]["count"] == 1
    assert summary["metrics"]["val"]["count"] == 2
    checkpoint = torch.load(output / "candidate.pt", weights_only=False)
    assert checkpoint["human_only_training"] is True
    assert checkpoint["counts"] == {"train": 1, "val": 2, "test": 2}
    summary_file = json.loads(
        (output / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary_file["human_only_training"] is True


def test_human_only_training_fails_closed_without_human_train_rows(
    tmp_path: Path,
) -> None:
    handoff, _, run = _fixture(tmp_path)
    names = [f"tile-{index}.png" for index in range(6)]
    # Move the sole human label to a val row so the train rows are all weak.
    pd.DataFrame(
        {
            "image_path": names,
            "scenic_human": [np.nan, np.nan, 8.0, np.nan, np.nan, np.nan],
            "skip": [False] * 6,
        }
    ).to_csv(run / "absolute_annotations.csv", index=False)
    pd.DataFrame(
        {
            "image_path": names,
            "scenic_score": [1.0, 2.0, 8.0, 4.0, 5.0, 6.0],
            "label_source": [
                "heuristic",
                "heuristic",
                "human_override",
                "heuristic",
                "heuristic",
                "heuristic",
            ],
            "scenic_human": [np.nan, np.nan, 8.0, np.nan, np.nan, np.nan],
        }
    ).to_csv(run / "mixed_labels.csv", index=False)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["artifacts"]["absolute_annotations"]["sha256"] = _sha256(
        run / "absolute_annotations.csv"
    )
    payload["artifacts"]["mixed_labels"]["sha256"] = _sha256(
        run / "mixed_labels.csv"
    )
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    with pytest.raises(
        ActiveTrainingError, match="at least one human-labeled train row"
    ):
        train_active_model(
            dataset,
            run / "geographic_splits.csv",
            run / "training",
            ActiveTrainingConfig(
                epochs=1, batch_size=1, device="cpu", human_only_training=True
            ),
        )


def test_human_only_training_resume_binding(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    paused = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(
            epochs=2,
            batch_size=1,
            max_steps=1,
            device="cpu",
            human_only_training=True,
        ),
    )
    assert paused["state"] == "paused"
    assert paused["global_step"] == 1
    assert paused["human_only_training"] is True

    # The scope is hash-bound: resuming with the flag off must fail closed.
    with pytest.raises(ActiveTrainingError, match="hash mismatch"):
        train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(
                epochs=2, batch_size=1, max_steps=8, device="cpu"
            ),
            resume=True,
        )

    resumed = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(
            epochs=2,
            batch_size=1,
            max_steps=8,
            device="cpu",
            human_only_training=True,
        ),
        resume=True,
    )
    assert resumed["state"] == "completed"
    # Cursor consistency holds against the filtered train length: one batch.
    assert resumed["global_step"] == 2
    assert resumed["human_only_training"] is True
    assert resumed["counts"] == {"train": 1, "val": 2, "test": 2}

    # Completed reuse is equally bound to the scope.
    with pytest.raises(
        ActiveTrainingError, match="hashes do not match requested inputs"
    ):
        train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(
                epochs=2, batch_size=1, max_steps=8, device="cpu"
            ),
            resume=True,
        )


def test_human_only_training_default_behavior_unchanged(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    summary = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(epochs=1, batch_size=1, device="cpu"),
    )
    assert summary["state"] == "completed"
    assert summary["human_only_training"] is False
    assert summary["train_scope"] == {
        "mode": "all",
        "total_train_rows": 2,
        "human_train_rows": 1,
    }
    # Weak train rows still contribute: two batches of one.
    assert summary["counts"] == {"train": 2, "val": 2, "test": 2}
    assert summary["global_step"] == 2
    checkpoint = torch.load(output / "candidate.pt", weights_only=False)
    assert checkpoint["human_only_training"] is False
    assert checkpoint["counts"]["train"] == 2


def test_human_only_training_is_hash_bound_and_validated() -> None:
    with pytest.raises(ValueError, match="human_only_training must be boolean"):
        ActiveTrainingConfig(human_only_training=1).validate()
    with pytest.raises(ValueError, match="human_only_training must be boolean"):
        ActiveTrainingConfig(human_only_training="yes").validate()
    enabled = ActiveTrainingConfig(human_only_training=True)
    enabled.validate()
    assert enabled.as_dict()["human_only_training"] is True
    assert _config_hash(ActiveTrainingConfig(), "cpu") != _config_hash(
        ActiveTrainingConfig(human_only_training=True), "cpu"
    )


def test_resume_requires_matching_human_weight_config(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    output = run / "training"
    train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(
            epochs=2,
            batch_size=2,
            max_steps=1,
            device="cpu",
            human_sample_weight_multiplier=4.0,
        ),
    )
    with pytest.raises(ActiveTrainingError, match="hash mismatch"):
        train_active_model(
            dataset,
            split_csv,
            output,
            ActiveTrainingConfig(
                epochs=2,
                batch_size=2,
                max_steps=5,
                device="cpu",
                human_sample_weight_multiplier=1.0,
            ),
            resume=True,
        )


def test_resume_does_not_require_initial_checkpoint_file(tmp_path: Path) -> None:
    handoff, split_csv, run = _fixture(tmp_path)
    dataset = run / "prepared.npz"
    prepare_active_dataset(handoff, dataset)
    source = ScenicRegressionModel(vit_dim=4, terrain_dim=2, num_classes=3)
    initial = _write_initial_checkpoint(
        run,
        {"vit_dim": 4, "terrain_dim": 2, "num_classes": 3, "hidden_dim": 256},
        source.state_dict(),
    )
    initial_sha = _sha256(initial)
    output = run / "training"
    config = ActiveTrainingConfig(
        epochs=2,
        batch_size=2,
        max_steps=1,
        device="cpu",
        initial_checkpoint_path=str(initial),
        initial_checkpoint_sha256=initial_sha,
    )
    paused = train_active_model(dataset, split_csv, output, config)
    assert paused["state"] == "paused"
    initial.unlink()
    resumed = train_active_model(
        dataset,
        split_csv,
        output,
        ActiveTrainingConfig(
            epochs=2,
            batch_size=2,
            max_steps=8,
            device="cpu",
            initial_checkpoint_path=str(initial),
            initial_checkpoint_sha256=initial_sha,
        ),
        resume=True,
    )
    assert resumed["state"] == "completed"
    assert resumed["initial_checkpoint"] is not None
    assert resumed["initial_checkpoint"]["sha256"] == initial_sha
