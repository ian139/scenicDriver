from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.active_learning.common import sha256_file
from src.active_learning.finalize import finalize_stage1
from src.active_learning.selection import audit_geographic_leakage, select_candidates


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_path": "z14/100_100.png",
                "region": "west",
                "z": 14,
                "x": 100,
                "y": 100,
                "heuristic_score": 0.1,
                "model_score": 0.9,
                "cluster_id": "a",
            },
            {
                "image_path": "z14/102_100.png",
                "region": "west",
                "z": 14,
                "x": 102,
                "y": 100,
                "heuristic_score": 0.8,
                "model_score": 0.2,
                "cluster_id": "b",
            },
            {
                "image_path": "z14/110_100.png",
                "region": "south",
                "z": 14,
                "x": 110,
                "y": 100,
                "heuristic_score": 0.4,
                "model_score": 0.5,
                "cluster_id": "c",
            },
            {
                "image_path": "z14/120_100.png",
                "region": "south",
                "z": 14,
                "x": 120,
                "y": 100,
                "heuristic_score": 0.5,
                "model_score": 0.5,
                "cluster_id": "d",
            },
        ]
    )


def test_selection_is_deterministic_and_does_not_fabricate_uncertainty() -> None:
    candidates = _candidates()
    first = select_candidates(candidates, batch_size=3, seed=11, run_name="fixture")
    second = select_candidates(candidates, batch_size=3, seed=11, run_name="fixture")
    pd.testing.assert_frame_equal(first, second)
    assert first["uncertainty_score"].eq(0).all()
    assert first["uncertainty_observed"].eq(False).all()
    assert first["selection_reason"].notna().all()
    assert first["batch_id"].nunique() == 1


def test_selection_excludes_unavailable_pairs() -> None:
    candidates = _candidates()
    candidates["satellite_present"] = [True, False, True, True]
    candidates["terrain_present"] = [True, True, True, True]
    selected = select_candidates(candidates, batch_size=4, seed=11, run_name="fixture")
    assert "z14/102_100.png" not in set(selected["image_path"])


def test_leakage_audit_detects_duplicate_and_adjacent_cross_split_tiles() -> None:
    split = pd.DataFrame(
        [
            {"image_path": "a.png", "z": 14, "x": 10, "y": 10, "split": "train"},
            {"image_path": "a-copy.png", "z": 14, "x": 10, "y": 10, "split": "test"},
            {"image_path": "b.png", "z": 14, "x": 11, "y": 10, "split": "val"},
        ]
    )
    report = audit_geographic_leakage(split)
    assert report["valid"] is False
    assert report["duplicate_cross_split"] is True
    assert report["adjacent_cross_split"] is True


def test_finalizer_fails_closed_for_incomplete_run(tmp_path) -> None:
    handoff = finalize_stage1(tmp_path, run_name="fixture")
    assert handoff["ready_for_stage2"] is False
    assert handoff["blockers"]
    assert (tmp_path / "stage1_handoff.json").exists()


def test_finalizer_accepts_complete_validated_fixture(tmp_path: Path) -> None:
    paths = ["a.png", "b.png", "c.png"]
    for name in paths:
        (tmp_path / name).write_bytes(b"image")
    for name in (
        "region_manifest.json",
        "inventory_report.json",
        "batch_manifest.json",
        "selection_diagnostics.json",
    ):
        (tmp_path / name).write_text(json.dumps({"schema_version": 1}))
    (tmp_path / "acquisition_preflight.json").write_text(
        json.dumps({"schema_version": 1, "budget_valid": True}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "region": "fixture",
                "z": 14,
                "x": 10 + index * 10,
                "y": 20,
                "lat": 40 + index,
                "lon": -75 - index,
                "satellite_path": name,
                "terrain_path": name,
                "satellite_present": True,
                "terrain_present": True,
            }
            for index, name in enumerate(paths)
        ]
    ).to_csv(tmp_path / "tile_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "image_path": name,
                "selection_reason": "fixture",
                "selection_score": 1,
                "selection_rank": index + 1,
                "batch_id": "batch",
                "run_id": "run",
            }
            for index, name in enumerate(paths)
        ]
    ).to_csv(tmp_path / "annotation_batch.csv", index=False)
    pd.DataFrame(
        [
            {
                "image_path": name,
                "z": 14,
                "x": 10 + index * 10,
                "y": 20,
                "split": split,
            }
            for index, (name, split) in enumerate(
                zip(paths, ("train", "val", "test"), strict=True)
            )
        ]
    ).to_csv(tmp_path / "geographic_splits.csv", index=False)
    (tmp_path / "leakage_audit.json").write_text(
        json.dumps({"schema_version": 1, "valid": True})
    )
    candidate = pd.DataFrame(
        [
            {
                "image_path": name,
                "source_identity": f"fixture/{name}",
                "satellite_path": name,
                "terrain_path": name,
                "region": "fixture",
                "score_status": "scored",
                "selector_eligible": True,
                "heuristic_score": 4.0 + index,
                "scenic_score": 4.0 + index,
                "scenic_score_heuristic": 4.0 + index,
                "regression_prediction": 5.0 + index,
                "normalized_class_entropy": 0.5,
                "embedding_row_index": index,
            }
            for index, name in enumerate(paths)
        ]
    )
    candidate.to_csv(tmp_path / "candidate_pool.csv", index=False)
    np.savez_compressed(
        tmp_path / "feature_embeddings.npz",
        embeddings=np.ones((len(paths), 2), dtype=np.float32),
        row_indices=np.arange(len(paths), dtype=np.int64),
    )
    scoring_manifest = {
        "schema_version": 1,
        "state": {"complete": True, "ready_for_selection": True},
        "counts": {
            "manifest_rows": len(paths),
            "scored_rows": len(paths),
            "missing_rows": 0,
            "error_rows": 0,
        },
        "artifacts": {
            "candidate_pool.csv": {
                "path": "candidate_pool.csv",
                "sha256": sha256_file(tmp_path / "candidate_pool.csv"),
            },
            "feature_embeddings.npz": {
                "path": "feature_embeddings.npz",
                "sha256": sha256_file(tmp_path / "feature_embeddings.npz"),
            },
        },
    }
    (tmp_path / "scoring_manifest.json").write_text(
        json.dumps(scoring_manifest),
        encoding="utf-8",
    )
    annotations = pd.DataFrame(
        [
            {
                "image_path": name,
                "scenic_human": 5 + index,
                "confidence": "high",
                "skip": False,
                "annotator_id": "fixture",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            }
            for index, name in enumerate(paths)
        ]
    )
    annotations.to_csv(tmp_path / "annotations.csv", index=False)
    pd.DataFrame(
        [
            {"image_path": name, "scenic_score": 5 + index}
            for index, name in enumerate(paths)
        ]
    ).to_csv(tmp_path / "mixed_labels.csv", index=False)
    pd.DataFrame(
        [
            {"image_path": name, "scenic_human": 5 + index}
            for index, name in enumerate(paths)
        ]
    ).to_csv(tmp_path / "benchmark.csv", index=False)
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": str(checkpoint)}}))
    registry_before = registry.read_bytes()

    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=tmp_path / "annotations.csv",
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )

    assert handoff["ready_for_stage2"] is True
    assert not handoff["blockers"]
    assert registry.read_bytes() == registry_before
