from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.active_learning.common import sha256_bytes, sha256_file
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
    selection_contract = {"selected_identities": paths}
    selection_digest = sha256_bytes(
        json.dumps(selection_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    batch_id = f"batch-{selection_digest[:16]}"
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
                "batch_id": batch_id,
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
            {
                "image_path": name,
                "scenic_score": 5 + index,
                "label_source": "human_override",
            }
            for index, name in enumerate(paths)
        ]
    ).to_csv(tmp_path / "mixed_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "image_path": name,
                "scenic_human": 5 + index,
                "split": ("train", "val", "test")[index],
            }
            for index, name in enumerate(paths)
        ]
    ).to_csv(tmp_path / "benchmark.csv", index=False)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "source_hashes": {
                    "annotations_csv": sha256_file(tmp_path / "annotations.csv"),
                    "geographic_splits_csv": sha256_file(
                        tmp_path / "geographic_splits.csv"
                    ),
                    "benchmark_split_csv": sha256_file(tmp_path / "benchmark.csv"),
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": str(checkpoint)}}))
    registry_before = registry.read_bytes()

    (tmp_path / "batch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_count": len(paths),
                "batch_id": batch_id,
                "selection_contract": selection_contract,
                "selection_contract_sha256": selection_digest,
                "candidate_input": {
                    "path": "candidate_pool.csv",
                    "sha256": sha256_file(tmp_path / "candidate_pool.csv"),
                },
                "outputs": {
                    "annotation_batch_csv": {
                        "path": str(tmp_path / "annotation_batch.csv"),
                        "sha256": sha256_file(tmp_path / "annotation_batch.csv"),
                        "bytes": int(
                            (tmp_path / "annotation_batch.csv").stat().st_size
                        ),
                    },
                    "geographic_splits_csv": {
                        "path": str(tmp_path / "geographic_splits.csv"),
                        "sha256": sha256_file(tmp_path / "geographic_splits.csv"),
                        "bytes": int(
                            (tmp_path / "geographic_splits.csv").stat().st_size
                        ),
                    },
                    "leakage_audit_json": {
                        "path": str(tmp_path / "leakage_audit.json"),
                        "sha256": sha256_file(tmp_path / "leakage_audit.json"),
                        "bytes": int((tmp_path / "leakage_audit.json").stat().st_size),
                    },
                },
            }
        )
    )
    (tmp_path / "selection_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_input": {
                    "path": "candidate_pool.csv",
                    "sha256": sha256_file(tmp_path / "candidate_pool.csv"),
                },
            }
        )
    )
    (tmp_path / "scoring_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": {"complete": True, "ready_for_selection": True},
                "counts": {
                    "manifest_rows": len(paths),
                    "scored_rows": len(paths),
                    "missing_rows": 0,
                    "error_rows": 0,
                },
                "source": {
                    "tile_manifest": {
                        "path": "tile_manifest.csv",
                        "sha256": sha256_file(tmp_path / "tile_manifest.csv"),
                    }
                },
                "models": {
                    "regression_checkpoint_sha256": sha256_file(checkpoint),
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
        )
    )
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=tmp_path / "annotations.csv",
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    handoff_bytes = (tmp_path / "stage1_handoff.json").read_bytes()
    annotation_snapshot_bytes = (tmp_path / "absolute_annotations.csv").read_bytes()
    filtered_index_bytes = (tmp_path / "filtered_index.csv").read_bytes()
    identical = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=tmp_path / "annotations.csv",
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    assert identical == handoff
    assert (tmp_path / "stage1_handoff.json").read_bytes() == handoff_bytes
    tile_manifest = tmp_path / "tile_manifest.csv"
    tile_bytes = tile_manifest.read_bytes()
    tile_manifest.write_bytes(tile_bytes + b"\n")
    stale_tile = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=tmp_path / "annotations.csv",
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    assert "scoring manifest tile source hash mismatch" in stale_tile["blockers"]
    tile_manifest.write_bytes(tile_bytes)

    candidate_path = tmp_path / "candidate_pool.csv"
    candidate_bytes = candidate_path.read_bytes()
    candidate_path.write_bytes(candidate_bytes + b"\n")
    stale_candidate = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=tmp_path / "annotations.csv",
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    assert (
        "batch_manifest candidate source hash mismatch" in stale_candidate["blockers"]
    )
    candidate_path.write_bytes(candidate_bytes)

    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"stale")
    stale_checkpoint = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=tmp_path / "annotations.csv",
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    assert (
        "scoring regression checkpoint does not match baseline"
        in stale_checkpoint["blockers"]
    )
    checkpoint.write_bytes(checkpoint_bytes)

    annotations_path = tmp_path / "annotations.csv"
    annotations_bytes = annotations_path.read_bytes()
    pd.read_csv(annotations_path).iloc[:2].to_csv(annotations_path, index=False)
    incomplete_batch = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=annotations_path,
        mixed_labels_csv=tmp_path / "mixed_labels.csv",
        benchmark_csv=tmp_path / "benchmark.csv",
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    assert any(
        "without completed review decisions" in blocker
        for blocker in incomplete_batch["blockers"]
    )
    annotations_path.write_bytes(annotations_bytes)
    assert (tmp_path / "stage1_handoff.json").read_bytes() == handoff_bytes
    assert (
        tmp_path / "absolute_annotations.csv"
    ).read_bytes() == annotation_snapshot_bytes
    assert (tmp_path / "filtered_index.csv").read_bytes() == filtered_index_bytes
    filtered_index = pd.read_csv(tmp_path / "filtered_index.csv")
    assert filtered_index["candidate_row_index"].tolist() == [0, 1, 2]
    assert "filtered_index" in handoff["artifacts"]

    assert handoff["ready_for_stage2"] is True
    assert not handoff["blockers"]
    assert registry.read_bytes() == registry_before


def test_selection_rejects_invalid_run_names(tmp_path: Path) -> None:
    from src.active_learning.selection import SelectionConfig, run_selection

    candidates = _candidates()
    csv_path = tmp_path / "candidates.csv"
    candidates.to_csv(csv_path, index=False)

    for bad_name in (
        "",
        "/abs/path",
        "../traversal",
        "dot/dot",
        ".",
        "..",
        "invalid name",
    ):
        with pytest.raises(ValueError):
            select_candidates(candidates, batch_size=2, seed=1, run_name=bad_name)
        with pytest.raises(ValueError):
            SelectionConfig(run_name=bad_name)
        with pytest.raises(ValueError):
            run_selection(csv_path, output_dir=tmp_path / "runs", run_name=bad_name)


def test_prepare_for_selection_filters_scorer_columns_and_fails_closed() -> None:
    from src.active_learning.selection import _prepare_for_selection

    df = pd.DataFrame(
        [
            {
                "image_path": "z14/100_100.png",
                "region": "west",
                "z": 14,
                "x": 100,
                "y": 100,
                "satellite_present": True,
                "terrain_present": True,
                "selector_eligible": True,
                "score_status": "scored",
                "model_score": 0.9,
            },
            {
                "image_path": "z14/101_100.png",
                "region": "west",
                "z": 14,
                "x": 101,
                "y": 100,
                "satellite_present": True,
                "terrain_present": True,
                "selector_eligible": False,
                "score_status": "error",
                "model_score": 0.1,
            },
            {
                "image_path": "z14/102_100.png",
                "region": "west",
                "z": 14,
                "x": 102,
                "y": 100,
                "satellite_present": True,
                "terrain_present": True,
                "selector_eligible": True,
                "score_status": "missing",
                "model_score": 0.2,
            },
        ]
    )

    prepared = _prepare_for_selection(df, seed=42)
    assert len(prepared) == 1
    assert prepared.iloc[0]["image_path"] == "z14/100_100.png"

    # Fail closed when none eligible
    all_ineligible = df.copy()
    all_ineligible["selector_eligible"] = False
    with pytest.raises(ValueError, match="no eligible scored candidates"):
        _prepare_for_selection(all_ineligible, seed=42)


def test_modifying_annotation_batch_content_preserving_row_count_and_batch_id_is_rejected(
    tmp_path: Path,
) -> None:
    from src.active_learning.finalize import _validate_lineage
    from src.active_learning.selection import run_selection

    candidates = _candidates()
    csv_path = tmp_path / "candidate_pool.csv"
    candidates.to_csv(csv_path, index=False)

    artifacts = run_selection(
        csv_path, output_dir=tmp_path / "runs", run_name="active_learning"
    )
    run_dir = tmp_path / "runs" / "active_learning"
    batch_path = run_dir / "annotation_batch.csv"
    manifest_path = run_dir / "batch_manifest.json"

    # Admitted/published batch manifest contains exact output hash and byte count
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "outputs" in manifest_data
    assert "annotation_batch_csv" in manifest_data["outputs"]
    output_record = manifest_data["outputs"]["annotation_batch_csv"]
    assert isinstance(output_record, dict)
    exact_hash = sha256_file(batch_path)
    assert output_record["sha256"] == exact_hash
    assert output_record["bytes"] == batch_path.stat().st_size

    # Prepare environment for finalizer validation
    paths = list(artifacts.selected["image_path"])
    annotations_path = run_dir / "annotations.csv"
    pd.DataFrame(
        [
            {
                "image_path": name,
                "scenic_human": 5,
                "confidence": "high",
                "skip": False,
                "annotator_id": "test",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            }
            for name in paths
        ]
    ).to_csv(annotations_path, index=False)

    # Modify annotation_batch content while keeping row count and batch_id unchanged
    df_batch = pd.read_csv(batch_path)
    original_rows = len(df_batch)
    original_batch_id = df_batch["batch_id"].iloc[0]

    # Modify content column
    df_batch["selection_reason"] = "tampered_reason"
    df_batch.to_csv(batch_path, index=False, lineterminator="\n")

    # Confirm row count and batch_id are strictly preserved
    modified_df = pd.read_csv(batch_path)
    assert len(modified_df) == original_rows
    assert (modified_df["batch_id"] == original_batch_id).all()

    # Verify finalizer selection validation rejects the modified annotation batch
    lineage_paths = {
        "annotation_batch": batch_path,
        "absolute_annotations": annotations_path,
        "batch_manifest": manifest_path,
        "selection_diagnostics": run_dir / "selection_diagnostics.json",
        "scoring_manifest": run_dir / "scoring_manifest.json",
        "candidate_pool": csv_path,
        "tile_manifest": run_dir / "tile_manifest.csv",
    }
    blockers: list[str] = []
    valid = _validate_lineage(
        lineage_paths, baseline={"checkpoint_sha256": "dummy"}, blockers=blockers
    )
    assert valid is False
    assert "batch_manifest annotation batch output hash mismatch" in blockers
