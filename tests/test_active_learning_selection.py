from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.active_learning.common import sha256_bytes, sha256_file
from src.active_learning.finalize import finalize_stage1
from src.active_learning.scoring import SCORING_SCHEMA_VERSION
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


def _write_complete_stage1(
    tmp_path: Path, *, scoring_schema_version: int = SCORING_SCHEMA_VERSION
) -> dict[str, Path]:
    """Write every artifact a validated stage-one run needs under tmp_path."""
    names = ["a.png", "b.png", "c.png"]
    selection_contract = {"selected_identities": names}
    selection_digest = sha256_bytes(
        json.dumps(selection_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    batch_id = f"batch-{selection_digest[:16]}"
    for name in names:
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
            for index, name in enumerate(names)
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
            for index, name in enumerate(names)
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
                zip(names, ("train", "val", "test"), strict=True)
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
            for index, name in enumerate(names)
        ]
    )
    candidate.to_csv(tmp_path / "candidate_pool.csv", index=False)
    np.savez_compressed(
        tmp_path / "feature_embeddings.npz",
        embeddings=np.ones((len(names), 2), dtype=np.float32),
        row_indices=np.arange(len(names), dtype=np.int64),
    )
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": str(checkpoint)}}))
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
            for index, name in enumerate(names)
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
            for index, name in enumerate(names)
        ]
    ).to_csv(tmp_path / "mixed_labels.csv", index=False)
    benchmark = pd.DataFrame(
        [
            {
                "image_path": name,
                "scenic_human": 5 + index,
                "split": ("train", "val", "test")[index],
            }
            for index, name in enumerate(names)
        ]
    )
    benchmark.to_csv(tmp_path / "benchmark.csv", index=False)
    # Control benchmark must keep split=test identities disjoint from the
    # expanded benchmark's (expanded test image is "c.png").
    pd.DataFrame(
        [
            {"image_path": "a.png", "scenic_human": 5, "split": "train"},
            {"image_path": "b.png", "scenic_human": 6, "split": "val"},
            {"image_path": "d.png", "scenic_human": 7, "split": "test"},
        ]
    ).to_csv(tmp_path / "control_benchmark.csv", index=False)
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
    (tmp_path / "batch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_count": len(names),
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
                "schema_version": scoring_schema_version,
                "state": {"complete": True, "ready_for_selection": True},
                "counts": {
                    "manifest_rows": len(names),
                    "scored_rows": len(names),
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
    return {
        "annotations": tmp_path / "annotations.csv",
        "mixed_labels": tmp_path / "mixed_labels.csv",
        "benchmark": tmp_path / "benchmark.csv",
        "registry": registry,
        "checkpoint": checkpoint,
    }


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
    inputs = _write_complete_stage1(tmp_path)
    registry_before = inputs["registry"].read_bytes()
    checkpoint = inputs["checkpoint"]
    registry = inputs["registry"]
    annotations_path = inputs["annotations"]
    common = dict(
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=registry,
        checkpoint_path=checkpoint,
    )
    handoff = finalize_stage1(tmp_path, **common)
    handoff_bytes = (tmp_path / "stage1_handoff.json").read_bytes()
    annotation_snapshot_bytes = (tmp_path / "absolute_annotations.csv").read_bytes()
    filtered_index_bytes = (tmp_path / "filtered_index.csv").read_bytes()
    identical = finalize_stage1(tmp_path, **common)
    assert identical == handoff
    assert (tmp_path / "stage1_handoff.json").read_bytes() == handoff_bytes
    tile_manifest = tmp_path / "tile_manifest.csv"
    tile_bytes = tile_manifest.read_bytes()
    tile_manifest.write_bytes(tile_bytes + b"\n")
    stale_tile = finalize_stage1(tmp_path, **common)
    assert "scoring manifest tile source hash mismatch" in stale_tile["blockers"]
    tile_manifest.write_bytes(tile_bytes)

    candidate_path = tmp_path / "candidate_pool.csv"
    candidate_bytes = candidate_path.read_bytes()
    candidate_path.write_bytes(candidate_bytes + b"\n")
    stale_candidate = finalize_stage1(tmp_path, **common)
    assert (
        "batch_manifest candidate source hash mismatch" in stale_candidate["blockers"]
    )
    candidate_path.write_bytes(candidate_bytes)

    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"stale")
    stale_checkpoint = finalize_stage1(tmp_path, **common)
    assert (
        "scoring regression checkpoint does not match baseline"
        in stale_checkpoint["blockers"]
    )
    checkpoint.write_bytes(checkpoint_bytes)

    annotations_path = tmp_path / "annotations.csv"
    annotations_bytes = annotations_path.read_bytes()
    pd.read_csv(annotations_path).iloc[:2].to_csv(annotations_path, index=False)
    incomplete_batch = finalize_stage1(tmp_path, **common)
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


def test_resolve_registry_checkpoint_prefers_existing_and_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.active_learning.finalize import (
        _registry_checkpoint_candidates,
        _resolve_registry_checkpoint,
    )

    monkeypatch.chdir(tmp_path)
    registry_dir = tmp_path / "data" / "processed" / "regression"
    registry_dir.mkdir(parents=True)
    registry = registry_dir / "model_registry.json"
    registry.write_text("{}", encoding="utf-8")

    # Registry-relative checkpoints/<sha>.pt resolves via the registry directory
    sha = "ab" * 32
    reg_ckpt = registry_dir / "checkpoints" / f"{sha}.pt"
    reg_ckpt.parent.mkdir(parents=True)
    reg_ckpt.write_bytes(b"checkpoint")
    assert (
        _resolve_registry_checkpoint(registry, f"checkpoints/{sha}.pt").resolve()
        == reg_ckpt.resolve()
    )

    # Project-root-relative models/... resolves via the working directory
    model_ckpt = tmp_path / "models" / "baseline.pt"
    model_ckpt.parent.mkdir(parents=True)
    model_ckpt.write_bytes(b"model")
    assert (
        _resolve_registry_checkpoint(registry, "models/baseline.pt").resolve()
        == model_ckpt.resolve()
    )

    # Missing checkpoint returns None and exposes every tried candidate
    missing_value = "checkpoints/missing.pt"
    assert _resolve_registry_checkpoint(registry, missing_value) is None
    tried = _registry_checkpoint_candidates(registry, missing_value)
    assert tried[0] == Path(missing_value)
    assert tried[1] == reg_ckpt.parent / "missing.pt"
    assert tried[2] == tmp_path / missing_value


def test_validate_registry_resolves_registry_relative_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.active_learning.finalize import _validate_registry

    monkeypatch.chdir(tmp_path)
    registry_dir = tmp_path / "data" / "processed" / "regression"
    registry_dir.mkdir(parents=True)
    registry = registry_dir / "model_registry.json"
    sha = "cd" * 32
    ckpt = registry_dir / "checkpoints" / f"{sha}.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"checkpoint bytes")
    registry.write_text(
        json.dumps(
            {
                "active": {
                    "checkpoint": f"checkpoints/{sha}.pt",
                    "sha256": sha256_file(ckpt),
                }
            }
        ),
        encoding="utf-8",
    )
    blockers: list[str] = []
    ok, baseline = _validate_registry(registry, None, blockers)
    assert ok
    assert blockers == []
    assert baseline["checkpoint_path"] == str(ckpt)
    assert baseline["checkpoint_sha256"] == sha256_file(ckpt)

    # Missing checkpoint fails closed and names every tried candidate
    registry.write_text(
        json.dumps({"active": {"checkpoint": "checkpoints/missing.pt"}}),
        encoding="utf-8",
    )
    blockers = []
    ok, baseline = _validate_registry(registry, None, blockers)
    assert not ok
    assert any("tried:" in blocker for blocker in blockers)


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


def test_selection_defensive_water_filtering_and_external_tables() -> None:
    # Legacy / external candidate table lacking selector_eligible, or with water fractions/reasons
    df = pd.DataFrame(
        [
            {
                "image_path": "z14/100_100.png",
                "region": "west",
                "z": 14,
                "x": 100,
                "y": 100,
                "water_fraction": 0.10,
                "model_score": 0.9,
                "heuristic_score": 0.1,
                "cluster_id": "a",
            },
            {
                "image_path": "z14/101_100.png",
                "region": "west",
                "z": 14,
                "x": 101,
                "y": 100,
                "water_fraction": 0.60,  # > 0.50, legacy table
                "model_score": 0.9,
                "heuristic_score": 0.1,
                "cluster_id": "b",
            },
            {
                "image_path": "z14/102_100.png",
                "region": "west",
                "z": 14,
                "x": 102,
                "y": 100,
                "unusable_reason": "Excessive_Water",
                "model_score": 0.8,
                "heuristic_score": 0.2,
                "cluster_id": "c",
            },
            {
                "image_path": "z14/103_100.png",
                "region": "west",
                "z": 14,
                "x": 103,
                "y": 100,
                "water_filter_status": "excessive_water",
                "model_score": 0.8,
                "heuristic_score": 0.2,
                "cluster_id": "d",
            },
            {
                "image_path": "z14/104_100.png",
                "region": "west",
                "z": 14,
                "x": 104,
                "y": 100,
                "effective_water_fraction": 0.50,  # == 0.50 inclusive threshold
                "model_score": 0.8,
                "heuristic_score": 0.2,
                "cluster_id": "e",
            },
            {
                "image_path": "z14/105_100.png",
                "region": "west",
                "z": 14,
                "x": 105,
                "y": 100,
                "water_fraction": 0.45,  # < 0.50 shoreline
                "terrain_sea_level_fraction": 0.80,  # sea-level proxy is not authoritative
                "effective_water_fraction": 0.45,
                "model_score": 0.8,
                "heuristic_score": 0.2,
                "cluster_id": "f",
            },
        ]
    )

    selected = select_candidates(df, batch_size=5, seed=42, run_name="water_test")
    selected_paths = list(selected["image_path"])
    assert "z14/100_100.png" in selected_paths
    assert "z14/105_100.png" in selected_paths
    assert "z14/101_100.png" not in selected_paths
    assert "z14/102_100.png" not in selected_paths
    assert "z14/103_100.png" not in selected_paths
    assert "z14/104_100.png" not in selected_paths


def test_selection_all_water_failure() -> None:
    df = pd.DataFrame(
        [
            {
                "image_path": "z14/100_100.png",
                "region": "west",
                "z": 14,
                "x": 100,
                "y": 100,
                "water_fraction": 0.80,
                "model_score": 0.9,
                "heuristic_score": 0.1,
            },
            {
                "image_path": "z14/101_100.png",
                "region": "west",
                "z": 14,
                "x": 101,
                "y": 100,
                "unusable_reason": "excessive_water",
                "model_score": 0.9,
                "heuristic_score": 0.1,
            },
        ]
    )
    with pytest.raises(ValueError, match="no eligible scored candidates"):
        select_candidates(df, batch_size=2, seed=42, run_name="all_water_test")


def test_stable_identity_parses_canonical_path_layouts() -> None:
    from src.active_learning.finalize import _stable_identity

    assert (
        _stable_identity("images/satellite/z14/fixture/100_200.png")
        == "fixture/z14/x100/y200"
    )
    assert (
        _stable_identity("data/raw/images/satellite/z14/masswhites/100_200.png")
        == "masswhites/z14/x100/y200"
    )
    assert _stable_identity("z14/100_100.png") == "unknown/z14/x100/y100"
    assert _stable_identity("z14x100y200.png") == "unknown/z14/x100/y200"
    assert _stable_identity("z14/x100/y200") == "unknown/z14/x100/y200"
    assert _stable_identity("a.png") == "a.png"
    assert _stable_identity("") == ""


def test_snapshot_matches_alternate_canonical_paths_by_stable_identity(
    tmp_path: Path,
) -> None:
    from src.active_learning.finalize import ABSOLUTE_COLUMNS, _snapshot_annotations

    batch_path = tmp_path / "annotation_batch.csv"
    pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/101_200.png",
                "selection_reason": "fixture",
                "selection_score": 1,
                "selection_rank": 2,
                "batch_id": "batch-x",
                "run_id": "run",
            },
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "selection_reason": "fixture",
                "selection_score": 1,
                "selection_rank": 1,
                "batch_id": "batch-x",
                "run_id": "run",
            },
        ]
    ).to_csv(batch_path, index=False)
    source_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "image_path": f"data/raw/images/satellite/z14/fixture/{x}_200.png",
                "scenic_human": 5 + x,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            }
            for x in (100, 101, 102)
        ]
    ).to_csv(source_path, index=False)
    destination = tmp_path / "absolute_annotations.csv"
    blockers: list[str] = []
    result = _snapshot_annotations(
        source_path, batch_path, destination, blockers, write=True
    )
    assert result == destination
    assert not blockers
    snapshot = pd.read_csv(destination)
    # Deterministic source order (not batch order); seven columns preserved.
    assert snapshot["image_path"].tolist() == [
        "data/raw/images/satellite/z14/fixture/100_200.png",
        "data/raw/images/satellite/z14/fixture/101_200.png",
    ]
    assert list(snapshot.columns) == list(ABSOLUTE_COLUMNS)
    assert snapshot["scenic_human"].tolist() == [105, 106]


def test_lineage_matches_batch_decisions_by_stable_identity(tmp_path: Path) -> None:
    from src.active_learning.finalize import _validate_lineage

    tile_path = tmp_path / "tile_manifest.csv"
    pd.DataFrame(
        [
            {"region": "fixture", "z": 14, "x": 100, "y": 200},
            {"region": "fixture", "z": 14, "x": 101, "y": 200},
        ]
    ).to_csv(tile_path, index=False)
    candidate_path = tmp_path / "candidate_pool.csv"
    pd.DataFrame(
        [
            {
                "image_path": f"images/satellite/z14/fixture/{x}_200.png",
                "score_status": "scored",
            }
            for x in (100, 101)
        ]
    ).to_csv(candidate_path, index=False)
    batch_path = tmp_path / "annotation_batch.csv"
    batch_images = [
        "images/satellite/z14/fixture/101_200.png",
        "images/satellite/z14/fixture/100_200.png",
    ]
    selection_contract = {"selected_identities": batch_images}
    selection_digest = sha256_bytes(
        json.dumps(selection_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    batch_id = f"batch-{selection_digest[:16]}"
    pd.DataFrame(
        [
            {
                "image_path": image,
                "selection_reason": "fixture",
                "selection_score": 1,
                "selection_rank": index + 1,
                "batch_id": batch_id,
                "run_id": "run",
            }
            for index, image in enumerate(batch_images)
        ]
    ).to_csv(batch_path, index=False)
    (tmp_path / "scoring_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"tile_manifest": {"sha256": sha256_file(tile_path)}},
                "models": {"regression_checkpoint_sha256": "baseline-ckpt"},
            }
        )
    )
    (tmp_path / "batch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_count": 2,
                "batch_id": batch_id,
                "selection_contract": selection_contract,
                "selection_contract_sha256": selection_digest,
                "candidate_input": {"sha256": sha256_file(candidate_path)},
                "outputs": {
                    "annotation_batch_csv": {"sha256": sha256_file(batch_path)},
                },
            }
        )
    )
    (tmp_path / "selection_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_input": {"sha256": sha256_file(candidate_path)},
            }
        )
    )
    annotations_path = tmp_path / "annotations.csv"

    def write_annotations(x_values: tuple[int, ...]) -> None:
        pd.DataFrame(
            [
                {
                    "image_path": (
                        f"data/raw/images/satellite/z14/fixture/{x}_200.png"
                    ),
                    "scenic_human": 5 + x,
                    "confidence": "high",
                    "skip": False,
                    "annotator_id": "a1",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "notes": "",
                }
                for x in x_values
            ]
        ).to_csv(annotations_path, index=False)

    write_annotations((100, 101))
    lineage_paths = {
        "annotation_batch": batch_path,
        "absolute_annotations": annotations_path,
        "batch_manifest": tmp_path / "batch_manifest.json",
        "selection_diagnostics": tmp_path / "selection_diagnostics.json",
        "scoring_manifest": tmp_path / "scoring_manifest.json",
        "candidate_pool": candidate_path,
        "tile_manifest": tile_path,
    }
    blockers: list[str] = []
    assert (
        _validate_lineage(
            lineage_paths,
            baseline={"checkpoint_sha256": "baseline-ckpt"},
            blockers=blockers,
        )
        is True
    )
    assert not blockers

    # Missing decision on one tile must be reported exactly, by identity.
    write_annotations((100,))
    blockers = []
    assert (
        _validate_lineage(
            lineage_paths,
            baseline={"checkpoint_sha256": "baseline-ckpt"},
            blockers=blockers,
        )
        is False
    )
    assert any(
        "without completed review decisions" in blocker
        and "fixture/z14/x101/y200" in blocker
        for blocker in blockers
    )


@pytest.mark.parametrize(
    ("notes", "accepted"),
    [
        ("", False),
        ("unclear view", False),
        ("[unusable: missing_imagery]", True),
        ("[unusable: cloud_or_obstruction] heavy cloud cover", True),
        ("missing_imagery, duplicate", False),
    ],
)
def test_skipped_decisions_require_one_supported_unusable_reason(
    tmp_path: Path, notes: str, accepted: bool
) -> None:
    from src.active_learning.finalize import _validate_annotations

    path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "scenic_human": 5,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
            {
                "image_path": "images/satellite/z14/fixture/101_200.png",
                "scenic_human": "",
                "confidence": "medium",
                "skip": True,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": notes,
            },
        ]
    ).to_csv(path, index=False)
    blockers: list[str] = []
    valid, _ = _validate_annotations(path, blockers)
    assert valid is accepted
    if not accepted:
        assert any("unusable reason" in blocker for blocker in blockers)


def test_annotations_reject_duplicate_identity_for_same_annotator(
    tmp_path: Path,
) -> None:
    from src.active_learning.finalize import _validate_annotations

    path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "scenic_human": 5,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
            {
                "image_path": "data/raw/images/satellite/z14/fixture/100_200.png",
                "scenic_human": 6,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
            {
                "image_path": "images/satellite/z14/fixture/101_200.png",
                "scenic_human": 7,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a2",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
        ]
    ).to_csv(path, index=False)
    blockers: list[str] = []
    valid, _ = _validate_annotations(path, blockers)
    assert valid is False
    assert any(
        "duplicate or conflicting" in blocker and "fixture/z14/x100/y200" in blocker
        for blocker in blockers
    )


def test_annotations_allow_same_identity_across_different_annotators(
    tmp_path: Path,
) -> None:
    from src.active_learning.finalize import _validate_annotations

    path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "scenic_human": 5,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
            {
                "image_path": "data/raw/images/satellite/z14/fixture/100_200.png",
                "scenic_human": 6,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a2",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
            {
                "image_path": "images/satellite/z14/fixture/101_200.png",
                "scenic_human": 7,
                "confidence": "high",
                "skip": False,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
        ]
    ).to_csv(path, index=False)
    blockers: list[str] = []
    valid, _ = _validate_annotations(path, blockers)
    assert valid is True
    assert not blockers


def test_finalizer_admits_immutable_earlier_scoring_schema(tmp_path: Path) -> None:
    inputs = _write_complete_stage1(tmp_path, scoring_schema_version=1)
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is True
    assert handoff["scoring_valid"] is True
    assert not handoff["blockers"]


@pytest.mark.parametrize("bad_version", [0, -1, SCORING_SCHEMA_VERSION + 1])
def test_finalizer_rejects_unsupported_scoring_schema_version(
    tmp_path: Path, bad_version: int
) -> None:
    inputs = _write_complete_stage1(tmp_path, scoring_schema_version=bad_version)
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is False
    assert handoff["scoring_valid"] is False
    assert any(
        "scoring manifest schema version" in blocker for blocker in handoff["blockers"]
    )


def test_finalizer_fails_closed_without_control_benchmark(tmp_path: Path) -> None:
    inputs = _write_complete_stage1(tmp_path)
    (tmp_path / "control_benchmark.csv").unlink()
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is False
    assert handoff["control_benchmark_valid"] is False
    assert any("control_benchmark" in blocker for blocker in handoff["blockers"])


def test_finalizer_records_risks_and_treats_them_as_material_identity(
    tmp_path: Path,
) -> None:
    inputs = _write_complete_stage1(tmp_path)
    common = dict(
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    handoff = finalize_stage1(tmp_path, risks=["flood", "snow"], **common)
    assert handoff["ready_for_stage2"] is True
    assert handoff["risks"] == ["flood", "snow"]
    unchanged = finalize_stage1(tmp_path, risks=["flood", "snow"], **common)
    assert unchanged == handoff
    drifted = finalize_stage1(tmp_path, risks=["flood"], **common)
    assert drifted["ready_for_stage2"] is False
    assert "risks differ from previous ready handoff" in drifted["blockers"]


def _rewrite_benchmark_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("rows", "blocker_fragment"),
    [
        (
            [
                {"image_path": "a.png", "scenic_human": 5},
                {"image_path": "b.png", "scenic_human": 6},
                {"image_path": "d.png", "scenic_human": 7},
            ],
            "requires an explicit split column",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {"image_path": "b.png", "scenic_human": 6, "split": "val"},
            ],
            "contains no split=test rows",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {"image_path": "d.png", "scenic_score": 7, "split": "test"},
            ],
            "never weak/mixed scenic_score",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {"image_path": "d.png", "scenic_human": 11, "split": "test"},
            ],
            "finite and in [0, 10]",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {
                    "image_path": "d.png",
                    "scenic_human": "not-a-number",
                    "split": "test",
                },
            ],
            "finite and in [0, 10]",
        ),
        (
            [
                {
                    "image_path": "images/satellite/z14/fixture/100_200.png",
                    "scenic_human": 5,
                    "split": "test",
                },
                {
                    "image_path": "data/raw/images/satellite/z14/fixture/100_200.png",
                    "scenic_human": 6,
                    "split": "test",
                },
            ],
            "duplicate test image paths",
        ),
    ],
)
def test_benchmark_pre_dataset_rejects_invalid_contracts(
    tmp_path: Path, rows: list[dict[str, object]], blocker_fragment: str
) -> None:
    from src.active_learning.finalize import _validate_benchmark_pre_dataset

    path = tmp_path / "benchmark.csv"
    _rewrite_benchmark_csv(path, rows)
    blockers: list[str] = []
    assert _validate_benchmark_pre_dataset(path, blockers, name="benchmark") is False
    assert any(blocker_fragment in blocker for blocker in blockers)


def test_benchmark_pre_dataset_accepts_scenic_human_mean_target(
    tmp_path: Path,
) -> None:
    from src.active_learning.finalize import (
        _validate_benchmark_pre_dataset,
        _validate_human_table,
    )

    path = tmp_path / "benchmark.csv"
    pd.DataFrame(
        [
            {"image_path": "a.png", "scenic_human_mean": 5.0, "split": "train"},
            {"image_path": "b.png", "scenic_human_mean": 6.0, "split": "test"},
        ]
    ).to_csv(path, index=False)
    blockers: list[str] = []
    assert _validate_benchmark_pre_dataset(path, blockers, name="benchmark") is True
    assert not blockers
    valid, rows = _validate_human_table(path, [], benchmark=True)
    assert valid is True
    assert rows == 2


def test_benchmark_test_overlap_uses_stable_identity(tmp_path: Path) -> None:
    from src.active_learning.finalize import _validate_benchmark_test_overlap

    expanded = tmp_path / "benchmark.csv"
    control = tmp_path / "control_benchmark.csv"
    pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "scenic_human": 5,
                "split": "test",
            }
        ]
    ).to_csv(expanded, index=False)
    pd.DataFrame(
        [
            {
                "image_path": "data/raw/images/satellite/z14/fixture/100_200.png",
                "scenic_human": 6,
                "split": "test",
            }
        ]
    ).to_csv(control, index=False)
    blockers: list[str] = []
    assert _validate_benchmark_test_overlap(expanded, control, blockers) is False
    assert any(
        "Overlap detected between expanded and control benchmark" in blocker
        for blocker in blockers
    )


@pytest.mark.parametrize(
    ("rows", "blocker_fragment"),
    [
        (
            [
                {"image_path": "a.png", "scenic_human": 5},
                {"image_path": "b.png", "scenic_human": 6},
                {"image_path": "d.png", "scenic_human": 7},
            ],
            "control benchmark requires an explicit split column",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {"image_path": "b.png", "scenic_human": 6, "split": "val"},
            ],
            "control benchmark contains no split=test rows",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {"image_path": "d.png", "scenic_human": 11, "split": "test"},
            ],
            "control benchmark human targets must be finite and in [0, 10]",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {
                    "image_path": "d.png",
                    "scenic_human": "not-a-number",
                    "split": "test",
                },
            ],
            "control benchmark human targets must be finite and in [0, 10]",
        ),
        (
            [
                {"image_path": "d.png", "scenic_human": 7, "split": "test"},
                {"image_path": "d.png", "scenic_human": 8, "split": "test"},
            ],
            "control benchmark contains duplicate test image paths",
        ),
        (
            [
                {"image_path": "a.png", "scenic_human": 5, "split": "train"},
                {"image_path": "d.png", "scenic_score": 7, "split": "test"},
            ],
            "never weak/mixed scenic_score",
        ),
    ],
)
def test_finalizer_rejects_malformed_control_benchmark(
    tmp_path: Path, rows: list[dict[str, object]], blocker_fragment: str
) -> None:
    inputs = _write_complete_stage1(tmp_path)
    _rewrite_benchmark_csv(tmp_path / "control_benchmark.csv", rows)
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is False
    assert handoff["control_benchmark_valid"] is False
    assert handoff["benchmark_valid"] is True
    assert any(blocker_fragment in blocker for blocker in handoff["blockers"])


def test_finalizer_rejects_expanded_control_test_identity_overlap(
    tmp_path: Path,
) -> None:
    inputs = _write_complete_stage1(tmp_path)
    _rewrite_benchmark_csv(
        tmp_path / "control_benchmark.csv",
        [
            {"image_path": "a.png", "scenic_human": 5, "split": "train"},
            {"image_path": "b.png", "scenic_human": 6, "split": "val"},
            # Reuses the expanded benchmark's split=test image identity.
            {"image_path": "c.png", "scenic_human": 7, "split": "test"},
        ],
    )
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is False
    assert handoff["benchmark_valid"] is False
    assert handoff["control_benchmark_valid"] is False
    assert any(
        "Overlap detected between expanded and control benchmark" in blocker
        for blocker in handoff["blockers"]
    )


def test_finalizer_rejects_expanded_benchmark_without_test_rows(
    tmp_path: Path,
) -> None:
    inputs = _write_complete_stage1(tmp_path)
    benchmark = tmp_path / "benchmark.csv"
    _rewrite_benchmark_csv(
        benchmark,
        [
            {"image_path": "a.png", "scenic_human": 5, "split": "train"},
            {"image_path": "b.png", "scenic_human": 6, "split": "val"},
        ],
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    summary["source_hashes"]["benchmark_split_csv"] = sha256_file(benchmark)
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is False
    assert handoff["benchmark_valid"] is False
    assert any(
        "benchmark contains no split=test rows" in blocker
        for blocker in handoff["blockers"]
    )


def test_finalizer_rejects_blank_control_test_image_path(tmp_path: Path) -> None:
    inputs = _write_complete_stage1(tmp_path)
    _rewrite_benchmark_csv(
        tmp_path / "control_benchmark.csv",
        [
            {"image_path": "a.png", "scenic_human": 5, "split": "train"},
            {"image_path": "", "scenic_human": 7, "split": "test"},
        ],
    )
    handoff = finalize_stage1(
        tmp_path,
        run_name="fixture",
        annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"],
        benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"],
        checkpoint_path=inputs["checkpoint"],
    )
    assert handoff["ready_for_stage2"] is False
    assert handoff["control_benchmark_valid"] is False
    assert "control benchmark record lacks image_path" in handoff["blockers"]
