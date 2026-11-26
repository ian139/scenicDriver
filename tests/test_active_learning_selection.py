from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scenicdriver.active_learning.common import sha256_bytes, sha256_file
from scenicdriver.active_learning.finalize import finalize_stage1
from scenicdriver.active_learning.scoring import SCORING_SCHEMA_VERSION
from scenicdriver.active_learning.selection import audit_geographic_leakage, select_candidates


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
        terrain_features=np.ones((len(names), 6), dtype=np.float32),
        class_logits=np.zeros((len(names), 2), dtype=np.float32),
        class_probs=np.full((len(names), 2), 0.5, dtype=np.float32),
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
    split_registry = {
        "schema": "scenic-geographic-split-registry/1",
        "rule": {
            "name": "sha256-physical-block-threshold/1",
            "seed": 17,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "block_size": 32,
            "block_degrees": 0.25,
            "adjacency_radius": 1,
            "adjacency_distance_km": 0.0,
        },
        "regions": {"fixture": {"z14|b0|3": "train"}},
    }
    split_registry_path = tmp_path / "geographic_split_registry.json"
    split_registry_path.write_text(json.dumps(split_registry, sort_keys=True))
    split_registry_record = {
        "path": str(split_registry_path),
        "sha256": sha256_file(split_registry_path),
        "bytes": split_registry_path.stat().st_size,
        "schema": split_registry["schema"],
        "rule": split_registry["rule"],
    }
    (tmp_path / "batch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "row_count": len(names),
                "batch_id": batch_id,
                "selection_contract": selection_contract,
                "selection_contract_sha256": selection_digest,
                "geographic_split_registry": split_registry_record,
                "candidate_input": {
                    "path": "candidate_pool.csv",
                    "sha256": sha256_file(tmp_path / "candidate_pool.csv"),
                },
                "outputs": {
                    "geographic_split_registry_json": split_registry_record,
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
                    "unusable_rows": 0,
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


def _finalize_fixture(root: Path, inputs: dict[str, Path]) -> dict:
    return finalize_stage1(
        root, run_name="fixture", annotations_csv=inputs["annotations"],
        mixed_labels_csv=inputs["mixed_labels"], benchmark_csv=inputs["benchmark"],
        registry_path=inputs["registry"], checkpoint_path=inputs["checkpoint"],
    )


@pytest.mark.parametrize("corruption", ["missing", "missing_path", "missing_digest", "changed_snapshot"])
def test_finalizer_requires_bound_geographic_split_registry(
    tmp_path: Path, corruption: str,
) -> None:
    inputs = _write_complete_stage1(tmp_path)
    manifest_path = tmp_path / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "missing":
        manifest.pop("geographic_split_registry")
    elif corruption == "missing_path":
        manifest["geographic_split_registry"].pop("path")
    elif corruption == "missing_digest":
        manifest["geographic_split_registry"].pop("sha256")
    else:
        registry_path = tmp_path / "geographic_split_registry.json"
        registry_path.write_bytes(registry_path.read_bytes() + b"\n")
    manifest_path.write_text(json.dumps(manifest))

    handoff = _finalize_fixture(tmp_path, inputs)

    assert handoff["ready_for_stage2"] is False
    assert handoff["splits_valid"] is False
    assert any(
        "geographic split registry" in blocker or "geographic_split_registry" in blocker
        for blocker in handoff["blockers"]
    )


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "shape", "probabilities"])
def test_finalizer_rejects_incomplete_consuming_arrays(tmp_path: Path, corruption: str) -> None:
    inputs = _write_complete_stage1(tmp_path)
    feature_path = tmp_path / "feature_embeddings.npz"
    with np.load(feature_path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    if corruption == "missing":
        del arrays["terrain_features"]
    elif corruption == "nonfinite":
        arrays["class_logits"][0, 0] = np.nan
    elif corruption == "shape":
        arrays["class_probs"] = np.ones((2, 2), dtype=np.float32)
    else:
        arrays["class_probs"][0] = [0.9, 0.9]
    np.savez_compressed(feature_path, **arrays)
    manifest_path = tmp_path / "scoring_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["feature_embeddings.npz"]["sha256"] = sha256_file(feature_path)
    manifest_path.write_text(json.dumps(manifest))
    result = _finalize_fixture(tmp_path, inputs)
    assert result["ready_for_stage2"] is False
    assert result["readiness"]["scoring_valid"] is False


@pytest.mark.parametrize("corruption", ["nan", "range", "duplicate", "unknown", "missing", "weak_mismatch"])
def test_finalizer_rejects_invalid_mixed_label_contract(tmp_path: Path, corruption: str) -> None:
    inputs = _write_complete_stage1(tmp_path)
    mixed = pd.read_csv(inputs["mixed_labels"])
    if corruption == "nan":
        mixed.loc[0, "scenic_score"] = np.nan
    elif corruption == "range":
        mixed.loc[0, "scenic_score"] = 11
    elif corruption == "duplicate":
        mixed = pd.concat([mixed, mixed.iloc[:1]], ignore_index=True)
    elif corruption == "unknown":
        mixed.loc[0, "image_path"] = "unknown.png"
    elif corruption == "missing":
        mixed = mixed.iloc[1:]
    else:
        mixed.loc[0, "label_source"] = "heuristic"
        annotations = pd.read_csv(inputs["annotations"])
        annotations = annotations.iloc[1:]
        annotations.to_csv(inputs["annotations"], index=False)
    mixed.to_csv(inputs["mixed_labels"], index=False)
    result = _finalize_fixture(tmp_path, inputs)
    assert result["ready_for_stage2"] is False
    assert any("mixed" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("bad_exclusion", [None, "error", "eligibility", "reason"])
def test_finalizer_admits_only_explicit_successful_exclusions(
    tmp_path: Path, bad_exclusion: str | None,
) -> None:
    inputs = _write_complete_stage1(tmp_path)
    candidate_path = tmp_path / "candidate_pool.csv"
    candidate = pd.read_csv(candidate_path)
    excluded = candidate.iloc[-1].copy()
    excluded["image_path"] = "excluded.png"
    excluded["source_identity"] = "fixture/excluded.png"
    excluded["score_status"] = "unusable"
    excluded["selector_eligible"] = False
    excluded["embedding_row_index"] = -1
    excluded["availability_state"] = "unusable"
    excluded["water_filter_status"] = "excessive_water"
    excluded["unusable_reason"] = "excessive_water"
    excluded["error"] = None
    if bad_exclusion == "error":
        excluded["error"] = "feature extraction failed"
    elif bad_exclusion == "eligibility":
        excluded["selector_eligible"] = True
    elif bad_exclusion == "reason":
        excluded["unusable_reason"] = ""
    candidate = pd.concat([candidate, excluded.to_frame().T], ignore_index=True)
    candidate.to_csv(candidate_path, index=False)
    tile_path = tmp_path / "tile_manifest.csv"
    tiles = pd.read_csv(tile_path)
    tile = tiles.iloc[-1].copy()
    tile["x"] = 100
    tiles = pd.concat([tiles, tile.to_frame().T], ignore_index=True)
    tiles.to_csv(tile_path, index=False)
    scoring_path = tmp_path / "scoring_manifest.json"
    scoring = json.loads(scoring_path.read_text())
    scoring["counts"].update({"manifest_rows": 4, "unusable_rows": 1})
    scoring["artifacts"]["candidate_pool.csv"]["sha256"] = sha256_file(candidate_path)
    scoring["source"]["tile_manifest"]["sha256"] = sha256_file(tile_path)
    scoring_path.write_text(json.dumps(scoring))
    for name in ("batch_manifest.json", "selection_diagnostics.json"):
        path = tmp_path / name
        data = json.loads(path.read_text())
        data["candidate_input"]["sha256"] = sha256_file(candidate_path)
        path.write_text(json.dumps(data))
    result = _finalize_fixture(tmp_path, inputs)
    assert result["ready_for_stage2"] is (bad_exclusion is None), result["blockers"]
    if bad_exclusion is not None:
        assert result["readiness"]["scoring_valid"] is False
    filtered = pd.read_csv(tmp_path / "filtered_index.csv")
    assert filtered["image_path"].tolist() == ["a.png", "b.png", "c.png"]
    assert filtered["candidate_row_index"].tolist() == [0, 1, 2]


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
    split_registry_record = json.loads((tmp_path / "batch_manifest.json").read_text())[
        "geographic_split_registry"
    ]
    assert handoff["geographic_split_registry"] == split_registry_record
    assert handoff["artifacts"]["geographic_split_registry"]["path"] == (
        "geographic_split_registry.json"
    )
    assert handoff["artifacts"]["geographic_split_registry"]["sha256"] == (
        split_registry_record["sha256"]
    )
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
    from scenicdriver.active_learning.selection import SelectionConfig, run_selection

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


@pytest.mark.parametrize(
    "change",
    [
        "candidate_content",
        "candidate_path",
        "seed",
        "batch_size",
        "adjacency_radius",
        "min_separation_km",
        "qa_overlap_count",
        "qa_overlap_fraction",
        "random_control_count",
        "random_control_fraction",
        "weights",
        "prior_file",
        "prior_frame",
    ],
)
def test_selection_rejects_changed_run_identity(tmp_path: Path, change: str) -> None:
    from dataclasses import replace

    from scenicdriver.active_learning.selection import SelectionConfig, run_selection

    candidates = _candidates()
    source = tmp_path / "candidates.csv"
    candidates.to_csv(source, index=False)
    prior = pd.DataFrame([{"image_path": "other.png", "scenic_human": 5}])
    prior_path = tmp_path / "prior.csv"
    prior.to_csv(prior_path, index=False)
    prior_annotations = prior if change == "prior_frame" else prior_path
    config = SelectionConfig(batch_size=2, seed=7, run_name="fixed")
    output_dir = tmp_path / "runs"
    run_selection(
        source,
        output_dir=output_dir,
        run_name="fixed",
        config=config,
        prior_annotations=prior_annotations,
    )
    run_dir = output_dir / "fixed"
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_dir.iterdir()
    }

    if change == "candidate_content":
        candidates.loc[0, "model_score"] = 0.8
        candidates.to_csv(source, index=False)
    elif change == "candidate_path":
        other_source = tmp_path / "other_candidates.csv"
        other_source.write_bytes(source.read_bytes())
        source = other_source
    elif change in {"prior_file", "prior_frame"}:
        # Changing label values must reject reuse even when excluded identities
        # and the selected batch would remain identical.
        prior.loc[0, "scenic_human"] = 6
        if change == "prior_file":
            prior.to_csv(prior_path, index=False)
    elif change == "weights":
        config = replace(config, weights={"uncertainty": 0.3})
    else:
        config = replace(
            config,
            **{change: 0.1 if change.endswith(("fraction", "_km")) else 3},
        )

    with pytest.raises(ValueError, match="new run name"):
        run_selection(
            source,
            output_dir=output_dir,
            run_name="fixed",
            config=config,
            prior_annotations=prior_annotations,
        )
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_dir.iterdir()
    } == before


@pytest.mark.parametrize("prior_kind", ["none", "file", "frame"])
def test_selection_identical_rerun_preserves_fixed_artifacts(
    tmp_path: Path, prior_kind: str
) -> None:
    from scenicdriver.active_learning.selection import SelectionConfig, run_selection

    source = tmp_path / "candidates.csv"
    _candidates().to_csv(source, index=False)
    prior = pd.DataFrame([{"image_path": "other.png", "scenic_human": 5}])
    prior_path = tmp_path / "prior.csv"
    prior.to_csv(prior_path, index=False)
    prior_annotations = {"none": None, "file": prior_path, "frame": prior}[prior_kind]
    kwargs = {
        "output_dir": tmp_path / "runs",
        "run_name": "fixed",
        "config": SelectionConfig(batch_size=2, seed=7, run_name="fixed"),
        "prior_annotations": prior_annotations,
    }
    first = run_selection(source, **kwargs)
    run_dir = tmp_path / "runs" / "fixed"
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_dir.iterdir()
    }
    second = run_selection(source, **kwargs)

    pd.testing.assert_frame_equal(first.selected, second.selected)
    pd.testing.assert_frame_equal(first.geographic_splits, second.geographic_splits)
    assert first.batch_manifest == second.batch_manifest
    assert first.diagnostics == second.diagnostics
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_dir.iterdir()
    } == before


@pytest.mark.parametrize(
    "damage",
    [
        "annotation_batch.csv",
        "geographic_splits.csv",
        "leakage_audit.json",
        "selection_diagnostics.json",
        "batch_manifest.json",
        "run_identity",
        "selection_contract",
        "outputs",
        "invalid_json",
        "tampered_batch",
    ],
)
def test_selection_refuses_incomplete_existing_contract(
    tmp_path: Path, damage: str
) -> None:
    from scenicdriver.active_learning.selection import run_selection

    source = tmp_path / "candidates.csv"
    _candidates().to_csv(source, index=False)
    output_dir = tmp_path / "runs"
    run_selection(source, output_dir=output_dir, run_name="fixed")
    run_dir = output_dir / "fixed"
    manifest_path = run_dir / "batch_manifest.json"
    if damage.endswith((".csv", ".json")):
        (run_dir / damage).unlink()
    elif damage == "invalid_json":
        manifest_path.write_text("{", encoding="utf-8")
    elif damage == "tampered_batch":
        batch_path = run_dir / "annotation_batch.csv"
        batch_path.write_bytes(batch_path.read_bytes() + b"\n")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop(damage, None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}

    with pytest.raises(ValueError, match="new run name"):
        run_selection(source, output_dir=output_dir, run_name="fixed")
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_selection_allows_run_directory_without_selection_artifacts(
    tmp_path: Path,
) -> None:
    from scenicdriver.active_learning.selection import run_selection

    output_dir = tmp_path / "runs"
    run_dir = output_dir / "fixed"
    run_dir.mkdir(parents=True)
    source = run_dir / "candidate_pool.csv"
    _candidates().to_csv(source, index=False)
    original = source.read_bytes()

    selected = run_selection(source, output_dir=output_dir, run_name="fixed")

    assert source.read_bytes() == original
    assert pd.read_csv(run_dir / "annotation_batch.csv")["image_path"].tolist() == (
        selected.selected["image_path"].tolist()
    )


def test_prepare_for_selection_filters_scorer_columns_and_fails_closed() -> None:
    from scenicdriver.active_learning.selection import _prepare_for_selection

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
    from scenicdriver.active_learning.finalize import (
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
    cwd_shadow = tmp_path / "checkpoints" / f"{sha}.pt"
    cwd_shadow.parent.mkdir()
    cwd_shadow.write_bytes(b"unrelated checkpoint")
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

    # Missing checkpoint fails resolution rather than using a guessed path.
    missing_value = "checkpoints/missing.pt"
    assert _resolve_registry_checkpoint(registry, missing_value) is None


def test_validate_registry_resolves_registry_relative_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scenicdriver.active_learning.finalize import _validate_registry

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
    from scenicdriver.active_learning.finalize import _validate_lineage
    from scenicdriver.active_learning.selection import run_selection

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
    from scenicdriver.active_learning.finalize import _stable_identity

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
    from scenicdriver.active_learning.finalize import ABSOLUTE_COLUMNS, _snapshot_annotations

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
    from scenicdriver.active_learning.finalize import _validate_lineage

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
        ("[unusable: unsupported]", False),
        ("[unusable: missing_imagery]", True),
        ("[unusable: cloud_or_obstruction] heavy cloud cover", True),
        ("missing_imagery, duplicate", False),
    ],
)
def test_skipped_decisions_require_one_supported_unusable_reason(
    tmp_path: Path, notes: str, accepted: bool
) -> None:
    from scenicdriver.active_learning.finalize import _validate_annotations

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
    from scenicdriver.active_learning.finalize import _validate_annotations

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
    from scenicdriver.active_learning.finalize import _validate_annotations

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
    from scenicdriver.active_learning.finalize import _validate_benchmark_pre_dataset

    path = tmp_path / "benchmark.csv"
    _rewrite_benchmark_csv(path, rows)
    blockers: list[str] = []
    assert _validate_benchmark_pre_dataset(path, blockers, name="benchmark") is False
    assert any(blocker_fragment in blocker for blocker in blockers)


def test_benchmark_pre_dataset_accepts_scenic_human_mean_target(
    tmp_path: Path,
) -> None:
    from scenicdriver.active_learning.finalize import (
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
    from scenicdriver.active_learning.finalize import _validate_benchmark_test_overlap

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


def _spaced_selection_candidates(count: int) -> pd.DataFrame:
    """Keep tiles in distinct blocks with no spatial-exclusion ambiguity."""
    return pd.DataFrame(
        {
            "image_path": [f"z14/{index * 64}_100.png" for index in range(count)],
            "z": 14,
            "x": np.arange(count) * 64,
            "y": 100,
            "region": "fixture",
            "class_id": "forest",
            "model_score": np.linspace(0, 1, count),
            "satellite_present": True,
            "terrain_present": True,
            "selector_eligible": True,
            "score_status": "scored",
        }
    )


@pytest.mark.parametrize(
    "fresh_count, quota, expected_fresh, expected_overlap",
    [(0, 2, 0, 2), (1, 2, 1, 2), (6, 2, 4, 2), (6, 0, 6, 0), (1, 99, 0, 6)],
)
def test_selection_overlap_quota_does_not_fill_missing_fresh_rows(
    fresh_count: int, quota: int, expected_fresh: int, expected_overlap: int
) -> None:
    candidates = _spaced_selection_candidates(fresh_count + 8)
    candidates["already_annotated"] = candidates.index >= fresh_count
    selected = select_candidates(
        candidates,
        batch_size=6,
        seed=13,
        adjacency_radius=0,
        qa_overlap_count=quota,
    )
    assert int(selected["is_qa_overlap"].sum()) == expected_overlap
    assert int((~selected["is_qa_overlap"]).sum()) == expected_fresh
    assert len(selected) == expected_fresh + expected_overlap
    assert selected["image_path"].is_unique


def test_selection_overlap_fraction_stays_bounded_when_fresh_rows_are_blocked() -> None:
    candidates = _spaced_selection_candidates(10)
    candidates.loc[1, ["image_path", "x"]] = ["z14/1_100.png", 1]
    candidates["already_annotated"] = candidates.index >= 2
    selected = select_candidates(
        candidates,
        batch_size=10,
        seed=13,
        adjacency_radius=1,
        qa_overlap_fraction=0.2,
        random_control_count=99,
    )
    assert int(selected["is_qa_overlap"].sum()) == 2
    assert int(selected["is_random_control"].sum()) == 1
    assert len(selected) == 3
    diagnostics = selected.attrs["diagnostics"]
    assert diagnostics["requested_random_controls"] == 8
    assert diagnostics["requested_overlap_rows"] == 2


@pytest.mark.parametrize(
    "val_fraction, test_fraction, expected_splits",
    [
        (0.0, 0.0, {"train"}),
        (0.0, 0.25, {"train", "test"}),
        (0.25, 0.0, {"train", "val"}),
        (0.25, 0.25, {"train", "val", "test"}),
    ],
)
def test_geographic_splits_allocate_only_positive_fractions(
    val_fraction: float, test_fraction: float, expected_splits: set[str]
) -> None:
    from scenicdriver.active_learning.selection import build_geographic_splits

    candidates = _spaced_selection_candidates(24)
    splits = build_geographic_splits(
        candidates,
        seed=17,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        adjacency_radius=0,
    )
    reordered = build_geographic_splits(
        candidates.iloc[::-1],
        seed=17,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        adjacency_radius=0,
    )
    assert set(splits["split"]) == expected_splits
    assert set(splits["image_path"]) == set(candidates["image_path"])
    pd.testing.assert_frame_equal(splits, reordered)
    assert audit_geographic_leakage(splits, adjacency_radius=0)["valid"]


@pytest.mark.parametrize(
    "ineligible_fields",
    [
        {"satellite_present": False},
        {"terrain_present": False},
        {"selector_eligible": False},
        {"score_status": "error"},
        {"availability_state": "unusable"},
        {"effective_water_fraction": 0.6},
        {"unusable_reason": "excessive_water"},
        {"water_filter_status": "excessive_water"},
    ],
)
def test_selection_keeps_eligible_physical_tile_alias(
    ineligible_fields: dict[str, object],
) -> None:
    candidates = _spaced_selection_candidates(1)
    healthy = candidates.iloc[0].to_dict()
    healthy["image_path"] = "z-healthy.png"
    unusable = {**healthy, "image_path": "a-unusable.png", **ineligible_fields}
    candidates = pd.DataFrame([unusable, healthy])
    selected = select_candidates(candidates, batch_size=2, seed=23)
    reordered = select_candidates(candidates.iloc[::-1], batch_size=2, seed=23)
    assert selected["image_path"].tolist() == ["z-healthy.png"]
    assert selected["selector_eligible"].tolist() == [True]
    pd.testing.assert_frame_equal(selected, reordered)


def test_random_control_uses_full_pool_before_stratum_score_shortlist() -> None:
    import hashlib

    candidates = _spaced_selection_candidates(5_001)
    candidates["region"] = [f"stratum-{index:05d}" for index in candidates.index]
    candidates["uncertainty"] = np.arange(1, len(candidates) + 1, dtype=float)
    seed = 31
    # The smallest seeded draw is deliberately low-scoring and the last stratum.
    # With more strata than shortlist slots, its representative is not shortlisted.
    control_index = min(
        candidates.index,
        key=lambda index: hashlib.sha256(
            f"{seed}\0random-control\0z14/{index * 64}/100".encode()
        ).digest()[:8],
    )
    candidates.loc[control_index, "region"] = "zz-control-stratum"
    candidates.loc[control_index, "uncertainty"] = 0.0
    best_index = int(candidates["uncertainty"].idxmax())
    kwargs = {
        "batch_size": 2,
        "seed": seed,
        "adjacency_radius": 0,
        "random_control_count": 1,
        "weights": {
            "disagreement": 0,
            "uncertainty": 1,
            "diversity": 0,
            "geographic": 0,
            "underrepresented": 0,
            "score_band": 0,
            "random_control": 0,
        },
    }
    selected = select_candidates(candidates, **kwargs)
    reordered = select_candidates(candidates.iloc[::-1], **kwargs)
    assert selected["image_path"].tolist() == candidates.loc[
        [control_index, best_index], "image_path"
    ].tolist()
    assert selected["is_random_control"].tolist() == [True, False]
    assert selected["is_qa_overlap"].tolist() == [False, False]
    assert selected.attrs["diagnostics"]["eligible_rows"] == 5_001
    assert selected.attrs["diagnostics"]["shortlist_rows"] == 5_000
    pd.testing.assert_frame_equal(selected, reordered)


def test_geographic_registry_freezes_blocks_across_pool_changes() -> None:
    from scenicdriver.active_learning.selection import build_geographic_splits

    original = _spaced_selection_candidates(40)
    kwargs = {"seed": 17, "adjacency_radius": 0}
    initial = build_geographic_splits(original.iloc[:20], **kwargs)
    registry = initial.attrs["geographic_split_registry"]
    growing = build_geographic_splits(original, registry=registry, **kwargs)
    shrinking = build_geographic_splits(
        original.iloc[5:15], registry=growing.attrs["geographic_split_registry"],
        seed=99, adjacency_radius=0,
    )
    initial_by_block = initial.set_index("geographic_block")["split"].to_dict()
    growing_by_block = growing.set_index("geographic_block")["split"].to_dict()
    shrinking_by_block = shrinking.set_index("geographic_block")["split"].to_dict()
    assert all(growing_by_block[block] == split for block, split in initial_by_block.items())
    assert all(growing_by_block[block] == split for block, split in shrinking_by_block.items())
    # Independently introduced blocks get the same assignment; pool size is irrelevant.
    new_only = build_geographic_splits(original.iloc[20:], **kwargs)
    assert all(
        growing_by_block[row.geographic_block] == row.split
        for row in new_only.itertuples()
    )
    assert growing.attrs["geographic_split_registry"] == shrinking.attrs["geographic_split_registry"]
    assert set(shrinking["split_seed"]) == {17}
    assert registry["regions"] != growing.attrs["geographic_split_registry"]["regions"]


def test_geographic_block_assignment_ignores_region_aliases() -> None:
    from scenicdriver.active_learning.selection import build_geographic_splits

    candidates = _spaced_selection_candidates(40)
    first = build_geographic_splits(candidates, seed=7, adjacency_radius=0)
    candidates["region"] = "neighboring-region"
    candidates["tile_identity"] = [
        f"neighboring-region/z{row.z}/x{row.x}/y{row.y}" for row in candidates.itertuples()
    ]
    candidates = candidates.drop(columns=["z", "x", "y"])
    candidates["image_path"] = "alias-" + candidates["image_path"]
    second = build_geographic_splits(candidates, seed=7, adjacency_radius=0)
    assert first.set_index("geographic_block")["split"].to_dict() == second.set_index(
        "geographic_block"
    )["split"].to_dict()


def test_selection_persists_registry_snapshot_for_historical_handoff(tmp_path: Path) -> None:
    from scenicdriver.active_learning.selection import SelectionConfig, run_selection

    source = tmp_path / "pool.csv"
    candidates = _spaced_selection_candidates(40)
    candidates.iloc[:20].to_csv(source, index=False)
    output = tmp_path / "rounds"
    first = run_selection(
        source, output_dir=output, run_name="first",
        config=SelectionConfig(batch_size=5, seed=17, adjacency_radius=0, run_name="first"),
    )
    snapshot = output / "first" / "geographic_split_registry.json"
    frozen = snapshot.read_bytes()
    record = first.batch_manifest["geographic_split_registry"]
    assert record["sha256"] == sha256_file(snapshot)
    assert record == first.batch_manifest["outputs"]["geographic_split_registry_json"]
    assert first.batch_manifest["selection_contract"]["geographic_split_registry_sha256"] == record["sha256"]
    candidates.to_csv(source, index=False)
    second = run_selection(
        source, output_dir=output, run_name="second",
        config=SelectionConfig(batch_size=5, seed=99, adjacency_radius=0, run_name="second"),
    )
    assert snapshot.read_bytes() == frozen
    assert record["sha256"] != second.batch_manifest["geographic_split_registry"]["sha256"]
    first_blocks = first.geographic_splits.set_index("geographic_block")["split"].to_dict()
    second_blocks = second.geographic_splits.set_index("geographic_block")["split"].to_dict()
    assert all(second_blocks[block] == split for block, split in first_blocks.items())
    # A prior round is reproducible even after the root registry grows.
    candidates.iloc[:20].to_csv(source, index=False)
    repeated = run_selection(
        source, output_dir=output, run_name="first",
        config=SelectionConfig(batch_size=5, seed=17, adjacency_radius=0, run_name="first"),
    )
    assert repeated.batch_manifest == first.batch_manifest
    assert snapshot.read_bytes() == frozen


def test_geographic_registry_rejects_assignment_and_rule_tampering() -> None:
    from scenicdriver.active_learning.selection import build_geographic_splits

    candidates = _spaced_selection_candidates(20)
    initial = build_geographic_splits(candidates, seed=17, adjacency_radius=0)
    registry = json.loads(json.dumps(initial.attrs["geographic_split_registry"]))
    block = next(iter(registry["regions"]["fixture"]))
    current = registry["regions"]["fixture"][block]
    registry["regions"]["fixture"][block] = "train" if current != "train" else "test"
    with pytest.raises(ValueError, match="assignment changed"):
        build_geographic_splits(candidates, registry=registry, adjacency_radius=0)
    with pytest.raises(ValueError, match="rule changed"):
        build_geographic_splits(
            candidates, registry=initial.attrs["geographic_split_registry"],
            block_size=16, adjacency_radius=0,
        )
