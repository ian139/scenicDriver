from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.evaluate_source_shift import (
    STRICT_HUMAN_COLUMNS,
    compute_bootstrap_ci,
    evaluate_source_shift,
    validate_human_annotations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fixed_splits(heuristic: Path) -> Path:
    splits = heuristic.with_name("geographic_splits.csv")
    frame = pd.read_csv(heuristic)[["image_path"]]
    frame["split"] = "train"
    frame.to_csv(splits, index=False)
    return splits


def test_build_mixed_labels_writes_audit_columns_and_retains_heuristics(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source\n"
        "raw/images/satellite/z14/region/1_2.png,4.0,7,heuristic\n"
        "raw/images/satellite/z14/region/3_4.png,6.0,8,heuristic\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,timestamp\n"
        "raw/images/satellite/z14/region/1_2.png,a,8.0,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/1_2.png,b,6.0,2026-01-01T00:01:00Z\n",
        encoding="utf-8",
    )
    splits = _write_fixed_splits(heuristic)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/modeling/build_mixed_labels.py"),
            "--heuristic-labels",
            str(heuristic),
            "--annotations-csv",
            str(annotations),
            "--geographic-splits-csv",
            str(splits),
            "--output",
            str(output),
            "--aggregate",
            "mean",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    mixed = pd.read_csv(output).sort_values("image_path").reset_index(drop=True)
    human_row = mixed.loc[mixed["image_path"].str.endswith("1_2.png")].iloc[0]
    heuristic_row = mixed.loc[mixed["image_path"].str.endswith("3_4.png")].iloc[0]

    assert human_row["label_source"] == "human_override"
    assert human_row["scenic_score"] == 7.0
    assert human_row["scenic_human_mean"] == 7.0
    assert human_row["scenic_human_median"] == 7.0
    assert human_row["human_annotation_count"] == 2
    assert human_row["human_annotator_count"] == 2
    assert round(float(human_row["scenic_human_std"]), 6) == round(2**0.5, 6)

    assert heuristic_row["label_source"] == "heuristic"
    assert heuristic_row["scenic_score"] == 6.0


def test_build_mixed_labels_excludes_skipped_and_retains_scored_and_heuristic_rows(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source,score_status\n"
        "raw/images/satellite/z14/region/path1.png,4.0,7,heuristic,scored\n"
        "raw/images/satellite/z14/region/path2.png,5.0,8,heuristic,scored\n"
        "raw/images/satellite/z14/region/path3.png,6.0,9,heuristic,scored\n"
        "raw/images/satellite/z14/region/path4.png,,10,heuristic,scored\n"
        "raw/images/satellite/z14/region/path5.png,2.0,11,heuristic,unusable\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,skip,timestamp\n"
        "raw/images/satellite/z14/region/path1.png,alice,8.0,False,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/path2.png,bob,,True,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    splits = _write_fixed_splits(heuristic)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/modeling/build_mixed_labels.py"),
            "--heuristic-labels",
            str(heuristic),
            "--annotations-csv",
            str(annotations),
            "--geographic-splits-csv",
            str(splits),
            "--output",
            str(output),
            "--aggregate",
            "mean",
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


    mixed = pd.read_csv(output).sort_values("image_path").reset_index(drop=True)
    paths = mixed["image_path"].tolist()

    assert "raw/images/satellite/z14/region/path2.png" not in paths
    assert "raw/images/satellite/z14/region/path4.png" not in paths
    assert "raw/images/satellite/z14/region/path5.png" not in paths
    assert len(mixed) == 2

    override_row = mixed.loc[mixed["image_path"].str.endswith("path1.png")].iloc[0]
    heuristic_row = mixed.loc[mixed["image_path"].str.endswith("path3.png")].iloc[0]

    assert override_row["label_source"] == "human_override"
    assert float(override_row["scenic_score"]) == 8.0

    assert heuristic_row["label_source"] == "heuristic"
    assert float(heuristic_row["scenic_score"]) == 6.0


def test_build_mixed_labels_latest_record_deduplication_for_skipped_status(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source\n"
        "raw/images/satellite/z14/region/resurrected.png,3.0,1,heuristic\n"
        "raw/images/satellite/z14/region/skipped_later.png,4.0,2,heuristic\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,skip,timestamp\n"
        "raw/images/satellite/z14/region/resurrected.png,alice,,True,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/resurrected.png,alice,9.0,False,2026-01-02T00:00:00Z\n"
        "raw/images/satellite/z14/region/skipped_later.png,bob,7.0,False,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/skipped_later.png,bob,,True,2026-01-02T00:00:00Z\n",
        encoding="utf-8",
    )
    splits = _write_fixed_splits(heuristic)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/modeling/build_mixed_labels.py"),
            "--heuristic-labels",
            str(heuristic),
            "--annotations-csv",
            str(annotations),
            "--geographic-splits-csv",
            str(splits),
            "--output",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    mixed = pd.read_csv(output)
    paths = mixed["image_path"].tolist()

    assert "raw/images/satellite/z14/region/skipped_later.png" not in paths
    assert "raw/images/satellite/z14/region/resurrected.png" in paths

    res_row = mixed.loc[mixed["image_path"].str.endswith("resurrected.png")].iloc[0]
    assert res_row["label_source"] == "human_override"
    assert float(res_row["scenic_score"]) == 9.0


def test_build_mixed_labels_multi_annotator_skip_excludes_image(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source\n"
        "raw/images/satellite/z14/region/conflict.png,5.0,1,heuristic\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,skip,timestamp\n"
        "raw/images/satellite/z14/region/conflict.png,alice,8.5,False,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/conflict.png,bob,,True,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    splits = _write_fixed_splits(heuristic)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/modeling/build_mixed_labels.py"),
            "--heuristic-labels",
            str(heuristic),
            "--annotations-csv",
            str(annotations),
            "--geographic-splits-csv",
            str(splits),
            "--output",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    mixed = pd.read_csv(output)
    assert len(mixed) == 0


def _build_snapshot(
    base: Path, annotations: Path, output: Path, splits: Path, *, keep: bool = False
) -> pd.DataFrame:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/modeling/build_mixed_labels.py"),
            "--heuristic-labels", str(base),
            "--annotations-csv", str(annotations),
            "--geographic-splits-csv", str(splits),
            "--output", str(output),
            "--keep-existing-human" if keep else "--no-keep-existing-human",
        ],
        check=True, cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return pd.read_csv(output)


@pytest.mark.parametrize("keep", [False, True])
def test_mixed_rebuild_replaces_audits_and_resets_or_retains_old_targets(
    tmp_path: Path, keep: bool
) -> None:
    base, ann, output = (tmp_path / name for name in ("base.csv", "ann.csv", "mixed.csv"))
    pd.DataFrame({
        "image_path": ["a.png", "b.png"],
        "scenic_score": [3.0, 4.0],
    }).to_csv(base, index=False)
    splits = _write_fixed_splits(base)
    pd.DataFrame({
        "image_path": ["a.png", "a.png", "b.png"],
        "annotator_id": ["alice", "bob", "alice"],
        "scenic_human": [6.0, 8.0, 9.0],
    }).to_csv(ann, index=False)
    _build_snapshot(base, ann, output, splits)
    pd.DataFrame({
        "image_path": ["a.png"], "annotator_id": ["alice"], "scenic_human": [5.0],
    }).to_csv(ann, index=False)
    rebuilt = _build_snapshot(output, ann, output, splits, keep=keep).set_index("image_path")
    assert rebuilt.loc["a.png", "scenic_score"] == 5.0
    assert rebuilt.loc["a.png", "scenic_score_heuristic"] == 3.0
    assert rebuilt.loc["a.png", "scenic_human_mean"] == 5.0
    assert rebuilt.loc["a.png", "scenic_human_median"] == 5.0
    assert rebuilt.loc["a.png", "scenic_human_std"] == 0.0
    assert rebuilt.loc["a.png", "human_annotation_count"] == 1
    assert rebuilt.loc["a.png", "human_annotator_count"] == 1
    assert rebuilt.loc["b.png", "scenic_score"] == (9.0 if keep else 4.0)
    assert rebuilt.loc["b.png", "label_source"] == ("human_override" if keep else "heuristic")
    if keep:
        assert rebuilt.loc["b.png", "scenic_human"] == 9.0
    else:
        assert pd.isna(rebuilt.loc["b.png", "scenic_human"])
    assert pd.isna(rebuilt.loc["b.png", "human_annotation_count"])


def test_mixed_blank_skip_and_scale_endpoints_are_admitted(tmp_path: Path) -> None:
    base, ann, output = (tmp_path / name for name in ("base.csv", "ann.csv", "mixed.csv"))
    pd.DataFrame({
        "image_path": ["zero.png", "ten.png"], "scenic_score": [3.0, 4.0],
    }).to_csv(base, index=False)
    ann.write_text(
        "image_path,annotator_id,scenic_human,skip\n"
        "zero.png,alice,0,\nten.png,alice,10,\n", encoding="utf-8",
    )
    mixed = _build_snapshot(base, ann, output, _write_fixed_splits(base))
    assert mixed["scenic_score"].tolist() == [0.0, 10.0]
    assert mixed["label_source"].tolist() == ["human_override", "human_override"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), -1.0, 11.0])
def test_mixed_invalid_latest_supervision_excludes_image(
    tmp_path: Path, score: float
) -> None:
    base, ann, output = (tmp_path / name for name in ("base.csv", "ann.csv", "mixed.csv"))
    pd.DataFrame({
        "image_path": ["bad.png", "weak.png"], "scenic_score": [3.0, 4.0],
    }).to_csv(base, index=False)
    pd.DataFrame({
        "image_path": ["bad.png", "bad.png", "bad.png"],
        "annotator_id": ["alice", "alice", "bob"],
        "scenic_human": [5.0, score, 5.0],
        "timestamp": ["2026-01-01", "2026-01-02", "2026-01-02"],
    }).to_csv(ann, index=False)
    mixed = _build_snapshot(base, ann, output, _write_fixed_splits(base))
    assert mixed["image_path"].tolist() == ["weak.png"]
    assert mixed["scenic_score"].tolist() == [4.0]


def test_mixed_snapshot_intersects_exact_run_paths_and_replaces_splits(tmp_path: Path) -> None:
    base, ann, output = (tmp_path / name for name in ("base.csv", "ann.csv", "mixed.csv"))
    pd.DataFrame({
        "image_path": [" run/a.png ", "run/b.png", "elsewhere/a.png", "outside.png"],
        "scenic_score": [3.0, 4.0, 7.0, 8.0],
        "split": ["train"] * 4, "geographic_block": ["stale"] * 4,
    }).to_csv(base, index=False)
    splits = tmp_path / "geographic_splits.csv"
    pd.DataFrame({
        "image_path": ["run/a.png", "run/b.png", "run/not_in_base.png"],
        "split": ["validation", "test", "train"],
        "geographic_block": ["a", "b", "c"], "split_seed": [42] * 3,
    }).to_csv(splits, index=False)
    pd.DataFrame({
        "image_path": ["run/a.png", "elsewhere/a.png", "outside.png", "run/not_in_base.png"],
        "annotator_id": ["alice"] * 4, "scenic_human": [6.0, 9.0, 9.0, 10.0],
    }).to_csv(ann, index=False)
    mixed = _build_snapshot(base, ann, output, splits)
    assert mixed["image_path"].tolist() == ["run/a.png", "run/b.png"]
    assert mixed["scenic_score"].tolist() == [6.0, 4.0]
    assert mixed["split"].tolist() == ["val", "test"]
    assert mixed["geographic_block"].tolist() == ["a", "b"]
    assert mixed["split_seed"].tolist() == [42, 42]


def test_mixed_invalid_weak_and_retained_targets_never_reach_snapshot(tmp_path: Path) -> None:
    base, ann, output = (tmp_path / name for name in ("base.csv", "ann.csv", "mixed.csv"))
    pd.DataFrame({
        "image_path": ["nan.png", "inf.png", "negative.png", "large.png", "weak.png"],
        "scenic_score": [float("nan"), float("inf"), -1.0, 11.0, 3.0],
        "scenic_human": [5.0, 5.0, 5.0, 5.0, float("inf")],
        "label_source": ["human_override"] * 5,
    }).to_csv(base, index=False)
    pd.DataFrame(columns=["image_path", "annotator_id", "scenic_human"]).to_csv(ann, index=False)
    mixed = _build_snapshot(base, ann, output, _write_fixed_splits(base), keep=True)
    assert mixed["image_path"].tolist() == ["weak.png"]
    assert mixed["scenic_score"].tolist() == [3.0]
    assert pd.isna(mixed.iloc[0]["scenic_human"])
    assert mixed["label_source"].tolist() == ["heuristic"]


@pytest.mark.parametrize("conflict", ["image", "block"])
def test_mixed_rejects_conflicting_fixed_assignments(tmp_path: Path, conflict: str) -> None:
    base, ann, output = (tmp_path / name for name in ("base.csv", "ann.csv", "mixed.csv"))
    pd.DataFrame({"image_path": ["a.png"], "scenic_score": [3.0]}).to_csv(base, index=False)
    pd.DataFrame(columns=["image_path", "annotator_id", "scenic_human"]).to_csv(ann, index=False)
    splits = tmp_path / "geographic_splits.csv"
    pd.DataFrame({
        "image_path": ["a.png", "a.png" if conflict == "image" else "b.png"],
        "split": ["train", "test"], "geographic_block": ["same", "same"],
    }).to_csv(splits, index=False)
    with pytest.raises(subprocess.CalledProcessError):
        _build_snapshot(base, ann, output, splits)
    assert not output.exists()


def _source_shift_inputs(tmp_path: Path) -> dict[str, Path]:
    batch = tmp_path / "batch.csv"
    rows = [
        {"blinded_id": f"{variant}-{i}", "pair_id": f"pair-{i}",
         "source_variant": variant, "image_path": f"{variant}/{i}.png"}
        for i in range(3) for variant in ("old", "new")
    ]
    pd.DataFrame(rows).to_csv(batch, index=False)
    summary = {
        "output_batch_csv_path": str(batch),
        "output_batch_csv_sha256": hashlib.sha256(batch.read_bytes()).hexdigest(),
        "old_source_identity": "old-source", "new_source_identity": "new-source",
        "old_manifest_sha256": "a" * 64, "new_manifest_sha256": "b" * 64,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    human_scores = {"old": [2.0, 4.0, 6.0], "new": [3.0, 6.0, 9.0]}
    annotations = tmp_path / "ann.csv"
    pd.DataFrame([
        {"image_path": row["image_path"],
         "scenic_human": human_scores[row["source_variant"]][int(row["pair_id"][-1])],
         "confidence": "high", "skip": False, "annotator_id": "alice",
         "timestamp": "2026-01-02T00:00:00Z", "notes": ""}
        for row in rows
    ], columns=STRICT_HUMAN_COLUMNS).to_csv(annotations, index=False)
    inputs = {
        "batch_summary_json": summary_path, "annotations_csv": annotations,
        "output_report_json": tmp_path / "report.json",
    }
    for variant in ("old", "new"):
        predictions = tmp_path / f"{variant}.csv"
        frame = pd.DataFrame({
            "image_path": [f"{variant}/{i}.png" for i in range(3)],
            "scenic_score": human_scores[variant],
        })
        frame.to_csv(predictions, index=False)
        inputs[f"{variant}_predictions_csv"] = predictions
        manifest = {
            "schema_version": 1, "source_variant": variant,
            "source_identity": summary[f"{variant}_source_identity"],
            "source_manifest_sha256": summary[f"{variant}_manifest_sha256"],
            "batch_csv_sha256": summary["output_batch_csv_sha256"],
            "predictions_csv_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "prediction_schema_sha256": hashlib.sha256(
                json.dumps(list(frame.columns), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "preprocessing_contract_sha256": "c" * 64,
            "grid_contract_sha256": "d" * 64,
            "checkpoint_sha256": "e" * 64, "calibration_sha256": "f" * 64,
        }
        predictions.with_suffix(".csv.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
    return inputs


def test_source_shift_aggregates_latest_decisions_not_history(tmp_path: Path) -> None:
    inputs = _source_shift_inputs(tmp_path)
    ann = pd.read_csv(inputs["annotations_csv"])
    older = ann.copy()
    older["scenic_human"] = 0.0
    older["timestamp"] = "2026-01-01T00:00:00Z"
    older.loc[0, "skip"] = True
    ann["skip"] = ""
    pd.concat([ann, older], ignore_index=True).to_csv(inputs["annotations_csv"], index=False)
    report = evaluate_source_shift(**inputs, n_bootstrap=100)
    assert report["passed"] is True
    assert report["metrics"]["mae_old"] == 0.0
    assert report["metrics"]["mae_new"] == 0.0
    assert report["metrics"]["mean_human_delta"] == 2.0
    assert report["confidence_intervals_95"]["mae_new"] == {
        "mean": 0.0, "ci_2_5": 0.0, "ci_97_5": 0.0,
    }


@pytest.mark.parametrize("latest", ["skip", "invalid", "tie_skip"])
def test_source_shift_latest_unusable_decision_blocks_coverage(
    tmp_path: Path, latest: str
) -> None:
    inputs = _source_shift_inputs(tmp_path)
    ann = pd.read_csv(inputs["annotations_csv"])
    decision = ann.iloc[[0]].copy()
    decision["timestamp"] = (
        "2026-01-02T00:00:00Z" if latest == "tie_skip" else "2026-01-03T00:00:00Z"
    )
    decision["skip"] = latest != "invalid"
    decision["scenic_human"] = 11.0 if latest == "invalid" else float("nan")
    other_annotator = ann.iloc[[0]].copy()
    other_annotator["annotator_id"] = "bob"
    # A tie skip wins even when the earlier scored row appears last in the CSV.
    pd.concat([decision, ann, other_annotator], ignore_index=True).to_csv(
        inputs["annotations_csv"], index=False,
    )
    with pytest.raises(ValueError, match="complete human coverage"):
        evaluate_source_shift(**inputs, n_bootstrap=100)
    assert not inputs["output_report_json"].exists()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), -1.0, 11.0])
def test_source_shift_rejects_invalid_predictions_before_gates(
    tmp_path: Path, score: float
) -> None:
    inputs = _source_shift_inputs(tmp_path)
    predictions = pd.read_csv(inputs["new_predictions_csv"])
    predictions.loc[0, "scenic_score"] = score
    predictions.to_csv(inputs["new_predictions_csv"], index=False)
    with pytest.raises(ValueError, match=r"finite and in \[0, 10\]"):
        evaluate_source_shift(
            **inputs, n_bootstrap=100, min_spearman_delta=-1.0,
            max_mae_new=100.0, min_correlation_new=-1.0, max_abs_delta_bias=100.0,
        )
    assert not inputs["output_report_json"].exists()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), -1.0, 11.0])
def test_source_shift_invalid_human_scores_are_not_averaged(
    tmp_path: Path, score: float
) -> None:
    inputs = _source_shift_inputs(tmp_path)
    ann = pd.read_csv(inputs["annotations_csv"])
    other = ann.iloc[[0]].copy()
    other["annotator_id"] = "bob"
    other["scenic_human"] = score
    pd.concat([ann, other], ignore_index=True).to_csv(inputs["annotations_csv"], index=False)
    grouped, _ = validate_human_annotations(inputs["annotations_csv"])
    assert "old/0.png" not in grouped["image_path"].tolist()
    with pytest.raises(ValueError, match="complete human coverage"):
        evaluate_source_shift(**inputs, n_bootstrap=100)


@pytest.mark.parametrize("mutation", ["duplicate_old", "extra_variant", "missing_new", "blank_pair"])
def test_source_shift_rejects_malformed_pair_protocol(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _source_shift_inputs(tmp_path)
    summary = json.loads(inputs["batch_summary_json"].read_text())
    batch_path = Path(summary["output_batch_csv_path"])
    batch = pd.read_csv(batch_path)
    if mutation in {"duplicate_old", "extra_variant"}:
        extra = batch.iloc[[0]].copy()
        extra["blinded_id"] = "extra"
        extra["image_path"] = "extra.png"
        extra["source_variant"] = "old" if mutation == "duplicate_old" else "unknown"
        batch = pd.concat([batch, extra], ignore_index=True)
    elif mutation == "missing_new":
        batch = batch.drop(index=1)
    else:
        batch.loc[0, "pair_id"] = ""
    batch.to_csv(batch_path, index=False)
    summary["output_batch_csv_sha256"] = hashlib.sha256(batch_path.read_bytes()).hexdigest()
    inputs["batch_summary_json"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="Batch corrupted"):
        evaluate_source_shift(**inputs, n_bootstrap=100)
    assert not inputs["output_report_json"].exists()


@pytest.mark.parametrize("attestation", [None, "", "xyz", "g" * 64, "0" * 64])
def test_source_shift_requires_matching_frozen_batch_hash(
    tmp_path: Path, attestation: str | None
) -> None:
    inputs = _source_shift_inputs(tmp_path)
    summary = json.loads(inputs["batch_summary_json"].read_text())
    if attestation is None:
        del summary["output_batch_csv_sha256"]
    else:
        summary["output_batch_csv_sha256"] = attestation
    inputs["batch_summary_json"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen output_batch_csv_sha256|SHA-256 mismatch"):
        evaluate_source_shift(**inputs, n_bootstrap=100)
    assert not inputs["output_report_json"].exists()


@pytest.mark.parametrize("iterations", [0, 1, 99, -1, True, 100.5])
def test_source_shift_bootstrap_minimum_enforced_at_api_boundaries(
    tmp_path: Path, iterations: int
) -> None:
    inputs = _source_shift_inputs(tmp_path)
    with pytest.raises(ValueError, match="integer >= 100"):
        evaluate_source_shift(**inputs, n_bootstrap=iterations)
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="integer >= 100"):
        compute_bootstrap_ci(values, values, values, values, n_bootstrap=iterations)
    assert not inputs["output_report_json"].exists()
