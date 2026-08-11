from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from scripts.annotation.build_source_shift_batch import build_source_shift_batch
from scripts.modeling.evaluate_source_shift import evaluate_source_shift


def _create_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create a tiny 1x1 raw binary file to act as a valid PNG/image file
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")


def _make_manifest_data(
    base_dir: Path,
    source_prefix: str,
    coords: list[tuple[str, int, int, int]],
    source_identity: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    sat_dir = base_dir / source_prefix / "satellite"
    terr_dir = base_dir / source_prefix / "terrain"

    rows = []
    for region, z, x, y in coords:
        sat_path = sat_dir / f"{region}_{z}_{x}_{y}.png"
        terr_path = terr_dir / f"{region}_{z}_{x}_{y}.png"
        _create_dummy_image(sat_path)
        _create_dummy_image(terr_path)

        row = {
            "region": region,
            "z": z,
            "x": x,
            "y": y,
            "satellite_path": str(sat_path),
            "terrain_path": str(terr_path),
        }
        if source_identity:
            row["source_identity"] = source_identity
        rows.append(row)

    df = pd.DataFrame(rows)
    manifest_path = base_dir / f"{source_prefix}_tile_manifest.csv"
    df.to_csv(manifest_path, index=False)
    return df, manifest_path


def _make_strict_human_annotations(
    image_paths: list[str], scores: dict[str, float] | None = None
) -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(image_paths):
        score = scores.get(path, 6.5) if scores else 6.5
        rows.append(
            {
                "image_path": path,
                "scenic_human": score,
                "confidence": "high",
                "skip": False,
                "annotator_id": "annotator_1",
                "timestamp": f"2026-08-10T12:{idx:02d}:00Z",
                "notes": "test annotation",
            }
        )
    return pd.DataFrame(rows)


def _make_predictions(
    image_paths: list[str], scores: dict[str, float] | None = None
) -> pd.DataFrame:
    rows = []
    for path in image_paths:
        score = scores.get(path, 6.0) if scores else 6.0
        rows.append({"image_path": path, "scenic_score": score})
    return pd.DataFrame(rows)


def _write_prediction_manifests(
    batch_summary_path: Path,
    batch_csv_path: Path,
    old_predictions_csv: Path,
    new_predictions_csv: Path,
) -> None:
    summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    batch_sha = hashlib.sha256(batch_csv_path.read_bytes()).hexdigest()
    shared = {
        "schema_version": 1,
        "batch_csv_sha256": batch_sha,
        "preprocessing_contract_sha256": "1" * 64,
        "grid_contract_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "calibration_sha256": "4" * 64,
    }
    for variant, predictions_csv in (
        ("old", old_predictions_csv),
        ("new", new_predictions_csv),
    ):
        prediction_columns = list(pd.read_csv(predictions_csv).columns)
        schema_bytes = json.dumps(
            prediction_columns, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest = {
            **shared,
            "source_variant": variant,
            "source_identity": summary[f"{variant}_source_identity"],
            "source_manifest_sha256": summary[f"{variant}_manifest_sha256"],
            "predictions_csv_sha256": hashlib.sha256(
                predictions_csv.read_bytes()
            ).hexdigest(),
            "prediction_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        }
        predictions_csv.with_suffix(
            predictions_csv.suffix + ".manifest.json"
        ).write_text(json.dumps(manifest), encoding="utf-8")


def test_deterministic_selection_and_order(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(120)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv1 = tmp_path / "batch1.csv"
    out_json1 = tmp_path / "batch1.json"
    summary1 = build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv1,
        output_summary_json=out_json1,
        sample_size=100,
        seed=42,
    )

    out_csv2 = tmp_path / "batch2.csv"
    out_json2 = tmp_path / "batch2.json"
    summary2 = build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv2,
        output_summary_json=out_json2,
        sample_size=100,
        seed=42,
    )

    assert summary1["output_batch_csv_sha256"] == summary2["output_batch_csv_sha256"]
    assert out_csv1.read_text() == out_csv2.read_text()
    assert summary1["pair_unblinding"] == summary2["pair_unblinding"]


def test_exact_coordinate_pairing(tmp_path: Path) -> None:
    coords_common = [("sne", 14, 4800 + i, 6000 + i) for i in range(105)]
    coords_old_only = [("sne", 14, 9000 + i, 9000 + i) for i in range(10)]
    coords_new_only = [("sne", 14, 9500 + i, 9500 + i) for i in range(10)]

    _, old_m = _make_manifest_data(
        tmp_path, "old", coords_common + coords_old_only, "mapbox_v1"
    )
    _, new_m = _make_manifest_data(
        tmp_path, "new", coords_common + coords_new_only, "naip_3dep_v1"
    )

    out_csv = tmp_path / "batch.csv"
    summary = build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv,
        sample_size=100,
        seed=42,
    )

    assert summary["sample_size_pairs"] == 100
    df = pd.read_csv(out_csv)
    assert len(df) == 200

    # Ensure all paired coordinates in batch were from common set
    common_set = set(coords_common)
    for _, row in df.iterrows():
        coord = (row["region"], row["z"], row["x"], row["y"])
        assert coord in common_set


def test_rejection_of_missing_data(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    # Remove one tile image file to trigger missing file detection
    old_df = pd.read_csv(old_m)
    missing_file = Path(old_df.iloc[0]["satellite_path"])
    missing_file.unlink()

    out_csv = tmp_path / "batch.csv"
    with pytest.raises(
        ValueError, match="Insufficient matching valid coordinate pairs"
    ):
        build_source_shift_batch(
            old_manifest=old_m,
            new_manifest=new_m,
            output_batch_csv=out_csv,
            sample_size=100,
            seed=42,
            check_files=True,
        )


def test_rejection_of_duplicate_data(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(105)]
    coords_with_dup = coords + [coords[0]]  # duplicate entry

    sat_dir = tmp_path / "old" / "satellite"
    terr_dir = tmp_path / "old" / "terrain"
    rows = []
    for region, z, x, y in coords_with_dup:
        sat_path = sat_dir / f"{region}_{z}_{x}_{y}.png"
        terr_path = terr_dir / f"{region}_{z}_{x}_{y}.png"
        _create_dummy_image(sat_path)
        _create_dummy_image(terr_path)
        rows.append(
            {
                "region": region,
                "z": z,
                "x": x,
                "y": y,
                "satellite_path": str(sat_path),
                "terrain_path": str(terr_path),
                "source_identity": "mapbox_v1",
            }
        )

    old_m = tmp_path / "old_dup_manifest.csv"
    pd.DataFrame(rows).to_csv(old_m, index=False)

    _, new_m = _make_manifest_data(tmp_path, "new", coords[:105], "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    with pytest.raises(ValueError, match="contains duplicate coordinate entries"):
        build_source_shift_batch(
            old_manifest=old_m,
            new_manifest=new_m,
            output_batch_csv=out_csv,
            sample_size=100,
            seed=42,
        )


def test_rejection_of_identity_ambiguous_data(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(105)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "naip_3dep_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    with pytest.raises(ValueError, match="Source identity ambiguity"):
        build_source_shift_batch(
            old_manifest=old_m,
            new_manifest=new_m,
            output_batch_csv=out_csv,
            sample_size=100,
            seed=42,
        )


def test_rejection_of_missing_source_identity(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_manifest = _make_manifest_data(tmp_path, "old", coords)
    _, new_manifest = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    with pytest.raises(ValueError, match="lacks source_identity"):
        build_source_shift_batch(
            old_manifest=old_manifest,
            new_manifest=new_manifest,
            output_batch_csv=tmp_path / "batch.csv",
            sample_size=100,
            seed=42,
        )


def test_rejection_of_insufficient_pairs(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(50)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    with pytest.raises(ValueError, match="sample_size must be between 100 and 300"):
        build_source_shift_batch(
            old_manifest=old_m,
            new_manifest=new_m,
            output_batch_csv=out_csv,
            sample_size=50,
            seed=42,
        )


def test_strict_complete_human_coverage(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    out_summary = tmp_path / "batch.summary.json"
    build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv,
        output_summary_json=out_summary,
        sample_size=100,
        seed=42,
    )

    batch_df = pd.read_csv(out_csv)
    all_paths = batch_df["image_path"].tolist()

    # Drop 1 path from human annotations
    ann_df = _make_strict_human_annotations(all_paths[:-1])
    ann_csv = tmp_path / "labels_human.csv"
    ann_df.to_csv(ann_csv, index=False)

    old_preds_csv = tmp_path / "old_preds.csv"
    _make_predictions(all_paths).to_csv(old_preds_csv, index=False)

    new_preds_csv = tmp_path / "new_preds.csv"
    _make_predictions(all_paths).to_csv(new_preds_csv, index=False)
    _write_prediction_manifests(out_summary, out_csv, old_preds_csv, new_preds_csv)

    out_report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="Strict complete human coverage violated"):
        evaluate_source_shift(
            batch_summary_json=out_summary,
            annotations_csv=ann_csv,
            old_predictions_csv=old_preds_csv,
            new_predictions_csv=new_preds_csv,
            output_report_json=out_report,
        )


def test_strict_complete_prediction_coverage(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    out_summary = tmp_path / "batch.summary.json"
    build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv,
        output_summary_json=out_summary,
        sample_size=100,
        seed=42,
    )

    batch_df = pd.read_csv(out_csv)
    all_paths = batch_df["image_path"].tolist()

    ann_csv = tmp_path / "labels_human.csv"
    _make_strict_human_annotations(all_paths).to_csv(ann_csv, index=False)

    old_preds_csv = tmp_path / "old_preds.csv"
    # Drop 1 path from old predictions
    _make_predictions(all_paths[:-1]).to_csv(old_preds_csv, index=False)

    new_preds_csv = tmp_path / "new_preds.csv"
    _make_predictions(all_paths).to_csv(new_preds_csv, index=False)
    _write_prediction_manifests(out_summary, out_csv, old_preds_csv, new_preds_csv)

    out_report = tmp_path / "report.json"
    with pytest.raises(
        ValueError, match="Strict complete prediction coverage violated"
    ):
        evaluate_source_shift(
            batch_summary_json=out_summary,
            annotations_csv=ann_csv,
            old_predictions_csv=old_preds_csv,
            new_predictions_csv=new_preds_csv,
            output_report_json=out_report,
        )


def test_deterministic_metrics_and_ci(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    out_summary = tmp_path / "batch.summary.json"
    build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv,
        output_summary_json=out_summary,
        sample_size=100,
        seed=42,
    )

    batch_df = pd.read_csv(out_csv)
    all_paths = batch_df["image_path"].tolist()

    # Synthetic realistic scores
    scores_human = {p: 4.0 + (idx % 5) * 1.0 for idx, p in enumerate(all_paths)}
    scores_old_pred = {p: scores_human[p] + 0.1 for p in all_paths}
    scores_new_pred = {p: scores_human[p] + 0.2 for p in all_paths}

    ann_csv = tmp_path / "labels_human.csv"
    _make_strict_human_annotations(all_paths, scores_human).to_csv(ann_csv, index=False)

    old_preds_csv = tmp_path / "old_preds.csv"
    _make_predictions(all_paths, scores_old_pred).to_csv(old_preds_csv, index=False)

    new_preds_csv = tmp_path / "new_preds.csv"
    _make_predictions(all_paths, scores_new_pred).to_csv(new_preds_csv, index=False)
    _write_prediction_manifests(out_summary, out_csv, old_preds_csv, new_preds_csv)

    out_report1 = tmp_path / "report1.json"
    rep1 = evaluate_source_shift(
        batch_summary_json=out_summary,
        annotations_csv=ann_csv,
        old_predictions_csv=old_preds_csv,
        new_predictions_csv=new_preds_csv,
        output_report_json=out_report1,
        seed=42,
        n_bootstrap=200,
    )

    out_report2 = tmp_path / "report2.json"
    rep2 = evaluate_source_shift(
        batch_summary_json=out_summary,
        annotations_csv=ann_csv,
        old_predictions_csv=old_preds_csv,
        new_predictions_csv=new_preds_csv,
        output_report_json=out_report2,
        seed=42,
        n_bootstrap=200,
    )

    assert rep1 == rep2
    assert out_report1.read_text() == out_report2.read_text()
    assert rep1["passed"] is True


def test_prediction_manifests_must_bind_one_model_pipeline(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_manifest = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_manifest = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")
    batch_csv = tmp_path / "batch.csv"
    batch_summary = tmp_path / "batch.summary.json"
    build_source_shift_batch(
        old_manifest=old_manifest,
        new_manifest=new_manifest,
        output_batch_csv=batch_csv,
        output_summary_json=batch_summary,
        sample_size=100,
    )
    paths = pd.read_csv(batch_csv)["image_path"].tolist()
    annotations_csv = tmp_path / "labels_human.csv"
    _make_strict_human_annotations(paths).to_csv(annotations_csv, index=False)
    old_predictions = tmp_path / "old.csv"
    new_predictions = tmp_path / "new.csv"
    _make_predictions(paths).to_csv(old_predictions, index=False)
    _make_predictions(paths).to_csv(new_predictions, index=False)
    _write_prediction_manifests(
        batch_summary, batch_csv, old_predictions, new_predictions
    )
    new_manifest_path = new_predictions.with_suffix(".csv.manifest.json")
    identity = json.loads(new_manifest_path.read_text(encoding="utf-8"))
    identity["checkpoint_sha256"] = "9" * 64
    new_manifest_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree on checkpoint_sha256"):
        evaluate_source_shift(
            batch_summary_json=batch_summary,
            annotations_csv=annotations_csv,
            old_predictions_csv=old_predictions,
            new_predictions_csv=new_predictions,
            output_report_json=tmp_path / "report.json",
        )


def test_failing_threshold(tmp_path: Path) -> None:
    coords = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]
    _, old_m = _make_manifest_data(tmp_path, "old", coords, "mapbox_v1")
    _, new_m = _make_manifest_data(tmp_path, "new", coords, "naip_3dep_v1")

    out_csv = tmp_path / "batch.csv"
    out_summary = tmp_path / "batch.summary.json"
    build_source_shift_batch(
        old_manifest=old_m,
        new_manifest=new_m,
        output_batch_csv=out_csv,
        output_summary_json=out_summary,
        sample_size=100,
        seed=42,
    )

    batch_df = pd.read_csv(out_csv)
    all_paths = batch_df["image_path"].tolist()

    ann_csv = tmp_path / "labels_human.csv"
    _make_strict_human_annotations(all_paths).to_csv(ann_csv, index=False)

    old_preds_csv = tmp_path / "old_preds.csv"
    _make_predictions(all_paths).to_csv(old_preds_csv, index=False)

    # Intentionally bad predictions for new source (e.g. MAE > 5.0)
    new_scores = {p: 1.0 for p in all_paths}
    new_preds_csv = tmp_path / "new_preds.csv"
    _make_predictions(all_paths, new_scores).to_csv(new_preds_csv, index=False)
    _write_prediction_manifests(out_summary, out_csv, old_preds_csv, new_preds_csv)

    out_report = tmp_path / "report.json"
    report = evaluate_source_shift(
        batch_summary_json=out_summary,
        annotations_csv=ann_csv,
        old_predictions_csv=old_preds_csv,
        new_predictions_csv=new_preds_csv,
        output_report_json=out_report,
        max_mae_new=1.5,
        strict=False,
    )

    assert report["passed"] is False
    assert report["threshold_checks"]["mae_new"]["passed"] is False

    with pytest.raises(ValueError, match="Source shift evaluation failed thresholds"):
        evaluate_source_shift(
            batch_summary_json=out_summary,
            annotations_csv=ann_csv,
            old_predictions_csv=old_preds_csv,
            new_predictions_csv=new_preds_csv,
            output_report_json=out_report,
            max_mae_new=1.5,
            strict=True,
        )
