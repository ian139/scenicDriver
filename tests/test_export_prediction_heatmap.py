"""Tests for scripts/modeling/export_prediction_heatmap.py.

Covers hash-bound fail-closed validation, strict row alignment by
embedding_row_index, coordinate/path/zoom validation, atomic publish, and
deterministic output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modeling.export_prediction_heatmap import (  # noqa: E402
    LABELS_COLUMNS,
    export_heatmap_run,
)

VIT_DIM, TERRAIN_DIM, NUM_CLASSES, HIDDEN_DIM = 4, 2, 3, 4
N_TILES = 24
ZOOM = 14
REGIONS = ["new_england_north", "west_south_inland"]


def _tile_center(z: int, x: int, y: int) -> tuple[float, float]:
    tile_count = 1 << z
    lon = ((x + 0.5) / tile_count) * 360.0 - 180.0
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1.0 - (2.0 * (y + 0.5) / tile_count))))
    )
    return lat, lon


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_checkpoint(path: Path) -> None:
    from scenicdriver.scenic_scorer.regression import ScenicRegressionModel

    torch.manual_seed(0)
    model = ScenicRegressionModel(
        vit_dim=VIT_DIM,
        terrain_dim=TERRAIN_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_state": "completed",
            "model_state_dict": model.state_dict(),
            "vit_dim": VIT_DIM,
            "terrain_dim": TERRAIN_DIM,
            "num_classes": NUM_CLASSES,
            "hidden_dim": HIDDEN_DIM,
        },
        path,
    )


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_pool(tmp_path: Path) -> dict[str, object]:
    """Synthetic feature pool using canonical original metadata row order."""
    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((N_TILES, VIT_DIM)).astype(np.float32)
    terrain = rng.standard_normal((N_TILES, TERRAIN_DIM)).astype(np.float32)
    class_logits = rng.standard_normal((N_TILES, NUM_CLASSES)).astype(np.float32)
    row_indices = np.arange(N_TILES, dtype=np.int64)

    npz_path = tmp_path / "feature_embeddings.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        terrain_features=terrain,
        class_logits=class_logits,
        row_indices=row_indices,
    )
    ckpt_path = tmp_path / "candidate.pt"
    _save_checkpoint(ckpt_path)

    rows: list[dict[str, object]] = []
    for idx in range(N_TILES):
        region = REGIONS[idx % 2]
        x = 4800 + idx // 100
        y = 5700 + idx
        lat, lon = _tile_center(ZOOM, x, y)
        rows.append(
            {
                "image_path": f"images/satellite/z{ZOOM}/{region}/{x}_{y}.png",
                "region": region,
                "z": ZOOM,
                "x": x,
                "y": y,
                "lat": lat,
                "lon": lon,
                "class_id": int(np.argmax(class_logits[idx])),
                "embedding_row_index": idx,
            }
        )
    meta_path = tmp_path / "candidate_pool.csv"
    _write_metadata(meta_path, rows)
    return {
        "npz": npz_path,
        "ckpt": ckpt_path,
        "meta": meta_path,
        "embeddings": embeddings,
        "terrain": terrain,
        "class_logits": class_logits,
        "rows": rows,
    }


@pytest.fixture
def pool(tmp_path: Path) -> dict[str, object]:
    return _make_pool(tmp_path)


def _make_valid_manifest(
    pool: dict[str, object], tmp_path: Path | None = None, **overrides
) -> Path:
    ds_sha = _sha256(pool["npz"])
    meta_sha = _sha256(pool["meta"])
    ckpt_sha = _sha256(pool["ckpt"])
    parent = tmp_path if tmp_path is not None else pool["npz"].parent
    man_path = (
        parent
        / f"test_manifest_{hashlib.md5(str(overrides).encode()).hexdigest()[:8]}.json"
    )
    content = {
        "dataset_sha256": ds_sha,
        "metadata_sha256": meta_sha,
        "regression_checkpoint_sha256": ckpt_sha,
        "source_contract_sha256": "1" * 64,
        "preprocessing_contract_sha256": "2" * 64,
        "grid_contract_sha256": "3" * 64,
        "classifier_checkpoint_sha256": "4" * 64,
        "calibration_artifact_sha256": "5" * 64,
        "score_schema_version": "scenic_score_v1",
        "label_schema_version": "scenic_label_v1",
        "zoom": ZOOM,
    }
    content.update(overrides)
    man_path.write_text(json.dumps(content), encoding="utf-8")
    return man_path


def _export(
    pool: dict[str, object], out_root: Path, run_name: str = "test_run", **kwargs
):
    if "identity_manifest_path" not in kwargs:
        kwargs["identity_manifest_path"] = _make_valid_manifest(pool)
    return export_heatmap_run(
        dataset_path=pool["npz"],
        metadata_path=pool["meta"],
        checkpoint_path=pool["ckpt"],
        expected_dataset_sha256=_sha256(pool["npz"]),
        expected_metadata_sha256=_sha256(pool["meta"]),
        expected_checkpoint_sha256=_sha256(pool["ckpt"]),
        run_name=run_name,
        output_root=out_root,
        device="cpu",
        batch_size=7,
        **kwargs,
    )


def _read_labels(labels_path: Path) -> list[dict[str, str]]:
    with open(labels_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_export_writes_standard_outputs(
    pool: dict[str, object], tmp_path: Path
) -> None:
    out_root = tmp_path / "nested" / "runs"
    result = _export(pool, out_root)

    run_dir = out_root / "test_run"
    assert result["run_dir"] == str(run_dir)
    assert result["total_tiles"] == N_TILES
    assert run_dir.is_dir()
    labels_path = run_dir / "labels.csv"
    run_json_path = run_dir / "run.json"
    report_path = run_dir / "report" / "report.json"
    assert labels_path.is_file()
    assert run_json_path.is_file()
    assert report_path.is_file()
    assert (run_dir / "report" / "index.html").is_file()
    assert not (run_dir / "report" / "thumbs").exists()

    # labels.csv columns and count
    labels = _read_labels(labels_path)
    assert len(labels) == N_TILES
    assert list(labels[0].keys()) == LABELS_COLUMNS

    # report.json tiles carry the minimal heatmap record
    report = json.loads(report_path.read_text())
    assert report["summary"]["total_tiles"] == N_TILES
    assert report["grid"]["has_coords"] is True
    assert report["grid"]["zoom"] == ZOOM
    assert len(report["tiles"]) == N_TILES
    for tile in report["tiles"]:
        assert set(tile) == {
            "image_path",
            "scenic_score",
            "class_id",
            "region",
            "z",
            "x",
            "y",
            "lat",
            "lon",
            "index",
        }
        assert tile["z"] == ZOOM

    # run.json provenance
    run_record = json.loads(run_json_path.read_text())
    assert run_record["run_name"] == "test_run"
    assert run_record["raw_labels_path"] is None
    assert run_record["labels_path"] == str(run_dir / "labels.csv")
    assert run_record["report_dir"] == str(run_dir / "report")
    assert run_record["summary"]["total_tiles"] == N_TILES
    run_info = run_record["run_info"]
    assert run_info["derived_visualization"] is True
    assert "NOT promotion-gate" in run_info["purpose_statement"]
    assert run_info["split"] == {
        "applicable": False,
        "note": (
            "No train/val/test split is assigned; this is full-pool inference "
            "visualization, not evaluation evidence."
        ),
    }
    assert run_info["hashes"]["dataset_sha256"] == _sha256(pool["npz"])
    assert run_info["hashes"]["metadata_sha256"] == _sha256(pool["meta"])
    assert run_info["hashes"]["checkpoint_sha256"] == _sha256(pool["ckpt"])
    assert run_info["hashes"]["source_contract_sha256"] == "1" * 64
    assert run_info["hashes"]["preprocessing_contract_sha256"] == "2" * 64
    assert run_info["hashes"]["grid_contract_sha256"] == "3" * 64
    assert run_info["hashes"]["classifier_checkpoint_sha256"] == "4" * 64
    assert run_info["hashes"]["regression_checkpoint_sha256"] == _sha256(pool["ckpt"])
    assert run_info["hashes"]["calibration_artifact_sha256"] == "5" * 64
    assert "identity_manifest_sha256" in run_info["hashes"]

    assert set(run_info["identity"].keys()) == {
        "source_contract_sha256",
        "preprocessing_contract_sha256",
        "grid_contract_sha256",
        "classifier_checkpoint_sha256",
        "regression_checkpoint_sha256",
        "score_schema_version",
        "label_schema_version",
        "calibration_artifact_sha256",
    }
    assert run_info["identity"]["grid_contract_sha256"] == "3" * 64
    assert run_info["identity"]["regression_checkpoint_sha256"] == _sha256(pool["ckpt"])
    assert run_info["counts"]["total"] == N_TILES
    assert run_info["counts"]["per_region"] == {
        "new_england_north": 12,
        "west_south_inland": 12,
    }
    assert run_info["bounds"]["zoom"] == ZOOM
    assert run_record["config"]["batch_size"] == 7
    assert run_record["config"]["device"] == "cpu"
    assert run_record["config"]["thumbnails"] is False


def test_row_mapping_strict_alignment(pool: dict[str, object], tmp_path: Path) -> None:
    """Predictions must follow the original-row/dense-feature index contract."""
    from scenicdriver.scenic_scorer.active_evaluation import (
        load_model_checkpoint,
        predict_dataset,
    )

    out_root = tmp_path / "runs"
    _export(pool, out_root)
    labels = _read_labels(out_root / "test_run" / "labels.csv")
    by_path = {row["image_path"]: float(row["scenic_score"]) for row in labels}

    model = load_model_checkpoint(pool["ckpt"], device="cpu", is_candidate=True)
    expected = predict_dataset(
        model,
        pool["embeddings"],
        pool["terrain"],
        pool["class_logits"],
        batch_size=7,
        device="cpu",
    )
    for row in pool["rows"]:
        idx = int(row["embedding_row_index"])
        got = by_path[row["image_path"]]
        assert math.isclose(got, float(expected[idx]), rel_tol=1e-6, abs_tol=1e-6), (
            f"row {idx}: {got} != {float(expected[idx])}"
        )

    # labels and report agree exactly on the same predictions
    report = json.loads((out_root / "test_run" / "report" / "report.json").read_text())
    report_scores = {t["image_path"]: float(t["scenic_score"]) for t in report["tiles"]}
    assert report_scores == by_path


@pytest.mark.parametrize(
    "which",
    ["dataset", "metadata", "checkpoint"],
)
def test_hash_mismatch_fails_closed(
    pool: dict[str, object], tmp_path: Path, which: str
) -> None:
    out_root = tmp_path / "runs"
    expected = {
        "dataset": _sha256(pool["npz"]),
        "metadata": _sha256(pool["meta"]),
        "checkpoint": _sha256(pool["ckpt"]),
    }
    expected[which] = "0" * 64  # definitely wrong
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=pool["meta"],
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=expected["dataset"],
            expected_metadata_sha256=expected["metadata"],
            expected_checkpoint_sha256=expected["checkpoint"],
            run_name="test_run",
            output_root=out_root,
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )
    assert not (out_root / "test_run").exists()
    assert not list(out_root.glob(".test_run.tmp-*"))


def test_duplicate_coordinate_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    rows = [dict(r) for r in pool["rows"]]
    rows[1]["x"] = rows[0]["x"]
    rows[1]["y"] = rows[0]["y"]
    rows[1]["image_path"] = rows[0]["image_path"]
    rows[1]["region"] = rows[0]["region"]
    rows[1]["lat"] = rows[0]["lat"]
    rows[1]["lon"] = rows[0]["lon"]
    meta_path = tmp_path / "dup_coord.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="Duplicate tile coordinate"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_class_id_mismatch_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    rows = [dict(r) for r in pool["rows"]]
    rows[0]["class_id"] = int(rows[0]["class_id"]) + 1  # no longer argmax
    meta_path = tmp_path / "bad_class.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="class_id mismatch"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_missing_row_index_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    rows = [dict(r) for r in pool["rows"]][:-1]  # drop the final original row
    meta_path = tmp_path / "missing_idx.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="expected exactly"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_path_grammar_fails_closed(pool: dict[str, object], tmp_path: Path) -> None:
    rows = [dict(r) for r in pool["rows"]]
    rows[0]["image_path"] = "images/satellite/z14/new_england_north/4846.png"
    meta_path = tmp_path / "bad_path.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="does not match"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_out_of_domain_tile_coordinate_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    rows = [dict(r) for r in pool["rows"]]
    rows[0]["x"] = 1 << ZOOM
    rows[0]["image_path"] = (
        f"images/satellite/z{ZOOM}/{rows[0]['region']}/{rows[0]['x']}_{rows[0]['y']}.png"
    )
    meta_path = tmp_path / "bad_domain.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="outside the z14 Web Mercator domain"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_mismatched_tile_center_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    rows = [dict(r) for r in pool["rows"]]
    rows[0]["lat"] = float(rows[0]["lat"]) + 0.01
    meta_path = tmp_path / "bad_center.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="lat/lon do not match"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_multiple_zooms_fails_closed(pool: dict[str, object], tmp_path: Path) -> None:
    rows = [dict(r) for r in pool["rows"]]
    rows[0]["z"] = 13
    rows[0]["image_path"] = rows[0]["image_path"].replace("/z14/", "/z13/")
    rows[0]["lat"], rows[0]["lon"] = _tile_center(
        13, int(rows[0]["x"]), int(rows[0]["y"])
    )
    meta_path = tmp_path / "two_zooms.csv"
    _write_metadata(meta_path, rows)
    with pytest.raises(ValueError, match="Multiple zooms"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=meta_path,
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(meta_path),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="test_run",
            output_root=tmp_path / "runs",
            device="cpu",
            batch_size=7,
            identity_manifest_path=_make_valid_manifest(pool, tmp_path),
        )


def test_pre_existing_run_rejected_and_not_mutated(
    pool: dict[str, object], tmp_path: Path
) -> None:
    out_root = tmp_path / "runs"
    run_dir = out_root / "test_run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "sentinel.txt"
    marker.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _export(pool, out_root)
    assert marker.read_text(encoding="utf-8") == "pre-existing"
    assert not list(out_root.glob(".test_run.tmp-*"))


def test_invalid_run_name_rejected(pool: dict[str, object], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single path segment"):
        _export(pool, tmp_path / "runs", run_name="a/b")


def test_deterministic_output(pool: dict[str, object], tmp_path: Path) -> None:
    out1 = tmp_path / "runs1"
    out2 = tmp_path / "runs2"
    _export(pool, out1)
    _export(pool, out2)
    labels1 = (out1 / "test_run" / "labels.csv").read_bytes()
    labels2 = (out2 / "test_run" / "labels.csv").read_bytes()
    assert labels1 == labels2
    report1 = (out1 / "test_run" / "report" / "report.json").read_bytes()
    report2 = (out2 / "test_run" / "report" / "report.json").read_bytes()
    assert report1 == report2


def test_absent_identity_manifest_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="requires an explicit --identity-manifest"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=pool["meta"],
            checkpoint_path=pool["ckpt"],
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(pool["meta"]),
            expected_checkpoint_sha256=_sha256(pool["ckpt"]),
            run_name="no_man_run",
            output_root=tmp_path / "runs",
            identity_manifest_path=None,
        )


def test_identity_manifest_validates_before_model_loading_or_inference(
    pool: dict[str, object], tmp_path: Path
) -> None:
    corrupt_ckpt = tmp_path / "corrupt_checkpoint.pt"
    corrupt_ckpt.write_bytes(b"not a PyTorch checkpoint")
    ckpt_sha = _sha256(corrupt_ckpt)

    man = _make_valid_manifest(
        pool,
        tmp_path,
        regression_checkpoint_sha256=ckpt_sha,
        source_contract_sha256="not_a_sha",
    )

    with pytest.raises(ValueError, match="source_contract_sha256"):
        export_heatmap_run(
            dataset_path=pool["npz"],
            metadata_path=pool["meta"],
            checkpoint_path=corrupt_ckpt,
            expected_dataset_sha256=_sha256(pool["npz"]),
            expected_metadata_sha256=_sha256(pool["meta"]),
            expected_checkpoint_sha256=ckpt_sha,
            run_name="fail_fast_run",
            output_root=tmp_path / "runs",
            identity_manifest_path=man,
        )


def test_mismatched_dataset_sha_in_manifest_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    man = _make_valid_manifest(pool, tmp_path, dataset_sha256="0" * 64)
    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        _export(pool, tmp_path / "runs", identity_manifest_path=man)


def test_mismatched_metadata_sha_in_manifest_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    man = _make_valid_manifest(pool, tmp_path, metadata_sha256="0" * 64)
    with pytest.raises(ValueError, match="metadata_sha256 mismatch"):
        _export(pool, tmp_path / "runs", identity_manifest_path=man)


def test_mismatched_checkpoint_sha_in_manifest_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    man = _make_valid_manifest(pool, tmp_path, regression_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="regression_checkpoint_sha256 mismatch"):
        _export(pool, tmp_path / "runs", identity_manifest_path=man)


def test_invalid_or_missing_64hex_sha_in_manifest_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    ds_sha = _sha256(pool["npz"])
    meta_sha = _sha256(pool["meta"])
    ckpt_sha = _sha256(pool["ckpt"])

    # Non-64-hex string
    man1 = _make_valid_manifest(pool, tmp_path, source_contract_sha256="not_a_sha")
    with pytest.raises(ValueError, match="source_contract_sha256"):
        _export(pool, tmp_path / "runs", identity_manifest_path=man1)

    # Missing required contract SHA
    bad_man = tmp_path / "bad_man.json"
    bad_man.write_text(
        json.dumps(
            {
                "dataset_sha256": ds_sha,
                "metadata_sha256": meta_sha,
                "regression_checkpoint_sha256": ckpt_sha,
                "source_contract_sha256": "1" * 64,
                "preprocessing_contract_sha256": "2" * 64,
                "grid_contract_sha256": "3" * 64,
                # missing classifier_checkpoint_sha256
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="classifier_checkpoint_sha256"):
        _export(pool, tmp_path / "runs", identity_manifest_path=bad_man)


def test_mismatched_declared_zoom_in_manifest_fails_closed(
    pool: dict[str, object], tmp_path: Path
) -> None:
    man = _make_valid_manifest(pool, tmp_path, zoom=99)
    with pytest.raises(ValueError, match="declared zoom 99 does not match"):
        _export(pool, tmp_path / "runs", identity_manifest_path=man)


def test_export_with_identity_manifest(pool: dict[str, object], tmp_path: Path) -> None:
    manifest_path = _make_valid_manifest(
        pool,
        tmp_path,
        source_contract_sha256="a" * 64,
        grid_contract_sha256="b" * 64,
        classifier_checkpoint_sha256="c" * 64,
    )

    out_root = tmp_path / "manifest_runs"
    _export(
        pool,
        out_root,
        run_name="manifest_run",
        identity_manifest_path=manifest_path,
    )

    run_record = json.loads((out_root / "manifest_run" / "run.json").read_text())
    identity = run_record["run_info"]["identity"]
    assert identity["source_contract_sha256"] == "a" * 64
    assert identity["grid_contract_sha256"] == "b" * 64
    assert identity["classifier_checkpoint_sha256"] == "c" * 64


def test_export_union_compatibility(pool: dict[str, object], tmp_path: Path) -> None:
    from scenicdriver.heuristics.report import union_reports

    out_root = tmp_path / "union_runs"
    _export(pool, out_root, run_name="run_a")
    _export(pool, out_root, run_name="run_b")

    # Matching identity contracts union successfully
    res = union_reports([out_root / "run_a", out_root / "run_b"])
    man_path = _make_valid_manifest(
        pool, tmp_path, classifier_checkpoint_sha256="e" * 64
    )
    _export(
        pool,
        out_root,
        run_name="run_c",
        identity_manifest_path=man_path,
    )

    with pytest.raises(ValueError, match="'classifier_checkpoint'"):
        union_reports([out_root / "run_a", out_root / "run_c"])
    assert res["summary"]["total_tiles"] == N_TILES


def test_sparse_original_rows_map_predictions(pool: dict[str, object], tmp_path: Path) -> None:
    from scenicdriver.scenic_scorer.active_evaluation import load_model_checkpoint, predict_dataset

    originals = np.array([19, 3, 22, 8], dtype=np.int64)
    rows = [dict(row) for row in pool["rows"]]
    for row in rows:
        row["embedding_row_index"] = -1
    for feature, original in enumerate(originals):
        rows[original]["embedding_row_index"] = feature
    _write_metadata(pool["meta"], rows)
    np.savez(
        pool["npz"],
        embeddings=pool["embeddings"][originals],
        terrain_features=pool["terrain"][originals],
        class_logits=pool["class_logits"][originals],
        row_indices=originals,
    )
    result = _export(pool, tmp_path / "runs")
    labels = _read_labels(Path(result["run_dir"]) / "labels.csv")
    model = load_model_checkpoint(pool["ckpt"], device="cpu", is_candidate=True)
    predictions = predict_dataset(
        model, pool["embeddings"][originals], pool["terrain"][originals],
        pool["class_logits"][originals], batch_size=7, device="cpu",
    )
    assert [row["image_path"] for row in labels] == [rows[i]["image_path"] for i in originals]
    np.testing.assert_allclose([float(row["scenic_score"]) for row in labels], predictions)


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [0, N_TILES]])
def test_invalid_original_rows_fail_closed(pool, tmp_path, indices) -> None:
    np.savez(
        pool["npz"], embeddings=pool["embeddings"][:2],
        terrain_features=pool["terrain"][:2], class_logits=pool["class_logits"][:2],
        row_indices=np.array(indices, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="row_indices|expected exactly"):
        _export(pool, tmp_path / "runs")
    assert not (tmp_path / "runs" / "test_run").exists()


def _regression_export(tmp_path, paths=("train", "test-b", "test-a", "validation")):
    path = tmp_path / "regression.npz"
    n = len(paths)
    np.savez(
        path, vit_embeddings=np.arange(n, dtype=np.float32).reshape(n, 1),
        terrain_features=np.zeros((n, 1), dtype=np.float32),
        class_logits=np.zeros((n, 1), dtype=np.float32),
        scenic_scores=np.full(n, 5, dtype=np.float32),
        image_paths=np.array(paths),
    )
    return path


def _benchmark_files(tmp_path):
    import pandas as pd

    split_path = tmp_path / "benchmark.csv"
    labels_path = tmp_path / "mixed.csv"
    pd.DataFrame({
        "image_path": ["test-a", "validation", "train", "test-b"],
        "split": ["test", "validation", "train", "test"],
    }).to_csv(split_path, index=False)
    pd.DataFrame({
        "image_path": ["validation", "test-a", "train", "test-b", "skipped-export"],
        "scenic_score": [5, 5, 5, 5, 5],
        "scenic_human": [6, 2, np.nan, 8, np.nan],
    }).to_csv(labels_path, index=False)
    return labels_path, split_path


def test_baseline_excludes_fixed_heldout_by_physical_tile(tmp_path) -> None:
    import pandas as pd
    from scripts.modeling.regression_split import holdout_population_mask

    paths = (
        "images/satellite/z14/east/10_10.png",  # benchmark test tile
        "images/satellite/z14/west/10_10.png",  # region alias of the same physical tile
        "images/satellite/z14/east/11_10.png",  # inside the one-tile buffer
        "images/satellite/z14/east/20_20.png",
    )
    dataset = _regression_export(tmp_path, paths)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"image_path": paths, "scenic_score": [5] * len(paths)}).to_csv(labels, index=False)
    split = tmp_path / "benchmark.csv"
    pd.DataFrame({"image_path": [paths[0], paths[3]], "split": ["test", "train"]}).to_csv(
        split, index=False
    )
    assert holdout_population_mask(dataset, labels, split, holdout_buffer_tiles=1).tolist() == [
        False, False, False, True,
    ]


def test_benchmark_uses_human_targets_with_reordered_and_skipped_exports(tmp_path) -> None:
    from scripts.modeling.compare_regression_on_benchmark import benchmark_dataset

    dataset = _regression_export(tmp_path)
    labels, split = _benchmark_files(tmp_path)
    selected = benchmark_dataset(dataset, labels, split)
    assert [float(selected[i][0][0]) for i in range(len(selected))] == [2, 1]
    assert [float(selected[i][3][0]) for i in range(len(selected))] == [2, 8]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "weak-only", "nan", "out-of-range", "duplicate-split"])
def test_benchmark_rejects_incomplete_or_nonhuman_population(tmp_path, problem) -> None:
    import pandas as pd
    from scripts.modeling.compare_regression_on_benchmark import benchmark_dataset

    paths = ("train", "test-a", "validation") if problem == "missing" else (
        ("train", "test-a", "test-a", "test-b") if problem == "duplicate"
        else ("train", "test-b", "test-a", "validation")
    )
    dataset = _regression_export(tmp_path, paths)
    labels, split = _benchmark_files(tmp_path)
    frame = pd.read_csv(labels)
    if problem == "weak-only":
        frame = frame.drop(columns="scenic_human")
    elif problem in {"nan", "out-of-range"}:
        frame.loc[frame["image_path"] == "test-a", "scenic_human"] = (
            np.nan if problem == "nan" else 11
        )
    frame.to_csv(labels, index=False)
    if problem == "duplicate-split":
        frame = pd.read_csv(split)
        pd.concat([frame, frame.iloc[:1]]).to_csv(split, index=False)
    with pytest.raises(ValueError, match="absent|duplicate|human score"):
        benchmark_dataset(dataset, labels, split)


def test_comparison_measures_both_checkpoints_on_same_human_population(tmp_path, monkeypatch) -> None:
    from scripts.modeling.compare_regression_on_benchmark import main
    from scenicdriver.scenic_scorer.regression import ScenicRegressionModel

    dataset = _regression_export(tmp_path)
    labels, split = _benchmark_files(tmp_path)
    checkpoints = []
    for name, prediction in (("baseline", 5), ("candidate", 7)):
        model = ScenicRegressionModel(vit_dim=1, terrain_dim=1, num_classes=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-2].bias.fill_(math.log(prediction / (10 - prediction)))
        checkpoint = tmp_path / f"{name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(), "vit_dim": 1,
            "terrain_dim": 1, "num_classes": 1,
        }, checkpoint)
        checkpoints.append(checkpoint)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "compare", "--dataset", str(dataset), "--labels-csv", str(labels),
        "--benchmark-split-csv", str(split), "--baseline-checkpoint", str(checkpoints[0]),
        "--candidate-checkpoint", str(checkpoints[1]), "--output-json", str(output),
        "--device", "cpu",
    ])
    main()
    summary = json.loads(output.read_text())
    assert summary["baseline"]["samples"] == summary["candidate"]["samples"] == 2
    assert summary["baseline"]["mae"] == pytest.approx(3)
    assert summary["baseline"]["rmse"] == pytest.approx(3)
    assert summary["candidate"]["mae"] == pytest.approx(3)
    assert summary["candidate"]["rmse"] == pytest.approx(math.sqrt(13))


@pytest.mark.parametrize("epochs", ["0", "-1"])
def test_zero_epoch_baseline_rejected_before_saved_claim(tmp_path, monkeypatch, capsys, epochs) -> None:
    from scripts.modeling.train_regression_baseline import main

    output = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(sys, "argv", [
        "train", "--dataset", str(tmp_path / "absent.npz"),
        "--benchmark-split-csv", str(tmp_path / "absent.csv"),
        "--labels-csv", str(tmp_path / "absent-labels.csv"),
        "--classifier-checkpoint", str(tmp_path / "absent-classifier.pt"),
        "--output", str(output), "--epochs", epochs,
    ])
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "--epochs must be >= 1" in captured.err
    assert "Saved checkpoint" not in captured.out
    assert not output.exists()


@pytest.mark.parametrize("code", ["NoSuchKey", "NotFound", "404", "AccessDenied", "NoSuchBucket", "SlowDown"])
def test_s3_export_skip_missing_only_for_missing_objects(code) -> None:
    from botocore.exceptions import ClientError
    from scripts.modeling.export_regression_dataset import _open_image

    error = ClientError({"Error": {"Code": code}}, "GetObject")

    class Client:
        def get_object(self, **kwargs):
            raise error

    if code in {"NoSuchKey", "NotFound", "404"}:
        with pytest.raises(FileNotFoundError, match="s3://bucket/raw/tile.png"):
            _open_image("tile.png", "s3://bucket/raw", Client())
    else:
        with pytest.raises(ClientError) as caught:
            _open_image("tile.png", "s3://bucket/raw", Client())
        assert caught.value is error


def test_s3_export_propagates_network_and_invalid_imagery() -> None:
    from io import BytesIO
    from botocore.exceptions import EndpointConnectionError
    from PIL import UnidentifiedImageError
    from scripts.modeling.export_regression_dataset import _open_image

    class OfflineClient:
        def get_object(self, **kwargs):
            raise EndpointConnectionError(endpoint_url="https://example.invalid")

    class InvalidImageClient:
        def get_object(self, **kwargs):
            return {"Body": BytesIO(b"not an image")}

    with pytest.raises(EndpointConnectionError):
        _open_image("tile.png", "s3://bucket/raw", OfflineClient())
    with pytest.raises(UnidentifiedImageError):
        _open_image("tile.png", "s3://bucket/raw", InvalidImageClient())


def test_regression_export_uses_one_backbone_per_batch(tmp_path, monkeypatch) -> None:
    from PIL import Image
    from scripts.modeling import export_regression_dataset as exporter

    class FixtureBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def forward(self, batch):
            self.batch_sizes.append(len(batch))
            return batch.mean(dim=(2, 3))

    class FixtureClassifier(torch.nn.Module):
        def __init__(self) -> None:
            from scenicdriver.classifier.inference import PreprocessingContract

            super().__init__()
            self.backbone = FixtureBackbone()
            self.preprocessing = PreprocessingContract.for_normalization(
                "resisc45", class_names=("first", "second")
            )

        def forward(self, _batch):
            raise AssertionError("export should use the combined forward")

        def get_features(self, _batch):
            raise AssertionError("export should not run a separate feature forward")

        def forward_with_features(self, batch):
            features = self.backbone(batch)
            return features[:, :2], features

    classifier = FixtureClassifier()
    colors = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    image_paths = [f"images/satellite/{i}.png" for i in range(len(colors))]
    labels = tmp_path / "labels.csv"
    _write_metadata(labels, [
        {"image_path": path, "scenic_score": i + 1, "class_id": i}
        for i, path in enumerate(image_paths)
    ])
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.touch()
    output = tmp_path / "dataset.npz"
    monkeypatch.setattr(exporter, "LandscapeClassifier", lambda **_kwargs: classifier)
    monkeypatch.setattr(
        exporter, "_open_image",
        lambda path, *_args: Image.new("RGB", (4, 4), colors[int(Path(path).stem)]),
    )
    monkeypatch.setattr(
        exporter, "get_inference_transform",
        lambda _preprocessing: lambda image: torch.from_numpy(
            np.array(image, dtype=np.float32)
        ).permute(2, 0, 1),
    )
    monkeypatch.setattr(
        exporter, "_terrain_features_from_images",
        lambda *_args: np.zeros(6, dtype=np.float32),
    )
    monkeypatch.setattr(sys, "argv", [
        "export", "--labels-csv", str(labels), "--classifier-ckpt", str(checkpoint),
        "--output", str(output), "--batch-size", "2", "--device", "cpu",
    ])

    exporter.main()

    assert classifier.backbone.batch_sizes == [2, 1]
    features = torch.tensor(colors, dtype=torch.float32)
    logits = features[:, :2]
    with np.load(output) as dataset:
        np.testing.assert_array_equal(dataset["vit_embeddings"], features.numpy())
        np.testing.assert_array_equal(dataset["class_logits"], logits.numpy())
        np.testing.assert_array_equal(
            dataset["class_probs"], torch.softmax(logits, dim=1).numpy()
        )
        assert dataset["vit_embeddings"].dtype == np.float32
        assert dataset["class_logits"].dtype == np.float32
        assert dataset["class_probs"].dtype == np.float32
        assert dataset["image_paths"].tolist() == image_paths


def test_heatmap_rejects_conflicting_original_and_dense_indices(pool, tmp_path) -> None:
    rows = [dict(row) for row in pool["rows"]]
    rows[0]["embedding_row_index"], rows[1]["embedding_row_index"] = 1, 0
    _write_metadata(pool["meta"], rows)
    with pytest.raises(ValueError, match="does not match feature position"):
        _export(pool, tmp_path / "runs")
    assert not (tmp_path / "runs" / "test_run").exists()


@pytest.mark.parametrize("consumer", ["training", "benchmark"])
@pytest.mark.parametrize("problem", ["missing-identities", "conflicting-split"])
def test_regression_consumers_fail_closed_on_identity_contract(tmp_path, consumer, problem) -> None:
    from scripts.modeling.compare_regression_on_benchmark import benchmark_dataset
    from scripts.modeling.regression_split import holdout_population_mask

    dataset = _regression_export(tmp_path)
    labels, split = _benchmark_files(tmp_path)
    with np.load(dataset, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if problem == "missing-identities":
        del arrays["image_paths"]
    else:
        arrays["splits"] = np.array(["train", "train", "test", "val"])
    np.savez(dataset, **arrays)
    with pytest.raises(ValueError, match="exported image_paths|conflicts"):
        if consumer == "training":
            holdout_population_mask(dataset, labels, split, holdout_buffer_tiles=1)
        else:
            benchmark_dataset(dataset, labels, split)


def test_training_excludes_embedded_heldout_not_in_benchmark(tmp_path) -> None:
    import pandas as pd
    from scripts.modeling.regression_split import holdout_population_mask

    paths = (
        "images/satellite/z14/east/10_10.png",
        "images/satellite/z14/east/30_30.png",  # embedded test row, absent from benchmark
        "images/satellite/z14/east/31_31.png",  # adjacent to the embedded test row
        "images/satellite/z14/east/50_50.png",  # benchmark test tile
    )
    dataset = _regression_export(tmp_path, paths)
    with np.load(dataset, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["splits"] = np.array(["train", "test", "train", "test"])
    np.savez(dataset, **arrays)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"image_path": paths, "scenic_score": [5] * len(paths)}).to_csv(labels, index=False)
    split = tmp_path / "benchmark.csv"
    pd.DataFrame({"image_path": [paths[3]], "split": ["test"]}).to_csv(split, index=False)
    assert holdout_population_mask(dataset, labels, split, holdout_buffer_tiles=1).tolist() == [
        True, False, False, False,
    ]


def test_benchmark_prefers_explicit_benchmark_human_mean(tmp_path) -> None:
    import pandas as pd
    from scripts.modeling.compare_regression_on_benchmark import benchmark_dataset

    dataset = _regression_export(tmp_path)
    labels, split = _benchmark_files(tmp_path)
    frame = pd.read_csv(split)
    frame["scenic_human_mean"] = [4, np.nan, np.nan, 9]
    frame.to_csv(split, index=False)
    selected = benchmark_dataset(dataset, labels, split)
    assert [float(selected[i][3][0]) for i in range(len(selected))] == [4, 9]
