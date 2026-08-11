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
    from src.scenic_scorer.regression import ScenicRegressionModel

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
    """Synthetic feature pool with shuffled metadata order.

    The metadata CSV rows are emitted in a shuffled embedding_row_index order
    while the NPZ row_indices stay contiguous 0..n-1, so any positional
    (non-index) row alignment fails the strict-alignment test.
    """
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

    shuffled = list(range(N_TILES))
    rng.shuffle(shuffled)
    rows: list[dict[str, object]] = []
    for idx in shuffled:
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
    """Predictions must follow embedding_row_index, not CSV row position."""
    from src.scenic_scorer.active_evaluation import (
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
    rows = [dict(r) for r in pool["rows"]][1:]  # drop embedding_row_index 0
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
    from src.heuristics.report import union_reports

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
