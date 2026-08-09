"""Focused tests for the Stage Two preflight validator.

Covers the unchanged handoff-only contract (byte-compatible metrics), the
supplemental benchmark success path, and fail-closed supplemental failures:
hash, split, target, completeness, duplicate, adjacency, control overlap,
malformed decisions, and split support.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.modeling.validate_stage2_preflight import (
    REQUIRED_READINESS,
    main,
    validate_handoff,
    validate_supplemental,
)

REAL_RUN = Path(
    "data/processed/active_learning/run_v2_annotation_expansion_20260807"
)
REAL_HANDOFF = Path(
    "data/processed/active_learning/run_v1_expanded_20260805/stage1_handoff.json"
)
REAL_ANNOTATIONS_SHA256 = (
    "adb25c1dc50c7b4bf1a6743a233ba3681511fed31da26f3c33334199d5cd0b1d"
)
REAL_BENCHMARK_SHA256 = (
    "b157013d15a95c0d57c3baa60b5ab17a721ecbbdc05ed944610b74a66fed795b"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record(path: Path, relative: str, required: bool = True) -> dict:
    return {
        "bytes": path.stat().st_size,
        "path": relative,
        "required": required,
        "sha256": _sha256_file(path),
    }


def _write_handoff(
    root: Path,
    supplemental_split_rows: list[dict] | None = None,
    control_extra_identities: list[str] | None = None,
) -> Path:
    """Write a fully self-consistent synthetic stage-one handoff.

    The expanded benchmark is 24 rows (19 train / 1 val / 4 test) to satisfy
    the validator's hard-coded expanded-split contract; every other declared
    count matches the artifact it names.
    """
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)

    expanded = []
    split_rows = []
    for index in range(24):
        split = "train" if index < 19 else ("val" if index == 19 else "test")
        identity = f"images/satellite/z14/expanded/exp{index:03d}.png"
        expanded.append({"image_path": identity, "split": split})
        split_rows.append(
            {"image_path": identity, "split": split, "z": 14, "x": 2000 + index, "y": 5000}
        )
    for index, split in enumerate(
        ["train", "val", "test", "train", "train", "val"]
    ):
        identity = f"images/satellite/z14/extra/extra{index:03d}.png"
        split_rows.append(
            {"image_path": identity, "split": split, "z": 14, "x": 3000 + index, "y": 5000}
        )
    if supplemental_split_rows:
        split_rows.extend(supplemental_split_rows)

    split_counts: dict[str, int] = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    control_rows = [
        {
            "image_path": f"images/satellite/z14/control/ctl{index:03d}.png",
            "split": split,
        }
        for index, split in enumerate(
            ["train", "train", "val", "test", "test", "test"]
        )
    ]
    for identity in control_extra_identities or []:
        control_rows.append({"image_path": identity, "split": "test"})

    generic = [
        {"row_id": index, "value": float(index)}
        for index in range(5)
    ]
    annotation_rows = [
        {
            "image_path": f"images/satellite/z14/expanded/exp{index:03d}.png",
            "scenic_human": str(score),
            "confidence": "medium",
            "skip": "False",
            "annotator_id": "ian",
            "timestamp": "2026-08-09T00:00:00+00:00",
            "notes": "",
        }
        for index, score in enumerate([5.0, 6.0, 4.0])
    ]

    (run / "annotation_batch.csv").write_text(
        "row_id,value\n" + "".join(f"{i},{float(i)}\n" for i in range(4)),
        encoding="utf-8",
    )
    _write_csv(run / "candidate_pool.csv", generic)
    _write_csv(run / "mixed_labels.csv", generic)
    _write_csv(run / "tile_manifest.csv", generic)
    _write_csv(run / "absolute_annotations.csv", annotation_rows)
    _write_csv(run / "benchmark.csv", expanded)
    _write_csv(run / "control_benchmark.csv", control_rows)
    _write_csv(run / "geographic_splits.csv", split_rows)

    leakage_audit = {
        "schema_version": 1,
        "valid": True,
        "checked_rows": len(split_rows),
        "duplicate_cross_split": False,
        "adjacent_cross_split": False,
        "violation_count": 0,
        "violations": [],
        "split_counts": split_counts,
    }
    (run / "leakage_audit.json").write_text(
        json.dumps(leakage_audit), encoding="utf-8"
    )
    active = {
        "checkpoint": "models/synthetic.pt",
        "metrics": {"corr": 0.5, "mae": 0.1, "rmse": 0.3, "samples": 10},
        "run_name": "synthetic",
        "source_metrics": "data/synthetic.json",
        "updated_at": "2026-08-09T00:00:00+00:00",
    }
    (run / "baseline_registry.json").write_text(
        json.dumps({"active": active}), encoding="utf-8"
    )
    np.savez(run / "feature_embeddings.npz", embeddings=np.zeros((5, 4), dtype=np.float32))

    file_artifacts = {
        "absolute_annotations": "absolute_annotations.csv",
        "annotation_batch": "annotation_batch.csv",
        "baseline_registry": "baseline_registry.json",
        "benchmark": "benchmark.csv",
        "candidate_pool": "candidate_pool.csv",
        "control_benchmark": "control_benchmark.csv",
        "feature_embeddings": "feature_embeddings.npz",
        "geographic_splits": "geographic_splits.csv",
        "leakage_audit": "leakage_audit.json",
        "mixed_labels": "mixed_labels.csv",
        "tile_manifest": "tile_manifest.csv",
    }
    artifacts = {
        name: _record(run / relative, relative)
        for name, relative in file_artifacts.items()
    }
    artifacts["baseline_checkpoint"] = {
        "bytes": 1,
        "path": "missing_checkpoint.pt",
        "required": False,
        "sha256": "0" * 64,
    }
    artifact_hashes = {name: record["sha256"] for name, record in artifacts.items()}

    counts = {
        "annotation_rows": len(annotation_rows),
        "batch_rows": 4,
        "benchmark_rows": len(expanded),
        "candidate_pool_rows": len(generic),
        "control_benchmark_rows": len(control_rows),
        "embedding_rows": 5,
        "mixed_label_rows": len(generic),
        "split_rows": len(split_rows),
        "tile_rows": len(generic),
    }
    handoff = {
        "schema_version": 1,
        "run_name": "synthetic_run",
        "run_root": str(run),
        "ready_for_stage2": True,
        "blockers": [],
        "incomplete_work": [],
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "counts": counts,
        "leakage_audit": leakage_audit,
        "baseline": {
            "active": active,
            "checkpoint_path": "models/synthetic.pt",
            "checkpoint_sha256": "0" * 64,
            "registry_path": "data/synthetic_registry.json",
            "registry_sha256": artifact_hashes["baseline_registry"],
        },
        "material_config": {
            "split_strategy": "fixed geographic blocks with adjacency_radius=1"
        },
        "readiness": {key: True for key in REQUIRED_READINESS},
    }
    for key in REQUIRED_READINESS:
        handoff[key] = True
    handoff_path = run / "stage1_handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    return handoff_path


def _success_tiles() -> list[dict]:
    specs = [
        ("a", 4.0, False, "train", 1000, True),
        ("b", 5.0, False, "train", 1010, True),
        ("c", 6.0, False, "train", 1020, True),
        ("d", 7.0, False, "val", 1030, True),
        ("e", 3.0, False, "val", 1040, True),
        ("f", 5.5, False, "test", 1050, True),
        ("g", 2.0, False, "test", 1060, True),
        ("h", 8.0, False, "test", 1070, True),
        ("i", 4.0, False, None, 1080, False),  # non-skipped, no split membership
        ("j", 6.0, False, None, 1090, False),  # non-skipped, no split membership
        ("k", None, True, None, 1100, False),  # skipped
        ("l", None, True, None, 1110, False),  # skipped
    ]
    return [
        {
            "id": f"images/satellite/z14/sup/{name}.png",
            "target": target,
            "skip": skip,
            "split": split,
            "x": x,
            "y": 5000,
            "in_benchmark": in_benchmark,
        }
        for name, target, skip, split, x, in_benchmark in specs
    ]


def _supplemental_split_rows(tiles: list[dict]) -> list[dict]:
    return [
        {
            "image_path": tile["id"],
            "split": tile["split"],
            "z": 14,
            "x": tile.get("x", 1000),
            "y": tile.get("y", 5000),
        }
        for tile in tiles
        if tile.get("split")
    ]


def _write_supplemental(root: Path, tiles: list[dict]) -> tuple[Path, Path]:
    """Write supplemental annotations and benchmark CSVs from tile specs.

    Tile keys: id, target (float), skip (bool), split (assignment or None),
    x, y, in_benchmark (bool), plus overrides: scenic_human_override,
    skip_value, benchmark_target, benchmark_split, benchmark_only.
    """
    ann_rows: list[dict] = []
    bench_rows: list[dict] = []
    for tile in tiles:
        identity = tile["id"]
        skip = tile.get("skip", False)
        if not tile.get("benchmark_only"):
            ann_rows.append(
                {
                    "image_path": identity,
                    "scenic_human": tile.get(
                        "scenic_human_override", "" if skip else str(tile["target"])
                    ),
                    "confidence": "medium",
                    "skip": str(tile.get("skip_value", "True" if skip else "False")),
                    "annotator_id": "ian",
                    "timestamp": "2026-08-09T00:00:00+00:00",
                    "notes": "[unusable: other]" if skip else "",
                }
            )
        if tile.get("in_benchmark"):
            target = tile.get("benchmark_target", tile["target"])
            bench_rows.append(
                {
                    "image_path": identity,
                    "scenic_human_mean": str(target),
                    "scenic_human_median": str(target),
                    "scenic_human_std": "0.0",
                    "scenic_human_min": str(target),
                    "scenic_human_max": str(target),
                    "n_annotations": "1",
                    "n_annotators": "1",
                    "annotator_variance": "0.0",
                    "annotator_range": "0.0",
                    "class_id": "1",
                    "scenic_score_heuristic": "1.0",
                    "label_source": "active_regression_prediction",
                    "split": tile.get("benchmark_split", tile["split"]),
                    "geographic_block": f"block|{tile.get('x', 0)}",
                    "split_seed": "0",
                }
            )
    ann_path = root / "supplemental_annotations.csv"
    bench_path = root / "supplemental_benchmark.csv"
    _write_csv(ann_path, ann_rows)
    _write_csv(bench_path, bench_rows)
    return ann_path, bench_path


def _validate_supplemental(
    tmp_path: Path,
    tiles: list[dict],
    control_extra: list[str] | None = None,
    control_path: Path | None = None,
    ann_sha: str | None = None,
    bench_sha: str | None = None,
) -> dict[str, int]:
    handoff = _write_handoff(
        tmp_path,
        supplemental_split_rows=_supplemental_split_rows(tiles),
        control_extra_identities=control_extra,
    )
    ann, bench = _write_supplemental(tmp_path, tiles)
    return validate_supplemental(
        handoff,
        ann,
        bench,
        ann_sha if ann_sha is not None else _sha256_file(ann),
        bench_sha if bench_sha is not None else _sha256_file(bench),
        control_path,
    )


# ---------------------------------------------------------------- handoff only


def test_handoff_only_metrics_byte_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    handoff = _write_handoff(tmp_path)
    metrics = validate_handoff(handoff)
    assert metrics == {
        "artifacts": 11,
        "candidate_rows": 5,
        "split_rows": 30,
        "expanded_rows": 24,
        "control_rows": 6,
        "leakage_violations": 0,
    }
    monkeypatch.setattr(
        sys, "argv", ["validate_stage2_preflight.py", "--handoff", str(handoff)]
    )
    main()
    assert capsys.readouterr().out.splitlines() == [
        "METRIC handoff_ready=1",
        "METRIC artifacts=11",
        "METRIC candidate_rows=5",
        "METRIC split_rows=30",
        "METRIC expanded_rows=24",
        "METRIC control_rows=6",
        "METRIC leakage_violations=0",
    ]


def test_supplemental_requires_all_paired_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff = _write_handoff(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_stage2_preflight.py",
            "--handoff",
            str(handoff),
            "--supplemental-annotations",
            "unused.csv",
            "--supplemental-annotations-sha256",
            "0" * 64,
            "--supplemental-benchmark-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2


# --------------------------------------------------------------- supplemental


def test_supplemental_success_metrics(tmp_path: Path) -> None:
    metrics = _validate_supplemental(tmp_path, _success_tiles())
    assert metrics == {
        "supplemental_benchmark_valid": 1,
        "supplemental_rows": 8,
        "supplemental_val_rows": 2,
        "supplemental_test_rows": 3,
        "supplemental_skipped_rows": 2,
    }


def test_supplemental_cli_prints_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    tiles = _success_tiles()
    handoff = _write_handoff(tmp_path, supplemental_split_rows=_supplemental_split_rows(tiles))
    ann, bench = _write_supplemental(tmp_path, tiles)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_stage2_preflight.py",
            "--handoff",
            str(handoff),
            "--supplemental-annotations",
            str(ann),
            "--supplemental-annotations-sha256",
            _sha256_file(ann),
            "--supplemental-benchmark",
            str(bench),
            "--supplemental-benchmark-sha256",
            _sha256_file(bench),
        ],
    )
    main()
    lines = capsys.readouterr().out.splitlines()
    assert "METRIC handoff_ready=1" in lines
    assert "METRIC supplemental_benchmark_valid=1" in lines
    assert "METRIC supplemental_rows=8" in lines
    assert "METRIC supplemental_val_rows=2" in lines
    assert "METRIC supplemental_test_rows=3" in lines
    assert "METRIC supplemental_skipped_rows=2" in lines


def test_supplemental_skip_with_vestigial_score_allowed(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles[10]["scenic_human_override"] = "5.0"  # skipped row still carries a score
    metrics = _validate_supplemental(tmp_path, tiles)
    assert metrics["supplemental_skipped_rows"] == 2


# ------------------------------------------------------------------ failures


def test_supplemental_annotations_hash_mismatch(tmp_path: Path) -> None:
    tiles = _success_tiles()
    handoff = _write_handoff(tmp_path, supplemental_split_rows=_supplemental_split_rows(tiles))
    ann, bench = _write_supplemental(tmp_path, tiles)
    with pytest.raises(ValueError, match="supplemental annotations SHA-256 mismatch"):
        validate_supplemental(handoff, ann, bench, "0" * 64, _sha256_file(bench))


def test_supplemental_benchmark_hash_mismatch(tmp_path: Path) -> None:
    tiles = _success_tiles()
    handoff = _write_handoff(tmp_path, supplemental_split_rows=_supplemental_split_rows(tiles))
    ann, bench = _write_supplemental(tmp_path, tiles)
    with pytest.raises(ValueError, match="supplemental benchmark SHA-256 mismatch"):
        validate_supplemental(handoff, ann, bench, _sha256_file(ann), "0" * 64)


def test_supplemental_invalid_sha256_format(tmp_path: Path) -> None:
    tiles = _success_tiles()
    with pytest.raises(ValueError, match="64-character hex SHA-256"):
        _validate_supplemental(tmp_path, tiles, ann_sha="not-a-digest")


def test_supplemental_split_mismatch(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles[5]["benchmark_split"] = "val"  # assigned test, benchmark claims val
    with pytest.raises(ValueError, match="benchmark split mismatch"):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_target_mismatch(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles[0]["benchmark_target"] = 4.5  # annotation 4.0, benchmark mean 4.5
    with pytest.raises(ValueError, match="benchmark target mismatch"):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_missing_benchmark_row(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles[2]["in_benchmark"] = False  # assigned and scored, absent from benchmark
    with pytest.raises(ValueError, match="benchmark missing annotated tile"):
        _validate_supplemental(tmp_path, tiles)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unknown", "no valid non-skipped annotation"),
        ("skipped", "no valid non-skipped annotation"),
        ("unassigned", "no valid non-skipped annotation"),
    ],
)
def test_supplemental_extra_benchmark_row(
    tmp_path: Path, mutation: str, match: str
) -> None:
    tiles = _success_tiles()
    if mutation == "unknown":
        tiles.append(
            {
                "id": "images/satellite/z14/sup/unknown.png",
                "target": 5.0,
                "skip": False,
                "split": "train",
                "x": 1200,
                "y": 5000,
                "in_benchmark": True,
                "benchmark_only": True,
            }
        )
    elif mutation == "skipped":
        tiles[10]["in_benchmark"] = True
        tiles[10]["benchmark_target"] = 5.0
        tiles[10]["benchmark_split"] = "test"
    else:
        tiles[8]["in_benchmark"] = True
        tiles[8]["benchmark_split"] = "train"
    with pytest.raises(ValueError, match=match):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_duplicate_annotation_identity(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles.append(dict(tiles[0]))
    with pytest.raises(ValueError, match="duplicate identity"):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_duplicate_benchmark_identity(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles.append(
        {
            "id": tiles[0]["id"],
            "target": tiles[0]["target"],
            "skip": False,
            "split": tiles[0]["split"],
            "x": tiles[0]["x"],
            "y": 5000,
            "in_benchmark": True,
            "benchmark_only": True,
        }
    )
    with pytest.raises(ValueError, match="duplicate image_path"):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_duplicate_split_assignment(tmp_path: Path) -> None:
    tiles = _success_tiles()
    handoff = _write_handoff(
        tmp_path, supplemental_split_rows=_supplemental_split_rows(tiles) * 2
    )
    ann, bench = _write_supplemental(tmp_path, tiles)
    with pytest.raises(ValueError, match="assigns identity to multiple rows"):
        validate_supplemental(handoff, ann, bench, _sha256_file(ann), _sha256_file(bench))


def test_supplemental_adjacency_violation(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles[5]["x"] = 1001  # test tile now adjacent to train tile at x=1000
    with pytest.raises(ValueError, match="adjacent tiles assigned to different splits"):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_control_test_overlap(tmp_path: Path) -> None:
    tiles = _success_tiles()
    with pytest.raises(ValueError, match="overlap control benchmark"):
        _validate_supplemental(
            tmp_path, tiles, control_extra=["images/satellite/z14/sup/f.png"]
        )


def test_supplemental_explicit_control_hash_mismatch(tmp_path: Path) -> None:
    tiles = _success_tiles()
    handoff = _write_handoff(tmp_path, supplemental_split_rows=_supplemental_split_rows(tiles))
    ann, bench = _write_supplemental(tmp_path, tiles)
    tampered = tmp_path / "tampered_control.csv"
    tampered.write_text(
        "image_path,split\nimages/satellite/z14/control/ctl000.png,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="control benchmark SHA-256 mismatch"):
        validate_supplemental(
            handoff, ann, bench, _sha256_file(ann), _sha256_file(bench), tampered
        )


def test_supplemental_missing_val_support(tmp_path: Path) -> None:
    tiles = _success_tiles()
    for tile in tiles:
        if tile["split"] == "val":
            tile["split"] = None
            tile["in_benchmark"] = False
    with pytest.raises(ValueError, match="lacks train/val/test support"):
        _validate_supplemental(tmp_path, tiles)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"scenic_human_override": ""}, "neither scored nor skipped"),
        ({"scenic_human_override": "NaN"}, "annotation score out of range"),
        ({"scenic_human_override": "11.0"}, "annotation score out of range"),
        ({"skip_value": "maybe"}, "malformed skip decision"),
    ],
)
def test_supplemental_malformed_decisions(
    tmp_path: Path, override: dict, match: str
) -> None:
    tiles = _success_tiles()
    tiles[0].update(override)
    with pytest.raises(ValueError, match=match):
        _validate_supplemental(tmp_path, tiles)


def test_supplemental_benchmark_target_out_of_range(tmp_path: Path) -> None:
    tiles = _success_tiles()
    tiles[5]["benchmark_target"] = 10.5
    with pytest.raises(ValueError, match="benchmark target out of range"):
        _validate_supplemental(tmp_path, tiles)


# ------------------------------------------------------------ real fixture


def test_supplemental_real_fixture_contract() -> None:
    annotations = REAL_RUN / "annotations.csv"
    benchmark = REAL_RUN / "benchmark_split.csv"
    if not (
        REAL_HANDOFF.is_file() and annotations.is_file() and benchmark.is_file()
    ):
        pytest.skip("real supplemental fixtures not present")
    metrics = validate_supplemental(
        REAL_HANDOFF,
        annotations,
        benchmark,
        REAL_ANNOTATIONS_SHA256,
        REAL_BENCHMARK_SHA256,
    )
    assert metrics == {
        "supplemental_benchmark_valid": 1,
        "supplemental_rows": 457,
        "supplemental_val_rows": 64,
        "supplemental_test_rows": 68,
        "supplemental_skipped_rows": 8,
    }
