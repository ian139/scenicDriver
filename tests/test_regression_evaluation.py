"""Geographic membership and run provenance for baseline regression.

Training splits exported rows by region-independent physical tiles: benchmark
held-out tiles plus a declared buffer are excluded, physical blocks and
declared geographic groups never straddle train/validation, and no training
tile is adjacent to a validation tile. A run manifest binds every input and
output by content digest; evaluation admits a run only when every supplied file
matches (relocated identical bytes are accepted) and then scores exactly the
rows the trained model validated on.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scenicdriver.classifier.inference import PreprocessingContract
from scenicdriver.classifier.model import TERRAIN_CLASSES, save_classifier_checkpoint
from scenicdriver.scenic_scorer import regression
from scripts.modeling import evaluate_regression_baseline, train_regression_baseline
from scripts.modeling.regression_split import (
    BOUND_FILES,
    GeographicSplitConfig,
    derive_validation_membership,
    holdout_population_mask,
    load_validation_membership,
    physical_tile,
    sha256_file,
)

SPACED = [("alpha", 14, 3 * i, 0) for i in range(40)]
ADJACENT_PAIR = [("alpha", 14, 1000, 50), ("alpha", 14, 1001, 50)]
# Seed 0 at val_split 0.3 with one-tile blocks assigns 1000 to train and 1001 to
# validation, and leaves both splits of SPACED non-empty.
SPLIT = GeographicSplitConfig(val_split=0.3, seed=0, block_size=1, holdout_buffer_tiles=1)


def _tile_path(region: str, z: int, x: int, y: int) -> str:
    return f"images/satellite/z{z}/{region}/{x}_{y}.png"


def _write_inputs(
    root: Path,
    tiles: list[tuple[str, int, int, int]],
    *,
    groups: dict[int, str] | None = None,
    benchmark: dict[str, str] | None = None,
    embedded: list[str] | None = None,
    normalization: str = "resisc45",
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    paths = [_tile_path(*tile) for tile in tiles]
    n = len(paths)
    rng = np.random.default_rng(1234)
    arrays: dict[str, np.ndarray] = {
        "image_paths": np.array(paths, dtype=str),
        "vit_embeddings": rng.uniform(size=(n, 2)).astype(np.float32),
        "terrain_features": rng.uniform(size=(n, 1)).astype(np.float32),
        "class_logits": rng.uniform(size=(n, 3)).astype(np.float32),
        "scenic_scores": rng.uniform(0.0, 10.0, size=(n,)).astype(np.float32),
        "sample_weights": rng.uniform(0.5, 4.0, size=(n,)).astype(np.float32),
    }
    if embedded is not None:
        arrays["splits"] = np.array(embedded, dtype=str)
    np.savez(root / "dataset.npz", **arrays)
    labels = pd.DataFrame(
        {
            "image_path": paths,
            "scenic_score": arrays["scenic_scores"],
            "region": [tile[0] for tile in tiles],
            "z": [tile[1] for tile in tiles],
            "x": [tile[2] for tile in tiles],
            "y": [tile[3] for tile in tiles],
        }
    )
    if groups is not None:
        labels["geographic_group"] = [groups.get(index) for index in range(n)]
    labels.to_csv(root / "labels.csv", index=False)
    rows = benchmark or {_tile_path("alpha", 14, 500, 500): "test"}
    pd.DataFrame({"image_path": list(rows), "split": list(rows.values())}).to_csv(
        root / "benchmark.csv", index=False
    )
    save_classifier_checkpoint(
        root / "classifier.pt",
        model_state_dict={"weight": torch.zeros(1)},
        preprocessing=PreprocessingContract.for_normalization(
            normalization, class_names=TERRAIN_CLASSES
        ),
    )
    return {
        "dataset": root / "dataset.npz",
        "labels_csv": root / "labels.csv",
        "benchmark_split_csv": root / "benchmark.csv",
        "classifier_checkpoint": root / "classifier.pt",
        "arrays": arrays,
        "paths": paths,
    }


def _train(
    monkeypatch: pytest.MonkeyPatch, inputs: dict, root: Path, *extra: str
) -> dict[str, Path]:
    run = {
        "checkpoint": root / "model.pt",
        "split_artifact": root / "model.split.json",
        "run_manifest": root / "model.manifest.json",
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_regression_baseline.py",
            "--dataset", str(inputs["dataset"]),
            "--labels-csv", str(inputs["labels_csv"]),
            "--benchmark-split-csv", str(inputs["benchmark_split_csv"]),
            "--classifier-checkpoint", str(inputs["classifier_checkpoint"]),
            "--output", str(run["checkpoint"]),
            "--split-artifact", str(run["split_artifact"]),
            "--run-manifest", str(run["run_manifest"]),
            "--val-split", str(SPLIT.val_split),
            "--seed", str(SPLIT.seed),
            "--block-size", str(SPLIT.block_size),
            "--holdout-buffer-tiles", str(SPLIT.holdout_buffer_tiles),
            "--epochs", "1",
            "--batch-size", "4",
            "--num-workers", "0",
            "--device", "cpu",
            *extra,
        ],
    )
    train_regression_baseline.main()
    return run


def _admission_files(inputs: dict, run: dict[str, Path]) -> dict[str, Path]:
    return {
        "checkpoint": run["checkpoint"],
        "split_artifact": run["split_artifact"],
        "dataset": inputs["dataset"],
        "labels_csv": inputs["labels_csv"],
        "benchmark_split_csv": inputs["benchmark_split_csv"],
        "classifier_checkpoint": inputs["classifier_checkpoint"],
    }


def _admit(files: dict[str, Path], manifest: Path):
    return load_validation_membership(
        files["dataset"],
        files["split_artifact"],
        run_manifest=manifest,
        checkpoint=files["checkpoint"],
        labels_csv=files["labels_csv"],
        benchmark_split_csv=files["benchmark_split_csv"],
        classifier_checkpoint=files["classifier_checkpoint"],
    )


def _relocate(files: dict[str, Path], manifest: Path, destination: Path) -> tuple[dict[str, Path], Path]:
    destination.mkdir(parents=True)
    moved = {}
    for role, source in files.items():
        moved[role] = destination / f"renamed-{role}{source.suffix}"
        shutil.copyfile(source, moved[role])
    moved_manifest = destination / "renamed-manifest.json"
    shutil.copyfile(manifest, moved_manifest)
    return moved, moved_manifest


def _expected_val_loss(
    arrays: dict[str, np.ndarray], checkpoint: dict, rows: list[int], weighted: bool
) -> float:
    model = regression.ScenicRegressionModel(vit_dim=2, terrain_dim=1, num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        pred = model(
            torch.from_numpy(arrays["vit_embeddings"][rows]),
            torch.from_numpy(arrays["terrain_features"][rows]),
            torch.from_numpy(arrays["class_logits"][rows]),
        ).numpy().reshape(-1)
    error = (pred - arrays["scenic_scores"][rows]) ** 2
    weights = arrays["sample_weights"][rows] if weighted else np.ones_like(error)
    return float((error * weights).sum() / weights.sum())


def test_physical_tile_ignores_region_and_rejects_disagreement() -> None:
    assert physical_tile({"image_path": _tile_path("alpha", 14, 5, 6)}) == (14, 5, 6)
    assert physical_tile({"image_path": _tile_path("beta", 14, 5, 6)}) == (14, 5, 6)
    assert physical_tile(
        {"image_path": _tile_path("beta", 14, 5, 6), "z": 14, "x": 5.0, "y": 6,
         "tile_identity": "gamma/z14/x5/y6"}
    ) == (14, 5, 6)
    assert physical_tile({"image_path": "img_00"}) is None
    with pytest.raises(ValueError, match="disagrees"):
        physical_tile({"image_path": _tile_path("alpha", 14, 5, 6), "z": 14, "x": 5, "y": 7})
    with pytest.raises(ValueError, match="partial"):
        physical_tile({"image_path": "img_00", "z": 14, "x": 5})


def test_holdout_buffer_excludes_aliases_adjacent_cross_zoom_and_embedded(tmp_path: Path) -> None:
    tiles = [
        ("beta", 14, 100, 100),  # region alias of the benchmark test tile
        ("alpha", 14, 101, 101),  # adjacent to it
        ("alpha", 14, 102, 100),  # two tiles away
        ("alpha", 16, 401, 402),  # finer tile inside it
        ("alpha", 16, 412, 400),  # finer tile three z14 tiles away
        ("alpha", 14, 300, 300),  # embedded held-out export row
        ("alpha", 14, 301, 300),  # adjacent to the embedded held-out row
        ("alpha", 14, 50, 50),  # benchmark train row
    ]
    embedded = ["train"] * len(tiles)
    embedded[5] = "val"
    benchmark = {
        _tile_path("alpha", 14, 100, 100): "test",
        _tile_path(*tiles[7]): "train",
    }
    inputs = _write_inputs(tmp_path, tiles, benchmark=benchmark, embedded=embedded)
    args = (inputs["dataset"], inputs["labels_csv"], inputs["benchmark_split_csv"])

    assert holdout_population_mask(*args, holdout_buffer_tiles=1).tolist() == [
        False, False, True, False, True, False, False, True,
    ]
    assert holdout_population_mask(*args, holdout_buffer_tiles=2).tolist() == [
        False, False, False, False, True, False, False, True,
    ]
    with pytest.raises(ValueError, match="at least one tile"):
        holdout_population_mask(*args, holdout_buffer_tiles=0)


def test_holdout_requires_physical_identity(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, SPACED[:4])
    pd.DataFrame({"image_path": ["unlocated"], "split": ["test"]}).to_csv(
        inputs["benchmark_split_csv"], index=False
    )
    with pytest.raises(ValueError, match="no physical z/x/y tile"):
        holdout_population_mask(
            inputs["dataset"], inputs["labels_csv"], inputs["benchmark_split_csv"],
            holdout_buffer_tiles=1,
        )


def test_geographic_membership_protects_blocks_groups_and_adjacency(tmp_path: Path) -> None:
    tiles = [
        *SPACED,
        *ADJACENT_PAIR,
        ("alpha", 14, 2000, 0),
        ("alpha", 14, 2010, 0),
        ("alpha", 14, 3000, 0),
        ("beta", 14, 3000, 0),
    ]
    groups = {len(SPACED) + 2: "g1", len(SPACED) + 3: "g1"}
    inputs = _write_inputs(tmp_path, tiles, groups=groups)
    paths = inputs["paths"]

    membership = derive_validation_membership(
        inputs["dataset"], inputs["labels_csv"], inputs["benchmark_split_csv"], SPLIT
    )

    train, val = set(membership.train_paths), set(membership.val_paths)
    assert train and val and not (train & val)
    assert train | val | set(membership.boundary_excluded_paths) == set(paths)
    assert membership.holdout_excluded_paths == ()
    for group in ((paths[-4], paths[-3]), (paths[-2], paths[-1])):
        assert set(group) <= train or set(group) <= val
    assert membership.boundary_excluded_paths == (_tile_path(*ADJACENT_PAIR[0]),)
    assert _tile_path(*ADJACENT_PAIR[1]) in val
    tile_of = dict(zip(paths, tiles, strict=True))
    for train_path in train:
        _, _, tx, ty = tile_of[train_path]
        for val_path in val:
            _, _, vx, vy = tile_of[val_path]
            assert max(abs(tx - vx), abs(ty - vy)) > 1
    population = np.asarray(paths)[membership.population_indices]
    assert population[list(membership.val_indices)].tolist() == list(membership.val_paths)
    assert population[list(membership.train_indices)].tolist() == list(membership.train_paths)


def test_geographic_membership_is_stable_when_unrelated_rows_arrive(tmp_path: Path) -> None:
    base = _write_inputs(tmp_path / "base", SPACED)
    grown = _write_inputs(tmp_path / "grown", [*SPACED, ("alpha", 14, 9000, 9000)])
    first = derive_validation_membership(
        base["dataset"], base["labels_csv"], base["benchmark_split_csv"], SPLIT
    )
    second = derive_validation_membership(
        grown["dataset"], grown["labels_csv"], grown["benchmark_split_csv"], SPLIT
    )
    assert set(first.val_paths) <= set(second.val_paths)
    assert set(first.train_paths) <= set(second.train_paths)


def test_trainer_uses_explicit_validation_membership(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, SPACED[:20])
    arrays = inputs["arrays"]
    train_idx = list(range(15))
    val_idx = [15, 16, 17, 18, 19]
    output = tmp_path / "model.pt"

    regression.train_regression_model(
        inputs["dataset"],
        output,
        epochs=1,
        batch_size=2,
        device="cpu",
        num_workers=0,
        train_indices=train_idx,
        val_indices=val_idx,
    )

    checkpoint = torch.load(output, weights_only=False)
    expected = _expected_val_loss(arrays, checkpoint, val_idx, weighted=True)
    assert checkpoint["best_val_loss"] == pytest.approx(expected, rel=1e-5)


def test_trainer_rejects_partial_explicit_membership(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, SPACED[:20])

    with pytest.raises(ValueError, match="together"):
        regression.train_regression_model(
            inputs["dataset"],
            tmp_path / "model.pt",
            epochs=1,
            device="cpu",
            num_workers=0,
            train_indices=list(range(15)),
        )


def test_train_and_evaluate_end_to_end_bind_and_admit_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", [*SPACED, *ADJACENT_PAIR])
    run = _train(monkeypatch, inputs, tmp_path / "run", "--normalization", "resisc45")
    files = _admission_files(inputs, run)

    manifest = json.loads(run["run_manifest"].read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(BOUND_FILES)
    for role, path in files.items():
        assert manifest["artifacts"][role]["sha256"] == sha256_file(path)
    assert manifest["preprocessing"]["classifier"]["normalization"] == "resisc45"
    split = json.loads(run["split_artifact"].read_text(encoding="utf-8"))
    assert split["inputs"]["dataset_sha256"] == sha256_file(inputs["dataset"])
    assert split["boundary_excluded_image_paths"] == [_tile_path(*ADJACENT_PAIR[0])]
    val_paths = split["val_image_paths"]
    assert str(tmp_path) not in run["split_artifact"].read_text(encoding="utf-8")

    metrics_json = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_regression_baseline.py",
            "--dataset", str(files["dataset"]),
            "--checkpoint", str(files["checkpoint"]),
            "--split-artifact", str(files["split_artifact"]),
            "--run-manifest", str(run["run_manifest"]),
            "--labels-csv", str(files["labels_csv"]),
            "--benchmark-split-csv", str(files["benchmark_split_csv"]),
            "--classifier-checkpoint", str(files["classifier_checkpoint"]),
            "--batch-size", "4",
            "--device", "cpu",
            "--metrics-json", str(metrics_json),
        ],
    )
    evaluate_regression_baseline.main()

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert metrics["samples"] == len(val_paths)
    assert metrics["run_manifest"] == str(run["run_manifest"])
    arrays = inputs["arrays"]
    row_of = {path: i for i, path in enumerate(arrays["image_paths"].tolist())}
    rows = [row_of[path] for path in val_paths]
    ckpt = torch.load(files["checkpoint"], weights_only=False)
    model = regression.ScenicRegressionModel(vit_dim=2, terrain_dim=1, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        pred = model(
            torch.from_numpy(arrays["vit_embeddings"][rows]),
            torch.from_numpy(arrays["terrain_features"][rows]),
            torch.from_numpy(arrays["class_logits"][rows]),
        ).numpy().reshape(-1).astype(np.float64)
    target = arrays["scenic_scores"][rows].astype(np.float64)
    assert metrics["mae"] == pytest.approx(float(np.mean(np.abs(pred - target))), rel=1e-4)
    assert metrics["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((pred - target) ** 2))), rel=1e-4
    )


def test_admission_accepts_relocated_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", [*SPACED, *ADJACENT_PAIR])
    run = _train(monkeypatch, inputs, tmp_path / "run")
    original = _admit(_admission_files(inputs, run), run["run_manifest"])

    moved, manifest = _relocate(
        _admission_files(inputs, run), run["run_manifest"], tmp_path / "elsewhere"
    )
    shutil.rmtree(tmp_path / "inputs")
    shutil.rmtree(tmp_path / "run")
    relocated = _admit(moved, manifest)

    assert relocated.val_paths == original.val_paths
    assert relocated.train_paths == original.train_paths
    assert relocated.val_dataset_indices.tolist() == original.val_dataset_indices.tolist()


@pytest.mark.parametrize("role", BOUND_FILES)
def test_admission_rejects_tampered_bound_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", [*SPACED, *ADJACENT_PAIR])
    run = _train(monkeypatch, inputs, tmp_path / "run")
    files, manifest = _relocate(
        _admission_files(inputs, run), run["run_manifest"], tmp_path / "copy"
    )
    with files[role].open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match=f"{role} content digest mismatch"):
        _admit(files, manifest)


def test_admission_rejects_substituted_valid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", [*SPACED, *ADJACENT_PAIR])
    run = _train(monkeypatch, inputs, tmp_path / "run")
    other = _write_inputs(
        tmp_path / "other", [*SPACED, *ADJACENT_PAIR], normalization="imagenet"
    )
    for role in ("labels_csv", "classifier_checkpoint"):
        files = _admission_files(inputs, run)
        files[role] = other[role]
        if role == "labels_csv":
            frame = pd.read_csv(other[role])
            frame.loc[0, "scenic_score"] = 1.0
            frame.to_csv(other[role], index=False)
        with pytest.raises(ValueError, match=f"{role} content digest mismatch"):
            _admit(files, run["run_manifest"])


def test_admission_rejects_manifest_drift_and_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", [*SPACED, *ADJACENT_PAIR])
    run = _train(monkeypatch, inputs, tmp_path / "run")
    files = _admission_files(inputs, run)
    manifest = json.loads(run["run_manifest"].read_text(encoding="utf-8"))

    drifted = json.loads(json.dumps(manifest))
    drifted["preprocessing"]["classifier"]["normalization"] = "imagenet"
    drifted_path = tmp_path / "preprocessing.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing identity"):
        _admit(files, drifted_path)

    drifted = json.loads(json.dumps(manifest))
    drifted["split_rule"]["seed"] = 1
    drifted_path = tmp_path / "rule.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="rule does not match"):
        _admit(files, drifted_path)

    drifted = json.loads(json.dumps(manifest))
    del drifted["artifacts"]["classifier_checkpoint"]
    drifted_path = tmp_path / "unbound.json"
    drifted_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="must bind exactly"):
        _admit(files, drifted_path)

    with pytest.raises(FileNotFoundError):
        _admit(files, tmp_path / "missing.json")
    missing = dict(files, split_artifact=tmp_path / "missing.split.json")
    with pytest.raises(FileNotFoundError):
        _admit(missing, run["run_manifest"])


def test_trainer_rejects_conflicting_normalization_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path / "inputs", [*SPACED, *ADJACENT_PAIR])
    with pytest.raises(ValueError, match="conflicts"):
        _train(monkeypatch, inputs, tmp_path / "run", "--normalization", "imagenet")
    assert not (tmp_path / "run" / "model.pt").exists()
    assert not (tmp_path / "run" / "model.manifest.json").exists()


def test_evaluation_cli_requires_run_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_regression_baseline.py",
            "--dataset", "dataset.npz",
            "--checkpoint", "checkpoint.pt",
            "--split-artifact", "checkpoint.split.json",
        ],
    )
    with pytest.raises(SystemExit):
        evaluate_regression_baseline.parse_args()
