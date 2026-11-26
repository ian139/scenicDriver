from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scenicdriver.active_learning.common import sha256_file
from scenicdriver.active_learning.finalize import finalize_stage1
from scenicdriver.active_learning.scoring import (
    DEFAULT_SCORING_BATCH_SIZE,
    DEFAULT_SCORING_NUM_WORKERS,
    SCORING_PREFETCH_FACTOR,
    ScoringDependencies,
    _call_classifier,
    _loader_options,
    normalized_class_entropy,
    resolve_active_regression_checkpoint,
    run_active_learning_scoring,
    score_tile_manifest,
)

_WORKER_LOG_ENV = "SCENIC_SCORING_TEST_WORKER_LOG"
_PRODUCER_THREAD_NAME = "scoring-source-producer"


def _log_worker_event(kind: str) -> None:
    """Record which process/DataLoader worker ran a preprocessing step."""
    path = os.environ.get(_WORKER_LOG_ENV)
    if not path:
        return
    from torch.utils.data import get_worker_info

    info = get_worker_info()
    worker = "main" if info is None else f"{info.id} {info.num_workers}"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{kind} {os.getpid()} {worker}\n")


def _worker_transform(image: Image.Image) -> np.ndarray:
    """Top-level, picklable classifier transform for spawn workers."""
    _log_worker_event("transform")
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _worker_terrain(
    _terrain: Image.Image, _satellite: Image.Image, *, valid_mask: np.ndarray,
    pixel_spacing_m: float,
) -> dict[str, object]:
    """Top-level, picklable terrain feature function for spawn workers."""
    _log_worker_event("terrain")
    return {
        "features": np.asarray([0.2, 0.3, 0.4, 0.5, 0.0, 0.0], dtype=np.float32),
        "relief": 100.0,
        "roughness": 20.0,
        "slope_mean": 4.0,
    }


def _png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 4), color=color).save(path, format="PNG")


def _manifest(root: Path) -> pd.DataFrame:
    rows = []
    for index, color in enumerate(((40, 120, 70), (120, 70, 90), (160, 130, 80))):
        satellite = root / f"sat-{index}.png"
        terrain = root / f"terrain-{index}.png"
        _png(satellite, color)
        _png(terrain, (120, 120, 120))
        rows.append(
            {
                "image_path": f"images/satellite/z14/fixture/{100 + index}_200.png",
                "provider": "mapbox_v1",
                "region": "fixture",
                "z": 14,
                "x": 100 + index,
                "y": 200,
                "lat": 40.0 + index,
                "lon": -70.0,
                "satellite_path": satellite.name,
                "terrain_path": terrain.name,
                "satellite_present": True,
                "terrain_present": True,
            }
        )
    rows[-1]["satellite_present"] = False
    return pd.DataFrame(rows)


def _dependencies(counters: dict[str, int]) -> ScoringDependencies:
    def transform(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    def terrain(
        _terrain: Image.Image, _satellite: Image.Image, *, valid_mask: np.ndarray,
        pixel_spacing_m: float,
    ) -> dict[str, object]:
        return {
            "features": np.asarray([0.2, 0.3, 0.4, 0.5, 0.0, 0.0], dtype=np.float32),
            "relief": 100.0,
            "roughness": 20.0,
            "slope_mean": 4.0,
        }

    def classifier(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counters["classifier"] += 1
        size = len(batch)
        return (
            np.tile(np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32), (size, 1)),
            np.arange(size * 4, dtype=np.float32).reshape(size, 4) + 1.0,
        )

    def regression(
        embeddings: np.ndarray, terrain_features: np.ndarray, logits: np.ndarray
    ) -> np.ndarray:
        counters["regression"] += 1
        assert embeddings.shape[0] == terrain_features.shape[0] == logits.shape[0]
        return np.linspace(2.0, 8.0, len(embeddings), dtype=np.float32)

    return ScoringDependencies(
        classifier_transform=transform,
        classifier_predictor=classifier,
        regression_predictor=regression,
        terrain_feature_fn=terrain,
        class_names=("forest", "mountain", "lake"),
        classifier_hash="classifier-fixture",
        regression_hash="regression-fixture",
        classifier_preprocess_identity="fixture-transform:/255:v1",
        terrain_feature_identity="fixture-terrain:v1",
        device="cpu",
    )


def _row_frame(root: Path, count: int, prefix: str) -> pd.DataFrame:
    """Rows whose sources live in ``root``; score with run_root=root / "run"."""
    rows = []
    for index in range(count):
        satellite = root / f"{prefix}_sat_{index}.png"
        terrain = root / f"{prefix}_ter_{index}.png"
        _png(satellite, (40 + index % 20, 120, 70))
        _png(terrain, (120, 120, 120))
        rows.append(
            {
                "image_path": f"images/satellite/z14/{prefix}/{index}.png",
                "provider": "mapbox_v1",
                "region": "fixture",
                "z": 14,
                "x": index,
                "y": 200,
                "lat": 40.0,
                "lon": -70.0,
                "satellite_path": satellite.name,
                "terrain_path": terrain.name,
                "satellite_present": True,
                "terrain_present": True,
            }
        )
    return pd.DataFrame(rows)


def _count_loaders(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Spy DataLoader constructions; returns recorded constructor kwargs."""
    import torch.utils.data as torch_data

    created: list[dict[str, object]] = []
    base = torch_data.DataLoader

    class CountingLoader(base):  # type: ignore[misc, valid-type]
        def __init__(self, dataset, *args, **kwargs) -> None:
            created.append(dict(kwargs))
            super().__init__(dataset, *args, **kwargs)

    monkeypatch.setattr(torch_data, "DataLoader", CountingLoader)
    return created


def _producer_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == _PRODUCER_THREAD_NAME and thread.is_alive()
    ]


@pytest.mark.parametrize("changed_model", ["classifier", "regression_model"])
def test_injected_same_class_state_changes_invalidate_cache(tmp_path, changed_model):
    class Model:
        def __init__(self, weight):
            self.weight = np.asarray([weight], dtype=np.float32)

        def state_dict(self):
            return {"weight": self.weight}

        def __call__(self, batch, *features):
            if features:
                return batch.mean(axis=1) * self.weight[0]
            return np.tile([[2.0, 0.0, -1.0]], (len(batch), 1))

        def get_features(self, batch):
            return np.full((len(batch), 4), self.weight[0], dtype=np.float32)

    manifest = _manifest(tmp_path).iloc[:1]
    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.classifier_predictor = deps.regression_predictor = None
    deps.classifier = Model(1)
    deps.regression_model = Model(1)
    root = tmp_path / "run"
    first = score_tile_manifest(manifest, run_root=root, dependencies=deps)
    assert pd.read_csv(root / "candidate_pool.csv").iloc[0]["model_prediction"] == 1
    same_state = score_tile_manifest(manifest, run_root=root, dependencies=deps)
    assert same_state["counts"]["cache_hits"] == 1
    setattr(deps, changed_model, Model(2))
    changed = score_tile_manifest(manifest, run_root=root, dependencies=deps)
    assert changed["counts"]["cache_hits"] == 0
    assert changed["counts"]["scored_rows"] == 1
    assert pd.read_csv(root / "candidate_pool.csv").iloc[0]["model_prediction"] == 2
    identity_key = (
        "classifier_checkpoint_sha256" if changed_model == "classifier"
        else "regression_checkpoint_sha256"
    )
    assert changed["models"][identity_key] != first["models"][identity_key]


def test_opaque_injected_models_require_content_identity(tmp_path):
    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.classifier_predictor = None
    deps.classifier_hash = None
    deps.classifier = object()
    with pytest.raises(ValueError, match="classifier requires a content hash or state_dict"):
        score_tile_manifest(
            _manifest(tmp_path).iloc[:1], run_root=tmp_path / "run", dependencies=deps
        )



def test_scoring_defaults_and_dry_run_create_no_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "runs"
    result = run_active_learning_scoring(
        manifest,
        output_dir=output,
        run_name="fixture",
        dry_run=True,
    )
    assert result["configuration"]["batch_size"] == DEFAULT_SCORING_BATCH_SIZE == 256
    assert result["configuration"]["num_workers"] == DEFAULT_SCORING_NUM_WORKERS
    assert result["configuration"]["pin_memory"] is False
    assert result["artifacts"] == {}
    assert not output.exists()


def test_cuda_loader_options_enable_pinned_prefetch_workers() -> None:
    dependencies = ScoringDependencies(
        classifier_checkpoint=Path("classifier.pt"),
        regression_checkpoint=Path("regression.pt"),
    )
    options, effective = _loader_options(
        device="cuda",
        num_workers=3,
        dependencies=dependencies,
    )
    assert effective == 3
    assert options["num_workers"] == 3
    assert options["pin_memory"] is True
    assert options["prefetch_factor"] == 2
    assert options["persistent_workers"] is True


def test_classifier_combined_forward_uses_one_backbone() -> None:
    torch = pytest.importorskip("torch")

    class FixtureClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, _batch):
            raise AssertionError("scoring should use the combined forward")

        def forward_with_features(self, batch):
            self.calls += 1
            features = batch.mean(dim=(2, 3))
            return features[:, :2], features

    classifier = FixtureClassifier()
    dependencies = ScoringDependencies(
        classifier=classifier,
        classifier_transform=lambda _image: torch.ones(3, 4, 4),
        device="cpu",
    )
    logits, embeddings = _call_classifier(
        dependencies,
        [Image.new("RGB", (4, 4), color=(1, 2, 3))],
    )
    assert classifier.calls == 1
    assert logits.shape == (1, 2)
    assert embeddings.shape == (1, 3)


def test_scoring_cli_dry_run_has_no_run_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.modeling.score_active_learning_pool import main

    manifest_path = tmp_path / "manifest.csv"
    _manifest(tmp_path).to_csv(manifest_path, index=False)
    output = tmp_path / "cli-runs"
    main(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
            "--run-name",
            "fixture",
            "--dry-run",
        ]
    )
    assert not output.exists()
    assert '"dry_run": true' in capsys.readouterr().out


def test_resume_skips_unchanged_rows_and_preserves_error_state(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    counters = {"classifier": 0, "regression": 0}
    dependencies = _dependencies(counters)
    first = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=dependencies, batch_size=2
    )
    assert first["counts"] == {
        "manifest_rows": 3,
        "scored_rows": 2,
        "selector_eligible_rows": 2,
        "missing_rows": 1,
        "error_rows": 0,
        "unusable_rows": 0,
        "cache_hits": 0,
        "cache_misses": 3,
    }
    assert counters == {"classifier": 1, "regression": 1}
    second = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=dependencies, batch_size=2
    )
    assert second["counts"]["cache_hits"] == 2
    assert counters == {"classifier": 1, "regression": 1}
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    assert len(candidates) == 3
    assert (
        candidates.loc[candidates["score_status"] == "missing", "selector_eligible"]
        .eq(False)
        .all()
    )
    assert candidates.loc[0, "image_path"] == "images/satellite/z14/fixture/100_200.png"


def test_model_prediction_stays_separate_from_weak_and_human_names(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path).iloc[:1]
    result = score_tile_manifest(
        manifest,
        run_root=tmp_path / "run",
        dependencies=_dependencies({"classifier": 0, "regression": 0}),
    )
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    row = candidates.iloc[0]
    assert "scenic_human" not in candidates.columns
    assert row["label_source"] == "active_regression_prediction"
    assert (
        row["scenic_score"] == row["scenic_score_heuristic"] == row["heuristic_score"]
    )
    assert row["regression_prediction"] != row["scenic_score"]
    assert result["models"]["label_semantics"].startswith("regression_prediction")


def test_normalized_entropy_is_class_entropy_and_bounded() -> None:
    assert normalized_class_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalized_class_entropy([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert 0.0 <= normalized_class_entropy([0.2, 0.3, 0.5]) <= 1.0
    assert 0.0 <= normalized_class_entropy([np.nan, -2.0, 0.0]) <= 1.0


def test_embedding_npz_shape_and_stable_row_indices(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path).iloc[:2]
    result = score_tile_manifest(
        manifest,
        run_root=tmp_path / "run",
        dependencies=_dependencies({"classifier": 0, "regression": 0}),
        lsh_seed=7,
        lsh_bits=8,
    )
    with np.load(
        tmp_path / "run" / "feature_embeddings.npz", allow_pickle=False
    ) as arrays:
        assert arrays["embeddings"].shape == (2, 4)
        assert arrays["embeddings"].dtype == np.float32
        assert arrays["row_indices"].tolist() == [0, 1]
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    assert candidates["embedding_row_index"].tolist() == [0, 1]
    assert candidates["embedding_cluster_id"].notna().all()
    assert result["embedding"]["dimension"] == 4


def test_malformed_registry_fails_closed(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"active": {"checkpoint": "missing.pt"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        resolve_active_regression_checkpoint(registry)
    registry.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_active_regression_checkpoint(registry)


def test_finalizer_requires_scoring_handoff_artifacts(tmp_path: Path) -> None:
    handoff = finalize_stage1(tmp_path, write=False)
    assert handoff["ready_for_stage2"] is False
    assert handoff["scoring_valid"] is False
    assert any("candidate_pool" in blocker for blocker in handoff["blockers"])


def test_scorer_flushes_pending_images_at_batch_size(tmp_path: Path) -> None:
    rows = []
    for index in range(5):
        sat = tmp_path / f"batch_sat_{index}.png"
        ter = tmp_path / f"batch_ter_{index}.png"
        _png(sat, (50 + index * 10, 150, 80))
        _png(ter, (120, 120, 120))
        rows.append(
            {
                "image_path": f"images/satellite/z14/batch/{index}.png",
                "provider": "mapbox_v1",
                "region": "fixture",
                "z": 14,
                "x": index,
                "y": 200,
                "lat": 40.0 + index,
                "lon": -70.0,
                "satellite_path": sat.name,
                "terrain_path": ter.name,
                "satellite_present": True,
                "terrain_present": True,
            }
        )
    manifest_df = pd.DataFrame(rows)
    manifest_path = tmp_path / "tile_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    recorded_batch_sizes: list[int] = []

    def transform(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    def terrain(
        _terrain: Image.Image, _satellite: Image.Image, *, valid_mask: np.ndarray,
        pixel_spacing_m: float,
    ) -> dict[str, object]:
        return {
            "features": np.asarray([0.2, 0.3, 0.4, 0.5, 0.0, 0.0], dtype=np.float32),
            "relief": 100.0,
            "roughness": 20.0,
            "slope_mean": 4.0,
        }

    def spy_classifier(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = len(batch)
        recorded_batch_sizes.append(size)
        return (
            np.tile(np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32), (size, 1)),
            np.arange(size * 4, dtype=np.float32).reshape(size, 4) + 1.0,
        )

    def spy_regression(
        embeddings: np.ndarray, terrain_features: np.ndarray, logits: np.ndarray
    ) -> np.ndarray:
        return np.linspace(2.0, 8.0, len(embeddings), dtype=np.float32)

    deps = ScoringDependencies(
        classifier_transform=transform,
        classifier_predictor=spy_classifier,
        regression_predictor=spy_regression,
        terrain_feature_fn=terrain,
        class_names=("forest", "mountain", "lake"),
        classifier_hash="classifier-fixture",
        regression_hash="regression-fixture",
        classifier_preprocess_identity="fixture-transform:v1",
        terrain_feature_identity="fixture-terrain:v1",
        device="cpu",
    )
    result = score_tile_manifest(
        manifest_df, run_root=tmp_path / "run", dependencies=deps, batch_size=2
    )
    assert result["state"]["ready_for_selection"] is True
    assert recorded_batch_sizes == [2, 2, 1]
    assert all(size <= 2 for size in recorded_batch_sizes)


def test_source_residency_is_bounded_by_queue_batch_and_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scenicdriver.active_learning.scoring as scoring_module

    total_rows, batch_size = 20, 2
    # Injected predictors force effective workers=0, so the queue holds
    # batch_size * max(1, 0) * prefetch rows; the producer may hold one more pair.
    capacity = batch_size * 1 * SCORING_PREFETCH_FACTOR
    row_bound = capacity + batch_size + 1
    manifest = _row_frame(tmp_path, total_rows, "resident")
    original_read = scoring_module._read_source
    live_sources = [0]
    satellite_reads = [0]

    def release() -> None:
        live_sources[0] -= 1

    def tracked_read(**kwargs: object):
        source = original_read(**kwargs)
        live_sources[0] += 1
        weakref.finalize(source, release)
        if kwargs.get("style") == "satellite":
            satellite_reads[0] += 1
        return source

    monkeypatch.setattr(scoring_module, "_read_source", tracked_read)
    samples: list[tuple[int, int, int]] = []
    consumed = [0]

    def classifier(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        consumed[0] += len(batch)
        gc.collect()
        samples.append((satellite_reads[0], consumed[0], live_sources[0]))
        size = len(batch)
        return (
            np.tile(np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32), (size, 1)),
            np.arange(size * 4, dtype=np.float32).reshape(size, 4) + 1.0,
        )

    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.classifier_predictor = classifier
    result = score_tile_manifest(
        manifest,
        run_root=tmp_path / "run",
        dependencies=deps,
        batch_size=batch_size,
    )

    assert result["counts"]["scored_rows"] == total_rows, result["errors"]
    assert len(samples) == total_rows // batch_size
    first_reads = samples[0][0]
    assert first_reads < total_rows, "full manifest was acquired before inference"
    assert first_reads <= row_bound
    for reads, consumed_rows, live in samples:
        # Acquisition lead: queued rows plus one pair in the producer's hand.
        assert reads - consumed_rows <= capacity + 1
        assert live <= 2 * row_bound


def test_loader_is_constructed_once_across_multiple_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scenicdriver.active_learning.scoring as scoring_module

    created = _count_loaders(monkeypatch)
    datasets: list[object] = []
    original_dataset = scoring_module._ScoringDataset

    class CountingDataset(original_dataset):
        def __init__(self, *args: object, **kwargs: object) -> None:
            datasets.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(scoring_module, "_ScoringDataset", CountingDataset)
    counters = {"classifier": 0, "regression": 0}
    result = score_tile_manifest(
        _row_frame(tmp_path, 7, "once"),
        run_root=tmp_path / "run",
        dependencies=_dependencies(counters),
        batch_size=2,
        num_workers=3,
    )

    assert result["counts"]["scored_rows"] == 7, result["errors"]
    assert counters == {"classifier": 4, "regression": 4}
    assert len(created) == 1
    assert len(datasets) == 1
    # Injected CPU predictors keep the loader in-process.
    assert created[0]["num_workers"] == 0
    assert created[0]["batch_size"] == 2
    assert not _producer_threads()


def test_scoring_rejects_invalid_run_names(tmp_path: Path) -> None:
    from scenicdriver.active_learning.scoring import run_active_learning_scoring

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("x,y\n1,1\n")
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
            run_active_learning_scoring(
                manifest,
                output_dir=tmp_path / "runs",
                run_name=bad_name,
            )


def test_validate_canonical_regression_checkpoint(tmp_path: Path) -> None:
    import torch
    from scenicdriver.active_learning.scoring import _validate_regression_checkpoint
    from scenicdriver.scenic_scorer.regression import ScenicRegressionModel

    # New canonical format: top-level dims + hidden_dim
    model_new = ScenicRegressionModel(
        vit_dim=4, terrain_dim=2, num_classes=3, hidden_dim=16
    )
    ckpt_new = tmp_path / "model_new.pt"
    torch.save(
        {
            "model_state_dict": model_new.state_dict(),
            "vit_dim": 4,
            "terrain_dim": 2,
            "num_classes": 3,
            "hidden_dim": 16,
        },
        ckpt_new,
    )
    loaded_new, dims_new = _validate_regression_checkpoint(ckpt_new, device="cpu")
    assert dims_new == {
        "vit_dim": 4,
        "terrain_dim": 2,
        "num_classes": 3,
        "hidden_dim": 16,
    }
    assert loaded_new is not None

    # Preserved active v6 format: top-level dims without hidden_dim (defaults to 256)
    model_v6 = ScenicRegressionModel(
        vit_dim=4, terrain_dim=2, num_classes=3, hidden_dim=256
    )
    ckpt_v6 = tmp_path / "model_v6.pt"
    torch.save(
        {
            "model_state_dict": model_v6.state_dict(),
            "vit_dim": 4,
            "terrain_dim": 2,
            "num_classes": 3,
        },
        ckpt_v6,
    )
    loaded_v6, dims_v6 = _validate_regression_checkpoint(ckpt_v6, device="cpu")
    assert dims_v6 == {
        "vit_dim": 4,
        "terrain_dim": 2,
        "num_classes": 3,
        "hidden_dim": 256,
    }
    assert loaded_v6 is not None


def test_resolve_active_regression_checkpoint_sha256_read_only_validation(
    tmp_path: Path,
) -> None:
    from scenicdriver.active_learning.scoring import resolve_active_regression_checkpoint

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy_checkpoint_data")
    actual_sha = sha256_file(ckpt)

    # Legacy registry without sha256 -> accepts read-only without modifying registry
    registry_file = tmp_path / "model_registry.json"
    initial_content = json.dumps({"active": {"checkpoint": str(ckpt.name)}})
    registry_file.write_text(initial_content, encoding="utf-8")
    resolved = resolve_active_regression_checkpoint(registry_file)
    assert resolved == ckpt
    assert registry_file.read_text(encoding="utf-8") == initial_content

    # Matching sha256 -> succeeds
    registry_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt.name), "sha256": actual_sha}}),
        encoding="utf-8",
    )
    resolved_matching = resolve_active_regression_checkpoint(registry_file)
    assert resolved_matching == ckpt

    # Mismatching sha256 -> fails closed
    registry_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt.name), "sha256": "wrong_hash"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        resolve_active_regression_checkpoint(registry_file)

@pytest.mark.parametrize("with_hash", [False, True])
def test_registry_relative_checkpoint_wins_over_cwd(tmp_path, monkeypatch, with_hash):
    published = tmp_path / "published"
    cwd = tmp_path / "cwd"
    published.mkdir()
    cwd.mkdir()
    (published / "checkpoints").mkdir()
    (cwd / "checkpoints").mkdir()
    checkpoint = published / "checkpoints" / "model.pt"
    checkpoint.write_bytes(b"published weights")
    (cwd / "checkpoints" / "model.pt").write_bytes(b"unrelated CWD weights")
    active = {"checkpoint": "checkpoints/model.pt"}
    if with_hash:
        active["sha256"] = sha256_file(checkpoint)
    registry = published / "model_registry.json"
    registry.write_text(json.dumps({"active": active}), encoding="utf-8")
    monkeypatch.chdir(cwd)
    assert resolve_active_regression_checkpoint(registry) == checkpoint



def test_water_scoring_classification_fresh_and_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Row 0: satellite water 0.0 (green), terrain water 0.10 -> effective 0.10 (< 0.50) -> scored
    # Row 1: satellite water 1.0 (blue), terrain water 0.10 -> effective 1.0 (>= 0.50) -> excessive_water
    # Row 2: satellite water 0.0 (green), terrain water 0.60 -> effective 0.60 (>= 0.50) -> excessive_water
    sat0, terr0 = tmp_path / "sat-0.png", tmp_path / "terrain-0.png"
    sat1, terr1 = tmp_path / "sat-1.png", tmp_path / "terrain-1.png"
    sat2, terr2 = tmp_path / "sat-2.png", tmp_path / "terrain-2.png"

    _png(sat0, (40, 120, 70))  # water_fraction = 0.0
    _png(terr0, (120, 120, 120))

    _png(sat1, (70, 90, 180))  # water_fraction = 1.0
    _png(terr1, (120, 120, 120))

    _png(sat2, (41, 120, 70))  # water_fraction = 0.0
    _png(terr2, (120, 120, 120))

    manifest = pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "region": "fixture",
                "z": 14,
                "x": 100,
                "y": 200,
                "lat": 40.0,
                "lon": -70.0,
                "satellite_path": sat0.name,
                "terrain_path": terr0.name,
                "satellite_present": True,
                "terrain_present": True,
            },
            {
                "image_path": "images/satellite/z14/fixture/101_200.png",
                "region": "fixture",
                "z": 14,
                "x": 101,
                "y": 200,
                "lat": 40.1,
                "lon": -70.0,
                "satellite_path": sat1.name,
                "terrain_path": terr1.name,
                "satellite_present": True,
                "terrain_present": True,
            },
            {
                "image_path": "images/satellite/z14/fixture/102_200.png",
                "region": "fixture",
                "z": 14,
                "x": 102,
                "y": 200,
                "lat": 40.2,
                "lon": -70.0,
                "satellite_path": sat2.name,
                "terrain_path": terr2.name,
                "satellite_present": True,
                "terrain_present": True,
            },
        ]
    )
    manifest["provider"] = "mapbox_v1"

    counters = {"classifier": 0, "regression": 0}

    def terrain_fn(
        _terrain: Image.Image, satellite: Image.Image, *, valid_mask: np.ndarray,
        pixel_spacing_m: float,
    ) -> dict[str, object]:
        # Custom terrain function providing a 0.60 sea-level proxy for item 2.
        arr = np.asarray(satellite, dtype=np.float32)
        is_sat2 = tuple(arr[0, 0, :3]) == (41.0, 120.0, 70.0)
        return {
            "features": np.ones((6,), dtype=np.float32),
            "relief": 0.5,
            "roughness": 0.5,
            "slope_mean": 0.5,
            "terrain_sea_level_fraction": 0.60 if is_sat2 else 0.10,
        }

    dependencies = _dependencies(counters)
    dependencies.terrain_feature_fn = terrain_fn

    # Fresh scoring
    fresh_root = tmp_path / "run_fresh"
    fresh_result = score_tile_manifest(
        manifest,
        run_root=fresh_root,
        dependencies=dependencies,
    )
    assert fresh_result["state"]["complete"] is True
    assert fresh_result["counts"]["unusable_rows"] == 1

    pool_fresh = pd.read_csv(fresh_root / "candidate_pool.csv")
    assert len(pool_fresh) == 3

    # Row 0: Shoreline / low water (< 0.50) -> scored
    assert pool_fresh.loc[0, "score_status"] == "scored"
    assert pool_fresh.loc[0, "availability_state"] == "available"
    assert bool(pool_fresh.loc[0, "selector_eligible"])
    assert pool_fresh.loc[0, "water_filter_status"] == "pass"
    assert (
        pd.isna(pool_fresh.loc[0, "unusable_reason"])
        or pool_fresh.loc[0, "unusable_reason"] == ""
    )
    assert pd.notna(pool_fresh.loc[0, "regression_prediction"])

    # Row 1: High satellite water (1.0 >= 0.50) -> unusable
    assert pool_fresh.loc[1, "score_status"] == "unusable"
    assert pool_fresh.loc[1, "availability_state"] == "unusable"
    assert not bool(pool_fresh.loc[1, "selector_eligible"])
    assert pool_fresh.loc[1, "water_filter_status"] == "excessive_water"
    assert pool_fresh.loc[1, "unusable_reason"] == "excessive_water"
    assert pd.isna(pool_fresh.loc[1, "regression_prediction"])
    assert pd.isna(pool_fresh.loc[1, "heuristic_score"])
    assert pd.isna(pool_fresh.loc[1, "normalized_class_entropy"])
    # Evidence retained
    assert pool_fresh.loc[1, "water_fraction"] == 1.0

    # Row 2: Terrain-RGB is a sea-level proxy, but satellite imagery is
    # authoritative when present, preventing flat sea-level land false positives.
    assert pool_fresh.loc[2, "score_status"] == "scored"
    assert pool_fresh.loc[2, "availability_state"] == "available"
    assert bool(pool_fresh.loc[2, "selector_eligible"])
    assert pool_fresh.loc[2, "water_filter_status"] == "pass"
    assert pool_fresh.loc[2, "terrain_sea_level_fraction"] == 0.60
    assert pool_fresh.loc[2, "effective_water_fraction"] == 0.0

    # Feature embeddings exclude only the satellite-confirmed water row.
    npz_fresh = np.load(fresh_root / "feature_embeddings.npz")
    assert npz_fresh["embeddings"].shape[0] == 2
    assert list(npz_fresh["row_indices"]) == [0, 2]

    # Simulate a pre-filter cache containing embeddings for every row, then
    # prove resumed scoring reclassifies cached water rows fail-closed.
    import scenicdriver.active_learning.scoring as scoring_module

    original_evaluate = scoring_module.evaluate_water_status
    original_is_excessive = scoring_module.is_excessive_water
    legacy_root = tmp_path / "run_cached"
    monkeypatch.setattr(
        scoring_module,
        "evaluate_water_status",
        lambda sat, terrain: (
            max(float(sat or 0.0), float(terrain or 0.0)),
            "pass",
            None,
        ),
    )
    monkeypatch.setattr(scoring_module, "is_excessive_water", lambda _value: False)
    score_tile_manifest(manifest, run_root=legacy_root, dependencies=dependencies)
    monkeypatch.setattr(scoring_module, "evaluate_water_status", original_evaluate)
    monkeypatch.setattr(scoring_module, "is_excessive_water", original_is_excessive)
    res_cache = score_tile_manifest(
        manifest,
        run_root=legacy_root,
        dependencies=dependencies,
    )
    assert res_cache["counts"]["cache_hits"] == 3
    pool_cache = pd.read_csv(legacy_root / "candidate_pool.csv")
    assert pool_cache.loc[0, "score_status"] == "scored"
    assert pool_cache.loc[1, "score_status"] == "unusable"
    assert pool_cache.loc[1, "unusable_reason"] == "excessive_water"
    assert pool_cache.loc[2, "score_status"] == "scored"

    scoring_manifest_path = legacy_root / "scoring_manifest.json"
    stale_manifest = json.loads(scoring_manifest_path.read_text(encoding="utf-8"))
    stale_manifest["schema_version"] = 1
    scoring_manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")
    stale_result = score_tile_manifest(
        manifest,
        run_root=legacy_root,
        dependencies=dependencies,
    )
    assert stale_result["counts"]["cache_hits"] == 0


def test_all_water_manifest_scoring(tmp_path: Path) -> None:
    sat0, terr0 = tmp_path / "sat-0.png", tmp_path / "terrain-0.png"
    _png(sat0, (70, 90, 180))  # 100% water
    _png(terr0, (120, 120, 120))

    manifest = pd.DataFrame(
        [
            {
                "image_path": "images/satellite/z14/fixture/100_200.png",
                "region": "fixture",
                "z": 14,
                "x": 100,
                "y": 200,
                "lat": 40.0,
                "lon": -70.0,
                "satellite_path": sat0.name,
                "terrain_path": terr0.name,
                "satellite_present": True,
                "terrain_present": True,
            }
        ]
    )
    manifest["provider"] = "mapbox_v1"
    counters = {"classifier": 0, "regression": 0}
    dependencies = _dependencies(counters)
    all_water_root = tmp_path / "run_all_water"
    score_tile_manifest(
        manifest,
        run_root=all_water_root,
        dependencies=dependencies,
    )
    pool = pd.read_csv(all_water_root / "candidate_pool.csv")
    assert len(pool) == 1
    assert pool.loc[0, "score_status"] == "unusable"
    assert pool.loc[0, "unusable_reason"] == "excessive_water"
    assert not bool(pool.loc[0, "selector_eligible"])

    npz = np.load(all_water_root / "feature_embeddings.npz")
    assert npz["embeddings"].shape[0] == 0
    assert len(npz["row_indices"]) == 0


def test_stale_old_cache_invalidation_and_no_source_model_collision(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    counters = {"classifier": 0, "regression": 0}
    deps1 = _dependencies(counters)

    run_dir = tmp_path / "run_source_test"
    # 1. Score with mapbox source contract
    manifest_mapbox = manifest.copy()
    manifest_mapbox["provider"] = "mapbox"
    manifest_mapbox["satellite_source"] = "mapbox.satellite"
    manifest_mapbox["source_contract_sha256"] = "a" * 64

    res1 = score_tile_manifest(manifest_mapbox, run_root=run_dir, dependencies=deps1)
    assert res1["counts"]["cache_hits"] == 0
    assert res1["source"]["provider"] == "mapbox"

    # 2. Rescore with same mapbox source contract -> cache hit
    res2 = score_tile_manifest(manifest_mapbox, run_root=run_dir, dependencies=deps1)
    assert res2["counts"]["cache_hits"] == 2

    # 3. Rescore with naip source contract -> stale cache invalidated, 0 hits
    manifest_naip = manifest.copy()
    manifest_naip["provider"] = "naip"
    manifest_naip["satellite_source"] = "naip-visualization"
    manifest_naip["source_contract_sha256"] = "b" * 64

    res3 = score_tile_manifest(manifest_naip, run_root=run_dir, dependencies=deps1)
    assert res3["counts"]["cache_hits"] == 0
    assert res3["source"]["provider"] == "naip"


def test_old_active_data_readable_only_as_explicit_mapbox_v1_identities(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    counters = {"classifier": 0, "regression": 0}
    deps = _dependencies(counters)
    run_dir = tmp_path / "run_explicit_mapbox"

    # Explicit mapbox-v1 canonical fields
    manifest_explicit = manifest.copy()
    manifest_explicit["provider"] = "mapbox"
    manifest_explicit["satellite_source"] = "mapbox.satellite"

    res1 = score_tile_manifest(manifest_explicit, run_root=run_dir, dependencies=deps)
    assert res1["counts"]["cache_hits"] == 0

    res2 = score_tile_manifest(manifest_explicit, run_root=run_dir, dependencies=deps)
    assert res2["counts"]["cache_hits"] == 2

    # If canonical fields are missing when NAIP contract requested, no shimming/guessing mapbox
    manifest_missing_canonical = manifest.copy()
    manifest_missing_canonical["provider"] = "naip"
    manifest_missing_canonical["satellite_source"] = "naip-visualization"

    res3 = score_tile_manifest(
        manifest_missing_canonical, run_root=run_dir, dependencies=deps
    )
    assert res3["counts"]["cache_hits"] == 0


def test_mixed_shoreline_water_is_not_gated_by_land_texture():
    from scenicdriver.active_learning.water import compute_satellite_water_fraction

    satellite = np.full((100, 100, 3), [15, 30, 100], dtype=np.uint8)
    land = np.indices((100, 40)).sum(axis=0) % 2
    satellite[:, 60:] = np.where(land[..., None], 255, 0)
    assert compute_satellite_water_fraction(satellite) == pytest.approx(0.6, abs=0.02)


def test_water_geometry_counts_canonical_black_sea_with_sparse_land():
    from scenicdriver.active_learning.water import compute_satellite_water_fraction

    satellite = np.zeros((10, 10, 3), dtype=np.uint8)
    land = np.zeros((10, 10), dtype=bool)
    land[0, 0] = True
    satellite[0, 0] = [40, 120, 70]
    assert compute_satellite_water_fraction(satellite, land_mask=land) == pytest.approx(0.99)
    land[:] = False
    assert compute_satellite_water_fraction(satellite, land_mask=land) == 1.0


def test_flat_blue_water_with_clouds_retains_local_water_detection():
    from scenicdriver.active_learning.water import compute_satellite_water_fraction

    satellite = np.full((100, 100, 3), [15, 30, 100], dtype=np.uint8)
    satellite[30:70, 30:70] = 255
    satellite[10:12, 10:12] = 255  # small whitecap/cloud
    assert compute_satellite_water_fraction(satellite) == pytest.approx(0.84, abs=0.03)


@pytest.mark.parametrize("mask,error", [
    (np.ones((2, 2), dtype=np.uint8), TypeError),
    (np.ones((1, 2), dtype=bool), ValueError),
])
def test_water_rejects_invalid_land_masks(mask, error):
    from scenicdriver.active_learning.water import compute_satellite_water_fraction

    with pytest.raises(error):
        compute_satellite_water_fraction(np.zeros((2, 2, 3)), land_mask=mask)


def test_scoring_passes_tile_ground_spacing_and_requires_tile_xy(tmp_path):
    from scenicdriver.data_pipeline.web_mercator import tile_pixel_ground_spacing_m

    manifest = _manifest(tmp_path).iloc[:2].copy()
    manifest.loc[1, "x"] = np.nan
    deps = _dependencies({"classifier": 0, "regression": 0})
    fake = deps.terrain_feature_fn
    spacings = []

    def spy(terrain, satellite, *, valid_mask, pixel_spacing_m):
        spacings.append(pixel_spacing_m)
        return fake(terrain, satellite, valid_mask=valid_mask, pixel_spacing_m=pixel_spacing_m)

    deps.terrain_feature_fn = spy
    result = score_tile_manifest(manifest, run_root=tmp_path / "run", dependencies=deps)
    assert spacings == [pytest.approx(tile_pixel_ground_spacing_m(100, 200, 14, 4, 4))]
    assert result["counts"]["scored_rows"] == 1
    assert result["counts"]["error_rows"] == 1
    row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[1]
    assert "invalid_acquisition_tile" in row["error"]


def _water_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(satellite, terrain) 16x16 canonical tiles for shared water-metric cases."""
    satellite = np.full((16, 16, 3), [15, 30, 100], dtype=np.uint8)
    terrain = np.full((16, 16, 3), [1, 138, 112], dtype=np.uint8)
    if name == "textured_land":
        # Smooth water left; high-texture land right. A whole-image texture
        # gate would veto all water; local windows keep the water half.
        checker = (np.indices((16, 8)).sum(axis=0) % 2).astype(bool)
        satellite[:, 8:] = np.where(checker[..., None], 255, 0)
    elif name == "sea_fill":
        # Outside-land canonical black fill is known sea.
        terrain[:, :8] = 0
        satellite[:, :8] = 0
        satellite[:, 8:] = [40, 120, 70]
    elif name == "masked":
        # Bright fill under a masked DEM column must not add texture evidence.
        terrain[:, :3] = 0
        satellite[:, :3] = 255
    return satellite, terrain


_WATER_CASES = {"textured_land": None, "sea_fill": 0.5, "masked": 1.0}


def test_labeler_and_scorer_share_one_water_component(tmp_path, monkeypatch):
    from scenicdriver.active_learning import scoring, water
    from scenicdriver.heuristics import labeler
    from scenicdriver.terrain.features import compute_terrain_features

    assert labeler.compute_satellite_water_fraction is water.compute_satellite_water_fraction
    assert scoring.compute_satellite_water_fraction is water.compute_satellite_water_fraction
    assert labeler.weak_score_water_components is water.weak_score_water_components
    assert scoring.weak_score_water_components is water.weak_score_water_components

    monkeypatch.delenv("SCENIC_S3_BUCKET", raising=False)
    raw = tmp_path / "raw"
    sat_dir, terr_dir = raw / "images/satellite/z14", raw / "images/terrain/z14"
    sat_dir.mkdir(parents=True)
    terr_dir.mkdir(parents=True)
    rows = []
    for offset, name in enumerate(_WATER_CASES):
        satellite, terrain = _water_case(name)
        Image.fromarray(satellite).save(sat_dir / f"{100 + offset}_200.png")
        Image.fromarray(terrain).save(terr_dir / f"{100 + offset}_200.png")
        rows.append({
            "image_path": f"images/satellite/z14/{100 + offset}_200.png",
            "provider": "mapbox_v1", "region": "fixture",
            "z": 14, "x": 100 + offset, "y": 200, "lat": 40.0, "lon": -70.0,
            "satellite_path": str(sat_dir / f"{100 + offset}_200.png"),
            "terrain_path": str(terr_dir / f"{100 + offset}_200.png"),
            "satellite_present": True, "terrain_present": True,
        })

    _, tiles, _ = labeler.run_heuristic_labeling(
        satellite_dir=sat_dir, terrain_dir=terr_dir, raw_dir=raw, max_tiles=None,
        use_classifier=False, classifier_best_ckpt=tmp_path / "missing.pt",
        default_lat=0.0, default_lon=0.0, seed=0, device="cpu",
    )
    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.terrain_feature_fn = compute_terrain_features
    deps.terrain_feature_identity = None
    score_tile_manifest(pd.DataFrame(rows), run_root=tmp_path / "run", dependencies=deps)
    pool = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")

    by_x = {tile["x"]: tile for tile in tiles}
    for offset, (name, expected) in enumerate(_WATER_CASES.items()):
        tile = by_x[100 + offset]
        assert tile["water_fraction"] == pytest.approx(pool.loc[offset, "water_fraction"]), name
        if expected is not None:
            assert tile["water_fraction"] == pytest.approx(expected), name
        # Labeler weak score equals the scorer's heuristic for identical inputs.
        components = scoring._heuristic_components(
            tile["class_score"],
            {
                "relief": tile["relief"], "roughness": tile["roughness"],
                "slope_mean": tile["slope_mean"], "water_proximity": tile["water_proxy"],
                "vegetation_density": tile["veg_proxy"],
            },
            {"water_fraction": tile["water_fraction"]},
        )
        assert tile["scenic_score"] == pytest.approx(components["heuristic_score"]), name
    # Local texture keeps the smooth water half despite textured land.
    assert 0.4 <= by_x[100]["water_fraction"] <= 0.6


def test_labeler_run_metadata_records_water_and_terrain_contracts(tmp_path, monkeypatch):
    from scenicdriver.active_learning.scoring import TERRAIN_FEATURE_CONTRACT
    from scenicdriver.active_learning.water import WATER_METRIC
    from scenicdriver.heuristics import labeler

    monkeypatch.delenv("SCENIC_S3_BUCKET", raising=False)
    raw = tmp_path / "raw"
    sat_dir, terr_dir = raw / "images/satellite/z16", raw / "images/terrain/z16"
    sat_dir.mkdir(parents=True)
    terr_dir.mkdir(parents=True)
    satellite, terrain = _water_case("sea_fill")
    Image.fromarray(satellite).save(sat_dir / "19073_24793.png")
    Image.fromarray(terrain).save(terr_dir / "19073_24793.png")
    kwargs = dict(
        raw_dir=raw, max_tiles=None, use_classifier=False,
        classifier_best_ckpt=tmp_path / "missing.pt", default_lat=0.0, default_lon=0.0,
        seed=0, device="cpu",
    )
    _, _, run_info = labeler.run_heuristic_labeling(
        satellite_dir=sat_dir, terrain_dir=terr_dir, **kwargs
    )
    assert run_info["water_metric"] == WATER_METRIC
    assert run_info["terrain_feature_contract"] == TERRAIN_FEATURE_CONTRACT
    assert TERRAIN_FEATURE_CONTRACT["version"] == 2
    assert TERRAIN_FEATURE_CONTRACT["slope_units"] == "degrees"

    flat_sat, flat_terr = raw / "flat/sat", raw / "flat/terr"
    flat_sat.mkdir(parents=True)
    flat_terr.mkdir(parents=True)
    Image.fromarray(satellite).save(flat_sat / "tile.png")
    Image.fromarray(terrain).save(flat_terr / "tile.png")
    with pytest.raises(ValueError, match="pixel spacing"):
        labeler.run_heuristic_labeling(
            satellite_dir=flat_sat, terrain_dir=flat_terr, **kwargs
        )


@pytest.mark.parametrize("provider", [None, "", "  ", "mapbox_v1"])
def test_scoring_coast_requires_attribution_and_uses_land_only_features(tmp_path, provider):
    from scenicdriver.terrain.features import compute_terrain_features

    manifest = _manifest(tmp_path).iloc[:1].copy()
    if provider is None:
        manifest = manifest.drop(columns=["provider"])
    else:
        manifest["provider"] = provider
    terrain = np.full((20, 20, 3), [1, 134, 160], dtype=np.uint8)
    terrain[:, :8] = 0
    satellite = np.full((20, 20, 3), [10, 80, 10], dtype=np.uint8)
    satellite[:, :8] = 0
    Image.fromarray(terrain).save(tmp_path / "terrain-0.png")
    Image.fromarray(satellite).save(tmp_path / "sat-0.png")
    counters = {"classifier": 0, "regression": 0}
    deps = _dependencies(counters)
    deps.terrain_feature_fn = compute_terrain_features
    deps.terrain_feature_identity = None
    captured = []

    def regression(embeddings, terrain_features, logits):
        captured.append(terrain_features.copy())
        return np.zeros(len(embeddings), dtype=np.float32)

    deps.regression_predictor = regression
    result = score_tile_manifest(manifest, run_root=tmp_path / "run", dependencies=deps)
    if provider != "mapbox_v1":
        assert result["counts"]["error_rows"] == 1
        assert result["counts"]["scored_rows"] == 0
        assert counters == {"classifier": 0, "regression": 0}
        assert captured == []
        row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[0]
        assert "missing_source_attribution" in row["error"]
        assert not row["selector_eligible"]
        return
    assert result["counts"]["scored_rows"] == 1
    assert result["counts"]["selector_eligible_rows"] == 1
    row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[0]
    assert row["relief"] == row["roughness"] == row["slope_mean"] == 0.0
    assert row["vegetation_density"] == pytest.approx(0.8)
    assert row["water_fraction"] == pytest.approx(0.4)
    assert captured[0].shape == (1, 6)
    assert captured[0][0, 0] == captured[0][0, 1] == 0.0


def _acquisition_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _bind_acquisition_identity(manifest):
    from scenicdriver.data_pipeline.source_contracts import compute_acquisition_tile_identity

    row = manifest.iloc[0]
    manifest["acquisition_tile_identity_sha256"] = compute_acquisition_tile_identity(
        z=int(row["z"]), x=int(row["x"]), y=int(row["y"]), region=row["region"],
        target_grid_sha256=row["grid_sha256"],
        source_contract_sha256=row["source_contract_sha256"],
        preprocessing_contract_sha256=row["preprocessing_contract_sha256"],
        satellite_output_sha256=row["satellite_output_sha256"],
        terrain_output_sha256=row["terrain_output_sha256"],
    ).sha256()
    return manifest


def _acquisition_manifest(root):
    manifest = _manifest(root).iloc[:1].copy()
    manifest["provider"] = "naip"
    manifest["satellite_source"] = "naip-visualization"
    manifest["terrain_source"] = "usgs-3dep"
    preprocessing = {
        "zoom": 14, "tile_width": 4, "tile_height": 4, "target_crs": "EPSG:3857",
        "satellite_resampling": "nearest", "terrain_resampling": "cubic",
    }
    contract = root / "source_contract.json"
    contract.write_text(json.dumps({"preprocessing": preprocessing}), encoding="utf-8")
    contract_hash = sha256_file(contract)
    preprocessing_hash = _acquisition_digest(preprocessing)
    (root / "execution_plan.json").write_text(json.dumps({
        "zoom": 14,
        "grid_contract": {"tile_width": 4, "tile_height": 4, "crs": "EPSG:3857"},
        "inputs": {"source_contract_path": str(contract)},
        "hashes": {
            "source_contract_sha256": contract_hash,
            "preprocessing_contract_sha256": preprocessing_hash,
        },
    }), encoding="utf-8")
    row = manifest.iloc[0]
    manifest["grid_sha256"] = _acquisition_digest({
        "z": int(row["z"]), "x": int(row["x"]), "y": int(row["y"]),
        "width": 4, "height": 4, "crs": "EPSG:3857",
    })
    manifest["source_contract_sha256"] = contract_hash
    manifest["preprocessing_contract_sha256"] = preprocessing_hash
    manifest["satellite_output_sha256"] = sha256_file(root / "sat-0.png")
    manifest["terrain_output_sha256"] = sha256_file(root / "terrain-0.png")
    return _bind_acquisition_identity(manifest)


def test_canonical_scoring_requires_acquisition_provenance(tmp_path):
    manifest = _manifest(tmp_path).iloc[:1].copy()
    manifest["provider"] = "naip"
    manifest["satellite_source"] = "naip-visualization"
    counters = {"classifier": 0, "regression": 0}
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=_dependencies(counters)
    )
    assert result["counts"]["error_rows"] == 1
    assert result["counts"]["scored_rows"] == 0
    assert counters == {"classifier": 0, "regression": 0}
    row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[0]
    assert "missing_acquisition_provenance" in row["error"]
    assert not row["selector_eligible"]


def test_canonical_scoring_accepts_matching_acquisition(tmp_path):
    manifest = _acquisition_manifest(tmp_path)
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run",
        dependencies=_dependencies({"classifier": 0, "regression": 0}),
    )
    assert result["counts"]["scored_rows"] == 1
    assert result["counts"]["error_rows"] == 0
    assert result["grid"] == {
        "zoom": 14, "width": 4, "height": 4, "crs": "EPSG:3857", "has_coords": True,
    }
    assert result["preprocessing"]["satellite_resampling"] == "nearest"
    assert result["preprocessing"]["terrain_resampling"] == "cubic"
    assert result["preprocessing"]["resample_alg"] is None


@pytest.mark.parametrize("field,value,error", [
    ("z", 13, "acquisition_zoom_mismatch"),
    ("grid_sha256", "a" * 64, "acquisition_grid_mismatch"),
    ("width", 5, "acquisition_width_mismatch"),
    ("height", 5, "acquisition_height_mismatch"),
    ("crs", "EPSG:4326", "acquisition_crs_mismatch"),
    ("satellite_resampling", "average", "acquisition_satellite_resampling_mismatch"),
    ("terrain_resampling", "bilinear", "acquisition_terrain_resampling_mismatch"),
])
def test_acquisition_metadata_mismatch_cannot_reuse_cache(tmp_path, field, value, error):
    manifest = _acquisition_manifest(tmp_path)
    counters = {"classifier": 0, "regression": 0}
    deps = _dependencies(counters)
    root = tmp_path / "run"
    score_tile_manifest(manifest, run_root=root, dependencies=deps)
    counters.update(classifier=0, regression=0)
    manifest[field] = value
    _bind_acquisition_identity(manifest)
    result = score_tile_manifest(manifest, run_root=root, dependencies=deps)
    assert result["counts"]["cache_hits"] == result["counts"]["scored_rows"] == 0
    assert counters == {"classifier": 0, "regression": 0}
    assert error in pd.read_csv(root / "candidate_pool.csv").iloc[0]["error"]


def test_acquisition_dimensions_are_verified_against_png_bytes(tmp_path):
    manifest = _acquisition_manifest(tmp_path)
    for style in ("satellite", "terrain"):
        path = tmp_path / manifest.iloc[0][f"{style}_path"]
        Image.new("RGB", (5, 4), (40, 120, 70)).save(path)
        manifest[f"{style}_output_sha256"] = sha256_file(path)
    _bind_acquisition_identity(manifest)
    counters = {"classifier": 0, "regression": 0}
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=_dependencies(counters),
    )
    assert result["counts"]["scored_rows"] == 0
    assert counters == {"classifier": 0, "regression": 0}
    row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[0]
    assert "acquisition_dimensions_mismatch" in row["error"]


@pytest.mark.parametrize("change", ["missing_plan", "modified_contract"])
def test_acquisition_metadata_requires_hash_verified_contract(tmp_path, change):
    manifest = _acquisition_manifest(tmp_path)
    if change == "missing_plan":
        (tmp_path / "execution_plan.json").unlink()
        expected = "missing_acquisition_metadata"
    else:
        contract = tmp_path / "source_contract.json"
        payload = json.loads(contract.read_text())
        payload["preprocessing"]["terrain_resampling"] = "nearest"
        contract.write_text(json.dumps(payload))
        expected = "invalid_acquisition_metadata"
    counters = {"classifier": 0, "regression": 0}
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=_dependencies(counters),
    )
    assert result["counts"]["scored_rows"] == 0
    assert counters == {"classifier": 0, "regression": 0}
    row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[0]
    assert expected in row["error"]


def test_legacy_grid_reports_observed_size_and_unknown_resampling(tmp_path):
    manifest = _manifest(tmp_path).iloc[:1]
    manifest["z"] = 13
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run",
        dependencies=_dependencies({"classifier": 0, "regression": 0}),
    )
    assert result["counts"]["scored_rows"] == 1
    assert result["grid"]["zoom"] == 13
    assert (result["grid"]["width"], result["grid"]["height"]) == (4, 4)
    assert result["preprocessing"]["satellite_resampling"] is None
    assert result["preprocessing"]["terrain_resampling"] is None
    assert result["preprocessing"]["resample_alg"] is None


@pytest.mark.parametrize("style", ["satellite", "terrain"])
def test_canonical_scoring_rejects_tampered_png_before_cache_or_models(tmp_path, style):
    manifest = _acquisition_manifest(tmp_path)
    counters = {"classifier": 0, "regression": 0}
    dependencies = _dependencies(counters)
    run_root = tmp_path / "run"
    score_tile_manifest(manifest, run_root=run_root, dependencies=dependencies)
    counters.update(classifier=0, regression=0)
    path = manifest.iloc[0][f"{style}_path"]
    _png(tmp_path / path, (60, 130, 90))
    result = score_tile_manifest(manifest, run_root=run_root, dependencies=dependencies)
    assert result["counts"]["scored_rows"] == result["counts"]["cache_hits"] == 0
    assert result["counts"]["error_rows"] == 1
    assert counters == {"classifier": 0, "regression": 0}
    row = pd.read_csv(run_root / "candidate_pool.csv").iloc[0]
    assert f"{style}_output_sha256_mismatch" in row["error"]


def test_canonical_scoring_rejects_forged_acquisition_identity(tmp_path):
    manifest = _acquisition_manifest(tmp_path)
    manifest["acquisition_tile_identity_sha256"] = "d" * 64
    counters = {"classifier": 0, "regression": 0}
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=_dependencies(counters)
    )
    assert result["counts"]["error_rows"] == 1
    assert counters == {"classifier": 0, "regression": 0}
    row = pd.read_csv(tmp_path / "run" / "candidate_pool.csv").iloc[0]
    assert "acquisition_identity_mismatch" in row["error"]


def test_planner_coastal_inventory_counts_black_sea(tmp_path):
    from scripts.ingest.plan_active_learning_region import _assess_water_inventory

    terrain = np.full((10, 10, 3), [1, 134, 160], dtype=np.uint8)
    satellite = np.full((10, 10, 3), [10, 80, 10], dtype=np.uint8)
    terrain[:, :6] = 0
    satellite[:, :6] = 0
    terrain_path = tmp_path / "coast-terrain.png"
    satellite_path = tmp_path / "coast-satellite.png"
    Image.fromarray(terrain).save(terrain_path)
    Image.fromarray(satellite).save(satellite_path)
    row = {
        "satellite_path": str(satellite_path),
        "terrain_path": str(terrain_path),
        "satellite_present": True,
        "terrain_present": True,
    }

    result = _assess_water_inventory([row])

    assert row["satellite_water_fraction"] == pytest.approx(0.6)
    assert row["effective_water_fraction"] == pytest.approx(0.6)
    assert row["water_filter_status"] == "excessive_water"
    assert result["assessed"] == result["excessive"] == 1
    assert result["eligible"] == result["unknown"] == 0


@pytest.mark.parametrize("geometry", ["missing", "different_grid"])
def test_planner_inventory_without_matching_geometry_remains_unknown(tmp_path, geometry):
    from scripts.ingest.plan_active_learning_region import _assess_water_inventory

    satellite_path = tmp_path / "satellite.png"
    terrain_path = tmp_path / "terrain.png"
    _png(satellite_path, (0, 0, 0))
    if geometry == "different_grid":
        Image.new("RGB", (2, 2), color=(1, 134, 160)).save(terrain_path)
    row = {
        "satellite_path": str(satellite_path),
        "terrain_path": str(terrain_path),
        "satellite_present": True,
        "terrain_present": geometry != "missing",
    }

    result = _assess_water_inventory([row])

    assert row["satellite_water_fraction"] == ""
    assert row["effective_water_fraction"] == ""
    assert row["water_filter_status"] == "unknown"
    assert result["unknown"] == 1
    assert result["assessed"] == result["eligible"] == 0


def test_cache_identity_binds_checkpoint_transform_config(tmp_path: Path) -> None:
    from scenicdriver.active_learning.scoring import (
        _preprocessing_contract,
        classifier_preprocess_identity,
    )
    from scenicdriver.classifier.inference import PreprocessingContract

    names = ("forest", "mountain", "lake")

    def digest(preprocessing: PreprocessingContract) -> str:
        deps = ScoringDependencies(
            classifier_transform=lambda image: image,
            classifier_preprocess_identity=classifier_preprocess_identity(preprocessing),
            device="cpu",
        )
        return _preprocessing_contract(deps)["pipeline_sha256"]

    resisc45 = PreprocessingContract.for_normalization("resisc45", class_names=names)
    baseline = digest(resisc45)
    assert digest(PreprocessingContract.for_normalization("resisc45", class_names=names)) == baseline
    assert digest(PreprocessingContract.for_normalization("imagenet", class_names=names)) != baseline
    assert digest(
        PreprocessingContract.for_normalization("resisc45", class_names=names, image_size=256)
    ) != baseline
    assert digest(
        PreprocessingContract.for_normalization("resisc45", class_names=names[::-1])
    ) != baseline


def test_cache_identity_requires_injected_preprocessor_identities() -> None:
    from scenicdriver.active_learning.scoring import (
        TERRAIN_FEATURE_CONTRACT,
        _preprocessing_contract,
    )

    with pytest.raises(ValueError, match="classifier_preprocess_identity"):
        _preprocessing_contract(
            ScoringDependencies(classifier_transform=lambda image: image, device="cpu")
        )
    with pytest.raises(ValueError, match="terrain_feature_identity"):
        _preprocessing_contract(
            ScoringDependencies(terrain_feature_fn=lambda *a, **k: {}, device="cpu")
        )
    default = _preprocessing_contract(ScoringDependencies(device="cpu"))
    assert json.loads(default["terrain_features"]) == TERRAIN_FEATURE_CONTRACT
    injected = ScoringDependencies(
        terrain_feature_fn=lambda *a, **k: {}, terrain_feature_identity="terrain:v1", device="cpu"
    )
    changed = ScoringDependencies(
        terrain_feature_fn=lambda *a, **k: {}, terrain_feature_identity="terrain:v2", device="cpu"
    )
    assert (
        _preprocessing_contract(injected)["pipeline_sha256"]
        != _preprocessing_contract(changed)["pipeline_sha256"]
        != default["pipeline_sha256"]
    )


def test_spawn_workers_persist_for_one_loader_with_production_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward_with_features(self, batch):
            features = batch.mean(dim=(2, 3))
            return features @ self.weight, features

    class Regression(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, embeddings, terrain, logits):
            return embeddings.sum(1) + terrain.sum(1) + logits.sum(1) + self.bias

    log_path = tmp_path / "workers.log"
    monkeypatch.setenv(_WORKER_LOG_ENV, str(log_path))
    created = _count_loaders(monkeypatch)
    deps = ScoringDependencies(
        classifier=Classifier(),
        regression_model=Regression(),
        classifier_transform=_worker_transform,
        terrain_feature_fn=_worker_terrain,
        class_names=("forest", "mountain", "lake"),
        classifier_checkpoint=Path("classifier.pt"),
        regression_checkpoint=Path("regression.pt"),
        classifier_hash="spawn-classifier",
        regression_hash="spawn-regression",
        classifier_preprocess_identity="spawn-transform:v1",
        terrain_feature_identity="spawn-terrain:v1",
        device="cpu",
    )
    options, effective = _loader_options(device="cpu", num_workers=2, dependencies=deps)
    assert effective == 2 and options["persistent_workers"] is True

    total_rows = 8
    result = score_tile_manifest(
        _row_frame(tmp_path, total_rows, "spawn"),
        run_root=tmp_path / "run",
        dependencies=deps,
        device="cpu",
        batch_size=2,
        num_workers=2,
    )

    assert result["counts"]["scored_rows"] == total_rows, result["errors"]
    assert len(created) == 1
    kwargs = created[0]
    assert kwargs["num_workers"] == 2
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == SCORING_PREFETCH_FACTOR
    context = kwargs["multiprocessing_context"]
    start_method = getattr(context, "get_start_method", lambda: context)()
    assert start_method == "spawn"

    events = [line.split() for line in log_path.read_text().splitlines()]
    assert len([event for event in events if event[0] == "transform"]) == total_rows
    assert len([event for event in events if event[0] == "terrain"]) == total_rows
    assert all(event[2] != "main" for event in events), "decode ran in the parent"
    pid_by_worker: dict[str, set[int]] = {}
    for _kind, pid, worker_id, num_workers in events:
        assert num_workers == "2"
        pid_by_worker.setdefault(worker_id, set()).add(int(pid))
    assert set(pid_by_worker) <= {"0", "1"}
    # One process per worker id for the whole invocation: no per-batch respawn.
    assert all(len(pids) == 1 for pids in pid_by_worker.values())
    worker_pids = set().union(*pid_by_worker.values())
    assert 1 <= len(worker_pids) <= 2
    assert os.getpid() not in worker_pids
    assert not _producer_threads()


def test_acquisition_continues_while_classifier_infers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    import scenicdriver.active_learning.scoring as scoring_module

    original_read = scoring_module._read_source
    inference_started = threading.Event()
    read_during_inference = threading.Event()
    reader_threads: set[int] = set()

    def slow_read(**kwargs: object):
        # Lagging acquisition keeps the bounded queue below capacity, so the
        # producer must still be reading when the first batch reaches inference.
        time.sleep(0.02)
        source = original_read(**kwargs)
        reader_threads.add(threading.get_ident())
        if inference_started.is_set():
            read_during_inference.set()
        return source

    monkeypatch.setattr(scoring_module, "_read_source", slow_read)
    overlap: list[bool] = []
    inference_threads: set[int] = set()

    def classifier(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inference_threads.add(threading.get_ident())
        if not overlap:
            inference_started.set()
            overlap.append(read_during_inference.wait(timeout=5.0))
        size = len(batch)
        return (
            np.tile(np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32), (size, 1)),
            np.arange(size * 4, dtype=np.float32).reshape(size, 4) + 1.0,
        )

    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.classifier_predictor = classifier
    result = score_tile_manifest(
        _row_frame(tmp_path, 12, "overlap"),
        run_root=tmp_path / "run",
        dependencies=deps,
        batch_size=2,
    )

    assert result["counts"]["scored_rows"] == 12, result["errors"]
    assert overlap == [True], "source acquisition stalled during classifier inference"
    assert reader_threads.isdisjoint(inference_threads)
    assert not _producer_threads()


def test_torch_heads_receive_identical_device_tensors_between_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    window = {"open": False}
    classifier_outputs: list[tuple[object, object]] = []
    regression_inputs: list[tuple[object, object, object]] = []
    transfers: list[tuple[str, tuple[int, ...]]] = []

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, _batch):
            raise AssertionError("scoring should use the combined forward")

        def forward_with_features(self, batch):
            features = batch.mean(dim=(2, 3))
            logits = features @ self.weight
            classifier_outputs.append((logits, features))
            window["open"] = True
            return logits, features

    class Regression(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, embeddings, terrain, logits):
            window["open"] = False
            regression_inputs.append((embeddings, terrain, logits))
            return embeddings.sum(1) + terrain.sum(1) + logits.sum(1) + self.bias

    def spy(name: str):
        original = getattr(torch.Tensor, name)

        def wrapper(self, *args, **kwargs):
            if window["open"]:
                transfers.append((name, tuple(self.shape)))
            return original(self, *args, **kwargs)

        return wrapper

    for name in ("cpu", "to", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, spy(name))

    deps = ScoringDependencies(
        classifier=Classifier(),
        regression_model=Regression(),
        classifier_transform=lambda image: torch.as_tensor(
            np.asarray(image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1),
        terrain_feature_fn=_worker_terrain,
        class_names=("forest", "mountain", "lake"),
        classifier_hash="device-classifier",
        regression_hash="device-regression",
        classifier_preprocess_identity="device-transform:v1",
        terrain_feature_identity="device-terrain:v1",
        device="cpu",
    )
    result = score_tile_manifest(
        _row_frame(tmp_path, 5, "device"),
        run_root=tmp_path / "run",
        dependencies=deps,
        device="cpu",
        batch_size=2,
    )

    assert result["counts"]["scored_rows"] == 5, result["errors"]
    assert len(classifier_outputs) == len(regression_inputs) == 3
    for (logits, features), (embeddings, terrain, head_logits) in zip(
        classifier_outputs, regression_inputs
    ):
        assert embeddings is features
        assert head_logits is logits
        assert isinstance(terrain, torch.Tensor)
        assert terrain.dtype == torch.float32
        assert tuple(terrain.shape) == (len(features), 6)
    # Only the terrain features may move to the device between forwards.
    assert all(name == "to" and shape[-1] == 6 for name, shape in transfers), transfers
    with np.load(tmp_path / "run" / "feature_embeddings.npz", allow_pickle=False) as arrays:
        expected = np.concatenate(
            [features.detach().numpy() for _logits, features in classifier_outputs]
        )
        assert arrays["embeddings"].dtype == np.float32
        np.testing.assert_allclose(arrays["embeddings"], expected, rtol=1e-6)


def test_injected_predictors_still_receive_numpy_batches(tmp_path: Path) -> None:
    seen: list[tuple[str, type, object]] = []

    def classifier(batch):
        seen.append(("classifier", type(batch), batch.dtype))
        size = len(batch)
        return (
            np.tile(np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32), (size, 1)),
            np.arange(size * 4, dtype=np.float32).reshape(size, 4) + 1.0,
        )

    def regression(embeddings, terrain_features, logits):
        for name, value in (
            ("embeddings", embeddings),
            ("terrain", terrain_features),
            ("logits", logits),
        ):
            seen.append((name, type(value), value.dtype))
        return np.full(len(embeddings), 5.0, dtype=np.float32)

    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.classifier_predictor = classifier
    deps.regression_predictor = regression
    result = score_tile_manifest(
        _row_frame(tmp_path, 3, "numpy"),
        run_root=tmp_path / "run",
        dependencies=deps,
        batch_size=2,
    )

    assert result["counts"]["scored_rows"] == 3, result["errors"]
    assert len(seen) == 8
    assert all(kind is np.ndarray and dtype == np.float32 for _, kind, dtype in seen)


def test_per_row_and_batch_errors_do_not_stop_single_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _count_loaders(monkeypatch)
    manifest = _row_frame(tmp_path, 7, "errors")
    (tmp_path / manifest.loc[1, "terrain_path"]).unlink()
    (tmp_path / manifest.loc[3, "satellite_path"]).write_bytes(b"not a png")
    calls = [0]

    def classifier(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("classifier batch failed")
        size = len(batch)
        return (
            np.tile(np.asarray([[2.0, 0.0, -1.0]], dtype=np.float32), (size, 1)),
            np.arange(size * 4, dtype=np.float32).reshape(size, 4) + 1.0,
        )

    deps = _dependencies({"classifier": 0, "regression": 0})
    deps.classifier_predictor = classifier
    result = score_tile_manifest(
        manifest, run_root=tmp_path / "run", dependencies=deps, batch_size=2
    )

    assert len(created) == 1
    assert calls[0] == 3
    assert result["counts"]["scored_rows"] == 3
    assert result["counts"]["missing_rows"] == 1
    assert result["counts"]["error_rows"] == 3
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    assert candidates.loc[1, "score_status"] == "missing"
    assert candidates.loc[3, "score_status"] != "scored"
    batch_failures = candidates["error"].fillna("").str.contains("classifier batch failed")
    assert int(batch_failures.sum()) == 2
    assert not _producer_threads()


def test_loader_failure_stops_and_joins_source_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scenicdriver.active_learning.scoring as scoring_module

    original_collate = scoring_module._collate_scoring_items
    collated = [0]

    def failing_collate(items):
        collated[0] += 1
        if collated[0] == 2:
            raise RuntimeError("collate boom")
        return original_collate(items)

    monkeypatch.setattr(scoring_module, "_collate_scoring_items", failing_collate)
    with pytest.raises(RuntimeError, match="collate boom"):
        score_tile_manifest(
            _row_frame(tmp_path, 20, "cleanup"),
            run_root=tmp_path / "run",
            dependencies=_dependencies({"classifier": 0, "regression": 0}),
            batch_size=2,
        )
    # The producer is blocked on a full bounded queue when the consumer fails;
    # cancellation must release and join it before the error propagates.
    assert not _producer_threads()
    assert not (tmp_path / "run" / "candidate_pool.csv").exists()
