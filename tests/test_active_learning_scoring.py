from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from src.active_learning.common import sha256_file
from src.active_learning.finalize import finalize_stage1
from src.active_learning.scoring import (
    DEFAULT_SCORING_BATCH_SIZE,
    DEFAULT_SCORING_NUM_WORKERS,
    ScoringDependencies,
    _call_classifier,
    _loader_options,
    normalized_class_entropy,
    resolve_active_regression_checkpoint,
    run_active_learning_scoring,
    score_tile_manifest,
)


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

    def terrain(_terrain: Image.Image, _satellite: Image.Image, *, valid_mask: np.ndarray) -> dict[str, object]:
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
        device="cpu",
    )

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

    def terrain(_terrain: Image.Image, _satellite: Image.Image, *, valid_mask: np.ndarray) -> dict[str, object]:
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
        device="cpu",
    )
    result = score_tile_manifest(
        manifest_df, run_root=tmp_path / "run", dependencies=deps, batch_size=2
    )
    assert result["state"]["ready_for_selection"] is True
    assert recorded_batch_sizes == [2, 2, 1]
    assert all(size <= 2 for size in recorded_batch_sizes)


def test_uncached_pending_payload_high_water_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.active_learning.scoring as scoring_module

    total_rows = 9
    batch_size = 2
    rows = []
    row_payload_bytes: list[int] = []
    for index in range(total_rows):
        satellite = tmp_path / f"bounded_sat_{index}.png"
        terrain = tmp_path / f"bounded_ter_{index}.png"
        _png(satellite, (50 + index, 150, 80))
        _png(terrain, (120, 120, 120))
        row_payload_bytes.append(satellite.stat().st_size + terrain.stat().st_size)
        rows.append(
            {
                "image_path": f"images/satellite/z14/bounded/{index}.png",
                "provider": "mapbox_v1",
                "region": "fixture",
                "z": 14,
                "x": index,
                "y": 200,
                "lat": 40.0 + index,
                "lon": -70.0,
                "satellite_path": satellite.name,
                "terrain_path": terrain.name,
                "satellite_present": True,
                "terrain_present": True,
            }
        )

    observed_payload_bytes: list[int] = []
    original_dataset = scoring_module._ScoringDataset

    class ObservedDataset(original_dataset):
        def __init__(self, inputs: list[object], **kwargs: object) -> None:
            observed_payload_bytes.append(
                sum(
                    len(source.payload or b"")
                    for item in inputs
                    for source in (item.satellite, item.terrain)
                )
            )
            super().__init__(inputs, **kwargs)

    monkeypatch.setattr(scoring_module, "_ScoringDataset", ObservedDataset)
    result = score_tile_manifest(
        pd.DataFrame(rows),
        run_root=tmp_path / "bounded-run",
        dependencies=_dependencies({"classifier": 0, "regression": 0}),
        batch_size=batch_size,
    )

    assert result["counts"]["scored_rows"] == total_rows, result["errors"]
    assert len(observed_payload_bytes) == (total_rows + batch_size - 1) // batch_size
    largest_row_payload = max(observed_payload_bytes)
    assert largest_row_payload < sum(row_payload_bytes)
    assert largest_row_payload <= max(row_payload_bytes) * batch_size


def test_scoring_rejects_invalid_run_names(tmp_path: Path) -> None:
    from src.active_learning.scoring import run_active_learning_scoring

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
    from src.active_learning.scoring import _validate_regression_checkpoint
    from src.scenic_scorer.regression import ScenicRegressionModel

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
    from src.active_learning.scoring import resolve_active_regression_checkpoint

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

    def terrain_fn(_terrain: Image.Image, satellite: Image.Image, *, valid_mask: np.ndarray) -> dict[str, object]:
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
    import src.active_learning.scoring as scoring_module

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
    from src.active_learning.water import compute_satellite_water_fraction

    satellite = np.full((100, 100, 3), [15, 30, 100], dtype=np.uint8)
    land = np.indices((100, 40)).sum(axis=0) % 2
    satellite[:, 60:] = np.where(land[..., None], 255, 0)
    assert compute_satellite_water_fraction(satellite) == pytest.approx(0.6, abs=0.02)


def test_water_geometry_counts_canonical_black_sea_with_sparse_land():
    from src.active_learning.water import compute_satellite_water_fraction

    satellite = np.zeros((10, 10, 3), dtype=np.uint8)
    land = np.zeros((10, 10), dtype=bool)
    land[0, 0] = True
    satellite[0, 0] = [40, 120, 70]
    assert compute_satellite_water_fraction(satellite, land_mask=land) == pytest.approx(0.99)
    land[:] = False
    assert compute_satellite_water_fraction(satellite, land_mask=land) == 1.0


def test_flat_blue_water_with_clouds_retains_local_water_detection():
    from src.active_learning.water import compute_satellite_water_fraction

    satellite = np.full((100, 100, 3), [15, 30, 100], dtype=np.uint8)
    satellite[30:70, 30:70] = 255
    satellite[10:12, 10:12] = 255  # small whitecap/cloud
    assert compute_satellite_water_fraction(satellite) == pytest.approx(0.84, abs=0.03)


@pytest.mark.parametrize("mask,error", [
    (np.ones((2, 2), dtype=np.uint8), TypeError),
    (np.ones((1, 2), dtype=bool), ValueError),
])
def test_water_rejects_invalid_land_masks(mask, error):
    from src.active_learning.water import compute_satellite_water_fraction

    with pytest.raises(error):
        compute_satellite_water_fraction(np.zeros((2, 2, 3)), land_mask=mask)


@pytest.mark.parametrize("provider", [None, "", "  ", "mapbox_v1"])
def test_scoring_coast_requires_attribution_and_uses_land_only_features(tmp_path, provider):
    from src.terrain.features import compute_terrain_features

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
    from src.data_pipeline.source_contracts import compute_acquisition_tile_identity

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
