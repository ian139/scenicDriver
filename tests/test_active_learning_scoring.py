from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from src.active_learning.finalize import finalize_stage1
from src.active_learning.scoring import (
    ScoringDependencies,
    normalized_class_entropy,
    resolve_active_regression_checkpoint,
    score_tile_manifest,
)


def _png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 4), color=color).save(path, format="PNG")


def _manifest(root: Path) -> pd.DataFrame:
    rows = []
    for index, color in enumerate(((40, 120, 70), (70, 90, 180), (160, 130, 80))):
        satellite = root / f"sat-{index}.png"
        terrain = root / f"terrain-{index}.png"
        _png(satellite, color)
        _png(terrain, (120, 120, 120))
        rows.append(
            {
                "image_path": f"images/satellite/z14/fixture/{100 + index}_200.png",
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

    def terrain(_terrain: Image.Image, _satellite: Image.Image) -> dict[str, object]:
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

    def regression(embeddings: np.ndarray, terrain_features: np.ndarray, logits: np.ndarray) -> np.ndarray:
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


def test_resume_skips_unchanged_rows_and_preserves_error_state(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    counters = {"classifier": 0, "regression": 0}
    dependencies = _dependencies(counters)
    first = score_tile_manifest(manifest, run_root=tmp_path / "run", dependencies=dependencies, batch_size=2)
    assert first["counts"] == {
        "manifest_rows": 3,
        "scored_rows": 2,
        "selector_eligible_rows": 2,
        "missing_rows": 1,
        "error_rows": 0,
        "cache_hits": 0,
        "cache_misses": 3,
    }
    assert counters == {"classifier": 1, "regression": 1}
    second = score_tile_manifest(manifest, run_root=tmp_path / "run", dependencies=dependencies, batch_size=2)
    assert second["counts"]["cache_hits"] == 2
    assert counters == {"classifier": 1, "regression": 1}
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    assert len(candidates) == 3
    assert candidates.loc[candidates["score_status"] == "missing", "selector_eligible"].eq(False).all()
    assert candidates.loc[0, "image_path"] == "images/satellite/z14/fixture/100_200.png"


def test_model_prediction_stays_separate_from_weak_and_human_names(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path).iloc[:1]
    result = score_tile_manifest(manifest, run_root=tmp_path / "run", dependencies=_dependencies({"classifier": 0, "regression": 0}))
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    row = candidates.iloc[0]
    assert "scenic_human" not in candidates.columns
    assert row["label_source"] == "active_regression_prediction"
    assert row["scenic_score"] == row["scenic_score_heuristic"] == row["heuristic_score"]
    assert row["regression_prediction"] != row["scenic_score"]
    assert result["models"]["label_semantics"].startswith("regression_prediction")


def test_normalized_entropy_is_class_entropy_and_bounded() -> None:
    assert normalized_class_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalized_class_entropy([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert 0.0 <= normalized_class_entropy([0.2, 0.3, 0.5]) <= 1.0
    assert 0.0 <= normalized_class_entropy([np.nan, -2.0, 0.0]) <= 1.0


def test_embedding_npz_shape_and_stable_row_indices(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path).iloc[:2]
    result = score_tile_manifest(manifest, run_root=tmp_path / "run", dependencies=_dependencies({"classifier": 0, "regression": 0}), lsh_seed=7, lsh_bits=8)
    with np.load(tmp_path / "run" / "feature_embeddings.npz", allow_pickle=False) as arrays:
        assert arrays["embeddings"].shape == (2, 4)
        assert arrays["embeddings"].dtype == np.float32
        assert arrays["row_indices"].tolist() == [0, 1]
    candidates = pd.read_csv(tmp_path / "run" / "candidate_pool.csv")
    assert candidates["embedding_row_index"].tolist() == [0, 1]
    assert candidates["embedding_cluster_id"].notna().all()
    assert result["embedding"]["dimension"] == 4


def test_malformed_registry_fails_closed(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": "missing.pt"}}), encoding="utf-8")
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