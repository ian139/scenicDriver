from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np

import pytest
import torch

from src.heuristics import labeler

from src.scenic_scorer import regression
from src.scenic_scorer.regression import ScenicRegressionModel


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_marker, (str(self.marker),)


def _write_checkpoint(path: Path, **extra: object) -> None:
    model = ScenicRegressionModel(vit_dim=2, terrain_dim=1, num_classes=3)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vit_dim": 2,
            "terrain_dim": 1,
            "num_classes": 3,
            **extra,
        },
        path,
    )


def test_load_learned_regressor_supports_numpy_rng_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint, rng_state={"numpy": np.random.get_state()})
    warnings: list[str] = []

    model = labeler._load_learned_regressor(
        learned_regression_ckpt=checkpoint,
        device="cpu",
        warnings=warnings,
    )

    assert model is not None
    assert warnings == [f"Using learned regression checkpoint: {checkpoint}"]


def test_load_learned_regressor_rejects_pickle_code_execution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malicious.pt"
    marker = tmp_path / "executed"
    _write_checkpoint(checkpoint, payload=_MaliciousPayload(marker))

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        labeler._load_learned_regressor(
            learned_regression_ckpt=checkpoint,
            device="cpu",
            warnings=[],
        )

    assert not marker.exists()


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_resolve_device_auto_priority_and_explicit_requests(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(regression.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        regression.torch.backends.mps, "is_available", lambda: mps_available
    )

    assert regression.resolve_device(None) == expected
    assert regression.resolve_device("auto") == expected
    assert regression.resolve_device("cpu") == "cpu"
    # Explicit CUDA stays explicit even when unavailable; fallback is caller policy.
    assert regression.resolve_device("cuda") == "cuda"


def _write_dataset(path: Path, n: int = 20, **overrides: object) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    arrays: dict[str, np.ndarray] = {
        "vit_embeddings": rng.normal(size=(n, 2)).astype(np.float32),
        "terrain_features": rng.normal(size=(n, 1)).astype(np.float32),
        "class_logits": rng.normal(size=(n, 3)).astype(np.float32),
        "scenic_scores": rng.uniform(0.0, 10.0, size=(n,)).astype(np.float32),
        "sample_weights": rng.uniform(0.5, 4.0, size=(n,)).astype(np.float32),
    }
    arrays.update(overrides)  # type: ignore[arg-type]
    np.savez(path, **arrays)
    return arrays


def test_dataset_loads_rank1_scores_as_column_targets(tmp_path: Path) -> None:
    data_path = tmp_path / "data.npz"
    _write_dataset(data_path, n=4)

    dataset = regression.ScenicScoreDataset(data_path)

    assert len(dataset) == 4
    vit, terrain, logits, score, weight = dataset[1]
    assert vit.shape == (2,) and terrain.shape == (1,) and logits.shape == (3,)
    assert score.shape == (1,) and weight.shape == (1,)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"scenic_scores": np.ones((4, 1), dtype=np.float32)}, "scenic_scores must be rank 1"),
        ({"scenic_scores": np.float32(1.0)}, "scenic_scores must be rank 1"),
        ({"scenic_scores": np.ones((3,), dtype=np.float32)}, "inconsistent lengths"),
        ({"scenic_scores": np.array([1.0, np.nan, 2.0, 3.0])}, "scenic_scores contains non-finite"),
        ({"scenic_scores": np.array([1.0, 1e39, 2.0, 3.0])}, "scenic_scores contains non-finite"),
        ({"scenic_scores": np.array(["a", "b", "c", "d"])}, "scenic_scores must be numeric"),
        ({"vit_embeddings": np.ones((4,), dtype=np.float32)}, "vit_embeddings must be rank 2"),
        ({"terrain_features": np.ones((4, 1, 1), dtype=np.float32)}, "terrain_features must be rank 2"),
        ({"class_logits": np.ones((4, 0), dtype=np.float32)}, "class_logits must have at least one"),
        ({"class_logits": np.ones((5, 3), dtype=np.float32)}, "inconsistent lengths"),
        ({"terrain_features": np.array([[0.0], [np.inf], [0.0], [0.0]])}, "terrain_features contains non-finite"),
        ({"vit_embeddings": np.full((4, 2), 1e300)}, "vit_embeddings contains non-finite"),
        ({"class_logits": np.full((4, 3), True)}, "class_logits must be numeric"),
    ],
)
def test_dataset_rejects_malformed_arrays(
    tmp_path: Path, overrides: dict[str, np.ndarray], match: str
) -> None:
    data_path = tmp_path / "data.npz"
    _write_dataset(data_path, n=4, **overrides)

    with pytest.raises(ValueError, match=match):
        regression.ScenicScoreDataset(data_path)


@pytest.mark.parametrize("use_sample_weights", [True, False])
def test_validation_loss_is_sample_aggregated_across_uneven_batches(
    tmp_path: Path, use_sample_weights: bool
) -> None:
    data_path = tmp_path / "data.npz"
    output_path = tmp_path / "model.pt"
    arrays = _write_dataset(data_path, n=20)
    seed = 7

    regression.train_regression_model(
        data_path,
        output_path,
        epochs=1,
        batch_size=2,
        val_split=0.15,
        seed=seed,
        device="cpu",
        use_sample_weights=use_sample_weights,
        num_workers=0,
    )

    # Mirror the module's deterministic split: 17 train / 3 val -> val batches of 2 and 1.
    np.random.seed(seed)
    indices = np.arange(20)
    np.random.shuffle(indices)
    val_idx = indices[17:]
    checkpoint = torch.load(output_path, weights_only=False)
    model = ScenicRegressionModel(vit_dim=2, terrain_dim=1, num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        pred = model(
            torch.from_numpy(arrays["vit_embeddings"][val_idx]),
            torch.from_numpy(arrays["terrain_features"][val_idx]),
            torch.from_numpy(arrays["class_logits"][val_idx]),
        ).numpy().reshape(-1).astype(np.float64)
    squared_error = (pred - arrays["scenic_scores"][val_idx].astype(np.float64)) ** 2
    weights = (
        arrays["sample_weights"][val_idx].astype(np.float64)
        if use_sample_weights
        else np.ones_like(squared_error)
    )
    expected = float((squared_error * weights).sum() / weights.sum())

    assert checkpoint["best_val_loss"] == pytest.approx(expected, rel=1e-5)
    assert checkpoint["history"]["val_loss"] == [pytest.approx(expected, rel=1e-5)]
