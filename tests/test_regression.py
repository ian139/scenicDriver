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
def test_resolve_device_auto_priority(
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
