from __future__ import annotations

import pytest

from src.scenic_scorer import regression


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
    monkeypatch.setattr(regression.torch.backends.mps, "is_available", lambda: mps_available)

    assert regression.resolve_device(None) == expected
    assert regression.resolve_device("auto") == expected

