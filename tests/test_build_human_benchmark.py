from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.annotation.build_human_benchmark import _apply_geographic_splits


def test_applies_fixed_geographic_splits_without_reassignment(tmp_path: Path) -> None:
    tile = pd.DataFrame(
        [
            {"image_path": "a.png", "scenic_human_mean": 4.0},
            {"image_path": "b.png", "scenic_human_mean": 7.0},
            {"image_path": "c.png", "scenic_human_mean": 8.0},
        ]
    )
    assignments = tmp_path / "geographic_splits.csv"
    pd.DataFrame(
        [
            {"image_path": "a.png", "split": "train", "geographic_block": "a"},
            {"image_path": "b.png", "split": "validation", "geographic_block": "b"},
            {"image_path": "c.png", "split": "test", "geographic_block": "c"},
        ]
    ).to_csv(assignments, index=False)

    result = _apply_geographic_splits(tile, assignments)

    assert result["split"].tolist() == ["train", "val", "test"]
    assert result["geographic_block"].tolist() == ["a", "b", "c"]


def test_fixed_geographic_splits_fail_closed_on_missing_assignment(tmp_path: Path) -> None:
    tile = pd.DataFrame(
        [
            {"image_path": "a.png", "scenic_human_mean": 4.0},
            {"image_path": "b.png", "scenic_human_mean": 7.0},
        ]
    )
    assignments = tmp_path / "geographic_splits.csv"
    pd.DataFrame([{"image_path": "a.png", "split": "train"}]).to_csv(
        assignments, index=False
    )

    with pytest.raises(ValueError, match="no assignment"):
        _apply_geographic_splits(tile, assignments)
