from __future__ import annotations

import pandas as pd

from src.active_learning.finalize import finalize_stage1
from src.active_learning.selection import audit_geographic_leakage, select_candidates


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_path": "z14/100_100.png", "region": "west", "z": 14, "x": 100, "y": 100, "heuristic_score": 0.1, "model_score": 0.9, "cluster_id": "a"},
            {"image_path": "z14/102_100.png", "region": "west", "z": 14, "x": 102, "y": 100, "heuristic_score": 0.8, "model_score": 0.2, "cluster_id": "b"},
            {"image_path": "z14/110_100.png", "region": "south", "z": 14, "x": 110, "y": 100, "heuristic_score": 0.4, "model_score": 0.5, "cluster_id": "c"},
            {"image_path": "z14/120_100.png", "region": "south", "z": 14, "x": 120, "y": 100, "heuristic_score": 0.5, "model_score": 0.5, "cluster_id": "d"},
        ]
    )


def test_selection_is_deterministic_and_does_not_fabricate_uncertainty() -> None:
    candidates = _candidates()
    first = select_candidates(candidates, batch_size=3, seed=11, run_name="fixture")
    second = select_candidates(candidates, batch_size=3, seed=11, run_name="fixture")
    pd.testing.assert_frame_equal(first, second)
    assert first["uncertainty_score"].eq(0).all()
    assert first["uncertainty_observed"].eq(False).all()
    assert first["selection_reason"].notna().all()
    assert first["batch_id"].nunique() == 1


def test_leakage_audit_detects_duplicate_and_adjacent_cross_split_tiles() -> None:
    split = pd.DataFrame(
        [
            {"image_path": "a.png", "z": 14, "x": 10, "y": 10, "split": "train"},
            {"image_path": "a-copy.png", "z": 14, "x": 10, "y": 10, "split": "test"},
            {"image_path": "b.png", "z": 14, "x": 11, "y": 10, "split": "val"},
        ]
    )
    report = audit_geographic_leakage(split)
    assert report["valid"] is False
    assert report["duplicate_cross_split"] is True
    assert report["adjacent_cross_split"] is True


def test_finalizer_fails_closed_for_incomplete_run(tmp_path) -> None:
    handoff = finalize_stage1(tmp_path, run_name="fixture")
    assert handoff["ready_for_stage2"] is False
    assert handoff["blockers"]
    assert (tmp_path / "stage1_handoff.json").exists()
