from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import json
import sys
from unittest.mock import patch

from scripts.annotation.build_human_benchmark import (
    _apply_geographic_splits,
    _filter_annotations_by_splits,
    _prepare_annotations,
    main,
)


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


def test_fixed_geographic_splits_fail_closed_on_missing_assignment(
    tmp_path: Path,
) -> None:
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


def test_filter_annotations_by_splits_drops_non_run_rows(tmp_path: Path) -> None:
    ann = pd.DataFrame(
        [
            {
                "image_path": "current_run_1.png",
                "scenic_human": 5.0,
                "annotator_id": "ann1",
            },
            {
                "image_path": "current_run_2.png",
                "scenic_human": 4.0,
                "annotator_id": "ann1",
            },
            {
                "image_path": "old_run_1.png",
                "scenic_human": 3.0,
                "annotator_id": "ann2",
            },
            {
                "image_path": "old_run_2.png",
                "scenic_human": 2.0,
                "annotator_id": "ann1",
            },
        ]
    )
    assignments = tmp_path / "geographic_splits.csv"
    pd.DataFrame(
        [
            {"image_path": "current_run_1.png", "split": "train"},
            {"image_path": "current_run_2.png", "split": "val"},
        ]
    ).to_csv(assignments, index=False)

    retained, dropped_count = _filter_annotations_by_splits(ann, assignments)

    assert dropped_count == 2
    assert retained["image_path"].tolist() == ["current_run_1.png", "current_run_2.png"]


def test_build_human_benchmark_end_to_end_cumulative_annotations(
    tmp_path: Path,
) -> None:
    annotations_csv = tmp_path / "labels_human.csv"
    pd.DataFrame(
        [
            {"image_path": "current_1.png", "scenic_human": 5.0, "annotator_id": "ian"},
            {"image_path": "current_2.png", "scenic_human": 4.0, "annotator_id": "ian"},
            {"image_path": "old_1.png", "scenic_human": 3.0, "annotator_id": "ian"},
        ]
    ).to_csv(annotations_csv, index=False)

    splits_csv = tmp_path / "geographic_splits.csv"
    pd.DataFrame(
        [
            {"image_path": "current_1.png", "split": "train"},
            {"image_path": "current_2.png", "split": "val"},
        ]
    ).to_csv(splits_csv, index=False)

    output_dir = tmp_path / "output"
    run_name = "test_run"

    test_args = [
        "build_human_benchmark.py",
        "--annotations-csv",
        str(annotations_csv),
        "--geographic-splits-csv",
        str(splits_csv),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]

    with patch.object(sys, "argv", test_args):
        main()

    run_dir = output_dir / run_name
    assert run_dir.exists()

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["non_run_rows_dropped"] == 1
    assert summary["rows_after_filter_and_dedupe"] == 2
    assert summary["unique_tiles"] == 2

    split_df = pd.read_csv(run_dir / "benchmark_split.csv")
    assert set(split_df["image_path"]) == {"current_1.png", "current_2.png"}


@pytest.mark.parametrize(
    "invalid_run_name", ["../invalid", "foo/bar", "/abs/path", ".."]
)
def test_build_human_benchmark_rejects_path_traversal_run_name(
    tmp_path: Path, invalid_run_name: str
) -> None:
    annotations_csv = tmp_path / "labels_human.csv"
    pd.DataFrame(
        [{"image_path": "a.png", "scenic_human": 5.0, "annotator_id": "ian"}]
    ).to_csv(annotations_csv, index=False)

    splits_csv = tmp_path / "geographic_splits.csv"
    pd.DataFrame([{"image_path": "a.png", "split": "train"}]).to_csv(
        splits_csv, index=False
    )

    test_args = [
        "build_human_benchmark.py",
        "--annotations-csv",
        str(annotations_csv),
        "--geographic-splits-csv",
        str(splits_csv),
        "--output-dir",
        str(tmp_path),
        "--run-name",
        invalid_run_name,
    ]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(ValueError, match="run_name"):
            main()


def test_prepare_annotations_rejects_missing_and_literal_nan_annotator_ids(
    tmp_path: Path,
) -> None:
    annotations_csv = tmp_path / "labels_human.csv"
    df = pd.DataFrame(
        [
            {"image_path": "a.png", "scenic_human": 5.0, "annotator_id": "alice"},
            {"image_path": "b.png", "scenic_human": 6.0, "annotator_id": None},
            {"image_path": "c.png", "scenic_human": 7.0, "annotator_id": ""},
            {"image_path": "d.png", "scenic_human": 8.0, "annotator_id": "   "},
            {"image_path": "e.png", "scenic_human": 9.0, "annotator_id": "nan"},
            {"image_path": "f.png", "scenic_human": 4.0, "annotator_id": "NaN"},
        ]
    )
    df.to_csv(annotations_csv, index=False)

    ann, total = _prepare_annotations(annotations_csv)
    assert total == 6
    assert len(ann) == 1
    assert ann["annotator_id"].tolist() == ["alice"]
    assert "nan" not in ann["annotator_id"].tolist()
    assert "unknown" not in ann["annotator_id"].tolist()
