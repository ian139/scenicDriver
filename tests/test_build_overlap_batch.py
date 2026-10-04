from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.annotation.build_overlap_batch import main


def _run(tmp_path: Path, annotations_csv: Path, *extra: str) -> tuple[pd.DataFrame, dict]:
    output_csv = tmp_path / "overlap.csv"
    argv = [
        "build_overlap_batch.py",
        "--annotations-csv",
        str(annotations_csv),
        "--labels-csv",
        str(tmp_path / "absent_labels.csv"),
        "--target-annotator",
        "bob",
        "--sample-size",
        "10",
        "--output-csv",
        str(output_csv),
        *extra,
    ]
    with patch.object(sys, "argv", argv):
        main()
    summary = json.loads(
        output_csv.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    return pd.read_csv(output_csv), summary


def test_auto_source_ignores_missing_and_blank_annotator_identities(
    tmp_path: Path,
) -> None:
    annotations_csv = tmp_path / "labels_human.csv"
    rows = [
        {"image_path": f"anon_{i}.png", "scenic_human": 5.0, "annotator_id": None}
        for i in range(4)
    ]
    rows += [
        {"image_path": f"blank_{i}.png", "scenic_human": 5.0, "annotator_id": "  "}
        for i in range(4)
    ]
    rows += [
        {"image_path": "alice_1.png", "scenic_human": 6.0, "annotator_id": "alice"},
        {"image_path": "alice_2.png", "scenic_human": 7.0, "annotator_id": "alice"},
        {"image_path": "alice_2.png", "scenic_human": 7.0, "annotator_id": "bob"},
    ]
    pd.DataFrame(rows).to_csv(annotations_csv, index=False)

    batch, summary = _run(tmp_path, annotations_csv, "--no-stratify-by-class")

    assert summary["source_annotator"] == "alice"
    assert batch["source_annotator"].tolist() == ["alice"]
    assert batch["image_path"].tolist() == ["alice_1.png"]
    assert summary["candidate_tiles"] == 1


def test_explicit_nan_source_has_no_candidates_from_missing_identities(
    tmp_path: Path,
) -> None:
    annotations_csv = tmp_path / "labels_human.csv"
    pd.DataFrame(
        [
            {"image_path": "anon.png", "scenic_human": 5.0, "annotator_id": None},
            {"image_path": "alice.png", "scenic_human": 6.0, "annotator_id": "alice"},
        ]
    ).to_csv(annotations_csv, index=False)

    with pytest.raises(ValueError, match="No overlap candidates"):
        _run(tmp_path, annotations_csv, "--source-annotator", "nan")
