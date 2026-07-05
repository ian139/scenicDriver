from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_mixed_labels_writes_audit_columns_and_retains_heuristics(tmp_path: Path) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source\n"
        "raw/images/satellite/z14/region/1_2.png,4.0,7,heuristic\n"
        "raw/images/satellite/z14/region/3_4.png,6.0,8,heuristic\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,timestamp\n"
        "raw/images/satellite/z14/region/1_2.png,a,8.0,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/1_2.png,b,6.0,2026-01-01T00:01:00Z\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/modeling/build_mixed_labels.py"),
            "--heuristic-labels",
            str(heuristic),
            "--annotations-csv",
            str(annotations),
            "--output",
            str(output),
            "--aggregate",
            "mean",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    mixed = pd.read_csv(output).sort_values("image_path").reset_index(drop=True)
    human_row = mixed.loc[mixed["image_path"].str.endswith("1_2.png")].iloc[0]
    heuristic_row = mixed.loc[mixed["image_path"].str.endswith("3_4.png")].iloc[0]

    assert human_row["label_source"] == "human_override"
    assert human_row["scenic_score"] == 7.0
    assert human_row["scenic_human_mean"] == 7.0
    assert human_row["scenic_human_median"] == 7.0
    assert human_row["human_annotation_count"] == 2
    assert human_row["human_annotator_count"] == 2
    assert round(float(human_row["scenic_human_std"]), 6) == round(2**0.5, 6)

    assert heuristic_row["label_source"] == "heuristic"
    assert heuristic_row["scenic_score"] == 6.0
