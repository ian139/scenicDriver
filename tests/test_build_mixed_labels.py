from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_mixed_labels_writes_audit_columns_and_retains_heuristics(
    tmp_path: Path,
) -> None:
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


def test_build_mixed_labels_excludes_skipped_and_retains_scored_and_heuristic_rows(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source,score_status\n"
        "raw/images/satellite/z14/region/path1.png,4.0,7,heuristic,scored\n"
        "raw/images/satellite/z14/region/path2.png,5.0,8,heuristic,scored\n"
        "raw/images/satellite/z14/region/path3.png,6.0,9,heuristic,scored\n"
        "raw/images/satellite/z14/region/path4.png,,10,heuristic,scored\n"
        "raw/images/satellite/z14/region/path5.png,2.0,11,heuristic,unusable\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,skip,timestamp\n"
        "raw/images/satellite/z14/region/path1.png,alice,8.0,False,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/path2.png,bob,,True,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )

    res = subprocess.run(
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
        capture_output=True,
        text=True,
    )

    assert "Excluded unusable tiles: 1" in res.stdout
    assert "Excluded unscored base tiles: 1" in res.stdout
    assert "Excluded unusable base tiles: 1" in res.stdout

    mixed = pd.read_csv(output).sort_values("image_path").reset_index(drop=True)
    paths = mixed["image_path"].tolist()

    assert "raw/images/satellite/z14/region/path2.png" not in paths
    assert "raw/images/satellite/z14/region/path4.png" not in paths
    assert "raw/images/satellite/z14/region/path5.png" not in paths
    assert len(mixed) == 2

    override_row = mixed.loc[mixed["image_path"].str.endswith("path1.png")].iloc[0]
    heuristic_row = mixed.loc[mixed["image_path"].str.endswith("path3.png")].iloc[0]

    assert override_row["label_source"] == "human_override"
    assert float(override_row["scenic_score"]) == 8.0

    assert heuristic_row["label_source"] == "heuristic"
    assert float(heuristic_row["scenic_score"]) == 6.0


def test_build_mixed_labels_latest_record_deduplication_for_skipped_status(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source\n"
        "raw/images/satellite/z14/region/resurrected.png,3.0,1,heuristic\n"
        "raw/images/satellite/z14/region/skipped_later.png,4.0,2,heuristic\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,skip,timestamp\n"
        "raw/images/satellite/z14/region/resurrected.png,alice,,True,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/resurrected.png,alice,9.0,False,2026-01-02T00:00:00Z\n"
        "raw/images/satellite/z14/region/skipped_later.png,bob,7.0,False,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/skipped_later.png,bob,,True,2026-01-02T00:00:00Z\n",
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
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    mixed = pd.read_csv(output)
    paths = mixed["image_path"].tolist()

    assert "raw/images/satellite/z14/region/skipped_later.png" not in paths
    assert "raw/images/satellite/z14/region/resurrected.png" in paths

    res_row = mixed.loc[mixed["image_path"].str.endswith("resurrected.png")].iloc[0]
    assert res_row["label_source"] == "human_override"
    assert float(res_row["scenic_score"]) == 9.0


def test_build_mixed_labels_multi_annotator_skip_excludes_image(
    tmp_path: Path,
) -> None:
    heuristic = tmp_path / "heuristic.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "mixed.csv"

    heuristic.write_text(
        "image_path,scenic_score,class_id,label_source\n"
        "raw/images/satellite/z14/region/conflict.png,5.0,1,heuristic\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "image_path,annotator_id,scenic_human,skip,timestamp\n"
        "raw/images/satellite/z14/region/conflict.png,alice,8.5,False,2026-01-01T00:00:00Z\n"
        "raw/images/satellite/z14/region/conflict.png,bob,,True,2026-01-01T00:00:00Z\n",
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
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    mixed = pd.read_csv(output)
    assert len(mixed) == 0
