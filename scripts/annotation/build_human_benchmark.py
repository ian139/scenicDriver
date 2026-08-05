"""
Build a deterministic human benchmark split and agreement report.

Example:
  uv run python scripts/annotation/build_human_benchmark.py \
    --annotations-csv data/raw/labels_human.csv \
    --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
    --output-dir data/processed/regression \
    --run-name masswhites_human_benchmark_v1 \
    --val-frac 0.2 \
    --test-frac 0.2 \
    --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active_learning.common import sha256_file, validate_run_name  # noqa: E402


BOOL_TRUE = {"1", "true", "yes", "y", "t", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build human benchmark split and agreement stats"
    )
    parser.add_argument(
        "--annotations-csv", type=Path, default=Path("data/raw/labels_human.csv")
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/processed/regression/labels_masswhites_z14_mixed5000.csv"),
        help="Optional labels file for class_id/heuristic joins",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/regression")
    )
    parser.add_argument("--run-name", type=str, default="human_benchmark_v1")
    parser.add_argument(
        "--geographic-splits-csv",
        type=Path,
        required=True,
        help="Fixed leakage-resistant image_path/split assignments",
    )
    return parser.parse_args()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in BOOL_TRUE


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 2:
        return None
    a_std = float(a.std(ddof=0))
    b_std = float(b.std(ddof=0))
    if a_std == 0.0 or b_std == 0.0:
        return None
    return float(a.corr(b))


def _prepare_annotations(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotations CSV: {path}")
    ann = pd.read_csv(path)
    total_rows = len(ann)

    required = {"image_path", "scenic_human", "annotator_id"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Missing required annotation columns: {sorted(missing)}")

    ann["image_path"] = ann["image_path"].astype(str).str.strip()
    ann["annotator_id"] = ann["annotator_id"].fillna("").astype(str).str.strip()
    ann["scenic_human"] = pd.to_numeric(ann["scenic_human"], errors="coerce")

    if "skip" in ann.columns:
        ann["skip"] = ann["skip"].map(_to_bool)
    else:
        ann["skip"] = False

    if "timestamp" in ann.columns:
        ann["_timestamp"] = pd.to_datetime(ann["timestamp"], errors="coerce", utc=True)
    else:
        ann["_timestamp"] = pd.NaT
    ann["_row_id"] = np.arange(len(ann))

    ann = ann.loc[
        ~ann["skip"]
        & ann["scenic_human"].notna()
        & ann["image_path"].ne("")
        & ann["annotator_id"].ne("")
        & ann["annotator_id"].str.lower().ne("nan")
    ].copy()
    ann = ann.sort_values(["_timestamp", "_row_id"])
    ann = ann.drop_duplicates(subset=["image_path", "annotator_id"], keep="last")
    ann = ann.drop(columns=["_row_id"])
    return ann, total_rows


def _filter_annotations_by_splits(
    ann: pd.DataFrame, splits_path: Path
) -> tuple[pd.DataFrame, int]:
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing geographic splits CSV: {splits_path}")
    assignments = pd.read_csv(splits_path)
    required = {"image_path", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Geographic splits CSV missing columns: {sorted(missing)}")
    valid_paths = set(assignments["image_path"].astype(str).str.strip())
    retained = ann.loc[ann["image_path"].isin(valid_paths)].copy()
    dropped_count = len(ann) - len(retained)
    return retained, dropped_count


def _build_tile_table(ann: pd.DataFrame, labels_path: Path | None) -> pd.DataFrame:
    tile = (
        ann.groupby("image_path", as_index=False)
        .agg(
            scenic_human_mean=("scenic_human", "mean"),
            scenic_human_median=("scenic_human", "median"),
            scenic_human_std=("scenic_human", "std"),
            scenic_human_min=("scenic_human", "min"),
            scenic_human_max=("scenic_human", "max"),
            n_annotations=("scenic_human", "size"),
            n_annotators=("annotator_id", "nunique"),
        )
        .sort_values("image_path")
    )
    tile["scenic_human_std"] = tile["scenic_human_std"].fillna(0.0)
    tile["annotator_variance"] = tile["scenic_human_std"] ** 2
    tile["annotator_range"] = tile["scenic_human_max"] - tile["scenic_human_min"]

    if labels_path is not None and labels_path.exists():
        labels = pd.read_csv(labels_path)
        keep_cols = [
            c
            for c in [
                "image_path",
                "class_id",
                "scenic_score_heuristic",
                "label_source",
            ]
            if c in labels.columns
        ]
        if keep_cols:
            labels_small = labels[keep_cols].copy()
            labels_small = labels_small.drop_duplicates(
                subset=["image_path"], keep="first"
            )
            tile = tile.merge(labels_small, on="image_path", how="left")
    return tile


def _apply_geographic_splits(tile: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing geographic splits CSV: {path}")
    assignments = pd.read_csv(path)
    required = {"image_path", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Geographic splits CSV missing columns: {sorted(missing)}")
    assignments = assignments.copy()
    assignments["image_path"] = assignments["image_path"].astype(str).str.strip()
    assignments["split"] = (
        assignments["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("validation", "val")
    )
    invalid = sorted(set(assignments["split"]) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"Geographic splits CSV has invalid split values: {invalid}")
    conflicts = assignments.groupby("image_path")["split"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Geographic splits CSV assigns one image to multiple splits")
    if "geographic_block" in assignments.columns:
        blocks = assignments.loc[
            assignments["geographic_block"].notna()
            & assignments["geographic_block"].astype(str).str.strip().ne("")
        ]
        block_conflicts = blocks.groupby("geographic_block")["split"].nunique()
        if (block_conflicts > 1).any():
            raise ValueError(
                "Geographic splits CSV assigns one geographic_block to multiple splits"
            )
    keep = [
        column
        for column in ("image_path", "split", "geographic_block", "split_seed")
        if column in assignments
    ]
    assignments = assignments[keep].drop_duplicates("image_path", keep="first")
    output = tile.merge(assignments, on="image_path", how="left", validate="one_to_one")
    missing_assignments = output["split"].isna()
    if missing_assignments.any():
        examples = output.loc[missing_assignments, "image_path"].head(3).tolist()
        raise ValueError(
            f"Geographic splits CSV has no assignment for {int(missing_assignments.sum())} "
            f"benchmark tiles; examples: {examples}"
        )
    return output


def _agreement_tables(ann: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_annotator = (
        ann.groupby("annotator_id", as_index=False)
        .agg(
            n_annotations=("scenic_human", "size"),
            unique_tiles=("image_path", "nunique"),
            mean_score=("scenic_human", "mean"),
            std_score=("scenic_human", "std"),
            min_score=("scenic_human", "min"),
            max_score=("scenic_human", "max"),
        )
        .sort_values(["n_annotations", "annotator_id"], ascending=[False, True])
    )
    by_annotator["std_score"] = by_annotator["std_score"].fillna(0.0)

    pairs: list[dict[str, Any]] = []
    annotators = sorted(ann["annotator_id"].dropna().unique().tolist())
    for a_id, b_id in itertools.combinations(annotators, 2):
        a_df = ann.loc[
            ann["annotator_id"] == a_id, ["image_path", "scenic_human"]
        ].rename(columns={"scenic_human": "score_a"})
        b_df = ann.loc[
            ann["annotator_id"] == b_id, ["image_path", "scenic_human"]
        ].rename(columns={"scenic_human": "score_b"})
        overlap = a_df.merge(b_df, on="image_path", how="inner")
        if overlap.empty:
            continue
        diff = overlap["score_a"] - overlap["score_b"]
        pairs.append(
            {
                "annotator_a": a_id,
                "annotator_b": b_id,
                "overlap_tiles": int(len(overlap)),
                "mean_diff": float(diff.mean()),
                "mae": float(diff.abs().mean()),
                "rmse": float(np.sqrt((diff**2).mean())),
                "corr": _corr(overlap["score_a"], overlap["score_b"]),
            }
        )

    pairwise = pd.DataFrame(
        pairs,
        columns=[
            "annotator_a",
            "annotator_b",
            "overlap_tiles",
            "mean_diff",
            "mae",
            "rmse",
            "corr",
        ],
    )
    if not pairwise.empty:
        pairwise = pairwise.sort_values(
            ["overlap_tiles", "mae"], ascending=[False, True]
        )
    return by_annotator, pairwise


def main() -> None:
    args = parse_args()
    run_name = validate_run_name(args.run_name)
    ann, total_rows_before_filter = _prepare_annotations(args.annotations_csv)
    ann, non_run_rows_dropped = _filter_annotations_by_splits(
        ann, args.geographic_splits_csv
    )
    tile = _build_tile_table(ann, args.labels_csv if args.labels_csv else None)
    split = _apply_geographic_splits(tile, args.geographic_splits_csv)
    by_annotator, pairwise = _agreement_tables(ann)

    base = args.output_dir / run_name
    base.mkdir(parents=True, exist_ok=True)

    tile_path = base / "benchmark_tiles.csv"
    split_path = base / "benchmark_split.csv"
    annotator_path = base / "agreement_by_annotator.csv"
    pairwise_path = base / "agreement_by_pair.csv"
    summary_path = base / "summary.json"

    tile.to_csv(tile_path, index=False)
    split.to_csv(split_path, index=False)
    by_annotator.to_csv(annotator_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    overlap_tiles = int((tile["n_annotators"] >= 2).sum())
    split_counts = split["split"].value_counts().to_dict()

    benchmark_vs_heuristic = None
    if (
        "scenic_score_heuristic" in split.columns
        and split["scenic_score_heuristic"].notna().any()
    ):
        valid = split.loc[
            split["scenic_score_heuristic"].notna(),
            ["scenic_human_mean", "scenic_score_heuristic"],
        ]
        if not valid.empty:
            diff = valid["scenic_human_mean"] - valid["scenic_score_heuristic"]
            benchmark_vs_heuristic = {
                "tiles": int(len(valid)),
                "mae": float(diff.abs().mean()),
                "rmse": float(np.sqrt((diff**2).mean())),
                "corr": _corr(
                    valid["scenic_human_mean"], valid["scenic_score_heuristic"]
                ),
            }

    summary = {
        "run_name": run_name,
        "source_annotations_csv": str(args.annotations_csv),
        "source_labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "source_geographic_splits_csv": str(args.geographic_splits_csv),
        "split_strategy": "fixed_geographic",
        "total_rows_before_filter": int(total_rows_before_filter),
        "non_run_rows_dropped": int(non_run_rows_dropped),
        "rows_after_filter_and_dedupe": int(len(ann)),
        "unique_tiles": int(tile["image_path"].nunique()),
        "annotators": int(ann["annotator_id"].nunique()),
        "tiles_with_overlap_ge_2": overlap_tiles,
        "mean_annotator_std": float(tile["scenic_human_std"].mean())
        if len(tile)
        else 0.0,
        "median_annotator_std": float(tile["scenic_human_std"].median())
        if len(tile)
        else 0.0,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "pairwise_overlap_rows": int(len(pairwise)),
        "benchmark_vs_heuristic": benchmark_vs_heuristic,
        "source_hashes": {
            "annotations_csv": sha256_file(args.annotations_csv),
            "geographic_splits_csv": sha256_file(args.geographic_splits_csv),
            "benchmark_split_csv": sha256_file(split_path),
        },
        "outputs": {
            "benchmark_tiles_csv": str(tile_path),
            "benchmark_split_csv": str(split_path),
            "agreement_by_annotator_csv": str(annotator_path),
            "agreement_by_pair_csv": str(pairwise_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {tile_path}")
    print(f"Wrote {split_path}")
    print(f"Wrote {annotator_path}")
    print(f"Wrote {pairwise_path}")
    print(f"Wrote {summary_path}")
    print(
        f"Rows(after dedupe): {summary['rows_after_filter_and_dedupe']} | "
        f"Non-run rows dropped: {summary['non_run_rows_dropped']} | "
        f"Unique tiles: {summary['unique_tiles']} | "
        f"Annotators: {summary['annotators']}"
    )
    print(
        f"Split counts: train={summary['split_counts'].get('train', 0)} "
        f"val={summary['split_counts'].get('val', 0)} "
        f"test={summary['split_counts'].get('test', 0)}"
    )


if __name__ == "__main__":
    main()
