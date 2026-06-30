"""
Build mixed supervision labels by overlaying aggregated human annotations
onto a heuristic labels CSV.

Example:
  uv run python scripts/modeling/build_mixed_labels.py \
    --heuristic-labels data/processed/regression/labels_masswhites_z14_mixed5000.csv \
    --annotations-csv data/raw/labels_human.csv \
    --output data/processed/regression/labels_masswhites_z14_mixed5000_v2.csv \
    --aggregate mean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


BOOL_TRUE = {"1", "true", "yes", "y", "on", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed labels with aggregated human overrides")
    parser.add_argument("--heuristic-labels", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path, default=Path("data/raw/labels_human.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--keep-existing-human",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, keep pre-existing scenic_human values in heuristic labels when no annotation is found",
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


def _prepare_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"annotations CSV not found: {path}")
    ann = pd.read_csv(path)
    required = {"image_path", "annotator_id", "scenic_human"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"annotations CSV missing required columns: {sorted(missing)}")

    ann = ann.copy()
    ann["image_path"] = ann["image_path"].astype(str).str.strip()
    ann["annotator_id"] = ann["annotator_id"].astype(str).str.strip()
    ann["scenic_human"] = pd.to_numeric(ann["scenic_human"], errors="coerce")
    if "skip" in ann.columns:
        ann["skip"] = ann["skip"].map(_to_bool)
    else:
        ann["skip"] = False

    if "timestamp" in ann.columns:
        ann["_timestamp"] = pd.to_datetime(ann["timestamp"], errors="coerce", utc=True)
    else:
        ann["_timestamp"] = pd.NaT
    ann["_row_id"] = range(len(ann))

    ann = ann.loc[ann["image_path"].ne("") & ann["annotator_id"].ne("") & ann["scenic_human"].notna() & ~ann["skip"]].copy()
    ann = ann.sort_values(["_timestamp", "_row_id"]).drop_duplicates(["image_path", "annotator_id"], keep="last")

    grouped = (
        ann.groupby("image_path", as_index=False)
        .agg(
            scenic_human_mean=("scenic_human", "mean"),
            scenic_human_median=("scenic_human", "median"),
            scenic_human_std=("scenic_human", "std"),
            human_annotation_count=("annotator_id", "size"),
            human_annotator_count=("annotator_id", "nunique"),
        )
        .sort_values("image_path")
    )
    grouped["scenic_human_std"] = grouped["scenic_human_std"].fillna(0.0)
    return grouped


def main() -> None:
    args = parse_args()
    if not args.heuristic_labels.exists():
        raise FileNotFoundError(f"heuristic labels not found: {args.heuristic_labels}")

    base = pd.read_csv(args.heuristic_labels)
    if "image_path" not in base.columns or "scenic_score" not in base.columns:
        raise ValueError("heuristic labels CSV must contain image_path and scenic_score")
    base = base.copy()
    base["image_path"] = base["image_path"].astype(str)

    if "scenic_score_heuristic" not in base.columns:
        base["scenic_score_heuristic"] = pd.to_numeric(base["scenic_score"], errors="coerce")

    human = _prepare_annotations(args.annotations_csv)
    merged = base.merge(human, on="image_path", how="left")

    pick_col = "scenic_human_mean" if args.aggregate == "mean" else "scenic_human_median"
    override_mask = merged[pick_col].notna()

    if args.keep_existing_human and "scenic_human" in merged.columns:
        merged["scenic_human"] = merged["scenic_human"].where(~override_mask, merged[pick_col])
    else:
        merged["scenic_human"] = merged[pick_col]

    merged["scenic_score"] = merged["scenic_score"].where(~override_mask, merged[pick_col])
    merged["label_source"] = merged.get("label_source", pd.Series(["heuristic"] * len(merged), index=merged.index)).astype(str)
    merged.loc[override_mask, "label_source"] = "human_override"
    merged.loc[~override_mask, "label_source"] = merged.loc[~override_mask, "label_source"].replace({"": "heuristic"})

    # Keep a stable, explicit column order for downstream scripts.
    preferred = [
        "image_path",
        "scenic_score",
        "lat",
        "lon",
        "class_id",
        "scenic_human",
        "scenic_score_heuristic",
        "label_source",
        "human_annotation_count",
        "human_annotator_count",
        "scenic_human_std",
    ]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(merged)}")
    print(f"Human override tiles: {int(override_mask.sum())}")
    if "human_annotator_count" in merged.columns:
        overlap_tiles = int((merged["human_annotator_count"].fillna(0) >= 2).sum())
        print(f"Tiles with >=2 annotators: {overlap_tiles}")


if __name__ == "__main__":
    main()
