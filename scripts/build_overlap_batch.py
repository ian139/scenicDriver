"""
Build a deterministic overlap-annotation batch.

This selects tiles that one annotator has already labeled but another annotator
has not, so the second annotator can create overlap labels for agreement stats.

Example:
  uv run python scripts/build_overlap_batch.py \
    --annotations-csv data/raw/labels_human.csv \
    --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
    --source-annotator ian \
    --target-annotator paperspace \
    --sample-size 200 \
    --seed 42 \
    --output-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BOOL_TRUE = {"1", "true", "yes", "y", "on", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build overlap annotation batch")
    parser.add_argument("--annotations-csv", type=Path, default=Path("data/raw/labels_human.csv"))
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/processed/regression/labels_masswhites_z14_mixed5000.csv"),
        help="Optional labels CSV for class_id joins (used for stratified sampling)",
    )
    parser.add_argument("--source-annotator", type=str, default="")
    parser.add_argument("--target-annotator", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/regression/overlap_batch.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary output path (defaults to <output-csv>.summary.json)",
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


def _read_annotations(path: Path) -> pd.DataFrame:
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
    ann = ann.sort_values(["_timestamp", "_row_id"])
    ann = ann.drop_duplicates(subset=["image_path", "annotator_id"], keep="last")
    return ann


def _sample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if "class_id" not in df.columns or df["class_id"].isna().all():
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    classes = sorted(df["class_id"].dropna().unique().tolist())
    per_class = max(1, n // max(1, len(classes)))
    chunks = []
    for class_id in classes:
        class_df = df.loc[df["class_id"] == class_id]
        if class_df.empty:
            continue
        chunks.append(class_df.sample(n=min(per_class, len(class_df)), random_state=seed))
    sampled = pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()
    needed = min(n, len(df)) - len(sampled)
    if needed > 0:
        remainder = df.loc[~df["image_path"].isin(sampled["image_path"])]
        if not remainder.empty:
            sampled = pd.concat(
                [sampled, remainder.sample(n=min(needed, len(remainder)), random_state=seed)],
                ignore_index=True,
            )
    return sampled.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def _pick_source_annotator(ann: pd.DataFrame, target_annotator: str) -> str:
    counts = (
        ann.loc[ann["annotator_id"] != target_annotator]
        .groupby("annotator_id", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    if counts.empty:
        raise ValueError("No source annotator candidates found (only target annotator present).")
    return str(counts.index[0])


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be > 0")

    ann = _read_annotations(args.annotations_csv)
    target = args.target_annotator.strip()
    if not target:
        raise ValueError("target-annotator is required")

    source = args.source_annotator.strip() or _pick_source_annotator(ann, target)
    if source == target:
        raise ValueError("source-annotator and target-annotator must be different")

    source_df = ann.loc[ann["annotator_id"] == source, ["image_path", "scenic_human"]].copy()
    source_df = source_df.rename(columns={"scenic_human": "source_scenic_human"})
    target_done = set(ann.loc[ann["annotator_id"] == target, "image_path"].astype(str))
    source_df = source_df.loc[~source_df["image_path"].isin(target_done)].copy()
    source_df = source_df.drop_duplicates(subset=["image_path"], keep="first")

    if source_df.empty:
        raise ValueError("No overlap candidates found: target already labeled all source tiles.")

    labels_path = args.labels_csv
    if labels_path and labels_path.exists():
        labels = pd.read_csv(labels_path)
        if "image_path" in labels.columns:
            keep_cols = [c for c in ["image_path", "class_id", "scenic_score_heuristic"] if c in labels.columns]
            if keep_cols:
                labels = labels[keep_cols].drop_duplicates(subset=["image_path"], keep="first")
                source_df = source_df.merge(labels, on="image_path", how="left")

    if args.stratify_by_class:
        batch = _sample_stratified(source_df, n=args.sample_size, seed=args.seed)
    else:
        batch = source_df.sample(n=min(args.sample_size, len(source_df)), random_state=args.seed).reset_index(drop=True)

    batch["source_annotator"] = source
    batch["target_annotator"] = target
    batch["batch_type"] = "overlap"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(args.output_csv, index=False)

    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary = {
        "annotations_csv": str(args.annotations_csv),
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "source_annotator": source,
        "target_annotator": target,
        "source_unique_tiles": int(ann.loc[ann["annotator_id"] == source, "image_path"].nunique()),
        "target_unique_tiles": int(ann.loc[ann["annotator_id"] == target, "image_path"].nunique()),
        "candidate_tiles": int(len(source_df)),
        "batch_size": int(len(batch)),
        "stratify_by_class": bool(args.stratify_by_class),
        "seed": int(args.seed),
        "output_csv": str(args.output_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {summary_path}")
    print(
        f"Source={source} Target={target} | "
        f"Candidates={summary['candidate_tiles']} | Batch={summary['batch_size']}"
    )


if __name__ == "__main__":
    main()
