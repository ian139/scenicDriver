"""
Build a deterministic matched-source human benchmark batch for source-shift evaluation.

Example:
  uv run python scripts/annotation/build_source_shift_batch.py \
    --old-manifest data/raw/sources/mapbox_v1/tile_manifest.csv \
    --new-manifest data/raw/sources/naip_3dep_v1/tile_manifest.csv \
    --output-batch-csv data/processed/annotation/source_shift_batch.csv \
    --output-summary-json data/processed/annotation/source_shift_batch.summary.json \
    --sample-size 200 \
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active_learning.common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)

REQUIRED_MANIFEST_COLUMNS = {"region", "z", "x", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic matched-source human benchmark batch for source-shift audit"
    )
    parser.add_argument(
        "--old-manifest",
        type=Path,
        required=True,
        help="Path to old source tile manifest CSV",
    )
    parser.add_argument(
        "--new-manifest",
        type=Path,
        required=True,
        help="Path to new source tile manifest CSV",
    )
    parser.add_argument(
        "--output-batch-csv",
        type=Path,
        required=True,
        help="Path to destination blinded annotation batch CSV",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        help="Path to destination batch summary JSON (defaults to output-batch-csv with .summary.json)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of matched coordinate pairs to sample (100 <= N <= 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and blinded ordering",
    )
    parser.add_argument(
        "--old-source-identity",
        type=str,
        default=None,
        help="Explicit source identity string for old manifest",
    )
    parser.add_argument(
        "--new-source-identity",
        type=str,
        default=None,
        help="Explicit source identity string for new manifest",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Disable checking file existence on disk for tile paths",
    )
    return parser.parse_args()


def _resolve_source_identity(
    df: pd.DataFrame,
    override: str | None,
    manifest_name: str,
) -> str:
    """Resolve one explicit source identity; never infer a provider default."""
    if override and override.strip():
        return override.strip()

    if "source_identity" not in df.columns:
        raise ValueError(
            f"Manifest for {manifest_name} lacks source_identity; provide an explicit "
            f"--{manifest_name}-source-identity"
        )
    values = df["source_identity"].dropna().astype(str).str.strip()
    values = values[values != ""]
    unique_values = sorted(values.unique())
    if not unique_values:
        raise ValueError(
            f"Manifest for {manifest_name} has no non-empty source_identity values"
        )
    if len(unique_values) != 1:
        raise ValueError(
            f"Manifest for {manifest_name} contains ambiguous source identities: "
            f"{unique_values}"
        )
    if len(values) != len(df):
        raise ValueError(
            f"Manifest for {manifest_name} has rows missing source_identity"
        )
    return unique_values[0]


def _compute_tile_sha256(path_str: str, check_files: bool) -> tuple[str, str]:
    path = Path(path_str)
    if check_files:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Tile file does not exist: {path}")
        digest = sha256_file(path)
    else:
        digest = hashlib.sha256(path_str.encode("utf-8")).hexdigest()
    return str(path), digest


def validate_manifest(
    manifest_input: Path | pd.DataFrame, manifest_name: str
) -> tuple[pd.DataFrame, str, str | None]:
    """Validate tile manifest dataframe/path and extract SHA-256 if from file."""
    if isinstance(manifest_input, Path):
        if not manifest_input.exists():
            raise FileNotFoundError(f"Manifest file missing: {manifest_input}")
        file_sha256 = sha256_file(manifest_input)
        df = pd.read_csv(manifest_input)
        path_str = str(manifest_input)
    else:
        df = manifest_input.copy()
        file_sha256 = None
        path_str = f"<DataFrame:{manifest_name}>"

    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{manifest_name} missing required coordinate columns: {sorted(missing)}"
        )

    # Ensure satellite and terrain paths exist in columns
    if "satellite_path" not in df.columns and "image_path" in df.columns:
        df["satellite_path"] = df["image_path"]
    if "satellite_path" not in df.columns:
        raise ValueError(
            f"{manifest_name} missing 'satellite_path' (or 'image_path') column"
        )
    if "terrain_path" not in df.columns:
        raise ValueError(f"{manifest_name} missing 'terrain_path' column")

    df["region"] = df["region"].astype(str).str.strip()
    df["z"] = pd.to_numeric(df["z"], errors="coerce").astype("Int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce").astype("Int64")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("Int64")

    if df["z"].isna().any() or df["x"].isna().any() or df["y"].isna().any():
        raise ValueError(f"{manifest_name} contains non-integer z, x, or y coordinates")

    # Reject duplicate coordinates within the same manifest
    coord_cols = ["region", "z", "x", "y"]
    if df.duplicated(subset=coord_cols).any():
        dup_coords = df[df.duplicated(subset=coord_cols, keep=False)][coord_cols].head(
            3
        )
        raise ValueError(
            f"{manifest_name} contains duplicate coordinate entries: {dup_coords.to_dict('records')}"
        )

    return df, path_str, file_sha256


def build_source_shift_batch(
    old_manifest: Path | pd.DataFrame,
    new_manifest: Path | pd.DataFrame,
    output_batch_csv: Path,
    output_summary_json: Path | None = None,
    sample_size: int = 200,
    seed: int = 42,
    old_source_identity: str | None = None,
    new_source_identity: str | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    """Build a deterministic matched-source human benchmark batch."""
    if not (100 <= sample_size <= 300):
        raise ValueError(
            f"sample_size must be between 100 and 300 (inclusive), got {sample_size}"
        )

    old_df, old_path_str, old_file_sha = validate_manifest(old_manifest, "old_manifest")
    new_df, new_path_str, new_file_sha = validate_manifest(new_manifest, "new_manifest")

    old_identity = _resolve_source_identity(old_df, old_source_identity, "old")
    new_identity = _resolve_source_identity(new_df, new_source_identity, "new")

    if old_identity.lower() == new_identity.lower():
        raise ValueError(
            f"Source identity ambiguity: old and new source identities are identical ({old_identity!r})"
        )

    # Join on exact coordinates (region, z, x, y)
    coord_cols = ["region", "z", "x", "y"]
    joined = pd.merge(
        old_df,
        new_df,
        on=coord_cols,
        suffixes=("_old", "_new"),
        how="inner",
    )

    if joined.empty:
        raise ValueError("No matching coordinates found between old and new manifests")

    # Filter complete valid satellite and terrain paths
    valid_records: list[dict[str, Any]] = []
    for idx, row in joined.iterrows():
        sat_old_raw = str(row["satellite_path_old"]).strip()
        terr_old_raw = str(row["terrain_path_old"]).strip()
        sat_new_raw = str(row["satellite_path_new"]).strip()
        terr_new_raw = str(row["terrain_path_new"]).strip()

        if not (sat_old_raw and terr_old_raw and sat_new_raw and terr_new_raw):
            continue

        try:
            sat_old_path, sat_old_sha = _compute_tile_sha256(sat_old_raw, check_files)
            terr_old_path, terr_old_sha = _compute_tile_sha256(
                terr_old_raw, check_files
            )
            sat_new_path, sat_new_sha = _compute_tile_sha256(sat_new_raw, check_files)
            terr_new_path, terr_new_sha = _compute_tile_sha256(
                terr_new_raw, check_files
            )
        except (FileNotFoundError, OSError):
            continue

        old_content_sha = hashlib.sha256(
            f"{sat_old_sha}:{terr_old_sha}".encode()
        ).hexdigest()
        new_content_sha = hashlib.sha256(
            f"{sat_new_sha}:{terr_new_sha}".encode()
        ).hexdigest()

        valid_records.append(
            {
                "region": str(row["region"]),
                "z": int(row["z"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "sat_old_path": sat_old_path,
                "sat_old_sha": sat_old_sha,
                "terr_old_path": terr_old_path,
                "terr_old_sha": terr_old_sha,
                "old_content_sha": old_content_sha,
                "sat_new_path": sat_new_path,
                "sat_new_sha": sat_new_sha,
                "terr_new_path": terr_new_path,
                "terr_new_sha": terr_new_sha,
                "new_content_sha": new_content_sha,
            }
        )

    if len(valid_records) < sample_size:
        raise ValueError(
            f"Insufficient matching valid coordinate pairs: found {len(valid_records)}, required {sample_size} (100 <= N <= 300)"
        )

    # Sort valid records deterministically by (region, z, x, y) before sampling
    valid_df = (
        pd.DataFrame(valid_records).sort_values(by=coord_cols).reset_index(drop=True)
    )

    # Sample exactly N pairs deterministically
    sampled_pairs = (
        valid_df.sample(n=sample_size, random_state=seed)
        .sort_values(by=coord_cols)
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    pair_unblinding: dict[str, Any] = {}

    for pair_idx, row in sampled_pairs.iterrows():
        pair_id = f"pair_{pair_idx + 1:04d}"

        pair_unblinding[pair_id] = {
            "region": row["region"],
            "z": int(row["z"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "old_image_path": row["sat_old_path"],
            "old_content_sha256": row["old_content_sha"],
            "new_image_path": row["sat_new_path"],
            "new_content_sha256": row["new_content_sha"],
        }

        # Old source row
        rows.append(
            {
                "pair_id": pair_id,
                "source_variant": "old",
                "source_identity": old_identity,
                "region": row["region"],
                "z": int(row["z"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "image_path": row["sat_old_path"],
                "satellite_path": row["sat_old_path"],
                "terrain_path": row["terr_old_path"],
                "satellite_sha256": row["sat_old_sha"],
                "terrain_sha256": row["terr_old_sha"],
                "content_sha256": row["old_content_sha"],
            }
        )

        # New source row
        rows.append(
            {
                "pair_id": pair_id,
                "source_variant": "new",
                "source_identity": new_identity,
                "region": row["region"],
                "z": int(row["z"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "image_path": row["sat_new_path"],
                "satellite_path": row["sat_new_path"],
                "terrain_path": row["terr_new_path"],
                "satellite_sha256": row["sat_new_sha"],
                "terrain_sha256": row["terr_new_sha"],
                "content_sha256": row["new_content_sha"],
            }
        )

    batch_df = pd.DataFrame(rows)

    # Deterministic Blinded Row Order
    blinded_df = batch_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    blinded_df.insert(
        0,
        "blinded_id",
        [f"shift_row_{i + 1:04d}" for i in range(len(blinded_df))],
    )

    output_batch_csv.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_batch_csv,
        blinded_df.to_csv(index=False, lineterminator="\n"),
    )
    batch_csv_sha256 = sha256_file(output_batch_csv)

    summary_path = (
        output_summary_json
        if output_summary_json is not None
        else output_batch_csv.with_suffix(".summary.json")
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "sample_size_pairs": int(sample_size),
        "total_batch_rows": int(len(blinded_df)),
        "old_manifest_path": old_path_str,
        "old_manifest_sha256": old_file_sha,
        "old_source_identity": old_identity,
        "new_manifest_path": new_path_str,
        "new_manifest_sha256": new_file_sha,
        "new_source_identity": new_identity,
        "output_batch_csv_path": str(output_batch_csv),
        "output_batch_csv_sha256": batch_csv_sha256,
        "pair_unblinding": pair_unblinding,
    }

    summary_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    summary["summary_sha256"] = hashlib.sha256(summary_bytes).hexdigest()

    atomic_write_json(summary_path, summary)

    return summary


def main() -> None:
    args = parse_args()
    summary = build_source_shift_batch(
        old_manifest=args.old_manifest,
        new_manifest=args.new_manifest,
        output_batch_csv=args.output_batch_csv,
        output_summary_json=args.output_summary_json,
        sample_size=args.sample_size,
        seed=args.seed,
        old_source_identity=args.old_source_identity,
        new_source_identity=args.new_source_identity,
        check_files=not args.no_check_files,
    )
    print(
        f"Built source-shift batch: Pairs={summary['sample_size_pairs']} | "
        f"Total rows={summary['total_batch_rows']} | CSV={summary['output_batch_csv_path']}"
    )


if __name__ == "__main__":
    main()
