"""Export regression predictions over a full feature pool to a heuristic run.

Creates a standard ``data/processed/heuristic_runs/<run_name>/`` run containing
``labels.csv``, ``run.json`` and ``report/report.json`` from:

* a prepared feature pool NPZ (``embeddings``/``vit_embeddings``,
  ``terrain_features``, ``class_logits``, ``row_indices``),
* a row-aligned metadata CSV (``image_path``, ``region``, ``z``, ``x``, ``y``,
  ``lat``, ``lon``, ``class_id``, ``embedding_row_index``), and
* a trained regression checkpoint (loaded via the canonical
  ``active_evaluation`` helpers).

This is a *derived heatmap visualization* run: it infers the complete pool and
never assigns train/val/test splits. It is NOT promotion-gate evaluation
evidence and must not be used to evaluate or promote a candidate checkpoint.

All inputs are hash-bound: dataset, metadata CSV and checkpoint SHA-256 values
are validated before anything is written. Rows are aligned strictly by
``embedding_row_index`` (the NPZ ``row_indices`` are required to be contiguous
``0..n-1`` and the metadata must provide exactly one row per index). Output is
published atomically via a sibling temporary directory, and pre-existing run
directories are rejected so a failure can never leave a partially written run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heuristics.report import build_report  # noqa: E402
from src.scenic_scorer.active_evaluation import (  # noqa: E402
    file_sha256,
    load_model_checkpoint,
    predict_dataset,
)
from src.scenic_scorer.regression import resolve_device  # noqa: E402

IMAGE_PATH_RE = re.compile(r"^images/satellite/z(\d+)/([^/]+)/(\d+)_(\d+)\.png$")

REQUIRED_NPZ_KEYS = {"terrain_features", "class_logits", "row_indices"}
REQUIRED_METADATA_COLUMNS = {
    "image_path",
    "region",
    "z",
    "x",
    "y",
    "lat",
    "lon",
    "class_id",
    "embedding_row_index",
}

LABELS_COLUMNS = [
    "image_path",
    "scenic_score",
    "lat",
    "lon",
    "class_id",
    "region",
    "z",
    "x",
    "y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export regression predictions over a full feature pool to a standard "
            "heuristic run (labels.csv, run.json, report/report.json). Derived "
            "heatmap visualization coverage; not promotion-gate evaluation evidence."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=(
            "Feature pool NPZ with embeddings (or vit_embeddings), terrain_features, "
            "class_logits, row_indices"
        ),
    )
    parser.add_argument(
        "--expected-dataset-sha256",
        required=True,
        help="Expected SHA-256 of the dataset NPZ; mismatch fails closed",
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        type=Path,
        help=(
            "Row-aligned metadata CSV with image_path, region, z, x, y, lat, lon, "
            "class_id, embedding_row_index"
        ),
    )
    parser.add_argument(
        "--expected-metadata-sha256",
        required=True,
        help="Expected SHA-256 of the metadata CSV; mismatch fails closed",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Regression checkpoint (candidate schema), loaded via active_evaluation",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="Expected SHA-256 of the checkpoint; mismatch fails closed",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output run directory name under --output-root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/heuristic_runs"),
        help="Root directory holding heuristic runs (default: data/processed/heuristic_runs)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Inference device (default: cpu for deterministic inference)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (default: 256)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def _load_feature_pool(dataset_path: Path) -> dict[str, Any]:
    """Load and validate the feature pool NPZ arrays and row-index contract."""
    data = np.load(dataset_path, allow_pickle=False)
    missing = sorted(REQUIRED_NPZ_KEYS - set(data.files))
    if missing:
        raise ValueError(
            f"Dataset NPZ {dataset_path} missing required fields: {missing}"
        )
    if "embeddings" not in data.files and "vit_embeddings" not in data.files:
        raise ValueError(
            f"Dataset NPZ {dataset_path} missing ViT array: expected "
            "'embeddings' or 'vit_embeddings'"
        )
    emb_key = "embeddings" if "embeddings" in data.files else "vit_embeddings"

    embeddings = data[emb_key]
    terrain = data["terrain_features"]
    class_logits = data["class_logits"]
    row_indices = data["row_indices"]
    n = len(embeddings)
    if n < 1:
        raise ValueError("Dataset NPZ must contain at least one sample")

    for name, arr in [
        (emb_key, embeddings),
        ("terrain_features", terrain),
        ("class_logits", class_logits),
    ]:
        if arr.ndim != 2 or len(arr) != n:
            raise ValueError(
                f"Dataset NPZ array {name!r} must be 2-D with {n} rows, "
                f"got shape {arr.shape}"
            )
    if row_indices.ndim != 1 or len(row_indices) != n:
        raise ValueError(
            f"row_indices must be 1-D with {n} entries, got shape {row_indices.shape}"
        )
    if not np.issubdtype(row_indices.dtype, np.integer):
        raise ValueError("row_indices must be an integer array")

    if (
        not np.isfinite(embeddings).all()
        or not np.isfinite(terrain).all()
        or not np.isfinite(class_logits).all()
    ):
        raise ValueError("Dataset NPZ contains non-finite feature values (NaN or Inf)")

    ri = row_indices.astype(np.int64, copy=False)
    if ri[0] != 0 or ri[-1] != n - 1 or not bool((np.diff(ri) == 1).all()):
        raise ValueError(
            "row_indices must be exactly contiguous 0..n-1 "
            f"(got min {int(ri.min())}, max {int(ri.max())}, "
            f"{len(np.unique(ri))} unique of {n})"
        )
    return {
        "embeddings": embeddings,
        "terrain": terrain,
        "class_logits": class_logits,
        "n": n,
    }


def _parse_int_field(row: dict[str, str], name: str, path: Path) -> int:
    try:
        return int(str(row[name]).strip())
    except ValueError:
        raise ValueError(
            f"Metadata CSV {path} has non-integer {name!r}: {row[name]!r}"
        ) from None


def _parse_float_field(row: dict[str, str], name: str, path: Path) -> float:
    try:
        return float(str(row[name]).strip())
    except ValueError:
        raise ValueError(
            f"Metadata CSV {path} has non-numeric {name!r}: {row[name]!r}"
        ) from None


def _load_metadata(metadata_path: Path, n: int) -> list[dict[str, Any]]:
    """Load metadata rows and strictly align them to NPZ rows by index.

    Returns a list where position ``i`` corresponds to NPZ row ``i``. Each
    ``embedding_row_index`` in ``0..n-1`` must appear exactly once.
    """
    rows_by_index: dict[int, dict[str, Any]] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV {metadata_path} is empty")
        missing = sorted(REQUIRED_METADATA_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Metadata CSV {metadata_path} missing required columns: {missing}"
            )
        for row in reader:
            idx = _parse_int_field(row, "embedding_row_index", metadata_path)
            if idx in rows_by_index:
                raise ValueError(
                    f"Metadata CSV {metadata_path} has duplicate "
                    f"embedding_row_index {idx}"
                )
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"Metadata CSV {metadata_path} embedding_row_index {idx} "
                    f"out of range 0..{n - 1}"
                )
            image_path = str(row["image_path"]).strip()
            match = IMAGE_PATH_RE.match(image_path)
            if match is None:
                raise ValueError(
                    f"Image path {image_path!r} does not match "
                    "images/satellite/z<z>/<region>/<x>_<y>.png"
                )
            parsed_z, parsed_region, parsed_x, parsed_y = (
                int(match.group(1)),
                match.group(2),
                int(match.group(3)),
                int(match.group(4)),
            )
            z = _parse_int_field(row, "z", metadata_path)
            x = _parse_int_field(row, "x", metadata_path)
            y = _parse_int_field(row, "y", metadata_path)
            class_id = _parse_int_field(row, "class_id", metadata_path)
            lat = _parse_float_field(row, "lat", metadata_path)
            lon = _parse_float_field(row, "lon", metadata_path)
            region = str(row["region"]).strip()
            if not region:
                raise ValueError(
                    f"Metadata CSV {metadata_path} has empty region for {image_path!r}"
                )
            if (z, x, y, region) != (parsed_z, parsed_x, parsed_y, parsed_region):
                raise ValueError(
                    f"Metadata CSV {metadata_path} z/x/y/region columns disagree "
                    f"with image path for {image_path!r}"
                )
            if z < 0 or z > 30 or class_id < 0:
                raise ValueError(
                    f"Metadata CSV {metadata_path} has invalid z/class_id for "
                    f"{image_path!r}: z={z} class_id={class_id}"
                )
            tile_count = 1 << z
            if not (0 <= x < tile_count and 0 <= y < tile_count):
                raise ValueError(
                    f"Metadata CSV {metadata_path} tile coordinate is outside "
                    f"the z{z} Web Mercator domain 0..{tile_count - 1} for "
                    f"{image_path!r}: x={x} y={y}"
                )
            if not (np.isfinite(lat) and np.isfinite(lon)):
                raise ValueError(
                    f"Metadata CSV {metadata_path} has non-finite lat/lon for {image_path!r}"
                )
            expected_lon = ((x + 0.5) / tile_count) * 360.0 - 180.0
            expected_lat = math.degrees(
                math.atan(math.sinh(math.pi * (1.0 - (2.0 * (y + 0.5) / tile_count))))
            )
            if not (
                math.isclose(lat, expected_lat, rel_tol=0.0, abs_tol=1e-6)
                and math.isclose(lon, expected_lon, rel_tol=0.0, abs_tol=1e-6)
            ):
                raise ValueError(
                    f"Metadata CSV {metadata_path} lat/lon do not match the "
                    f"z/x/y tile center for {image_path!r}: got ({lat}, {lon}), "
                    f"expected ({expected_lat}, {expected_lon})"
                )
            rows_by_index[idx] = {
                "image_path": image_path,
                "region": region,
                "z": z,
                "x": x,
                "y": y,
                "lat": lat,
                "lon": lon,
                "class_id": class_id,
            }
    if len(rows_by_index) != n:
        raise ValueError(
            f"Metadata CSV {metadata_path} has {len(rows_by_index)} rows with valid "
            f"indices; expected exactly {n} (one per NPZ row)"
        )
    return [rows_by_index[i] for i in range(n)]


def _validate_coverage(rows: list[dict[str, Any]], n: int) -> int:
    """Validate single zoom, unique coordinates, and unique image paths."""
    zoom = rows[0]["z"]
    coords: set[tuple[int, int, int]] = set()
    paths: set[str] = set()
    for row in rows:
        if row["z"] != zoom:
            raise ValueError(
                f"Multiple zooms found (z{zoom} and z{row['z']}); "
                "exactly one zoom is required"
            )
        coord = (row["z"], row["x"], row["y"])
        if coord in coords:
            raise ValueError(
                f"Duplicate tile coordinate z{coord[0]}/x{coord[1]}/y{coord[2]} "
                "in metadata"
            )
        coords.add(coord)
        if row["image_path"] in paths:
            raise ValueError(f"Duplicate image path {row['image_path']!r} in metadata")
        paths.add(row["image_path"])
    if len(coords) != n or len(paths) != n:
        raise ValueError(
            f"Metadata must provide {n} unique coordinates and image paths, "
            f"got {len(coords)} coordinates / {len(paths)} paths"
        )
    return zoom


def _validate_class_ids(rows: list[dict[str, Any]], class_logits: np.ndarray) -> None:
    """Fail closed when metadata class_id disagrees with argmax(class_logits)."""
    if class_logits.shape[0] != len(rows):
        raise ValueError("class_logits row count does not match metadata row count")
    argmax_ids = np.argmax(class_logits, axis=1).astype(np.int64)
    mismatches = [
        (i, row["class_id"], int(argmax_ids[i]))
        for i, row in enumerate(rows)
        if row["class_id"] != int(argmax_ids[i])
    ]
    if mismatches:
        shown = ", ".join(
            f"row {i}: metadata {meta} vs argmax {am}"
            for i, meta, am in mismatches[:10]
        )
        raise ValueError(
            f"class_id mismatch between metadata and argmax(class_logits) for "
            f"{len(mismatches)} rows: {shown}"
        )


def _build_run_info(
    *,
    rows: list[dict[str, Any]],
    zoom: int,
    ds_path: Path,
    meta_path: Path,
    ckpt_path: Path,
    ds_sha: str,
    meta_sha: str,
    ckpt_sha: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    region_counts: dict[str, int] = {}
    lats: list[float] = []
    lons: list[float] = []
    xs: list[int] = []
    ys: list[int] = []
    for row in rows:
        region_counts[row["region"]] = region_counts.get(row["region"], 0) + 1
        lats.append(row["lat"])
        lons.append(row["lon"])
        xs.append(row["x"])
        ys.append(row["y"])
    return {
        "scoring_mode": "learned",
        "derived_visualization": True,
        "purpose": "derived_expanded_heatmap_coverage",
        "purpose_statement": (
            "Derived heatmap visualization over the complete Stage One feature "
            f"pool ({len(rows)} tiles). This artifact is NOT promotion-gate "
            "evaluation evidence and MUST NOT be used to evaluate or promote a "
            "candidate checkpoint. No train/val/test split is applicable: "
            "full-pool inference coverage only."
        ),
        "split": {
            "applicable": False,
            "note": (
                "No train/val/test split is assigned; this is full-pool inference "
                "visualization, not evaluation evidence."
            ),
        },
        "hashes": {
            "dataset_sha256": ds_sha,
            "metadata_sha256": meta_sha,
            "checkpoint_sha256": ckpt_sha,
        },
        "inputs": {
            "dataset_path": str(ds_path),
            "metadata_path": str(meta_path),
            "checkpoint_path": str(ckpt_path),
        },
        "row_alignment": "embedding_row_index",
        "counts": {"total": len(rows), "per_region": region_counts},
        "bounds": {
            "zoom": zoom,
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "lat_min": min(lats),
            "lat_max": max(lats),
            "lon_min": min(lons),
            "lon_max": max(lons),
        },
        "device": device,
        "batch_size": batch_size,
    }


def _build_tiles(rows: list[dict[str, Any]], preds: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "image_path": row["image_path"],
            "scenic_score": float(pred),
            "class_id": row["class_id"],
            "region": row["region"],
            "z": row["z"],
            "x": row["x"],
            "y": row["y"],
            "lat": row["lat"],
            "lon": row["lon"],
        }
        for row, pred in zip(rows, preds, strict=True)
    ]


def _write_labels(labels_path: Path, tiles: list[dict[str, Any]]) -> None:
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABELS_COLUMNS)
        writer.writeheader()
        for tile in tiles:
            writer.writerow({col: tile[col] for col in LABELS_COLUMNS})


def export_heatmap_run(
    *,
    dataset_path: str | Path,
    metadata_path: str | Path,
    checkpoint_path: str | Path,
    expected_dataset_sha256: str,
    expected_metadata_sha256: str,
    expected_checkpoint_sha256: str,
    run_name: str,
    output_root: str | Path = "data/processed/heuristic_runs",
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Hash-guarded, atomically published heatmap run export. See module docstring."""
    ds_path = Path(dataset_path)
    meta_path = Path(metadata_path)
    ckpt_path = Path(checkpoint_path)
    root = Path(output_root)

    for path, label in [
        (ds_path, "Dataset NPZ"),
        (meta_path, "Metadata CSV"),
        (ckpt_path, "Checkpoint"),
    ]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    if not run_name or any(c in run_name for c in "/\\") or run_name in {".", ".."}:
        raise ValueError(
            f"Invalid run name {run_name!r}: must be a single path segment"
        )

    run_dir = root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Output run already exists: {run_dir}")

    ds_sha = file_sha256(ds_path)
    meta_sha = file_sha256(meta_path)
    ckpt_sha = file_sha256(ckpt_path)
    for label, actual, expected in [
        ("Dataset NPZ", ds_sha, expected_dataset_sha256),
        ("Metadata CSV", meta_sha, expected_metadata_sha256),
        ("Checkpoint", ckpt_sha, expected_checkpoint_sha256),
    ]:
        if actual != expected:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected}, got {actual}"
            )

    pool = _load_feature_pool(ds_path)
    rows = _load_metadata(meta_path, pool["n"])
    zoom = _validate_coverage(rows, pool["n"])
    _validate_class_ids(rows, pool["class_logits"])

    resolved_device = resolve_device(device)
    model = load_model_checkpoint(ckpt_path, device=resolved_device, is_candidate=True)
    preds = predict_dataset(
        model,
        pool["embeddings"],
        pool["terrain"],
        pool["class_logits"],
        batch_size=batch_size,
        device=resolved_device,
    )
    if preds.shape != (pool["n"],):
        raise ValueError(f"Prediction shape {preds.shape} != expected {(pool['n'],)}")
    if not np.isfinite(preds).all():
        raise ValueError("Model inference produced non-finite predictions")

    tiles = _build_tiles(rows, preds)
    run_info = _build_run_info(
        rows=rows,
        zoom=zoom,
        ds_path=ds_path,
        meta_path=meta_path,
        ckpt_path=ckpt_path,
        ds_sha=ds_sha,
        meta_sha=meta_sha,
        ckpt_sha=ckpt_sha,
        device=resolved_device,
        batch_size=batch_size,
    )

    root.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        staged = Path(tempfile.mkdtemp(prefix=f".{run_name}.tmp-", dir=root))
        labels_final = run_dir / "labels.csv"
        report_final = run_dir / "report"
        _write_labels(staged / "labels.csv", tiles)
        report_json = build_report(
            tiles=tiles,
            report_dir=staged / "report",
            raw_dir=staged,
            run_info=run_info,
            include_thumbs=False,
        )
        run_record = {
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "labels_path": str(labels_final),
            "raw_labels_path": None,
            "report_dir": str(report_final),
            "summary": report_json["summary"],
            "histogram": report_json["histogram"],
            "run_info": run_info,
            "config": {
                "dataset_path": str(ds_path),
                "metadata_path": str(meta_path),
                "checkpoint_path": str(ckpt_path),
                "run_name": run_name,
                "output_root": str(root),
                "device": resolved_device,
                "batch_size": batch_size,
                "scoring": "learned",
                "thumbnails": False,
            },
        }
        with open(staged / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)
        os.rename(staged, run_dir)
        staged = None
    finally:
        if staged is not None and Path(staged).exists():
            shutil.rmtree(staged, ignore_errors=True)

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "labels_path": str(labels_final),
        "report_dir": str(report_final),
        "total_tiles": int(report_json["summary"]["total_tiles"]),
        "summary": report_json["summary"],
    }


def main() -> int:
    args = parse_args()
    try:
        result = export_heatmap_run(
            dataset_path=args.dataset,
            metadata_path=args.metadata_csv,
            checkpoint_path=args.checkpoint,
            expected_dataset_sha256=args.expected_dataset_sha256,
            expected_metadata_sha256=args.expected_metadata_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            run_name=args.run_name,
            output_root=args.output_root,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
