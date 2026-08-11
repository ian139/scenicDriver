"""
Evaluate source shift using paired human benchmark and model predictions.

Example:
  uv run python scripts/modeling/evaluate_source_shift.py \
    --batch-summary-json data/processed/annotation/source_shift_batch.summary.json \
    --annotations-csv data/raw/labels_human.csv \
    --old-predictions-csv data/processed/modeling/old_predictions.csv \
    --new-predictions-csv data/processed/modeling/new_predictions.csv \
    --output-report-json data/processed/modeling/source_shift_evaluation.json \
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active_learning.common import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)

STRICT_HUMAN_COLUMNS = [
    "image_path",
    "scenic_human",
    "confidence",
    "skip",
    "annotator_id",
    "timestamp",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed evaluation of model source shift against human benchmark"
    )
    parser.add_argument(
        "--batch-summary-json",
        type=Path,
        required=True,
        help="Path to source-shift batch summary JSON file",
    )
    parser.add_argument(
        "--batch-csv",
        type=Path,
        help="Optional override path to batch CSV (defaults to path in summary JSON)",
    )
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        required=True,
        help="Path to strict 7-column human annotations CSV",
    )
    parser.add_argument(
        "--old-predictions-csv",
        type=Path,
        required=True,
        help="Path to model predictions CSV for old source tiles",
    )
    parser.add_argument(
        "--new-predictions-csv",
        type=Path,
        required=True,
        help="Path to model predictions CSV for new source tiles",
    )
    parser.add_argument(
        "--old-predictions-manifest-json",
        type=Path,
        help="Hash-bound old prediction manifest (defaults to <old-predictions-csv>.manifest.json)",
    )
    parser.add_argument(
        "--new-predictions-manifest-json",
        type=Path,
        help="Hash-bound new prediction manifest (defaults to <new-predictions-csv>.manifest.json)",
    )
    parser.add_argument(
        "--output-report-json",
        type=Path,
        required=True,
        help="Path to destination atomic evaluation report JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (>= 100)",
    )
    # Threshold CLI parameters
    parser.add_argument(
        "--min-spearman-delta",
        type=float,
        default=0.5,
        help="Minimum required Spearman correlation between human deltas and model deltas",
    )
    parser.add_argument(
        "--max-mae-new",
        type=float,
        default=1.5,
        help="Maximum allowed MAE between model and human on new source",
    )
    parser.add_argument(
        "--min-correlation-new",
        type=float,
        default=0.5,
        help="Minimum required Pearson correlation between model and human on new source",
    )
    parser.add_argument(
        "--max-abs-delta-bias",
        type=float,
        default=1.0,
        help="Maximum allowed absolute difference between mean model delta and mean human delta",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail closed with error exit if source shift evaluation does not pass thresholds",
    )
    return parser.parse_args()


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or pd.isna(val):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "t", "on"}


def validate_human_annotations(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing human annotations CSV: {path}")

    file_sha256 = sha256_file(path)
    df = pd.read_csv(path)

    if list(df.columns) != STRICT_HUMAN_COLUMNS:
        raise ValueError(
            f"annotations_csv must have exact 7 columns: {STRICT_HUMAN_COLUMNS}, got {list(df.columns)}"
        )

    df["image_path"] = df["image_path"].astype(str).str.strip()
    df["scenic_human"] = pd.to_numeric(df["scenic_human"], errors="coerce")
    df["skip"] = df["skip"].map(_to_bool)

    # Filter out skipped or missing human scores
    valid = df[~df["skip"] & df["scenic_human"].notna()].copy()

    # Aggregate by image_path (mean scenic_human across annotators)
    grouped = (
        valid.groupby("image_path")["scenic_human"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return grouped.rename(columns={"mean": "scenic_human_score"}), file_sha256


def validate_predictions_csv(path: Path, label: str) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} predictions CSV: {path}")

    file_sha256 = sha256_file(path)
    df = pd.read_csv(path)

    # Find image path column
    path_col = None
    for col in ("image_path", "satellite_path", "tile_path"):
        if col in df.columns:
            path_col = col
            break

    if not path_col:
        raise ValueError(f"{label} predictions CSV missing image_path column")

    # Find prediction score column
    score_col = None
    for col in ("scenic_score", "predicted_score", "score", "prediction"):
        if col in df.columns:
            score_col = col
            break

    if not score_col:
        raise ValueError(
            f"{label} predictions CSV missing prediction score column (scenic_score/predicted_score)"
        )

    df[path_col] = df[path_col].astype(str).str.strip()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    valid = df[df[score_col].notna()].copy()

    # Reject duplicate conflicting predictions for exact same path
    dups = valid.groupby(path_col)[score_col].agg(["min", "max", "count"]).reset_index()
    conflicts = dups[dups["min"] != dups["max"]]
    if not conflicts.empty:
        raise ValueError(
            f"{label} predictions CSV contains duplicate conflicting scores for paths: {conflicts[path_col].tolist()[:3]}"
        )
    agg = valid.groupby(path_col)[score_col].mean().reset_index()
    return (
        agg.rename(columns={path_col: "image_path", score_col: "predicted_score"}),
        file_sha256,
        _prediction_schema_sha256(df),
    )


def _prediction_schema_sha256(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        list(frame.columns), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_prediction_manifest(
    path: Path,
    *,
    label: str,
    predictions_csv_sha256: str,
    prediction_schema_sha256: str,
    batch_csv_sha256: str,
    batch_summary: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} prediction manifest JSON: {path}")
    manifest_sha256 = sha256_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} prediction manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{label} prediction manifest must use schema_version 1")

    expected = {
        "source_variant": label,
        "source_identity": batch_summary[f"{label}_source_identity"],
        "source_manifest_sha256": batch_summary[f"{label}_manifest_sha256"],
        "batch_csv_sha256": batch_csv_sha256,
        "predictions_csv_sha256": predictions_csv_sha256,
        "prediction_schema_sha256": prediction_schema_sha256,
    }
    for field, expected_value in expected.items():
        if not isinstance(expected_value, str) or not expected_value:
            raise ValueError(f"Batch summary lacks required {label} identity {field}")
        actual = manifest.get(field)
        if actual != expected_value:
            raise ValueError(
                f"{label} prediction manifest {field} mismatch: "
                f"{actual!r} != {expected_value!r}"
            )

    for field in (
        "preprocessing_contract_sha256",
        "grid_contract_sha256",
        "checkpoint_sha256",
        "calibration_sha256",
    ):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())
        ):
            raise ValueError(
                f"{label} prediction manifest {field} must be a SHA-256 hex digest"
            )
    return manifest, manifest_sha256


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    res = stats.spearmanr(x, y)
    val = res.statistic if hasattr(res, "statistic") else res[0]
    return float(val) if not np.isnan(val) else 0.0


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    res = stats.pearsonr(x, y)
    val = res.statistic if hasattr(res, "statistic") else res[0]
    return float(val) if not np.isnan(val) else 0.0


def compute_metrics(
    h_old: np.ndarray,
    h_new: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
) -> dict[str, float]:
    delta_h = h_new - h_old
    delta_p = p_new - p_old

    mae_old = float(np.mean(np.abs(p_old - h_old)))
    rmse_old = float(np.sqrt(np.mean((p_old - h_old) ** 2)))

    mae_new = float(np.mean(np.abs(p_new - h_new)))
    rmse_new = float(np.sqrt(np.mean((p_new - h_new) ** 2)))

    mean_delta_h = float(np.mean(delta_h))
    mean_delta_p = float(np.mean(delta_p))
    abs_delta_bias = float(abs(mean_delta_p - mean_delta_h))

    spearman_human_old_new = _safe_spearman(h_old, h_new)
    spearman_model_old_new = _safe_spearman(p_old, p_new)
    spearman_delta_human_model = _safe_spearman(delta_h, delta_p)

    pred_human_corr_old = _safe_pearson(p_old, h_old)
    pred_human_corr_new = _safe_pearson(p_new, h_new)

    return {
        "mae_old": mae_old,
        "rmse_old": rmse_old,
        "mae_new": mae_new,
        "rmse_new": rmse_new,
        "mean_human_delta": mean_delta_h,
        "mean_model_delta": mean_delta_p,
        "abs_delta_bias": abs_delta_bias,
        "spearman_human_old_new": spearman_human_old_new,
        "spearman_model_old_new": spearman_model_old_new,
        "spearman_delta_human_model": spearman_delta_human_model,
        "pred_human_corr_old": pred_human_corr_old,
        "pred_human_corr_new": pred_human_corr_new,
    }


def compute_bootstrap_ci(
    h_old: np.ndarray,
    h_new: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    n = len(h_old)
    rng = np.random.RandomState(seed)

    boot_results: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(h_old[idx], h_new[idx], p_old[idx], p_new[idx])
        for k, v in m.items():
            boot_results.setdefault(k, []).append(v)

    ci_dict: dict[str, dict[str, float]] = {}
    for k, values in boot_results.items():
        arr = np.array(values)
        ci_dict[k] = {
            "mean": float(np.mean(arr)),
            "ci_2_5": float(np.percentile(arr, 2.5)),
            "ci_97_5": float(np.percentile(arr, 97.5)),
        }

    return ci_dict


def evaluate_source_shift(
    batch_summary_json: Path,
    annotations_csv: Path,
    old_predictions_csv: Path,
    new_predictions_csv: Path,
    output_report_json: Path,
    old_predictions_manifest_json: Path | None = None,
    new_predictions_manifest_json: Path | None = None,
    batch_csv_override: Path | None = None,
    seed: int = 42,
    n_bootstrap: int = 1000,
    min_spearman_delta: float = 0.5,
    max_mae_new: float = 1.5,
    min_correlation_new: float = 0.5,
    max_abs_delta_bias: float = 1.0,
    strict: bool = False,
) -> dict[str, Any]:
    if not batch_summary_json.exists():
        raise FileNotFoundError(f"Batch summary JSON not found: {batch_summary_json}")

    summary_sha256 = sha256_file(batch_summary_json)
    summary_data = json.loads(batch_summary_json.read_text(encoding="utf-8"))

    batch_csv_path = (
        batch_csv_override
        if batch_csv_override is not None
        else Path(summary_data.get("output_batch_csv_path", ""))
    )

    if not batch_csv_path.exists():
        raise FileNotFoundError(f"Batch CSV not found: {batch_csv_path}")

    batch_csv_sha256 = sha256_file(batch_csv_path)

    expected_batch_sha = summary_data.get("output_batch_csv_sha256")
    if expected_batch_sha and expected_batch_sha.lower() != batch_csv_sha256.lower():
        raise ValueError(
            f"Batch CSV SHA-256 mismatch: expected {expected_batch_sha}, got {batch_csv_sha256}"
        )

    batch_df = pd.read_csv(batch_csv_path)
    required_batch_cols = {"blinded_id", "pair_id", "source_variant", "image_path"}
    missing_batch_cols = required_batch_cols - set(batch_df.columns)
    if missing_batch_cols:
        raise ValueError(
            f"Batch CSV missing required columns: {sorted(missing_batch_cols)}"
        )

    batch_df["image_path"] = batch_df["image_path"].astype(str).str.strip()

    # Load and validate human annotations
    ann_df, ann_sha256 = validate_human_annotations(annotations_csv)
    ann_map = dict(zip(ann_df["image_path"], ann_df["scenic_human_score"]))

    # Check complete 1-to-1 human coverage for all batch rows
    missing_human = [path for path in batch_df["image_path"] if path not in ann_map]
    if missing_human:
        raise ValueError(
            f"Strict complete human coverage violated! Missing human annotations for {len(missing_human)} batch image paths (e.g. {missing_human[:3]})"
        )

    # Load and validate old & new predictions
    old_pred_df, old_pred_sha256, old_schema_sha256 = validate_predictions_csv(
        old_predictions_csv, "old"
    )
    new_pred_df, new_pred_sha256, new_schema_sha256 = validate_predictions_csv(
        new_predictions_csv, "new"
    )
    old_manifest_path = (
        old_predictions_manifest_json
        or old_predictions_csv.with_suffix(
            old_predictions_csv.suffix + ".manifest.json"
        )
    )
    new_manifest_path = (
        new_predictions_manifest_json
        or new_predictions_csv.with_suffix(
            new_predictions_csv.suffix + ".manifest.json"
        )
    )
    old_prediction_manifest, old_prediction_manifest_sha = validate_prediction_manifest(
        old_manifest_path,
        label="old",
        predictions_csv_sha256=old_pred_sha256,
        prediction_schema_sha256=old_schema_sha256,
        batch_csv_sha256=batch_csv_sha256,
        batch_summary=summary_data,
    )
    new_prediction_manifest, new_prediction_manifest_sha = validate_prediction_manifest(
        new_manifest_path,
        label="new",
        predictions_csv_sha256=new_pred_sha256,
        prediction_schema_sha256=new_schema_sha256,
        batch_csv_sha256=batch_csv_sha256,
        batch_summary=summary_data,
    )
    for identity_field in (
        "preprocessing_contract_sha256",
        "grid_contract_sha256",
        "checkpoint_sha256",
        "calibration_sha256",
    ):
        if (
            old_prediction_manifest[identity_field]
            != new_prediction_manifest[identity_field]
        ):
            raise ValueError(
                f"Prediction manifests disagree on {identity_field}; "
                "source variants must use one identical model pipeline"
            )
    old_pred_map = dict(zip(old_pred_df["image_path"], old_pred_df["predicted_score"]))
    new_pred_map = dict(zip(new_pred_df["image_path"], new_pred_df["predicted_score"]))

    # Match pair-by-pair
    pair_ids = sorted(batch_df["pair_id"].unique())
    h_old_list, h_new_list = [], []
    p_old_list, p_new_list = [], []

    missing_preds: list[str] = []

    for pair_id in pair_ids:
        pair_rows = batch_df[batch_df["pair_id"] == pair_id]
        old_row = pair_rows[pair_rows["source_variant"] == "old"]
        new_row = pair_rows[pair_rows["source_variant"] == "new"]

        if old_row.empty or new_row.empty:
            raise ValueError(
                f"Batch corrupted: pair {pair_id} does not contain both old and new rows"
            )

        old_path = old_row["image_path"].iloc[0]
        new_path = new_row["image_path"].iloc[0]

        if old_path not in old_pred_map:
            missing_preds.append(f"old:{old_path}")
        if new_path not in new_pred_map:
            missing_preds.append(f"new:{new_path}")

        if not missing_preds:
            h_old_list.append(ann_map[old_path])
            h_new_list.append(ann_map[new_path])
            p_old_list.append(old_pred_map[old_path])
            p_new_list.append(new_pred_map[new_path])

    if missing_preds:
        raise ValueError(
            f"Strict complete prediction coverage violated! Missing predictions for paths: {missing_preds[:5]}"
        )

    h_old = np.array(h_old_list, dtype=float)
    h_new = np.array(h_new_list, dtype=float)
    p_old = np.array(p_old_list, dtype=float)
    p_new = np.array(p_new_list, dtype=float)

    # Compute point metrics
    point_metrics = compute_metrics(h_old, h_new, p_old, p_new)

    # Compute bootstrap CIs
    ci_metrics = compute_bootstrap_ci(
        h_old, h_new, p_old, p_new, n_bootstrap=n_bootstrap, seed=seed
    )

    # Threshold checks
    delta_corr_pass = point_metrics["spearman_delta_human_model"] >= min_spearman_delta
    mae_new_pass = point_metrics["mae_new"] <= max_mae_new
    pred_corr_new_pass = point_metrics["pred_human_corr_new"] >= min_correlation_new
    delta_bias_pass = point_metrics["abs_delta_bias"] <= max_abs_delta_bias

    overall_passed = bool(
        delta_corr_pass and mae_new_pass and pred_corr_new_pass and delta_bias_pass
    )

    threshold_checks = {
        "spearman_delta_human_model": {
            "metric_value": point_metrics["spearman_delta_human_model"],
            "threshold": min_spearman_delta,
            "passed": bool(delta_corr_pass),
        },
        "mae_new": {
            "metric_value": point_metrics["mae_new"],
            "threshold": max_mae_new,
            "passed": bool(mae_new_pass),
        },
        "pred_human_corr_new": {
            "metric_value": point_metrics["pred_human_corr_new"],
            "threshold": min_correlation_new,
            "passed": bool(pred_corr_new_pass),
        },
        "abs_delta_bias": {
            "metric_value": point_metrics["abs_delta_bias"],
            "threshold": max_abs_delta_bias,
            "passed": bool(delta_bias_pass),
        },
    }

    report = {
        "schema_version": 1,
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "n_pairs": int(len(pair_ids)),
        "input_hashes": {
            "batch_summary_sha256": summary_sha256,
            "batch_csv_sha256": batch_csv_sha256,
            "annotations_csv_sha256": ann_sha256,
            "old_predictions_csv_sha256": old_pred_sha256,
            "new_predictions_csv_sha256": new_pred_sha256,
            "old_predictions_manifest_sha256": old_prediction_manifest_sha,
            "new_predictions_manifest_sha256": new_prediction_manifest_sha,
        },
        "prediction_identities": {
            "old": old_prediction_manifest,
            "new": new_prediction_manifest,
        },
        "metrics": point_metrics,
        "confidence_intervals_95": ci_metrics,
        "threshold_checks": threshold_checks,
        "passed": overall_passed,
    }

    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_report_json, report)

    if strict and not overall_passed:
        failed_checks = [k for k, v in threshold_checks.items() if not v["passed"]]
        raise ValueError(
            f"Source shift evaluation failed thresholds! Failed checks: {failed_checks}"
        )

    return report


def main() -> None:
    args = parse_args()
    report = evaluate_source_shift(
        batch_summary_json=args.batch_summary_json,
        annotations_csv=args.annotations_csv,
        old_predictions_csv=args.old_predictions_csv,
        new_predictions_csv=args.new_predictions_csv,
        old_predictions_manifest_json=args.old_predictions_manifest_json,
        new_predictions_manifest_json=args.new_predictions_manifest_json,
        output_report_json=args.output_report_json,
        batch_csv_override=args.batch_csv,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        min_spearman_delta=args.min_spearman_delta,
        max_mae_new=args.max_mae_new,
        min_correlation_new=args.min_correlation_new,
        max_abs_delta_bias=args.max_abs_delta_bias,
        strict=args.strict,
    )
    status = "PASSED" if report["passed"] else "FAILED"
    print(
        f"Source shift evaluation {status}: Pairs={report['n_pairs']} | "
        f"Spearman Delta Corr={report['metrics']['spearman_delta_human_model']:.4f} | "
        f"New MAE={report['metrics']['mae_new']:.4f} | "
        f"Report={args.output_report_json}"
    )


if __name__ == "__main__":
    main()
