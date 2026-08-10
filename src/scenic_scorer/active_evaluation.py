"""
Active model evaluation, atomic promotion, and rollback module for Stage-Two scenic regression models.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from src.scenic_scorer.regression import ScenicRegressionModel, resolve_device


def file_sha256(path: str | Path) -> str:
    """Compute SHA256 hash of a file."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_model_checkpoint(
    checkpoint_path: str | Path, device: str = "cpu", is_candidate: bool = True
) -> ScenicRegressionModel:
    """Load and validate a canonical ScenicRegressionModel checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"Corrupt or invalid PyTorch checkpoint at {ckpt_path}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Invalid checkpoint structure in {ckpt_path}: expected dictionary mapping"
        )

    if is_candidate:
        required = {
            "checkpoint_schema_version",
            "checkpoint_state",
            "model_state_dict",
            "vit_dim",
            "terrain_dim",
            "num_classes",
        }
    else:
        required = {"model_state_dict", "vit_dim", "terrain_dim", "num_classes"}

    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Checkpoint {ckpt_path} missing required keys: {missing}")

    if is_candidate:
        schema_ver = payload.get("checkpoint_schema_version")
        if schema_ver != 1 or isinstance(schema_ver, bool):
            raise ValueError(
                f"Checkpoint {ckpt_path} checkpoint_schema_version must be 1, got {schema_ver!r}"
            )
        state = payload.get("checkpoint_state")
        if state != "completed":
            raise ValueError(
                f"Checkpoint {ckpt_path} checkpoint_state must be 'completed', got {state!r}"
            )

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError(
            f"Checkpoint {ckpt_path} model_state_dict must be a dictionary mapping"
        )

    dimension_values = {
        "vit_dim": payload["vit_dim"],
        "terrain_dim": payload["terrain_dim"],
        "num_classes": payload["num_classes"],
        "hidden_dim": payload.get("hidden_dim", 256),
    }
    dims: dict[str, int] = {}
    for key, value in dimension_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"Checkpoint {ckpt_path} field {key!r} must be a positive integer"
            )
        parsed = int(value)
        if parsed < 1:
            raise ValueError(
                f"Checkpoint {ckpt_path} field {key!r} must be a positive integer"
            )
        dims[key] = parsed

    try:
        model = ScenicRegressionModel(**dims).to(device)
        model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Checkpoint {ckpt_path} model state is incompatible with its architecture"
        ) from exc
    model.eval()
    return model


def _read_benchmark_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read human benchmark CSV into list of dicts with normalized keys."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")

    import csv

    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    if not records:
        raise ValueError(f"Benchmark CSV is empty: {path}")
    return records


def _predict_dataset(
    model: ScenicRegressionModel,
    vit_embeddings: np.ndarray,
    terrain_features: np.ndarray,
    class_logits: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Run model inference over dataset arrays in batches."""
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    n_samples = len(vit_embeddings)
    preds = []
    use_cuda = resolved_device.startswith("cuda")

    def move(values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(values).float()
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(resolved_device, non_blocking=use_cuda)

    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_cuda):
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            out = model(
                move(vit_embeddings[start_idx:end_idx]),
                move(terrain_features[start_idx:end_idx]),
                move(class_logits[start_idx:end_idx]),
            )
            preds.append(out.float().squeeze(-1).cpu().numpy())

    if not preds:
        return np.array([], dtype=np.float32)

    all_preds = np.concatenate(preds, axis=0)
    if np.isnan(all_preds).any() or np.isinf(all_preds).any():
        raise ValueError("Model inference produced NaN or Inf predictions")
    return all_preds


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics between ground truth and predictions."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Array length mismatch for metrics computation: {len(y_true)} vs {len(y_pred)}"
        )

    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    # R2 score
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(errors**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Pearson and Spearman correlation
    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr_matrix = np.corrcoef(y_true, y_pred)
        pearson_corr = float(corr_matrix[0, 1])

        # Spearman rank corr
        from scipy.stats import spearmanr

        try:
            spear_res = spearmanr(y_true, y_pred)
            spearman_corr = float(
                spear_res.statistic if hasattr(spear_res, "statistic") else spear_res[0]
            )
        except Exception:
            spearman_corr = pearson_corr
    else:
        pearson_corr = 0.0
        spearman_corr = 0.0
    return {
        "samples": int(len(y_true)),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    }


load_model_checkpoint = _load_model_checkpoint
read_benchmark_csv = _read_benchmark_csv
predict_dataset = _predict_dataset
compute_metrics = _compute_metrics


def _load_dataset_npz(
    ds_path: Path, *, require_embedded_splits: bool
) -> dict[str, Any]:
    """Load and validate a feature NPZ and its optional embedded split contract."""
    data_npz = np.load(ds_path, allow_pickle=False)
    required_npz_keys = {
        "vit_embeddings",
        "terrain_features",
        "class_logits",
        "scenic_scores",
        "sample_weights",
        "image_paths",
    }
    missing_npz = sorted(required_npz_keys - set(data_npz.files))
    if missing_npz:
        raise ValueError(
            f"Dataset NPZ {ds_path} missing required fields: {missing_npz}"
        )

    vit = data_npz["vit_embeddings"]
    terr = data_npz["terrain_features"]
    cls_logits = data_npz["class_logits"]
    npz_scores = data_npz["scenic_scores"]
    sample_weights = data_npz["sample_weights"]
    image_paths_raw = data_npz["image_paths"]
    if require_embedded_splits and "splits" not in data_npz.files:
        raise ValueError(f"Dataset NPZ {ds_path} lacks required embedded splits")
    splits_raw = data_npz["splits"] if "splits" in data_npz.files else None

    n_samples = len(vit)
    lengths_match = (
        len(terr) == n_samples
        and len(cls_logits) == n_samples
        and len(npz_scores) == n_samples
        and len(sample_weights) == n_samples
        and len(image_paths_raw) == n_samples
        and (splits_raw is None or len(splits_raw) == n_samples)
    )
    if not lengths_match:
        raise ValueError("NPZ arrays length mismatch across required fields")

    if not (
        np.isfinite(vit).all()
        and np.isfinite(terr).all()
        and np.isfinite(cls_logits).all()
        and np.isfinite(npz_scores).all()
        and np.isfinite(sample_weights).all()
    ):
        raise ValueError("NPZ contains non-finite values (NaN or Inf)")

    if np.any(sample_weights <= 0):
        raise ValueError("NPZ sample_weights must be strictly positive (> 0)")

    image_paths = [
        str(p.decode("utf-8") if isinstance(p, bytes) else p) for p in image_paths_raw
    ]
    prepared_splits = (
        [
            str(value.decode("utf-8") if isinstance(value, bytes) else value)
            .strip()
            .lower()
            for value in splits_raw
        ]
        if splits_raw is not None
        else []
    )
    if prepared_splits and any(
        value not in {"train", "val", "test"} for value in prepared_splits
    ):
        raise ValueError("NPZ splits contain invalid values")

    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate image_paths found in NPZ dataset")

    return {
        "path": ds_path,
        "vit": vit,
        "terr": terr,
        "cls_logits": cls_logits,
        "scores": npz_scores,
        "weights": sample_weights,
        "image_paths": image_paths,
        "splits": prepared_splits,
        "path_map": {p: i for i, p in enumerate(image_paths)},
        "split_map": (
            dict(zip(image_paths, prepared_splits, strict=True))
            if prepared_splits
            else {}
        ),
        "total": len(image_paths),
        "test": sum(1 for s in prepared_splits if s == "test"),
    }


def evaluate_active_baseline(
    dataset_path: str | Path,
    control_dataset_path: str | Path,
    checkpoint_path: str | Path,
    expanded_benchmark_csv: str | Path,
    control_benchmark_csv: str | Path,
    output_path: str | Path | None = None,
    min_supported_slice_samples: int = 5,
) -> dict[str, Any]:
    """
    Perform deterministic baseline-reproduction evaluation for the active checkpoint.

    Evaluates active baseline checkpoint on split=test for expanded and control human
    benchmark CSVs, loading and inferring the immutable expanded prepared NPZ and the
    canonical control dataset NPZ independently. Requires exact NPZ test split matching,
    disjoint datasets/benchmarks, and deterministic CPU prediction.
    """
    ds_path = Path(dataset_path)
    ctrl_ds_path = Path(control_dataset_path)
    ckpt_path = Path(checkpoint_path)
    exp_csv_path = Path(expanded_benchmark_csv)
    ctrl_csv_path = Path(control_benchmark_csv)

    for p, name in [
        (ds_path, "Dataset NPZ"),
        (ctrl_ds_path, "Control dataset NPZ"),
        (ckpt_path, "Checkpoint"),
        (exp_csv_path, "Expanded benchmark CSV"),
        (ctrl_csv_path, "Control benchmark CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    ds_sha256 = file_sha256(ds_path)
    ctrl_ds_sha256 = file_sha256(ctrl_ds_path)
    ckpt_sha256 = file_sha256(ckpt_path)
    exp_sha256 = file_sha256(exp_csv_path)
    ctrl_sha256 = file_sha256(ctrl_csv_path)

    # 1. Load and validate the expanded and control NPZ datasets independently
    expanded = _load_dataset_npz(ds_path, require_embedded_splits=True)
    control = _load_dataset_npz(ctrl_ds_path, require_embedded_splits=False)

    dataset_overlap = set(expanded["image_paths"]) & set(control["image_paths"])
    if dataset_overlap:
        raise ValueError(
            f"Overlap detected between expanded and control prepared datasets: "
            f"{sorted(dataset_overlap)}"
        )

    # 2. Load model and run double CPU inference for determinism check on both datasets
    model = _load_model_checkpoint(ckpt_path, device="cpu", is_candidate=False)

    preds_run1 = _predict_dataset(
        model, expanded["vit"], expanded["terr"], expanded["cls_logits"], device="cpu"
    )
    preds_run2 = _predict_dataset(
        model, expanded["vit"], expanded["terr"], expanded["cls_logits"], device="cpu"
    )
    max_delta_expanded = float(np.max(np.abs(preds_run1 - preds_run2)))

    ctrl_preds_run1 = _predict_dataset(
        model, control["vit"], control["terr"], control["cls_logits"], device="cpu"
    )
    ctrl_preds_run2 = _predict_dataset(
        model, control["vit"], control["terr"], control["cls_logits"], device="cpu"
    )
    max_delta_control = float(np.max(np.abs(ctrl_preds_run1 - ctrl_preds_run2)))

    max_delta = max(max_delta_expanded, max_delta_control)
    tolerance = 1e-7
    if max_delta > tolerance:
        raise ValueError(
            f"Deterministic CPU inference failure: max absolute difference {max_delta} exceeds tolerance {tolerance}"
        )

    pred_map = {p: float(pred) for p, pred in zip(expanded["image_paths"], preds_run1)}
    ctrl_pred_map = {
        p: float(pred) for p, pred in zip(control["image_paths"], ctrl_preds_run1)
    }

    # 3. Benchmark processing helper: a benchmark resolves only against its own dataset
    def process_benchmark(
        csv_path: Path,
        name: str,
        path_map: dict[str, int],
        split_map: dict[str, str],
        dataset_pred_map: dict[str, float],
    ) -> dict[str, Any]:
        records = _read_benchmark_csv(csv_path)
        if any("split" not in record for record in records):
            raise ValueError(f"Benchmark {name} requires an explicit split column")
        test_records = [
            record
            for record in records
            if str(record["split"]).strip().lower() == "test"
        ]
        if not test_records:
            raise ValueError(f"Benchmark {name} contains no split=test rows")

        matched_records = []
        missing_paths = []
        for record in test_records:
            score_val = record.get("scenic_human_mean")
            if score_val is None or score_val == "":
                score_val = record.get("scenic_human")
            if score_val is None or score_val == "":
                raise ValueError(
                    f"Benchmark {name} test target must be scenic_human_mean "
                    "or scenic_human, never weak/mixed scenic_score"
                )
            image_path = (
                record.get("image_path")
                or record.get("image_paths")
                or record.get("path")
                or record.get("image")
            )
            if not image_path:
                raise ValueError(f"Benchmark {name} record lacks image_path")
            image_path = str(image_path)
            if image_path not in path_map:
                missing_paths.append(image_path)
                continue
            if split_map and split_map[image_path] != "test":
                raise ValueError(
                    f"Benchmark {name} relabels non-test prepared identity: {image_path}"
                )
            try:
                target = float(score_val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Benchmark {name} target must be numeric") from exc
            if not math.isfinite(target) or not 0 <= target <= 10:
                raise ValueError(
                    f"Benchmark {name} human targets must be finite and in [0, 10]"
                )
            matched_records.append(
                {
                    "image_path": image_path,
                    "target": target,
                    "region": record.get("region", "default"),
                    "slice": record.get(
                        "slice",
                        record.get("terrain_type", record.get("region", "default")),
                    ),
                    "pred": dataset_pred_map[image_path],
                }
            )

        if missing_paths:
            raise ValueError(
                f"Benchmark {name} has {len(missing_paths)} test paths absent "
                "from the prepared dataset"
            )
        benchmark_paths = [record["image_path"] for record in matched_records]
        if len(benchmark_paths) != len(set(benchmark_paths)):
            raise ValueError(f"Benchmark {name} contains duplicate test image paths")
        if len(matched_records) != len(test_records):
            raise ValueError(f"Benchmark {name} test denominator mismatch")

        y_true = np.array([mr["target"] for mr in matched_records], dtype=np.float32)
        y_pred = np.array([mr["pred"] for mr in matched_records], dtype=np.float32)
        if not np.isfinite(y_true).all():
            raise ValueError(f"Benchmark {name} targets must be finite")

        metrics = _compute_metrics(y_true, y_pred)

        sliced_results = {}
        region_results = {}
        for group_name, group_key in (("slice", "slice"), ("region", "region")):
            groups: dict[str, list[dict]] = {}
            for rec in matched_records:
                groups.setdefault(str(rec[group_key]), []).append(rec)
            target_results = sliced_results if group_name == "slice" else region_results
            for s_name, s_recs in groups.items():
                if not s_recs:
                    raise ValueError(f"Supported {group_name} collapse: {s_name}")
                s_yt = np.array([r["target"] for r in s_recs], dtype=np.float32)
                s_yp = np.array([r["pred"] for r in s_recs], dtype=np.float32)
                s_m = _compute_metrics(s_yt, s_yp)
                is_supported = len(s_recs) >= min_supported_slice_samples
                target_results[s_name] = {
                    "metrics": s_m,
                    "samples": len(s_recs),
                    "supported": is_supported,
                }

        return {
            "records": matched_records,
            "metrics": metrics,
            "sliced_metrics": sliced_results,
            "region_metrics": region_results,
        }

    # 4. Evaluate each benchmark only against its corresponding dataset identity map
    exp_res = process_benchmark(
        exp_csv_path,
        "expanded_human_benchmark",
        expanded["path_map"],
        expanded["split_map"],
        pred_map,
    )
    ctrl_res = process_benchmark(
        ctrl_csv_path,
        "control_benchmark",
        control["path_map"],
        control["split_map"],
        ctrl_pred_map,
    )

    exp_test_paths = {r["image_path"] for r in exp_res["records"]}
    ctrl_test_paths = {r["image_path"] for r in ctrl_res["records"]}
    overlap = exp_test_paths & ctrl_test_paths
    if overlap:
        raise ValueError(
            f"Overlap detected between expanded and control benchmark split=test image identities: {sorted(overlap)}"
        )

    # 5. Calibration and Prediction Distribution Summary
    exp_yt = np.array([r["target"] for r in exp_res["records"]], dtype=np.float32)
    exp_yp = np.array([r["pred"] for r in exp_res["records"]], dtype=np.float32)

    score_bins = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
    bin_errors = []
    for bin_min, bin_max in score_bins:
        mask = (exp_yt >= bin_min) & (
            exp_yt < bin_max if bin_max < 10.0 else exp_yt <= bin_max
        )
        if np.any(mask):
            bin_mae = float(np.mean(np.abs(exp_yp[mask] - exp_yt[mask])))
            bin_errors.append(bin_mae)

    calibration_error = float(np.mean(bin_errors)) if bin_errors else 0.0
    pred_std = float(np.std(exp_yp))
    target_std = float(np.std(exp_yt))
    spread_ratio = pred_std / target_std if target_std > 1e-8 else 0.0
    mean_drift = float(abs(np.mean(exp_yp) - np.mean(exp_yt)))
    sat_count = int(np.sum((exp_yp <= 0.0) | (exp_yp >= 10.0)))
    sat_ratio = float(sat_count / len(exp_yp)) if len(exp_yp) > 0 else 0.0
    unique_vals = len(np.unique(np.round(exp_yp, 4)))
    unique_ratio = float(unique_vals / len(exp_yp)) if len(exp_yp) > 0 else 0.0

    calibration_distribution = {
        "expanded_human_benchmark": {
            "calibration_error": calibration_error,
            "binned_mae": bin_errors,
            "prediction_mean": float(np.mean(exp_yp)),
            "prediction_std": pred_std,
            "prediction_min": float(np.min(exp_yp)),
            "prediction_max": float(np.max(exp_yp)),
            "target_mean": float(np.mean(exp_yt)),
            "target_std": target_std,
            "spread_ratio": spread_ratio,
            "mean_drift_vs_target": mean_drift,
            "saturation_ratio": sat_ratio,
            "unique_ratio": unique_ratio,
        }
    }

    # 6. Build final response dictionary
    summary = {
        "hashes": {
            "dataset_sha256": ds_sha256,
            "control_dataset_sha256": ctrl_ds_sha256,
            "checkpoint_sha256": ckpt_sha256,
            "expanded_benchmark_sha256": exp_sha256,
            "control_benchmark_sha256": ctrl_sha256,
        },
        "deterministic_inference": {
            "tolerance": tolerance,
            "max_absolute_difference": max_delta,
            "expanded_dataset_max_absolute_difference": max_delta_expanded,
            "control_dataset_max_absolute_difference": max_delta_control,
        },
        "sample_counts": {
            "dataset_total": expanded["total"],
            "dataset_test": expanded["test"],
            "control_dataset_total": control["total"],
            "control_dataset_test": len(ctrl_res["records"]),
            "expanded_benchmark_test": len(exp_res["records"]),
            "control_benchmark_test": len(ctrl_res["records"]),
        },
        "benchmarks": {
            "expanded_human_benchmark": {
                "metrics": exp_res["metrics"],
                "sliced_metrics": exp_res["sliced_metrics"],
                "region_metrics": exp_res["region_metrics"],
            },
            "control_benchmark": {
                "metrics": ctrl_res["metrics"],
                "sliced_metrics": ctrl_res["sliced_metrics"],
                "region_metrics": ctrl_res["region_metrics"],
            },
        },
        "calibration_distribution_summary": calibration_distribution,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=str(out_p.parent), delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(json.dumps(summary, indent=2))
            tmp_name = tmp.name
        os.replace(tmp_name, out_p)

    return summary


reproduce_active_baseline = evaluate_active_baseline


def evaluate_stage_two(
    dataset_path: str | Path,
    control_dataset_path: str | Path,
    candidate_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
    expanded_benchmark_csv: str | Path,
    control_benchmark_csv: str | Path,
    route_qa_json: str | Path,
    thresholds: dict[str, Any] | str | Path | None,
    output_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Perform strict same-denominator Stage-Two evaluation of candidate vs baseline models.

    Loads and infers the immutable expanded prepared NPZ and the canonical control
    dataset NPZ independently; each benchmark is evaluated only against its own
    dataset identity map. Produces machine-readable compound decision JSON written
    to output_path.
    """
    dataset_path = Path(dataset_path)
    control_dataset_path = Path(control_dataset_path)
    output_path = Path(output_path)

    if thresholds is None:
        thresh_dict: dict[str, Any] = {}
    elif isinstance(thresholds, (str, Path)):
        t_path = Path(thresholds)
        if not t_path.exists():
            raise FileNotFoundError(f"Thresholds file not found: {t_path}")
        thresh_dict = json.loads(t_path.read_text(encoding="utf-8"))
    else:
        thresh_dict = dict(thresholds)
    min_expanded_corr = float(thresh_dict.get("min_expanded_corr", 0.80))
    max_expanded_mse = float(thresh_dict.get("max_expanded_mse", 2.0))
    min_expanded_mse_improvement = float(
        thresh_dict.get("min_expanded_mse_improvement", 0.0)
    )
    max_control_mse_regression = float(
        thresh_dict.get("max_control_mse_regression", 0.05)
    )
    min_control_corr = float(thresh_dict.get("min_control_corr", 0.75))
    max_worst_slice_mse = float(thresh_dict.get("max_worst_slice_mse", 2.5))
    max_calibration_error = float(thresh_dict.get("max_calibration_error", 1.5))
    # Route, stability, and complexity gates consume structured boolean evidence.
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_path}")
    if not control_dataset_path.exists():
        raise FileNotFoundError(
            f"Control dataset NPZ not found: {control_dataset_path}"
        )
    if any(float(v) < 0 for v in thresh_dict.values() if isinstance(v, (int, float))):
        raise ValueError("Thresholds must not contain negative numeric values")

    # Load and validate the expanded and control NPZ datasets independently
    expanded = _load_dataset_npz(dataset_path, require_embedded_splits=True)
    control = _load_dataset_npz(control_dataset_path, require_embedded_splits=False)

    dataset_overlap = set(expanded["image_paths"]) & set(control["image_paths"])
    if dataset_overlap:
        raise ValueError(
            f"Overlap detected between expanded and control prepared datasets: "
            f"{sorted(dataset_overlap)}"
        )

    min_supported_slice_samples = int(thresh_dict.get("min_supported_slice_samples", 5))
    resolved_device = resolve_device(device)

    # 2. Load models
    candidate_model = _load_model_checkpoint(
        candidate_checkpoint, device=resolved_device, is_candidate=True
    )
    baseline_model = _load_model_checkpoint(
        baseline_checkpoint, device=resolved_device, is_candidate=False
    )

    candidate_sha256 = file_sha256(candidate_checkpoint)
    baseline_sha256 = file_sha256(baseline_checkpoint)
    dataset_sha256 = file_sha256(dataset_path)
    control_dataset_sha256 = file_sha256(control_dataset_path)

    # 3. Model inference on both datasets independently
    cand_npz_preds = _predict_dataset(
        candidate_model,
        expanded["vit"],
        expanded["terr"],
        expanded["cls_logits"],
        device=resolved_device,
    )
    base_npz_preds = _predict_dataset(
        baseline_model,
        expanded["vit"],
        expanded["terr"],
        expanded["cls_logits"],
        device=resolved_device,
    )
    cand_ctrl_preds = _predict_dataset(
        candidate_model,
        control["vit"],
        control["terr"],
        control["cls_logits"],
        device=resolved_device,
    )
    base_ctrl_preds = _predict_dataset(
        baseline_model,
        control["vit"],
        control["terr"],
        control["cls_logits"],
        device=resolved_device,
    )

    cand_pred_map = {
        p: float(pred) for p, pred in zip(expanded["image_paths"], cand_npz_preds)
    }
    base_pred_map = {
        p: float(pred) for p, pred in zip(expanded["image_paths"], base_npz_preds)
    }
    cand_ctrl_pred_map = {
        p: float(pred) for p, pred in zip(control["image_paths"], cand_ctrl_preds)
    }
    base_ctrl_pred_map = {
        p: float(pred) for p, pred in zip(control["image_paths"], base_ctrl_preds)
    }

    # Helper function to process benchmark CSV against its own dataset's predictions
    def process_benchmark(
        csv_path: str | Path,
        name: str,
        path_map: dict[str, int],
        split_map: dict[str, str],
        cand_map: dict[str, float],
        base_map: dict[str, float],
    ) -> dict[str, Any]:
        records = _read_benchmark_csv(csv_path)
        if any("split" not in record for record in records):
            raise ValueError(f"Benchmark {name} requires an explicit split column")
        test_records = [
            record
            for record in records
            if str(record["split"]).strip().lower() == "test"
        ]
        if not test_records:
            raise ValueError(f"Benchmark {name} contains no split=test rows")
        matched_records = []
        missing_paths = []
        for record in test_records:
            score_val = record.get("scenic_human_mean")
            if score_val is None or score_val == "":
                score_val = record.get("scenic_human")
            if score_val is None or score_val == "":
                raise ValueError(
                    f"Benchmark {name} test target must be scenic_human_mean "
                    "or scenic_human, never weak/mixed scenic_score"
                )
            image_path = (
                record.get("image_path")
                or record.get("image_paths")
                or record.get("path")
                or record.get("image")
            )
            if not image_path:
                raise ValueError(f"Benchmark {name} record lacks image_path")
            image_path = str(image_path)
            if image_path not in path_map:
                missing_paths.append(image_path)
                continue
            if split_map and split_map[image_path] != "test":
                raise ValueError(
                    f"Benchmark {name} relabels non-test prepared identity: "
                    f"{image_path}"
                )
            try:
                target = float(score_val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Benchmark {name} target must be numeric") from exc
            if not math.isfinite(target) or not 0 <= target <= 10:
                raise ValueError(
                    f"Benchmark {name} human targets must be finite and in [0, 10]"
                )
            matched_records.append(
                {
                    "image_path": image_path,
                    "target": target,
                    "region": record.get("region", "default"),
                    "slice": record.get(
                        "slice",
                        record.get("terrain_type", record.get("region", "default")),
                    ),
                    "cand_pred": cand_map[image_path],
                    "base_pred": base_map[image_path],
                }
            )
        if missing_paths:
            raise ValueError(
                f"Benchmark {name} has {len(missing_paths)} test paths absent "
                "from the prepared dataset"
            )
        benchmark_paths = [record["image_path"] for record in matched_records]
        if len(benchmark_paths) != len(set(benchmark_paths)):
            raise ValueError(f"Benchmark {name} contains duplicate test image paths")
        if len(matched_records) != len(test_records):
            raise ValueError(f"Benchmark {name} test denominator mismatch")
        y_true = np.array([mr["target"] for mr in matched_records], dtype=np.float32)
        y_cand = np.array([mr["cand_pred"] for mr in matched_records], dtype=np.float32)
        y_base = np.array([mr["base_pred"] for mr in matched_records], dtype=np.float32)
        if not np.isfinite(y_true).all():
            raise ValueError(f"Benchmark {name} targets must be finite")

        cand_metrics = _compute_metrics(y_true, y_cand)
        base_metrics = _compute_metrics(y_true, y_base)

        sliced_results = {}
        region_results = {}
        worst_slice_mse = 0.0
        has_supported_slice = False
        for group_name, group_key in (("slice", "slice"), ("region", "region")):
            groups: dict[str, list[dict]] = {}
            for rec in matched_records:
                groups.setdefault(str(rec[group_key]), []).append(rec)
            target_results = sliced_results if group_name == "slice" else region_results
            for s_name, s_recs in groups.items():
                if not s_recs:
                    raise ValueError(f"Supported {group_name} collapse: {s_name}")
                s_yt = np.array([r["target"] for r in s_recs], dtype=np.float32)
                s_yc = np.array([r["cand_pred"] for r in s_recs], dtype=np.float32)
                s_yb = np.array([r["base_pred"] for r in s_recs], dtype=np.float32)
                s_cand_m = _compute_metrics(s_yt, s_yc)
                s_base_m = _compute_metrics(s_yt, s_yb)
                is_supported = len(s_recs) >= min_supported_slice_samples
                target_results[s_name] = {
                    "candidate": s_cand_m,
                    "baseline": s_base_m,
                    "samples": len(s_recs),
                    "supported": is_supported,
                }
                if is_supported:
                    worst_slice_mse = max(worst_slice_mse, s_cand_m["mse"])
                    has_supported_slice = True

        if not has_supported_slice:
            worst_slice_mse = cand_metrics["mse"]

        return {
            "records": matched_records,
            "candidate_metrics": cand_metrics,
            "baseline_metrics": base_metrics,
            "sliced_metrics": sliced_results,
            "region_metrics": region_results,
            "worst_slice_mse": worst_slice_mse,
        }

    # 4. Process each benchmark only against its corresponding dataset identity map
    exp_res = process_benchmark(
        expanded_benchmark_csv,
        "expanded_human_benchmark",
        expanded["path_map"],
        expanded["split_map"],
        cand_pred_map,
        base_pred_map,
    )
    ctrl_res = process_benchmark(
        control_benchmark_csv,
        "control_benchmark",
        control["path_map"],
        control["split_map"],
        cand_ctrl_pred_map,
        base_ctrl_pred_map,
    )

    exp_test_paths = {r["image_path"] for r in exp_res["records"]}
    ctrl_test_paths = {r["image_path"] for r in ctrl_res["records"]}
    benchmark_overlap = exp_test_paths & ctrl_test_paths
    if benchmark_overlap:
        raise ValueError(
            f"Overlap detected between expanded and control benchmark split=test image identities: {sorted(benchmark_overlap)}"
        )

    # 5. Calibration and Prediction Distribution QA
    exp_yt = np.array([r["target"] for r in exp_res["records"]], dtype=np.float32)
    exp_yc = np.array([r["cand_pred"] for r in exp_res["records"]], dtype=np.float32)
    exp_yb = np.array([r["base_pred"] for r in exp_res["records"]], dtype=np.float32)

    # Calculate binned calibration error
    score_bins = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
    bin_errors = []
    for bin_min, bin_max in score_bins:
        mask = (exp_yt >= bin_min) & (
            exp_yt < bin_max if bin_max < 10.0 else exp_yt <= bin_max
        )
        if np.any(mask):
            bin_mae = float(np.mean(np.abs(exp_yc[mask] - exp_yt[mask])))
            bin_errors.append(bin_mae)
    calibration_error = float(np.mean(bin_errors)) if bin_errors else 0.0

    cand_min = float(np.min(exp_yc))
    cand_max = float(np.max(exp_yc))
    cand_range = cand_max - cand_min
    cand_std = float(np.std(exp_yc))
    target_std = float(np.std(exp_yt))
    spread_ratio = cand_std / target_std if target_std > 1e-8 else 0.0
    mean_drift = float(abs(np.mean(exp_yc) - np.mean(exp_yt)))

    sat_count = int(np.sum((exp_yc <= 0.0) | (exp_yc >= 10.0)))
    sat_ratio = float(sat_count / len(exp_yc))
    unique_vals = len(np.unique(np.round(exp_yc, 4)))
    unique_ratio = float(unique_vals / len(exp_yc))

    min_spread_ratio = float(thresh_dict.get("min_spread_ratio", 0.15))
    max_mean_drift = float(thresh_dict.get("max_mean_drift", 2.0))
    max_saturation_ratio = float(thresh_dict.get("max_saturation_ratio", 0.20))
    min_unique_ratio = float(thresh_dict.get("min_unique_ratio", 0.05))

    pred_range_pass = bool(
        np.isfinite(exp_yc).all()
        and cand_min >= 0.0
        and cand_max <= 10.0
        and cand_range > 1e-4
    )
    spread_ratio_pass = bool(spread_ratio >= min_spread_ratio and cand_std > 1e-4)
    mean_drift_pass = bool(mean_drift <= max_mean_drift)
    saturation_pass = bool(sat_ratio <= max_saturation_ratio)
    tie_pass = bool(unique_ratio >= min_unique_ratio)
    distribution_pass = bool(
        pred_range_pass
        and spread_ratio_pass
        and mean_drift_pass
        and saturation_pass
        and tie_pass
    )

    distribution_qa = {
        "candidate": {
            "mean": float(np.mean(exp_yc)),
            "std": cand_std,
            "min": cand_min,
            "max": cand_max,
        },
        "baseline": {
            "mean": float(np.mean(exp_yb)),
            "std": float(np.std(exp_yb)),
            "min": float(np.min(exp_yb)),
            "max": float(np.max(exp_yb)),
        },
        "target": {
            "mean": float(np.mean(exp_yt)),
            "std": target_std,
            "min": float(np.min(exp_yt)),
            "max": float(np.max(exp_yt)),
        },
        "prediction_drift_vs_baseline_mean": float(
            np.abs(np.mean(exp_yc) - np.mean(exp_yb))
        ),
        "prediction_drift_vs_target_mean": mean_drift,
        "spread_ratio": spread_ratio,
        "saturation_ratio": sat_ratio,
        "unique_ratio": unique_ratio,
        "calibration_error": calibration_error,
    }

    # 6. Read and validate Route QA evidence
    rq_path = Path(route_qa_json)
    if not rq_path.exists():
        raise FileNotFoundError(f"Route QA JSON file not found: {rq_path}")
    try:
        route_qa_data = json.loads(rq_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed Route QA JSON at {rq_path}: {exc}") from exc

    if not isinstance(route_qa_data, dict):
        raise ValueError(f"Route QA JSON {rq_path} must be a dictionary")

    routes = route_qa_data.get("routes")

    def _get_sha(data_dict: dict[str, Any], *keys: str) -> str | None:
        for k in keys:
            val = data_dict.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
            elif isinstance(val, dict):
                sub = (
                    val.get("sha256")
                    or val.get("checkpoint_sha256")
                    or val.get("report_sha256")
                )
                if isinstance(sub, str) and sub.strip():
                    return sub.strip().lower()
        return None

    # Top-level candidate and baseline checkpoint SHA-256 validation
    top_cand_sha = _get_sha(
        route_qa_data, "candidate_checkpoint_sha256", "candidate_sha256", "candidate"
    )
    top_base_sha = _get_sha(
        route_qa_data, "baseline_checkpoint_sha256", "baseline_sha256", "baseline"
    )
    top_checkpoints_match = bool(
        top_cand_sha
        and top_base_sha
        and top_cand_sha == candidate_sha256.lower()
        and top_base_sha == baseline_sha256.lower()
    )

    all_invariants_pass = route_qa_data.get("all_invariants_pass") is True

    stability_confirmed = route_qa_data.get("stability_confirmed") is True or (
        isinstance(route_qa_data.get("stability"), dict)
        and route_qa_data["stability"].get("confirmed") is True
    )

    complexity_accepted = route_qa_data.get("complexity_accepted") is True or (
        isinstance(route_qa_data.get("complexity"), dict)
        and route_qa_data["complexity"].get("accepted") is True
    )

    # Detailed per-route validation
    has_candidate_role = False
    has_baseline_role = False
    routes_valid = True

    if isinstance(routes, list) and len(routes) > 0:
        for r in routes:
            if not isinstance(r, dict) or len(r) == 0:
                routes_valid = False
                break

            role_str = (
                str(
                    r.get("role")
                    or r.get("route_role")
                    or r.get("kind")
                    or r.get("route_kind")
                    or ""
                )
                .strip()
                .lower()
            )

            if role_str in ("candidate", "scenic"):
                has_candidate_role = True
                expected_ckpt_sha = candidate_sha256.lower()
            elif role_str == "baseline":
                has_baseline_role = True
                expected_ckpt_sha = baseline_sha256.lower()
            else:
                routes_valid = False
                break

            # Declared route checkpoint hash
            r_ckpt_sha = _get_sha(
                r, "checkpoint_sha256", "checkpoint_hash", "checkpoint_sha", "sha256"
            )
            if not r_ckpt_sha or r_ckpt_sha != expected_ckpt_sha:
                routes_valid = False
                break

            # Report path validation
            rep_path_val = r.get("report_path") or r.get("report") or r.get("path")
            if not rep_path_val:
                routes_valid = False
                break
            rep_path = Path(rep_path_val)
            if not rep_path.is_file():
                routes_valid = False
                break

            # Report SHA-256 validation
            declared_rep_sha = _get_sha(r, "report_sha256", "report_hash")
            if not declared_rep_sha:
                routes_valid = False
                break
            try:
                actual_rep_sha = file_sha256(rep_path).lower()
            except Exception:
                routes_valid = False
                break
            if declared_rep_sha != actual_rep_sha:
                routes_valid = False
                break

            # Route invariants check
            r_inv_pass = (
                r.get("invariants_pass") is not False
                and r.get("invariants_passed") is not False
                and r.get("pass") is not False
            )
            if not r_inv_pass:
                routes_valid = False
                break
    else:
        routes_valid = False

    route_evidence_pass = bool(
        top_checkpoints_match
        and has_candidate_role
        and has_baseline_role
        and routes_valid
        and all_invariants_pass
        and stability_confirmed
        and complexity_accepted
    )
    # 7. Compound Decision Gates Evaluation
    exp_cand_mse = exp_res["candidate_metrics"]["mse"]
    exp_base_mse = exp_res["baseline_metrics"]["mse"]
    exp_cand_corr = exp_res["candidate_metrics"]["pearson_corr"]

    ctrl_cand_mse = ctrl_res["candidate_metrics"]["mse"]
    ctrl_base_mse = ctrl_res["baseline_metrics"]["mse"]
    ctrl_cand_corr = ctrl_res["candidate_metrics"]["pearson_corr"]
    exp_mse_improvement = exp_base_mse - exp_cand_mse
    ctrl_mse_delta = ctrl_cand_mse - ctrl_base_mse
    worst_slice_mse = max(exp_res["worst_slice_mse"], ctrl_res["worst_slice_mse"])
    gates = {
        "integrity_pass": bool(candidate_sha256 and baseline_sha256),
        "expanded_benchmark_corr_pass": bool(exp_cand_corr >= min_expanded_corr),
        "expanded_benchmark_mse_pass": bool(exp_cand_mse <= max_expanded_mse),
        "expanded_benchmark_improvement_pass": bool(
            exp_mse_improvement >= min_expanded_mse_improvement
        ),
        "control_benchmark_non_regression_pass": bool(
            ctrl_mse_delta <= max_control_mse_regression
        ),
        "control_benchmark_corr_pass": bool(ctrl_cand_corr >= min_control_corr),
        "worst_slice_mse_pass": bool(worst_slice_mse <= max_worst_slice_mse),
        "calibration_pass": bool(calibration_error <= max_calibration_error),
        "distribution_pass": distribution_pass,
        "route_evidence_pass": route_evidence_pass,
    }

    all_gates_pass = all(gates.values())

    decision_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_gates_pass": all_gates_pass,
        "candidate": {
            "checkpoint": str(candidate_checkpoint),
            "sha256": candidate_sha256,
        },
        "baseline": {
            "checkpoint": str(baseline_checkpoint),
            "sha256": baseline_sha256,
        },
        "expanded_dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
        },
        "control_dataset": {
            "path": str(control_dataset_path),
            "sha256": control_dataset_sha256,
        },
        "expanded_human_benchmark": {
            "region_metrics": exp_res["region_metrics"],
            "candidate_metrics": exp_res["candidate_metrics"],
            "baseline_metrics": exp_res["baseline_metrics"],
            "sliced_metrics": exp_res["sliced_metrics"],
            "worst_slice_mse": exp_res["worst_slice_mse"],
            "mse_improvement": exp_mse_improvement,
        },
        "control_benchmark": {
            "candidate_metrics": ctrl_res["candidate_metrics"],
            "baseline_metrics": ctrl_res["baseline_metrics"],
            "sliced_metrics": ctrl_res["sliced_metrics"],
            "worst_slice_mse": ctrl_res["worst_slice_mse"],
            "mse_delta": ctrl_mse_delta,
        },
        "calibration_and_distribution": distribution_qa,
        "route_qa_evidence": {
            "routes": routes,
            "all_invariants_pass": all_invariants_pass,
            "stability_confirmed": stability_confirmed,
            "complexity_accepted": complexity_accepted,
            "details": route_qa_data,
        },
        "thresholds_evaluated": thresh_dict,
        "gates": gates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision_dict, indent=2), encoding="utf-8")
    return decision_dict


def evaluate_candidate_validation(
    dataset_path: str | Path,
    candidate_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
    expanded_benchmark_csv: str | Path,
    output_path: str | Path,
    min_supported_slice_samples: int = 5,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Perform validation-only screening of candidate vs baseline models on split=val.

    Loads and validates the expanded dataset NPZ requiring embedded splits, verifies
    split=val rows in expanded_benchmark_csv against NPZ split=val identities, and computes
    candidate and baseline metrics on the exact unique split=val denominator. Held-out
    split=test rows are not evaluated into metrics. Writes machine-readable JSON to output_path.
    """
    ds_path = Path(dataset_path)
    cand_ckpt_path = Path(candidate_checkpoint)
    base_ckpt_path = Path(baseline_checkpoint)
    exp_csv_path = Path(expanded_benchmark_csv)
    out_path = Path(output_path)

    for p, name in [
        (ds_path, "Dataset NPZ"),
        (cand_ckpt_path, "Candidate checkpoint"),
        (base_ckpt_path, "Baseline checkpoint"),
        (exp_csv_path, "Expanded benchmark CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    candidate_sha256 = file_sha256(cand_ckpt_path)
    baseline_sha256 = file_sha256(base_ckpt_path)
    dataset_sha256 = file_sha256(ds_path)
    benchmark_sha256 = file_sha256(exp_csv_path)

    # 1. Load and validate NPZ dataset requiring embedded splits
    expanded = _load_dataset_npz(ds_path, require_embedded_splits=True)

    resolved_device = resolve_device(device)

    # 2. Load models
    candidate_model = _load_model_checkpoint(
        cand_ckpt_path, device=resolved_device, is_candidate=True
    )
    baseline_model = _load_model_checkpoint(
        base_ckpt_path, device=resolved_device, is_candidate=False
    )

    # 3. Model inference over dataset arrays
    cand_npz_preds = _predict_dataset(
        candidate_model,
        expanded["vit"],
        expanded["terr"],
        expanded["cls_logits"],
        device=resolved_device,
    )
    base_npz_preds = _predict_dataset(
        baseline_model,
        expanded["vit"],
        expanded["terr"],
        expanded["cls_logits"],
        device=resolved_device,
    )

    cand_pred_map = {
        p: float(pred) for p, pred in zip(expanded["image_paths"], cand_npz_preds)
    }
    base_pred_map = {
        p: float(pred) for p, pred in zip(expanded["image_paths"], base_npz_preds)
    }

    # 4. Read and validate expanded benchmark CSV filtering ONLY split=val rows
    records = _read_benchmark_csv(exp_csv_path)
    if any("split" not in record for record in records):
        raise ValueError(
            "Benchmark expanded_validation_benchmark requires an explicit split column"
        )

    val_records = [
        record for record in records if str(record["split"]).strip().lower() == "val"
    ]
    if not val_records:
        raise ValueError(
            "Benchmark expanded_validation_benchmark contains no split=val rows"
        )

    matched_records = []
    missing_paths = []
    for record in val_records:
        score_val = record.get("scenic_human_mean")
        if score_val is None or score_val == "":
            score_val = record.get("scenic_human")
        if score_val is None or score_val == "":
            raise ValueError(
                "Benchmark expanded_validation_benchmark val target must be scenic_human_mean "
                "or scenic_human, never weak/mixed scenic_score"
            )
        image_path = (
            record.get("image_path")
            or record.get("image_paths")
            or record.get("path")
            or record.get("image")
        )
        if not image_path:
            raise ValueError(
                "Benchmark expanded_validation_benchmark record lacks image_path"
            )
        image_path = str(image_path)
        if image_path not in expanded["path_map"]:
            missing_paths.append(image_path)
            continue
        if expanded["split_map"] and expanded["split_map"][image_path] != "val":
            raise ValueError(
                "Benchmark expanded_validation_benchmark relabels non-val prepared identity: "
                f"{image_path}"
            )
        try:
            target = float(score_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Benchmark expanded_validation_benchmark target must be numeric"
            ) from exc
        if not math.isfinite(target) or not 0 <= target <= 10:
            raise ValueError(
                "Benchmark expanded_validation_benchmark human targets must be finite and in [0, 10]"
            )
        matched_records.append(
            {
                "image_path": image_path,
                "target": target,
                "region": record.get("region", "default"),
                "slice": record.get(
                    "slice",
                    record.get("terrain_type", record.get("region", "default")),
                ),
                "cand_pred": cand_pred_map[image_path],
                "base_pred": base_pred_map[image_path],
            }
        )

    if missing_paths:
        raise ValueError(
            f"Benchmark expanded_validation_benchmark has {len(missing_paths)} val paths absent "
            "from the prepared dataset"
        )
    benchmark_paths = [rec["image_path"] for rec in matched_records]
    if len(benchmark_paths) != len(set(benchmark_paths)):
        raise ValueError(
            "Benchmark expanded_validation_benchmark contains duplicate val image paths"
        )
    if len(matched_records) != len(val_records):
        raise ValueError(
            "Benchmark expanded_validation_benchmark val denominator mismatch"
        )

    # 5. Compute candidate & baseline metrics on exact val denominator
    y_true = np.array([mr["target"] for mr in matched_records], dtype=np.float32)
    y_cand = np.array([mr["cand_pred"] for mr in matched_records], dtype=np.float32)
    y_base = np.array([mr["base_pred"] for mr in matched_records], dtype=np.float32)
    if not np.isfinite(y_true).all():
        raise ValueError(
            "Benchmark expanded_validation_benchmark targets must be finite"
        )

    cand_metrics = _compute_metrics(y_true, y_cand)
    base_metrics = _compute_metrics(y_true, y_base)
    mse_improvement = float(base_metrics["mse"] - cand_metrics["mse"])

    # 6. Sliced and region metrics
    sliced_results = {}
    region_results = {}
    worst_slice_mse = 0.0
    has_supported_slice = False
    for group_name, group_key in (("slice", "slice"), ("region", "region")):
        groups: dict[str, list[dict]] = {}
        for rec in matched_records:
            groups.setdefault(str(rec[group_key]), []).append(rec)
        target_results = sliced_results if group_name == "slice" else region_results
        for s_name, s_recs in groups.items():
            if not s_recs:
                raise ValueError(f"Supported {group_name} collapse: {s_name}")
            s_yt = np.array([r["target"] for r in s_recs], dtype=np.float32)
            s_yc = np.array([r["cand_pred"] for r in s_recs], dtype=np.float32)
            s_yb = np.array([r["base_pred"] for r in s_recs], dtype=np.float32)
            s_cand_m = _compute_metrics(s_yt, s_yc)
            s_base_m = _compute_metrics(s_yt, s_yb)
            is_supported = len(s_recs) >= min_supported_slice_samples
            target_results[s_name] = {
                "candidate": s_cand_m,
                "baseline": s_base_m,
                "samples": len(s_recs),
                "supported": is_supported,
            }
            if group_name == "slice" and is_supported:
                worst_slice_mse = max(worst_slice_mse, s_cand_m["mse"])
                has_supported_slice = True

    if not has_supported_slice:
        worst_slice_mse = cand_metrics["mse"]

    validation_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "checkpoint": str(candidate_checkpoint),
            "sha256": candidate_sha256,
        },
        "baseline": {
            "checkpoint": str(baseline_checkpoint),
            "sha256": baseline_sha256,
        },
        "expanded_dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
        },
        "expanded_benchmark": {
            "path": str(expanded_benchmark_csv),
            "sha256": benchmark_sha256,
        },
        "candidate_metrics": cand_metrics,
        "baseline_metrics": base_metrics,
        "mse_improvement": mse_improvement,
        "sliced_metrics": sliced_results,
        "region_metrics": region_results,
        "worst_slice_mse": worst_slice_mse,
        "expanded_validation_benchmark": {
            "candidate_metrics": cand_metrics,
            "baseline_metrics": base_metrics,
            "mse_improvement": mse_improvement,
            "sliced_metrics": sliced_results,
            "region_metrics": region_results,
            "worst_slice_mse": worst_slice_mse,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    return validation_summary


def _locked_registry(reg_path: Path):
    """Hold an advisory lock for the complete registry transaction."""
    lock_path = reg_path.with_name(reg_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+b")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _write_registry_locked(reg_path: Path, registry_data: dict[str, Any]) -> str:
    """Durably replace and reread a registry while caller holds its lock."""
    temp_name: str | None = None
    try:
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=reg_path.parent, delete=False, encoding="utf-8"
        ) as tf:
            temp_name = tf.name
            json.dump(registry_data, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(temp_name, reg_path)
        temp_name = None
        dir_fd = os.open(reg_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        reread = json.loads(reg_path.read_text(encoding="utf-8"))
        if reread != registry_data:
            raise ValueError("Registry reread verification failed")
        return file_sha256(reg_path)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _publish_checkpoint(cand_path: Path, cand_sha: str, ckpt_dir: Path) -> Path:
    """Atomically publish a content-addressed checkpoint copy into ckpt_dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    published_ckpt_path = ckpt_dir / f"{cand_sha.lower()}.pt"
    if published_ckpt_path.exists():
        if file_sha256(published_ckpt_path).lower() != cand_sha.lower():
            raise ValueError(
                f"Existing content-addressed checkpoint at {published_ckpt_path} hash mismatch"
            )
    else:
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=ckpt_dir, delete=False) as tf:
                temp_name = tf.name
                with open(cand_path, "rb") as cf:
                    while chunk := cf.read(65536):
                        tf.write(chunk)
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(temp_name, published_ckpt_path)
            temp_name = None
            dir_fd = os.open(ckpt_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
        if file_sha256(published_ckpt_path).lower() != cand_sha.lower():
            raise ValueError(
                f"Published checkpoint hash mismatch at {published_ckpt_path}"
            )
    return published_ckpt_path


def promote_from_decision(
    decision_path: str | Path,
    candidate_checkpoint: str | Path,
    registry_path: str | Path,
    expected_registry_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    """Promote only a passing, hash-matched candidate in one locked transaction."""
    dec_path, cand_path, reg_path = (
        Path(decision_path),
        Path(candidate_checkpoint),
        Path(registry_path),
    )
    if not dec_path.exists():
        raise FileNotFoundError(f"Decision file not found: {dec_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {cand_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")
    try:
        decision_data = json.loads(dec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed decision JSON at {dec_path}: {exc}") from exc
    if not isinstance(decision_data, dict):
        raise ValueError(
            f"Promotion rejected: decision at {dec_path} is not a dictionary"
        )
    if decision_data.get("all_gates_pass") is not True:
        raise ValueError(
            f"Promotion rejected: decision at {dec_path} has all_gates_pass != True"
        )
    dec_sha = decision_data.get("candidate", {}).get("sha256")
    cand_sha = file_sha256(cand_path)
    if not isinstance(dec_sha, str) or dec_sha != cand_sha:
        raise ValueError("Candidate SHA256 mismatch or missing decision evidence")
    _load_model_checkpoint(cand_path, is_candidate=True)
    exp_bm = decision_data.get("expanded_human_benchmark")
    if not isinstance(exp_bm, dict):
        raise ValueError(
            "Promotion rejected: candidate metrics are missing expanded_human_benchmark"
        )
    raw_metrics = exp_bm.get("candidate_metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError("Promotion rejected: candidate metrics dictionary is missing")

    corr_val = raw_metrics.get("corr")
    if corr_val is None:
        corr_val = raw_metrics.get("pearson_corr")
    mae_val = raw_metrics.get("mae")
    rmse_val = raw_metrics.get("rmse")
    samples_val = raw_metrics.get("samples")
    for name, val in [("corr", corr_val), ("mae", mae_val), ("rmse", rmse_val)]:
        if (
            val is None
            or isinstance(val, bool)
            or not isinstance(val, (int, float, np.integer, np.floating))
            or not math.isfinite(float(val))
        ):
            raise ValueError(
                f"Promotion rejected: candidate metric {name!r} is missing or non-finite"
            )

    if (
        samples_val is None
        or isinstance(samples_val, bool)
        or not isinstance(samples_val, (int, np.integer))
        or int(samples_val) < 1
    ):
        raise ValueError(
            "Promotion rejected: candidate metric 'samples' is missing or not a positive integer"
        )

    formatted_metrics = dict(raw_metrics)
    formatted_metrics["corr"] = float(corr_val)
    formatted_metrics["mae"] = float(mae_val)
    formatted_metrics["rmse"] = float(rmse_val)
    formatted_metrics["samples"] = int(samples_val)
    lock = _locked_registry(reg_path)
    try:
        actual = file_sha256(reg_path)
        if actual != expected_registry_sha256:
            raise ValueError(
                f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual}"
            )
        registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
        if not isinstance(registry_data.get("active"), dict):
            raise ValueError(f"Malformed registry at {reg_path}: missing active entry")
        history = registry_data.get("history", [])
        if not isinstance(history, list):
            raise ValueError("Malformed registry history")
        prior_active = dict(registry_data["active"])
        raw_active_ckpt = str(prior_active.get("checkpoint", ""))
        active_checkpoint = Path(raw_active_ckpt)
        if not active_checkpoint.is_absolute():
            cand = reg_path.parent / active_checkpoint
            if cand.is_file():
                active_checkpoint = cand
        if not active_checkpoint.is_file():
            raise ValueError("Promotion rejected: registry-active checkpoint is absent")
        actual_active_sha = file_sha256(active_checkpoint)
        recorded_active_sha = prior_active.get("sha256")
        if isinstance(recorded_active_sha, str) and recorded_active_sha.strip():
            if recorded_active_sha.lower() != actual_active_sha.lower():
                raise ValueError(
                    "Promotion rejected: registry-active checkpoint hash mismatch"
                )
        decision_baseline = decision_data.get("baseline", {})
        if not isinstance(decision_baseline, dict):
            raise ValueError("Promotion rejected: decision has no baseline evidence")
        decision_baseline_sha = decision_baseline.get("sha256")
        if not isinstance(decision_baseline_sha, str) or not decision_baseline_sha:
            raise ValueError("Promotion rejected: decision baseline sha256 is missing")
        if decision_baseline_sha.lower() != actual_active_sha.lower():
            raise ValueError(
                "Promotion rejected: decision baseline does not match "
                "registry-active checkpoint"
            )
        ckpt_dir = reg_path.parent / "checkpoints"
        _publish_checkpoint(cand_path, cand_sha, ckpt_dir)

        now_iso = datetime.now(timezone.utc).isoformat()
        new_active = {
            "checkpoint": f"checkpoints/{cand_sha.lower()}.pt",
            "promoted_at": now_iso,
            "updated_at": now_iso,
            "run_name": run_name,
            "sha256": cand_sha,
            "metrics": formatted_metrics,
            "decision_path": str(dec_path.resolve()),
        }
        history.append(
            {
                "event": "promote",
                "record": new_active,
                "prior_active": prior_active,
                "decision": decision_data,
            }
        )
        registry_data["active"], registry_data["history"] = new_active, history
        new_hash = _write_registry_locked(reg_path, registry_data)
        return {
            "status": "promoted",
            "run_name": run_name,
            "new_registry_sha256": new_hash,
            "active": new_active,
        }
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def promote_with_user_override(
    override_path: str | Path,
    candidate_checkpoint: str | Path,
    registry_path: str | Path,
    expected_registry_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    """
    Activate a rejected candidate via an explicit, hash-guarded user override.

    The override JSON MUST declare schema_version 1, action
    'activate_rejected_candidate', activation_approved exactly True, the candidate
    checkpoint path and SHA-256, hash bindings to the existing final decision and
    evaluation evidence files, and nonempty acknowledged failed-gates and risk
    lists. Every mismatch fails before any registry mutation. On success the
    candidate is atomically activated under the registry lock, the event
    'promote_user_override' is appended binding the override file/hash and
    evidence, and prior_active is preserved with its verified checkpoint SHA.
    """
    ovr_path, cand_path, reg_path = (
        Path(override_path),
        Path(candidate_checkpoint),
        Path(registry_path),
    )
    if not ovr_path.exists():
        raise FileNotFoundError(f"Override file not found: {ovr_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {cand_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")
    try:
        override_data = json.loads(ovr_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed override JSON at {ovr_path}: {exc}") from exc
    if not isinstance(override_data, dict):
        raise ValueError(f"Override at {ovr_path} is not a dictionary")

    # 1. Exact schema, action, and boolean approval
    if override_data.get("schema_version") != 1 or isinstance(
        override_data.get("schema_version"), bool
    ):
        raise ValueError(
            f"Override rejected: schema_version must be 1, "
            f"got {override_data.get('schema_version')!r}"
        )
    if override_data.get("action") != "activate_rejected_candidate":
        raise ValueError(
            f"Override rejected: action must be 'activate_rejected_candidate', "
            f"got {override_data.get('action')!r}"
        )
    if override_data.get("activation_approved") is not True:
        raise ValueError(
            f"Override rejected: activation_approved must be exactly True, "
            f"got {override_data.get('activation_approved')!r}"
        )

    # 2. Candidate checkpoint path and SHA-256
    candidate = override_data.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Override rejected: candidate evidence is missing")
    declared_cand_path = candidate.get("checkpoint")
    declared_cand_sha = candidate.get("sha256")
    if not isinstance(declared_cand_path, str) or not declared_cand_path.strip():
        raise ValueError("Override rejected: candidate checkpoint path is missing")
    if not isinstance(declared_cand_sha, str) or not declared_cand_sha.strip():
        raise ValueError("Override rejected: candidate sha256 is missing")
    if Path(declared_cand_path).resolve() != cand_path.resolve():
        raise ValueError(
            "Override rejected: candidate checkpoint path does not match "
            "the override declaration"
        )
    cand_sha = file_sha256(cand_path)
    if declared_cand_sha.lower() != cand_sha.lower():
        raise ValueError("Override rejected: candidate SHA256 mismatch")

    # 3. Nonempty failed-gate and risk acknowledgements
    failed_gates = override_data.get("acknowledged_failed_gates")
    if (
        not isinstance(failed_gates, list)
        or not failed_gates
        or any(not isinstance(g, str) or not g.strip() for g in failed_gates)
    ):
        raise ValueError(
            "Override rejected: acknowledged_failed_gates must be a "
            "nonempty list of strings"
        )
    risks = override_data.get("acknowledged_risks")
    if (
        not isinstance(risks, list)
        or not risks
        or any(not isinstance(r, str) or not r.strip() for r in risks)
    ):
        raise ValueError(
            "Override rejected: acknowledged_risks must be a nonempty list of strings"
        )

    # 4. Evidence bindings to the existing final decision/evaluation files
    evidence = override_data.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Override rejected: evidence is missing")
    bound_files: dict[str, tuple[Path, str]] = {}
    for key in ("final_decision", "evaluation"):
        binding = evidence.get(key)
        if not isinstance(binding, dict):
            raise ValueError(f"Override rejected: evidence.{key} binding is missing")
        ev_path_val = binding.get("path")
        ev_sha_val = binding.get("sha256")
        if not isinstance(ev_path_val, str) or not ev_path_val.strip():
            raise ValueError(f"Override rejected: evidence.{key} path is missing")
        if not isinstance(ev_sha_val, str) or not ev_sha_val.strip():
            raise ValueError(f"Override rejected: evidence.{key} sha256 is missing")
        ev_file = Path(ev_path_val)
        if not ev_file.is_file():
            raise FileNotFoundError(
                f"Override rejected: evidence file not found: {ev_file}"
            )
        actual_sha = file_sha256(ev_file)
        if ev_sha_val.lower() != actual_sha.lower():
            raise ValueError(
                f"Override rejected: evidence.{key} SHA256 mismatch at {ev_file}"
            )
        bound_files[key] = (ev_file, ev_sha_val)

    try:
        decision_evidence = json.loads(
            bound_files["final_decision"][0].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Override rejected: final decision evidence is not valid JSON"
        ) from exc
    if not isinstance(decision_evidence, dict):
        raise ValueError("Override rejected: final decision evidence is not an object")
    fresh_human_test = decision_evidence.get("fresh_human_test")
    if not isinstance(fresh_human_test, dict) or not isinstance(
        fresh_human_test.get("candidate"), dict
    ):
        raise ValueError(
            "Override rejected: final decision lacks fresh_human_test candidate metrics"
        )
    raw_metrics = fresh_human_test["candidate"]
    corr_val = raw_metrics.get("corr", raw_metrics.get("pearson_corr"))
    mae_val = raw_metrics.get("mae")
    rmse_val = raw_metrics.get("rmse")
    samples_val = raw_metrics.get("samples")
    for name, value in (("corr", corr_val), ("mae", mae_val), ("rmse", rmse_val)):
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"Override rejected: candidate metric {name!r} is missing or non-finite"
            )
    if (
        samples_val is None
        or isinstance(samples_val, bool)
        or not isinstance(samples_val, (int, np.integer))
        or int(samples_val) < 1
    ):
        raise ValueError(
            "Override rejected: candidate metric 'samples' is missing or not a positive integer"
        )
    candidate_metrics = dict(raw_metrics)
    candidate_metrics["corr"] = float(corr_val)
    candidate_metrics["mae"] = float(mae_val)
    candidate_metrics["rmse"] = float(rmse_val)
    candidate_metrics["samples"] = int(samples_val)

    # 5. Candidate must be a valid canonical checkpoint
    _load_model_checkpoint(cand_path, is_candidate=True)

    override_sha = file_sha256(ovr_path)
    lock = _locked_registry(reg_path)
    try:
        actual = file_sha256(reg_path)
        if actual != expected_registry_sha256:
            raise ValueError(
                f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual}"
            )
        registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
        if not isinstance(registry_data.get("active"), dict):
            raise ValueError(f"Malformed registry at {reg_path}: missing active entry")
        history = registry_data.get("history", [])
        if not isinstance(history, list):
            raise ValueError("Malformed registry history")
        prior_active = dict(registry_data["active"])
        raw_active_ckpt = str(prior_active.get("checkpoint", ""))
        active_checkpoint = Path(raw_active_ckpt)
        if not active_checkpoint.is_absolute():
            cand = reg_path.parent / active_checkpoint
            if cand.is_file():
                active_checkpoint = cand
        if not active_checkpoint.is_file():
            raise ValueError("Override rejected: registry-active checkpoint is absent")
        actual_active_sha = file_sha256(active_checkpoint)
        recorded_active_sha = prior_active.get("sha256")
        if isinstance(recorded_active_sha, str) and recorded_active_sha.strip():
            if recorded_active_sha.lower() != actual_active_sha.lower():
                raise ValueError(
                    "Override rejected: registry-active checkpoint hash mismatch"
                )
        # Preserve prior active bound to its verified checkpoint SHA
        prior_active["sha256"] = actual_active_sha

        ckpt_dir = reg_path.parent / "checkpoints"
        _publish_checkpoint(cand_path, cand_sha, ckpt_dir)

        evidence_bindings = {
            key: {"path": str(ev_file.resolve()), "sha256": ev_sha}
            for key, (ev_file, ev_sha) in bound_files.items()
        }
        now_iso = datetime.now(timezone.utc).isoformat()
        new_active = {
            "checkpoint": f"checkpoints/{cand_sha.lower()}.pt",
            "promoted_at": now_iso,
            "updated_at": now_iso,
            "run_name": run_name,
            "sha256": cand_sha,
            "metrics": candidate_metrics,
            "override_path": str(ovr_path.resolve()),
            "override_sha256": override_sha,
            "evidence": evidence_bindings,
            "acknowledged_failed_gates": failed_gates,
            "acknowledged_risks": risks,
            "decision_path": str(bound_files["final_decision"][0].resolve()),
        }
        history.append(
            {
                "event": "promote_user_override",
                "record": new_active,
                "prior_active": prior_active,
                "override": {
                    "path": str(ovr_path.resolve()),
                    "sha256": override_sha,
                },
                "evidence": evidence_bindings,
            }
        )
        registry_data["active"], registry_data["history"] = new_active, history
        new_hash = _write_registry_locked(reg_path, registry_data)
        return {
            "status": "activated_by_user_override",
            "run_name": run_name,
            "new_registry_sha256": new_hash,
            "active": new_active,
        }
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def rollback_registry(
    registry_path: str | Path,
    target_history_index: int,
    expected_registry_sha256: str,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Restore the state before a recorded event, or its historical model record."""
    reg_path = Path(registry_path)
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")
    if isinstance(target_history_index, bool) or not isinstance(
        target_history_index, (int, np.integer)
    ):
        raise ValueError("target_history_index must be a non-negative integer")
    if int(target_history_index) < 0:
        raise ValueError("target_history_index must be a non-negative integer")
    lock = _locked_registry(reg_path)
    try:
        actual = file_sha256(reg_path)
        if actual != expected_registry_sha256:
            raise ValueError(
                f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual}"
            )
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        if not isinstance(data.get("active"), dict):
            raise ValueError(f"Malformed registry at {reg_path}: missing active entry")
        history = data.get("history", [])
        if not isinstance(history, list) or not history:
            raise ValueError(f"Cannot rollback registry {reg_path}: history is empty")
        try:
            target_event = history[target_history_index]
        except IndexError as exc:
            raise ValueError("Invalid target history index") from exc
        if not isinstance(target_event, dict):
            raise ValueError("Historical registry event has no model record")
        prior_active = target_event.get("prior_active")
        historical_record = target_event.get("record")
        if isinstance(prior_active, dict):
            new_active = dict(prior_active)
        elif isinstance(historical_record, dict):
            new_active = dict(historical_record)
        else:
            raise ValueError("Historical registry event has no model record")
        checkpoint = Path(str(new_active.get("checkpoint", "")))
        if not checkpoint.is_absolute():
            cand = reg_path.parent / checkpoint
            if cand.is_file():
                checkpoint = cand
        recorded_sha = new_active.get("sha256")
        target_sha = (
            recorded_sha
            if isinstance(recorded_sha, str)
            else expected_checkpoint_sha256
        )
        if (
            not checkpoint.is_file()
            or not isinstance(target_sha, str)
            or file_sha256(checkpoint) != target_sha
        ):
            raise ValueError("Historical checkpoint identity validation failed")
        new_active["sha256"] = target_sha
        _load_model_checkpoint(checkpoint, is_candidate=False)
        prior = dict(data["active"])
        history.append(
            {
                "event": "rollback",
                "record": new_active,
                "prior_active": prior,
                "target_history_index": target_history_index,
            }
        )
        data["active"], data["history"] = new_active, history
        new_hash = _write_registry_locked(reg_path, data)
        return {
            "status": "rolled_back",
            "target_history_index": target_history_index,
            "new_registry_sha256": new_hash,
            "active": new_active,
        }
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
