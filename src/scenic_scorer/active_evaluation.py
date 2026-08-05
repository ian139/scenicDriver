"""
Active model evaluation, atomic promotion, and rollback module for Stage-Two scenic regression models.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from src.scenic_scorer.regression import ScenicRegressionModel


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


def _load_model_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> ScenicRegressionModel:
    """Load and validate a ScenicRegressionModel checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid PyTorch checkpoint at {ckpt_path}: {exc}") from exc
    
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid checkpoint structure in {ckpt_path}: expected dictionary mapping")
    
    required = {"model_state_dict", "vit_dim", "terrain_dim", "num_classes"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Checkpoint {ckpt_path} missing required keys: {missing}")
    
    dims = {k: int(payload[k]) for k in ("vit_dim", "terrain_dim", "num_classes")}
    model = ScenicRegressionModel(**dims).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
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
    batch_size: int = 128
) -> np.ndarray:
    """Run model inference over dataset arrays in batches."""
    n_samples = len(vit_embeddings)
    preds = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            vit_b = torch.from_numpy(vit_embeddings[start_idx:end_idx]).float()
            terr_b = torch.from_numpy(terrain_features[start_idx:end_idx]).float()
            cls_b = torch.from_numpy(class_logits[start_idx:end_idx]).float()
            
            out = model(vit_b, terr_b, cls_b)
            scores = out.squeeze(-1).cpu().numpy()
            preds.append(scores)
            
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
        raise ValueError(f"Array length mismatch for metrics computation: {len(y_true)} vs {len(y_pred)}")
    
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    
    # R2 score
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    
    # Pearson and Spearman correlation
    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr_matrix = np.corrcoef(y_true, y_pred)
        pearson_corr = float(corr_matrix[0, 1])
        
        # Spearman rank corr
        from scipy.stats import spearmanr
        try:
            spear_res = spearmanr(y_true, y_pred)
            spearman_corr = float(spear_res.statistic if hasattr(spear_res, "statistic") else spear_res[0])
        except Exception:
            spearman_corr = pearson_corr
    else:
        pearson_corr = 0.0
        spearman_corr = 0.0
        
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "samples": len(y_true)
    }


def evaluate_stage_two(
    dataset_path: str | Path,
    candidate_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
    expanded_benchmark_csv: str | Path,
    control_benchmark_csv: str | Path,
    route_qa_json: str | Path,
    thresholds: dict[str, Any] | str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """
    Perform strict same-denominator Stage-Two evaluation of candidate vs baseline models.
    Produces machine-readable compound decision JSON written to output_path.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    if isinstance(thresholds, (str, Path)):
        t_path = Path(thresholds)
        if not t_path.exists():
            raise FileNotFoundError(f"Thresholds file not found: {t_path}")
        thresh_dict = json.loads(t_path.read_text(encoding="utf-8"))
    else:
        thresh_dict = dict(thresholds)
    min_expanded_corr = float(thresh_dict.get("min_expanded_corr", 0.80))
    max_expanded_mse = float(thresh_dict.get("max_expanded_mse", 2.0))
    min_expanded_mse_improvement = float(thresh_dict.get("min_expanded_mse_improvement", 0.0))
    max_control_mse_regression = float(thresh_dict.get("max_control_mse_regression", 0.05))
    min_control_corr = float(thresh_dict.get("min_control_corr", 0.75))
    max_worst_slice_mse = float(thresh_dict.get("max_worst_slice_mse", 2.5))
    max_calibration_error = float(thresh_dict.get("max_calibration_error", 1.5))
    # Route, stability, and complexity gates consume structured boolean evidence.
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_path}")
    data_npz = np.load(dataset_path, allow_pickle=False)
    required_npz_keys = {
        "vit_embeddings", "terrain_features", "class_logits", "scenic_scores",
        "sample_weights", "image_paths", "label_sources",
    }
    missing_npz = sorted(required_npz_keys - set(data_npz.files))
    if missing_npz:
        raise ValueError(f"Dataset NPZ {dataset_path} missing required fields: {missing_npz}")
    if any(float(v) < 0 for v in thresh_dict.values() if isinstance(v, (int, float))):
        raise ValueError("Thresholds must not contain negative numeric values")
    
    vit = data_npz["vit_embeddings"]
    terr = data_npz["terrain_features"]
    cls_logits = data_npz["class_logits"]
    npz_scores = data_npz["scenic_scores"]
    sample_weights = data_npz["sample_weights"]
    image_paths_raw = data_npz["image_paths"]
    
    n_samples = len(vit)
    if not (len(terr) == n_samples and len(cls_logits) == n_samples and len(npz_scores) == n_samples and len(sample_weights) == n_samples and len(image_paths_raw) == n_samples):
        raise ValueError("NPZ arrays length mismatch across required fields")
    
    if not (np.isfinite(vit).all() and np.isfinite(terr).all() and np.isfinite(cls_logits).all() and np.isfinite(npz_scores).all() and np.isfinite(sample_weights).all()):
        raise ValueError("NPZ contains non-finite values (NaN or Inf)")
    
    if np.any(sample_weights <= 0):
        raise ValueError("NPZ sample_weights must be strictly positive (> 0)")
    
    image_paths = [str(p.decode("utf-8") if isinstance(p, bytes) else p) for p in image_paths_raw]
    
    # Check for duplicate image_paths in NPZ
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate image_paths found in NPZ dataset")
    
    npz_path_map = {p: i for i, p in enumerate(image_paths)}
    min_supported_slice_samples = int(thresh_dict.get("min_supported_slice_samples", 5))
    
    # 2. Load models
    candidate_model = _load_model_checkpoint(candidate_checkpoint)
    baseline_model = _load_model_checkpoint(baseline_checkpoint)
    
    candidate_sha256 = file_sha256(candidate_checkpoint)
    baseline_sha256 = file_sha256(baseline_checkpoint)

    # 3. Model inference on entire NPZ
    cand_npz_preds = _predict_dataset(candidate_model, vit, terr, cls_logits)
    base_npz_preds = _predict_dataset(baseline_model, vit, terr, cls_logits)
    
    cand_pred_map = {p: float(pred) for p, pred in zip(image_paths, cand_npz_preds)}
    base_pred_map = {p: float(pred) for p, pred in zip(image_paths, base_npz_preds)}

    # Helper function to process benchmark CSV against NPZ predictions
    def process_benchmark(csv_path: str | Path, name: str) -> dict[str, Any]:
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
            if image_path not in npz_path_map:
                missing_paths.append(image_path)
                continue
            matched_records.append(
                {
                    "image_path": image_path,
                    "target": float(score_val),
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
                is_supported = (len(s_recs) >= min_supported_slice_samples)
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
            "records": matched_records, "candidate_metrics": cand_metrics,
            "baseline_metrics": base_metrics, "sliced_metrics": sliced_results,
            "region_metrics": region_results, "worst_slice_mse": worst_slice_mse,
        }

    # 4. Process Expanded Human Benchmark & Control Benchmark
    exp_res = process_benchmark(expanded_benchmark_csv, "expanded_human_benchmark")
    ctrl_res = process_benchmark(control_benchmark_csv, "control_benchmark")

    # 5. Calibration and Prediction Distribution QA
    exp_yt = np.array([r["target"] for r in exp_res["records"]], dtype=np.float32)
    exp_yc = np.array([r["cand_pred"] for r in exp_res["records"]], dtype=np.float32)
    exp_yb = np.array([r["base_pred"] for r in exp_res["records"]], dtype=np.float32)
    
    # Calculate binned calibration error
    score_bins = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
    bin_errors = []
    for bin_min, bin_max in score_bins:
        mask = (exp_yt >= bin_min) & (exp_yt < bin_max if bin_max < 10.0 else exp_yt <= bin_max)
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

    pred_range_pass = bool(np.isfinite(exp_yc).all() and cand_min >= 0.0 and cand_max <= 10.0 and cand_range > 1e-4)
    spread_ratio_pass = bool(spread_ratio >= min_spread_ratio and cand_std > 1e-4)
    mean_drift_pass = bool(mean_drift <= max_mean_drift)
    saturation_pass = bool(sat_ratio <= max_saturation_ratio)
    tie_pass = bool(unique_ratio >= min_unique_ratio)
    distribution_pass = bool(pred_range_pass and spread_ratio_pass and mean_drift_pass and saturation_pass and tie_pass)

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
        "prediction_drift_vs_baseline_mean": float(np.abs(np.mean(exp_yc) - np.mean(exp_yb))),
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
    has_nonempty_routes = isinstance(routes, (list, dict)) and len(routes) > 0
    all_invariants_pass = route_qa_data.get("all_invariants_pass") is True

    stability_confirmed = (
        route_qa_data.get("stability_confirmed") is True
        or (isinstance(route_qa_data.get("stability"), dict) and route_qa_data["stability"].get("confirmed") is True)
    )

    complexity_accepted = (
        route_qa_data.get("complexity_accepted") is True
        or (isinstance(route_qa_data.get("complexity"), dict) and route_qa_data["complexity"].get("accepted") is True)
    )

    route_evidence_pass = bool(has_nonempty_routes and all_invariants_pass and stability_confirmed and complexity_accepted)

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
        "expanded_benchmark_improvement_pass": bool(exp_mse_improvement >= min_expanded_mse_improvement),
        "control_benchmark_non_regression_pass": bool(ctrl_mse_delta <= max_control_mse_regression),
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
        with tempfile.NamedTemporaryFile("w", dir=reg_path.parent, delete=False, encoding="utf-8") as tf:
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


def promote_from_decision(
    decision_path: str | Path,
    candidate_checkpoint: str | Path,
    registry_path: str | Path,
    expected_registry_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    """Promote only a passing, hash-matched candidate in one locked transaction."""
    dec_path, cand_path, reg_path = Path(decision_path), Path(candidate_checkpoint), Path(registry_path)
    if not dec_path.exists():
        raise FileNotFoundError(f"Decision file not found: {dec_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {cand_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")
    decision_data = json.loads(dec_path.read_text(encoding="utf-8"))
    if not decision_data.get("all_gates_pass", False):
        raise ValueError(f"Promotion rejected: decision at {dec_path} has all_gates_pass == False")
    dec_sha = decision_data.get("candidate", {}).get("sha256")
    cand_sha = file_sha256(cand_path)
    if not isinstance(dec_sha, str) or dec_sha != cand_sha:
        raise ValueError("Candidate SHA256 mismatch or missing decision evidence")
    _load_model_checkpoint(cand_path)
    lock = _locked_registry(reg_path)
    try:
        actual = file_sha256(reg_path)
        if actual != expected_registry_sha256:
            raise ValueError(f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual}")
        registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
        if not isinstance(registry_data.get("active"), dict):
            raise ValueError(f"Malformed registry at {reg_path}: missing active entry")
        history = registry_data.get("history", [])
        if not isinstance(history, list):
            raise ValueError("Malformed registry history")
        prior_active = dict(registry_data["active"])
        exp_metrics = decision_data.get("expanded_human_benchmark", {}).get("candidate_metrics", {})
        new_active = {
            "checkpoint": str(cand_path.resolve()), "promoted_at": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name, "sha256": cand_sha, "metrics": exp_metrics,
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


def rollback_registry(
    registry_path: str | Path,
    target_history_index: int,
    expected_registry_sha256: str,
) -> dict[str, Any]:
    """Restore a validated historical raw record and append a rollback event."""
    reg_path = Path(registry_path)
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")
    lock = _locked_registry(reg_path)
    try:
        actual = file_sha256(reg_path)
        if actual != expected_registry_sha256:
            raise ValueError(f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual}")
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
        if (
            not isinstance(target_event, dict)
            or not isinstance(target_event.get("record"), dict)
        ):
            raise ValueError("Historical registry event has no model record")
        new_active = dict(target_event["record"])
        checkpoint = Path(str(new_active.get("checkpoint", "")))
        target_sha = new_active.get("sha256")
        if (
            not checkpoint.is_file()
            or not isinstance(target_sha, str)
            or file_sha256(checkpoint) != target_sha
        ):
            raise ValueError("Historical checkpoint identity validation failed")
        _load_model_checkpoint(checkpoint)
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
        return {"status": "rolled_back", "target_history_index": target_history_index, "new_registry_sha256": new_hash, "active": new_active}
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


