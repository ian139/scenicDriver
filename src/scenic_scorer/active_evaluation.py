"""
Active model evaluation, atomic promotion, and rollback module for Stage-Two scenic regression models.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
    min_route_qa_score = float(thresh_dict.get("min_route_qa_score", 0.80))
    min_route_stability_score = float(thresh_dict.get("min_route_stability_score", 0.80))
    min_complexity_score = float(thresh_dict.get("min_complexity_score", 0.0))

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_path}")
    data_npz = np.load(dataset_path, allow_pickle=True)
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
    image_paths_raw = data_npz["image_paths"]
    
    image_paths = [str(p) for p in image_paths_raw]
    
    # Check for duplicate image_paths in NPZ
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate image_paths found in NPZ dataset")
    
    npz_path_map = {p: i for i, p in enumerate(image_paths)}
    
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
    def process_benchmark(csv_path: str | Path, name: str):
        records = _read_benchmark_csv(csv_path)
        matched_records = []
        missing_paths = []
        
        for r in records:
            img_p = r.get("image_path") or r.get("image_paths") or r.get("path") or r.get("image")
            if not img_p:
                raise ValueError(f"Benchmark {name} record missing image path column: {r}")
            img_p_str = str(img_p)
            
            score_val = r.get("scenic_score") or r.get("human_score") or r.get("score") or r.get("human_label") or r.get("label")
            if score_val is None:
                raise ValueError(f"Benchmark {name} record missing score column: {r}")
            
            if img_p_str not in npz_path_map:
                missing_paths.append(img_p_str)
            else:
                matched_records.append({
                    "image_path": img_p_str,
                    "target": float(score_val),
                    "region": r.get("region", "default"),
                    "slice": r.get("slice", r.get("terrain_type", r.get("region", "default"))),
                    "cand_pred": cand_pred_map[img_p_str],
                    "base_pred": base_pred_map[img_p_str],
                })
                
        if missing_paths:
            raise ValueError(f"Benchmark {name} contains {len(missing_paths)} paths not present in NPZ dataset")
        if not matched_records:
            raise ValueError(f"Benchmark {name} has zero matched records with NPZ dataset")
        
        # Check duplicate paths within CSV
        csv_paths = [mr["image_path"] for mr in matched_records]
        if len(csv_paths) != len(set(csv_paths)):
            raise ValueError(f"Benchmark {name} CSV contains duplicate image paths")
            
        y_true = np.array([mr["target"] for mr in matched_records], dtype=np.float32)
        y_cand = np.array([mr["cand_pred"] for mr in matched_records], dtype=np.float32)
        y_base = np.array([mr["base_pred"] for mr in matched_records], dtype=np.float32)
        
        cand_metrics = _compute_metrics(y_true, y_cand)
        base_metrics = _compute_metrics(y_true, y_base)
        
        # Sliced metrics
        slice_map: dict[str, list[dict]] = {}
        for mr in matched_records:
            s_name = mr["slice"]
            slice_map.setdefault(s_name, []).append(mr)
            
        sliced_results = {}
        region_results = {}
        worst_slice_mse = 0.0
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
                target_results[s_name] = {"candidate": s_cand_m, "baseline": s_base_m, "samples": len(s_recs)}
                worst_slice_mse = max(worst_slice_mse, s_cand_m["mse"])
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
    
    distribution_qa = {
        "candidate": {
            "mean": float(np.mean(exp_yc)),
            "std": float(np.std(exp_yc)),
            "min": float(np.min(exp_yc)),
            "max": float(np.max(exp_yc)),
        },
        "baseline": {
            "mean": float(np.mean(exp_yb)),
            "std": float(np.std(exp_yb)),
            "min": float(np.min(exp_yb)),
            "max": float(np.max(exp_yb)),
        },
        "target": {
            "mean": float(np.mean(exp_yt)),
            "std": float(np.std(exp_yt)),
            "min": float(np.min(exp_yt)),
            "max": float(np.max(exp_yt)),
        },
        "prediction_drift_vs_baseline_mean": float(np.abs(np.mean(exp_yc) - np.mean(exp_yb))),
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
    
    required_rq_fields = {"route_qa_score", "stability_score", "complexity_score"}
    missing_rq = sorted(required_rq_fields - set(route_qa_data.keys()))
    if missing_rq:
        raise ValueError(f"Route QA JSON {rq_path} missing required fields: {missing_rq}")
    try:
        route_qa_score = float(route_qa_data["route_qa_score"])
        stability_score = float(route_qa_data["stability_score"])
        complexity_score = float(route_qa_data["complexity_score"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Route QA evidence scores must be numeric") from exc
    if not np.isfinite([route_qa_score, stability_score, complexity_score]).all():
        raise ValueError("Route QA evidence scores must be finite")

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
        "route_qa_score_pass": bool(route_qa_score >= min_route_qa_score),
        "stability_pass": bool(stability_score >= min_route_stability_score),
        "complexity_pass": bool(complexity_score >= min_complexity_score),
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
            "route_qa_score": route_qa_score,
            "stability_score": stability_score,
            "complexity_score": complexity_score,
            "details": route_qa_data,
        },
        "thresholds_evaluated": thresh_dict,
        "gates": gates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision_dict, indent=2), encoding="utf-8")
    return decision_dict


def promote_from_decision(
    decision_path: str | Path,
    candidate_checkpoint: str | Path,
    registry_path: str | Path,
    expected_registry_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    """
    Atomically promote candidate model into model_registry.json if decision passes.
    Rejection or hash mismatch leaves registry bytes 100% untouched.
    """
    dec_path = Path(decision_path)
    cand_path = Path(candidate_checkpoint)
    reg_path = Path(registry_path)

    if not dec_path.exists():
        raise FileNotFoundError(f"Decision file not found: {dec_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {cand_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")

    # Read and validate decision
    decision_data = json.loads(dec_path.read_text(encoding="utf-8"))
    if not decision_data.get("all_gates_pass", False):
        raise ValueError(f"Promotion rejected: decision at {dec_path} has all_gates_pass == False")

    # Check candidate SHA256 integrity
    dec_cand_sha256 = decision_data.get("candidate", {}).get("sha256")
    if not isinstance(dec_cand_sha256, str) or not dec_cand_sha256:
        raise ValueError("Promotion rejected: decision is missing candidate SHA256 evidence")
    cand_sha256 = file_sha256(cand_path)
    if dec_cand_sha256 != cand_sha256:
        raise ValueError(f"Candidate SHA256 mismatch: decision expected {dec_cand_sha256}, got {cand_sha256}")

    # Hash check on current registry
    actual_reg_sha256 = file_sha256(reg_path)
    if actual_reg_sha256 != expected_registry_sha256:
        raise ValueError(f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual_reg_sha256}")

    # Read registry
    registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
    if "active" not in registry_data or not isinstance(registry_data["active"], dict):
        raise ValueError(f"Malformed registry at {reg_path}: missing active entry")

    history = registry_data.get("history", [])
    if not isinstance(history, list):
        history = []

    # Preserve current active in history
    prior_active = dict(registry_data["active"])
    history.append(prior_active)

    # Form new active entry
    exp_cand_m = decision_data.get("expanded_human_benchmark", {}).get("candidate_metrics", {})
    new_active = {
        "checkpoint": str(cand_path.resolve()),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "sha256": cand_sha256,
        "metrics": exp_cand_m,
        "decision_path": str(dec_path.resolve()),
    }

    registry_data["active"] = new_active
    registry_data["history"] = history

    # Atomic write to registry
    reg_dir = reg_path.parent
    reg_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=reg_dir, delete=False, encoding="utf-8") as tf:
        json.dump(registry_data, tf, indent=2)
        temp_name = tf.name

    os.replace(temp_name, reg_path)

    # Post-write verification
    new_reg_sha256 = file_sha256(reg_path)
    reread = json.loads(reg_path.read_text(encoding="utf-8"))
    if reread.get("active", {}).get("sha256") != cand_sha256:
        raise ValueError("Promotion verification failed: active checkpoint hash mismatch")
    _load_model_checkpoint(cand_path)
    return {
        "status": "promoted",
        "run_name": run_name,
        "new_registry_sha256": new_reg_sha256,
        "active": new_active,
    }


def rollback_registry(
    registry_path: str | Path,
    target_history_index: int,
    expected_registry_sha256: str,
) -> dict[str, Any]:
    """
    Atomically rollback model registry active model to a historical entry.
    """
    reg_path = Path(registry_path)
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")

    actual_reg_sha256 = file_sha256(reg_path)
    if actual_reg_sha256 != expected_registry_sha256:
        raise ValueError(f"Registry SHA256 mismatch: expected {expected_registry_sha256}, got {actual_reg_sha256}")

    registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
    if "active" not in registry_data or not isinstance(registry_data["active"], dict):
        raise ValueError(f"Malformed registry at {reg_path}: missing active entry")

    history = registry_data.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"Cannot rollback registry {reg_path}: history is empty")

    try:
        target_entry = history[target_history_index]
    except IndexError as exc:
        raise ValueError(f"Invalid target history index {target_history_index} for history length {len(history)}") from exc

    # Perform swap: current active moved to history, target historical entry becomes active
    prior_active = dict(registry_data["active"])
    # If target_history_index was negative (e.g. -1), convert to positive index for pop
    real_idx = target_history_index if target_history_index >= 0 else len(history) + target_history_index
    new_active = history.pop(real_idx)
    history.append(prior_active)

    registry_data["active"] = new_active
    registry_data["history"] = history

    # Atomic write to registry
    reg_dir = reg_path.parent
    with tempfile.NamedTemporaryFile("w", dir=reg_dir, delete=False, encoding="utf-8") as tf:
        json.dump(registry_data, tf, indent=2)
        temp_name = tf.name

    os.replace(temp_name, reg_path)

    new_reg_sha256 = file_sha256(reg_path)
    return {
        "status": "rolled_back",
        "target_history_index": target_history_index,
        "new_registry_sha256": new_reg_sha256,
        "active": new_active,
    }
