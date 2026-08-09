#!/usr/bin/env python3
"""
Stage-Two Active Scenic Autoresearch Orchestrator.

Orchestrates Stage-Two active scenic model training, evaluation, threshold gating,
and optional model registry promotion from a validated Stage-One handoff.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.active_learning.common import validate_run_name  # noqa: E402
except (ImportError, SyntaxError):
    import importlib.util

    _common_path = PROJECT_ROOT / "src" / "active_learning" / "common.py"
    _spec = importlib.util.spec_from_file_location(
        "src_active_learning_common", _common_path
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        validate_run_name = _mod.validate_run_name
    else:
        raise

try:
    from scripts.modeling.validate_stage2_preflight import (  # noqa: E402
        validate_handoff,
        validate_supplemental,
    )
except (ImportError, SyntaxError):
    import importlib.util

    _preflight_path = (
        PROJECT_ROOT / "scripts" / "modeling" / "validate_stage2_preflight.py"
    )
    _spec = importlib.util.spec_from_file_location(
        "scripts_modeling_validate_stage2_preflight", _preflight_path
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        validate_handoff = _mod.validate_handoff
        validate_supplemental = getattr(_mod, "validate_supplemental", None)
    else:
        raise

from src.scenic_scorer.active_training import (  # noqa: E402
    ActiveTrainingConfig,
    prepare_active_dataset,
    train_active_model,
)
from src.scenic_scorer.active_evaluation import (  # noqa: E402
    evaluate_candidate_validation,
    evaluate_stage_two,
    promote_from_decision,
)


class DeadlineExceededError(TimeoutError):
    """Raised when a POSIX real-time deadline guard times out."""

    pass


@contextlib.contextmanager
def deadline_guard(max_seconds: Optional[float]):
    """
    Enforces a POSIX real-time deadline guard using signal.setitimer.
    Raises DeadlineExceededError if the wall-clock deadline expires during execution.
    """
    if max_seconds is None:
        yield
        return

    if max_seconds <= 0:
        raise DeadlineExceededError("Global deadline already exceeded.")

    def _alarm_handler(signum: int, frame: Any) -> None:
        raise DeadlineExceededError("Global deadline exceeded during execution phase.")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 digest of a file."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_experiment_digest(
    exp_id: str,
    config: Any,
    handoff_sha256: str,
    dataset_sha256: str,
    control_dataset_sha256: str,
    expanded_benchmark_sha256: str,
    control_benchmark_sha256: str,
    route_qa_sha256: str,
    baseline_checkpoint_sha256: str,
    thresholds_sha256: Optional[str] = None,
    supplemental_annotations_sha256: Optional[str] = None,
    supplemental_benchmark_sha256: Optional[str] = None,
) -> str:
    """Compute a deterministic SHA-256 digest binding experiment config and input artifact hashes."""
    cfg_dict = (
        dataclasses.asdict(config)
        if dataclasses.is_dataclass(config)
        else (config if isinstance(config, dict) else str(config))
    )
    payload = {
        "exp_id": exp_id,
        "config": cfg_dict,
        "handoff_sha256": handoff_sha256,
        "dataset_sha256": dataset_sha256,
        "control_dataset_sha256": control_dataset_sha256,
        "expanded_benchmark_sha256": expanded_benchmark_sha256,
        "control_benchmark_sha256": control_benchmark_sha256,
        "route_qa_sha256": route_qa_sha256,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "thresholds_sha256": thresholds_sha256,
        "supplemental_annotations_sha256": supplemental_annotations_sha256,
        "supplemental_benchmark_sha256": supplemental_benchmark_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sanitize_command(cmd: str | List[str]) -> str:
    """Sanitize command strings before persisting them in run artifacts."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    return re.sub(
        r"(?i)(--?(?:api[-_]?key|token|secret|password|auth))(?:=|\s+)(['\"]?)([^'\"\s]+)\2",
        r"\1=[REDACTED]",
        cmd_str,
    )


def resolve_stage_one_handoff(
    handoff_arg: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Resolve and validate stage-one handoff path.
    Fails closed if missing, invalid, or ambiguous (>1 ready handoff when omitted).
    """
    if handoff_arg:
        p = Path(handoff_arg)
        if p.is_dir():
            p = p / "stage1_handoff.json"
        if not p.exists():
            raise FileNotFoundError(f"Specified handoff file does not exist: {p}")

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON in specified handoff file {p}: {e}")

        return p, data

    # Search in default location: data/processed/active_learning/
    search_root = Path("data/processed/active_learning")
    if not search_root.exists():
        raise FileNotFoundError(
            f"No stage1 handoff specified and search root does not exist: {search_root}"
        )

    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for json_file in search_root.glob("**/stage1_handoff.json"):
        try:
            content = json.loads(json_file.read_text(encoding="utf-8"))
            is_ready = content.get("ready_for_stage2") or content.get("handoff_ready")
            blockers = content.get("blockers", [])
            if is_ready and len(blockers) == 0:
                candidates.append((json_file, content))
        except Exception:
            continue

    if len(candidates) == 0:
        raise RuntimeError(
            "No ready Stage-One handoff found in data/processed/active_learning/. "
            "Specify --handoff explicitly."
        )
    if len(candidates) > 1:
        paths_str = ", ".join(str(c[0]) for c in candidates)
        raise RuntimeError(
            f"Ambiguous Stage-One handoffs found ({len(candidates)} ready candidates: {paths_str}). "
            "Specify --handoff explicitly."
        )

    return candidates[0]


def validate_handoff_content(handoff_path: Path, handoff_data: Dict[str, Any]) -> None:
    """Validate handoff schema, ready flags, blockers, artifacts, and hashes."""
    is_ready = handoff_data.get("ready_for_stage2") or handoff_data.get("handoff_ready")
    if not is_ready:
        raise ValueError(f"Handoff {handoff_path} is not marked ready for Stage Two.")

    blockers = handoff_data.get("blockers", [])
    if len(blockers) > 0:
        raise ValueError(f"Handoff {handoff_path} has active blockers: {blockers}")

    artifacts = handoff_data.get("artifacts", {})
    root_dir = handoff_path.parent
    for name, record in artifacts.items():
        if isinstance(record, dict):
            rel_path = record.get("path")
            required = record.get("required", True)
            expected_hash = record.get("sha256")
            if rel_path:
                art_p = (
                    root_dir / rel_path
                    if not Path(rel_path).is_absolute()
                    else Path(rel_path)
                )
                if required and not art_p.exists():
                    raise FileNotFoundError(
                        f"Required artifact '{name}' missing at {art_p} in handoff {handoff_path}"
                    )
                if art_p.exists() and expected_hash:
                    actual_hash = compute_sha256(art_p)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"Artifact '{name}' SHA-256 mismatch! Expected {expected_hash}, got {actual_hash}"
                        )


def _count_expanded_human_samples(expanded_csv_path: str | Path, *, split: str) -> int:
    """Count unique finite human targets for one explicit benchmark split."""
    path = Path(expanded_csv_path)
    if not path.is_file():
        return 0
    unique_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_val = str(row.get("split", "")).strip().lower()
            if split_val == split:
                img_path = str(
                    row.get("image_path")
                    or row.get("image_paths")
                    or row.get("path")
                    or row.get("image")
                    or ""
                ).strip()
                score_val = (
                    row.get("scenic_human_mean")
                    if row.get("scenic_human_mean") is not None
                    and str(row.get("scenic_human_mean")).strip() != ""
                    else (
                        row.get("scenic_human")
                        if row.get("scenic_human") is not None
                        and str(row.get("scenic_human")).strip() != ""
                        else (
                            row.get("target")
                            if row.get("target") is not None
                            and str(row.get("target")).strip() != ""
                            else row.get("score")
                        )
                    )
                )
                if score_val is not None and img_path:
                    try:
                        t = float(score_val)
                        if math.isfinite(t) and 0 <= t <= 10:
                            unique_paths.add(img_path)
                    except (ValueError, TypeError):
                        pass
    return len(unique_paths)


def count_expanded_val_samples(expanded_csv_path: str | Path) -> int:
    return _count_expanded_human_samples(expanded_csv_path, split="val")


def build_candidate_ladder(
    base_config: ActiveTrainingConfig, max_experiments: int
) -> List[Dict[str, Any]]:
    """Generate a deterministic experiment candidate ladder."""
    ladder = []
    # Candidate 1: Baseline configuration
    c1 = dataclasses.replace(base_config)
    ladder.append(
        {
            "exp_id": "exp_01_baseline_control",
            "hypothesis": "Baseline training configuration with standard weights.",
            "config": c1,
        }
    )

    # Candidate 2: Region-balanced sample weighting
    if max_experiments >= 2:
        c2 = dataclasses.replace(base_config, sample_weight_scheme="region_balanced")
        ladder.append(
            {
                "exp_id": "exp_02_region_balanced",
                "hypothesis": "Region-balanced sample weighting improves low-support regional slices.",
                "config": c2,
            }
        )

    # Candidate 3: Robust Huber loss
    if max_experiments >= 3:
        c3 = dataclasses.replace(base_config, loss_function="huber")
        ladder.append(
            {
                "exp_id": "exp_03_robust_huber_loss",
                "hypothesis": "Huber loss reduces sensitivity to noisy human annotations.",
                "config": c3,
            }
        )

    # Candidate 4: Lower learning rate with weight decay
    if max_experiments >= 4:
        c4 = dataclasses.replace(base_config, learning_rate=5e-5, weight_decay=1e-3)
        ladder.append(
            {
                "exp_id": "exp_04_fine_learning_rate",
                "hypothesis": "Lower learning rate prevents overfitting on small active sets.",
                "config": c4,
            }
        )

    # Candidate 5: One additional epoch at the same capacity.
    if max_experiments >= 5:
        c5 = dataclasses.replace(base_config, epochs=base_config.epochs + 1)
        ladder.append(
            {
                "exp_id": "exp_05_extended_epochs",
                "hypothesis": "One additional epoch improves continuous scenic score ranking.",
                "config": c5,
            }
        )

    return ladder[:max_experiments]


# Aggregate candidate metrics used for validation-based selection. Records must
# carry finite values for every one of these that is present, and each value
# must exactly match the validation decision artifact's candidate_metrics.
VALIDATION_SELECTION_METRIC_KEYS = (
    "samples",
    "mse",
    "mae",
    "rmse",
    "pearson_corr",
    "pearson",
    "spearman_corr",
    "spearman",
)


def validate_experiment_record(
    rec: Dict[str, Any],
    expected_baseline_sha256: Optional[str] = None,
    expected_input_digest: Optional[str] = None,
) -> bool:
    """
    Validate a reused validated, completed, retained, or rejected experiment record.

    Returns True only when the record carries truthful, checkable evidence:
      - the candidate checkpoint exists and its current SHA-256 matches the
        validation decision file's candidate hash,
      - validated/completed/retained records carry finite validation_metrics
        plus a validation_decision_path whose candidate_metrics exactly match
        the record's selection metrics (legacy 'completed' records lacking
        validation evidence never bypass validation),
      - 'completed' records additionally carry full stage-two evidence
        (eval_decision_path with matching candidate hash and gate agreement),
      - input digest and baseline identity align when expected values are given.
    """
    if not isinstance(rec, dict):
        return False

    status = rec.get("status")
    if status not in ("validated", "completed", "retained", "rejected"):
        return False

    if expected_input_digest:
        rec_digest = rec.get("input_digest")
        if not rec_digest or rec_digest != expected_input_digest:
            return False

    ckpt_str = rec.get("candidate_checkpoint")
    if not ckpt_str:
        return False

    ckpt_path = Path(ckpt_str)
    if not ckpt_path.is_file():
        return False

    try:
        cand_sha256 = compute_sha256(ckpt_path)
    except Exception:
        return False

    if status == "rejected":
        rec_gate = rec.get("all_gates_pass")
        reason = rec.get("rejection_reason")
        if (
            rec_gate is not False
            or reason != "insufficient_expanded_human_validation_support"
        ):
            return False
        return True

    # Validation evidence is mandatory for every terminal non-rejected record.
    validation_metrics = rec.get("validation_metrics")
    if not isinstance(validation_metrics, dict) or not isinstance(
        validation_metrics.get("mse"), (int, float)
    ):
        return False

    # Aggregate candidate metrics used for selection must be finite numbers;
    # NaN/inf are never truthful evidence.
    for metric_key in VALIDATION_SELECTION_METRIC_KEYS:
        if metric_key not in validation_metrics:
            continue
        value = validation_metrics[metric_key]
        if not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ):
            return False

    val_dec_str = rec.get("validation_decision_path")
    if not val_dec_str:
        return False
    val_dec_path = Path(val_dec_str)
    if not val_dec_path.is_file():
        return False
    try:
        val_data = json.loads(val_dec_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(val_data, dict):
        return False

    val_cand = val_data.get("candidate", {})
    if not isinstance(val_cand, dict):
        return False

    val_cand_sha256 = val_cand.get("sha256")
    if not val_cand_sha256 or val_cand_sha256 != cand_sha256:
        return False

    val_cand_ckpt = val_cand.get("checkpoint")
    if val_cand_ckpt:
        try:
            if Path(val_cand_ckpt).resolve() != ckpt_path.resolve():
                return False
        except Exception:
            return False

    # The record's complete validation_metrics values used for selection must
    # equal the decision artifact's candidate_metrics. Stale, tampered, or
    # missing mismatches are rejected so legacy metrics can never influence
    # selection.
    val_candidate_metrics = val_data.get("candidate_metrics")
    if not isinstance(val_candidate_metrics, dict):
        return False
    record_metric_keys = {
        key for key in VALIDATION_SELECTION_METRIC_KEYS if key in validation_metrics
    }
    artifact_metric_keys = {
        key for key in VALIDATION_SELECTION_METRIC_KEYS if key in val_candidate_metrics
    }
    if record_metric_keys != artifact_metric_keys:
        return False
    for metric_key in record_metric_keys:
        artifact_value = val_candidate_metrics[metric_key]
        if not isinstance(artifact_value, (int, float)) or not math.isfinite(
            float(artifact_value)
        ):
            return False
        if float(artifact_value) != float(validation_metrics[metric_key]):
            return False

    if status == "completed":
        eval_dec_str = rec.get("eval_decision_path")
        if not eval_dec_str:
            return False
        eval_dec_path = Path(eval_dec_str)
        if not eval_dec_path.is_file():
            return False
        try:
            eval_data = json.loads(eval_dec_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        if not isinstance(eval_data, dict):
            return False

        eval_cand = eval_data.get("candidate", {})
        if not isinstance(eval_cand, dict):
            return False

        eval_cand_sha256 = eval_cand.get("sha256")
        if not eval_cand_sha256 or eval_cand_sha256 != cand_sha256:
            return False

        eval_cand_ckpt = eval_cand.get("checkpoint")
        if eval_cand_ckpt:
            try:
                if Path(eval_cand_ckpt).resolve() != ckpt_path.resolve():
                    return False
            except Exception:
                return False

        rec_gate = rec.get("all_gates_pass")
        eval_gate = eval_data.get("all_gates_pass")
        if not isinstance(rec_gate, bool) or not isinstance(eval_gate, bool):
            return False
        if rec_gate != eval_gate:
            return False

        if expected_baseline_sha256:
            eval_base = eval_data.get("baseline")
            if not isinstance(eval_base, dict):
                return False
            base_sha = eval_base.get("sha256")
            if (
                not isinstance(base_sha, str)
                or not base_sha
                or base_sha != expected_baseline_sha256
            ):
                return False
    elif expected_baseline_sha256:
        val_base = val_data.get("baseline")
        if not isinstance(val_base, dict):
            return False
        base_sha = val_base.get("sha256")
        if (
            not isinstance(base_sha, str)
            or not base_sha
            or base_sha != expected_baseline_sha256
        ):
            return False
    return True


def select_best_candidate(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Select best passing candidate based on highest correlation then lowest MAE.
    Tie-breaking: correlation descending, MAE ascending.
    Ignores rejected candidates.
    """
    passing_candidates = [
        r
        for r in records
        if r.get("all_gates_pass") is True and r.get("status") != "rejected"
    ]
    if not passing_candidates:
        return None

    def candidate_sort_key(r: Dict[str, Any]) -> Tuple[float, float]:
        metrics = r.get("metrics", {})
        corr = metrics.get("pearson_corr")
        if corr is None:
            corr = metrics.get("corr", 0.0)
        mae = metrics.get("mae", float("inf"))
        return (float(corr), -float(mae))

    passing_candidates.sort(key=candidate_sort_key, reverse=True)
    return passing_candidates[0]


def select_validation_finalist(
    records: List[Dict[str, Any]], ladder_exp_ids: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Select exactly one finalist by lowest validation MSE.

    Requires every ladder candidate's newest record to be a terminal
    validated/completed/retained record carrying validation_metrics; if any
    candidate is paused/stopped/failed/missing, returns None and no candidate
    is selected (and therefore no full evaluation runs). Ties are broken
    deterministically by ladder order (exp_id position in the ladder).
    """
    final_records: Dict[str, Dict[str, Any]] = {}
    for exp_id in ladder_exp_ids:
        exp_records = [r for r in records if r.get("exp_id") == exp_id]
        if not exp_records:
            return None
        rec = exp_records[-1]
        status = rec.get("status")
        if status not in ("validated", "completed", "retained"):
            return None
        validation_metrics = rec.get("validation_metrics")
        if not isinstance(validation_metrics, dict) or not isinstance(
            validation_metrics.get("mse"), (int, float)
        ):
            return None
        if not math.isfinite(float(validation_metrics["mse"])):
            return None
        final_records[exp_id] = rec

    ladder_index = {exp_id: i for i, exp_id in enumerate(ladder_exp_ids)}
    ranked = sorted(
        final_records.values(),
        key=lambda r: (
            float(r["validation_metrics"]["mse"]),
            ladder_index[r["exp_id"]],
        ),
    )
    return ranked[0]


def determine_aggregate_run_state(
    evaluated_records: List[Dict[str, Any]],
    intended_count: int,
    timed_out: bool = False,
) -> str:
    """
    Determine aggregate run_state for final summary and metrics.
    Returns one of: 'timed_out', 'paused', 'failed', 'completed'.
    Only returns 'completed' when intended ladder count is reached and all records succeeded.
    """
    if timed_out:
        return "timed_out"

    for r in evaluated_records:
        status = r.get("status")
        if status == "stopped":
            return "timed_out"

    for r in evaluated_records:
        status = r.get("status")
        if status == "failed":
            return "failed"

    for r in evaluated_records:
        status = r.get("status")
        if status == "paused":
            return "paused"

    if len(evaluated_records) < intended_count:
        return "timed_out"

    for r in evaluated_records:
        status = r.get("status")
        if status not in ("completed", "validated", "retained", "rejected"):
            return "failed"

    return "completed"


def is_valid_resumable_experiment(exp_dir: Path) -> bool:
    """Check if experiment directory contains a valid paused continuation or completed summary state for resume."""
    summary_path = exp_dir / "training_summary.json"
    if not summary_path.is_file():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        state = data.get("state")
        if state == "paused":
            return (exp_dir / "resume.pt").is_file()
        if state == "completed":
            return (exp_dir / "candidate.pt").is_file()
        return False
    except Exception:
        return False


is_valid_paused_experiment = is_valid_resumable_experiment


def get_completed_candidate_checkpoint(exp_dir: Path) -> Optional[str]:
    """
    Return the candidate checkpoint path when training finished a completed candidate.

    Used on resume to reuse already-trained candidate checkpoints without retraining
    (e.g. legacy records lacking validated evidence, or runs interrupted between
    training completion and validation). Returns None when no completed candidate
    checkpoint exists.
    """
    summary_path = exp_dir / "training_summary.json"
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("state") != "completed":
        return None
    ckpt_path = exp_dir / "candidate.pt"
    if not ckpt_path.is_file():
        return None
    return str(ckpt_path.resolve())


EXPOSURE_REJECTION_REASON = "heldout_test_exposure_before_finalist_selection"
EXPOSURE_MIN_FRESH_TEST_ROWS = 20
FIXED_VALIDATION_SELECTION_ROWS = 64


def has_validation_evidence(rec: Dict[str, Any]) -> bool:
    """True when a record carries validation-selection evidence (finite metrics + decision path)."""
    if not isinstance(rec, dict):
        return False
    validation_metrics = rec.get("validation_metrics")
    if not isinstance(validation_metrics, dict) or not isinstance(
        validation_metrics.get("mse"), (int, float)
    ):
        return False
    if not math.isfinite(float(validation_metrics["mse"])):
        return False
    return bool(rec.get("validation_decision_path"))


def detect_heldout_test_exposure(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return experiment records with full-eval evidence but no validation evidence.

    A completed record carrying a full stage-two evaluation (which touches the
    held-out test/control/route benchmarks) without any validation evidence
    means the held-out test set was exposed before finalist selection. Such
    records are returned for quarantine handling; they are never deleted.
    """
    exposed: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "completed":
            continue
        eval_path_str = rec.get("eval_decision_path")
        if not eval_path_str or not Path(str(eval_path_str)).is_file():
            continue
        if has_validation_evidence(rec):
            continue
        exposed.append(rec)
    return exposed


def build_exposure_annotation_request(
    observed_test_samples: int,
    observed_val_samples: int,
) -> Dict[str, Any]:
    """Next-data request for a quarantined run.

    Requests at least EXPOSURE_MIN_FRESH_TEST_ROWS fresh, untouched,
    geographically isolated human test rows with zero overlap against
    current-run test rows and control rows, while retaining the fixed
    FIXED_VALIDATION_SELECTION_ROWS-row validation selection unchanged.
    """
    return {
        "target_test_human_rows": EXPOSURE_MIN_FRESH_TEST_ROWS,
        "min_fresh_test_human_rows": EXPOSURE_MIN_FRESH_TEST_ROWS,
        "observed_expanded_test_samples": observed_test_samples,
        "needed_fresh_test_human_rows": EXPOSURE_MIN_FRESH_TEST_ROWS,
        "retained_validation_rows": FIXED_VALIDATION_SELECTION_ROWS,
        "validation_selection_unchanged": True,
        "sampling_strategy": (
            "fresh_geographically_isolated_no_current_or_control_overlap"
        ),
        "overlap_constraints": {
            "current_run_test_rows": 0,
            "control_rows": 0,
        },
        "description": (
            f"Request at least {EXPOSURE_MIN_FRESH_TEST_ROWS} fresh, untouched, "
            f"geographically isolated human test rows with zero overlap against "
            f"current-run test rows and control rows, while retaining the fixed "
            f"{FIXED_VALIDATION_SELECTION_ROWS}-row validation selection unchanged "
            f"(observed validation rows: {observed_val_samples})."
        ),
    }


def validate_run_manifest(
    existing_manifest: Dict[str, Any],
    run_name: str,
    handoff_sha256: str,
    baseline_registry_sha256: str,
    baseline_checkpoint_sha256: str,
    control_dataset_sha256: str,
    expanded_benchmark_sha256: str,
    control_benchmark_sha256: str,
    route_qa_sha256: str,
    thresholds_sha256: Optional[str],
    supplemental_annotations_sha256: Optional[str],
    supplemental_benchmark_sha256: Optional[str],
    observed_val_samples: int,
    observed_test_samples: int,
    is_data_limited: bool,
    supp_metrics: Optional[Dict[str, int]],
    args: argparse.Namespace,
) -> None:
    """Validate an existing run manifest against the current invocation."""
    mismatches = []

    if existing_manifest.get("run_name") != run_name:
        mismatches.append(
            f"run_name mismatch: manifest={existing_manifest.get('run_name')}, current={run_name}"
        )

    mode_checks = [
        ("dry_run", args.dry_run),
        ("promote_requested", args.promote),
    ]
    for key, current_val in mode_checks:
        if key not in existing_manifest:
            mismatches.append(f"missing key in manifest: {key}")
        elif existing_manifest[key] != current_val:
            mismatches.append(
                f"{key} mismatch: manifest={existing_manifest[key]}, current={current_val}"
            )

    hash_checks = [
        ("stage1_handoff_sha256", handoff_sha256),
        ("baseline_registry_sha256", baseline_registry_sha256),
        ("baseline_checkpoint_sha256", baseline_checkpoint_sha256),
        ("control_dataset_sha256", control_dataset_sha256),
        ("expanded_benchmark_sha256", expanded_benchmark_sha256),
        ("control_benchmark_sha256", control_benchmark_sha256),
        ("route_qa_sha256", route_qa_sha256),
        ("thresholds_sha256", thresholds_sha256),
    ]
    supplemental_enabled = args.supplemental_annotations is not None
    if supplemental_enabled:
        hash_checks.extend(
            [
                ("supplemental_annotations_sha256", supplemental_annotations_sha256),
                ("supplemental_benchmark_sha256", supplemental_benchmark_sha256),
            ]
        )
    for key, current_val in hash_checks:
        if key not in existing_manifest:
            mismatches.append(f"missing key in manifest: {key}")
        elif existing_manifest[key] != current_val:
            mismatches.append(
                f"{key} mismatch: manifest={existing_manifest[key]}, current={current_val}"
            )

    expected_supp_path = (
        str(args.supplemental_annotations) if supplemental_enabled else None
    )
    if supplemental_enabled:
        supplemental_checks = [
            ("supplemental_annotations_path", expected_supp_path),
            ("supplemental_metrics", supp_metrics),
            ("expanded_val_support", observed_val_samples),
            ("expanded_test_support", observed_test_samples),
            ("data_limited", is_data_limited),
        ]
        for key, current_val in supplemental_checks:
            if key not in existing_manifest:
                mismatches.append(f"missing key in manifest: {key}")
            elif existing_manifest[key] != current_val:
                mismatches.append(
                    f"{key} mismatch: manifest={existing_manifest[key]}, current={current_val}"
                )

    if "config" not in existing_manifest or not isinstance(
        existing_manifest["config"], dict
    ):
        mismatches.append("manifest config field is missing or not a dictionary")
    else:
        cfg = existing_manifest["config"]
        config_checks = [
            ("seed", args.seed),
            ("max_experiments", args.max_experiments),
            ("device", args.device),
            ("max_steps", args.max_steps),
            ("expanded_benchmark_csv", str(args.expanded_benchmark_csv)),
            ("control_benchmark_csv", str(args.control_benchmark_csv)),
            ("control_dataset", str(args.control_dataset)),
            ("route_qa_json", str(args.route_qa_json)),
            (
                "thresholds_json",
                str(args.thresholds_json) if args.thresholds_json else None,
            ),
        ]
        if supplemental_enabled:
            config_checks.extend(
                [
                    ("supplemental_annotations", expected_supp_path),
                    (
                        "supplemental_annotations_sha256",
                        supplemental_annotations_sha256,
                    ),
                    (
                        "supplemental_benchmark_sha256",
                        supplemental_benchmark_sha256,
                    ),
                    ("supplemental_metrics", supp_metrics),
                ]
            )
        for key, current_val in config_checks:
            if key not in cfg:
                mismatches.append(f"missing config key in manifest: {key}")
            elif cfg[key] != current_val:
                mismatches.append(
                    f"config.{key} mismatch: manifest={cfg[key]}, current={current_val}"
                )

    if mismatches:
        raise ValueError(
            "Run manifest validation failed on resume:\n" + "\n".join(mismatches)
        )

def load_existing_experiments(
    experiments_jsonl: Path,
    expected_baseline_sha256: Optional[str] = None,
    expected_input_digests: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Load existing completed experiment records for truthful resume handling."""
    completed_map: Dict[str, Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = []

    if not experiments_jsonl.exists():
        return completed_map, all_records

    for line in experiments_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            all_records.append(rec)
            exp_id = rec.get("exp_id")
            exp_digest = (
                expected_input_digests.get(exp_id)
                if expected_input_digests and exp_id
                else None
            )
            if exp_id and validate_experiment_record(
                rec,
                expected_baseline_sha256=expected_baseline_sha256,
                expected_input_digest=exp_digest,
            ):
                completed_map[exp_id] = rec
            elif exp_id and rec.get("status") in ("completed", "retained", "rejected"):
                print(
                    f"WARNING: Stale or invalid experiment record {exp_id} in {experiments_jsonl}. Will revalidate or rerun.",
                    file=sys.stderr,
                )
        except Exception:
            continue

    return completed_map, all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-Two Active Scenic Autoresearch Orchestrator"
    )
    parser.add_argument(
        "--handoff",
        type=str,
        default=None,
        help="Path to stage1 handoff JSON or directory containing stage1_handoff.json.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run_active_v1",
        help="Run identifier for directory under data/processed/modeling_autoresearch/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, write experiment plan, exit without dataset prep or training.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current run status and metrics without executing new experiments.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from existing experiment state truthfully.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=5,
        help="Maximum candidate experiments to evaluate in ladder.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per training experiment.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1800.0,
        help="Maximum runtime in seconds before stopping autoresearch loop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--expanded-benchmark-csv",
        type=Path,
        required=True,
        help="Expanded human benchmark CSV",
    )
    parser.add_argument(
        "--control-benchmark-csv",
        type=Path,
        required=True,
        help="New England control benchmark CSV",
    )
    parser.add_argument(
        "--control-dataset",
        type=Path,
        required=True,
        help="Canonical control feature dataset NPZ (disjoint from expanded prepared dataset)",
    )
    parser.add_argument(
        "--route-qa-json",
        type=Path,
        required=True,
        help="Route QA evidence JSON",
    )
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=None,
        help="Path to numeric decision gate thresholds JSON.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Attempt registry promotion if candidate passes all evaluation gates.",
    )
    parser.add_argument(
        "--supplemental-annotations",
        type=Path,
        default=None,
        help="Path to supplemental annotations CSV.",
    )
    parser.add_argument(
        "--supplemental-annotations-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 hash of supplemental annotations CSV.",
    )
    parser.add_argument(
        "--supplemental-benchmark-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 hash of supplemental benchmark CSV.",
    )
    args = parser.parse_args()
    args.run_name = validate_run_name(args.run_name)
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if not (1 <= args.max_experiments <= 5):
        raise ValueError("--max-experiments must be between 1 and 5")

    supp_flags = [
        args.supplemental_annotations,
        args.supplemental_annotations_sha256,
        args.supplemental_benchmark_sha256,
    ]
    num_supplied = sum(1 for f in supp_flags if f is not None)
    if num_supplied not in (0, 3):
        raise ValueError(
            "--supplemental-annotations, --supplemental-annotations-sha256, and "
            "--supplemental-benchmark-sha256 must be provided together or all omitted."
        )
    return args


def main() -> None:
    start_time = time.time()
    args = parse_args()
    global_deadline = start_time + args.max_seconds
    timed_out_flag = False

    # 1. Handoff Resolution and Fast Failure before training
    print("METRIC handoff_validating=1")
    try:
        handoff_path, handoff_data = resolve_stage_one_handoff(args.handoff)
        validate_handoff_content(handoff_path, handoff_data)
        validate_handoff(handoff_path)
        print(f"METRIC handoff_ready=1 handoff_path={handoff_path}")
    except Exception as err:
        print(f"ERROR: Stage-One handoff validation failed: {err}", file=sys.stderr)
        print("METRIC handoff_ready=0")
        sys.exit(1)

    # Require --thresholds-json for non-dry execution and parse before preparation/training
    thresholds: Optional[Dict[str, Any]] = None
    if args.thresholds_json:
        thresh_path = Path(args.thresholds_json)
        if not thresh_path.is_file():
            raise FileNotFoundError(f"Missing thresholds file: {args.thresholds_json}")
        try:
            thresholds = json.loads(thresh_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(
                f"Invalid JSON in thresholds file {args.thresholds_json}: {err}"
            ) from err

    required_inputs = {
        "expanded benchmark": args.expanded_benchmark_csv,
        "control benchmark": args.control_benchmark_csv,
        "control dataset NPZ": args.control_dataset,
        "route QA evidence": args.route_qa_json,
    }
    if args.supplemental_annotations:
        required_inputs["supplemental annotations"] = args.supplemental_annotations
    if args.thresholds_json:
        required_inputs["thresholds"] = Path(args.thresholds_json)
    for label, path in required_inputs.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    # Validate supplemental benchmark if enabled (before run directory setup)
    supp_metrics: Optional[Dict[str, int]] = None
    if args.supplemental_annotations is not None:
        if validate_supplemental is None:
            raise RuntimeError("validate_supplemental function not available in validate_stage2_preflight")
        supp_metrics = validate_supplemental(
            handoff_path=handoff_path,
            supplemental_annotations=Path(args.supplemental_annotations),
            supplemental_benchmark=Path(args.expanded_benchmark_csv),
            annotations_sha256=args.supplemental_annotations_sha256,
            benchmark_sha256=args.supplemental_benchmark_sha256,
            control_benchmark=Path(args.control_benchmark_csv),
        )
        for name, value in supp_metrics.items():
            print(f"METRIC {name}={value}")

    # 2. Setup Run Directory
    run_dir = Path("data/processed/modeling_autoresearch") / args.run_name
    artifacts_exist = False
    if run_dir.exists():
        artifact_files = [
            "run_manifest.json",
            "experiments.jsonl",
            "promotion_decision.json",
            "final_summary.json",
            "prepared_dataset.npz",
        ]
        if any((run_dir / f).exists() for f in artifact_files):
            artifacts_exist = True

    if artifacts_exist and not args.resume and not args.status:
        raise FileExistsError(
            f"Run directory '{run_dir}' already exists with run artifacts. "
            "Pass --resume to resume or use a new --run-name."
        )
    if (
        not args.dry_run
        and not args.status
        and not args.resume
        and not args.thresholds_json
    ):
        raise ValueError("--thresholds-json is required for non-dry execution")

    # 3. Capture Immutable Baseline Identity
    registry_path = Path("data/processed/regression/model_registry.json")
    if not registry_path.exists():
        print(
            f"ERROR: Active model registry missing at {registry_path}", file=sys.stderr
        )
        print("METRIC baseline_sha256=missing")
        sys.exit(1)

    baseline_registry_sha256 = compute_sha256(registry_path)
    print(f"METRIC baseline_sha256={baseline_registry_sha256}")

    registry_content = json.loads(registry_path.read_text(encoding="utf-8"))
    active_model = registry_content.get("active")
    if not isinstance(active_model, dict) or not active_model.get("checkpoint"):
        raise ValueError("active model registry lacks active.checkpoint")
    baseline_checkpoint_path = Path(active_model["checkpoint"])
    if not baseline_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Active baseline checkpoint missing: {baseline_checkpoint_path}"
        )
    baseline_checkpoint_sha256 = compute_sha256(baseline_checkpoint_path)

    # Compute Input Hashes
    expanded_benchmark_sha256 = compute_sha256(args.expanded_benchmark_csv)
    control_benchmark_sha256 = compute_sha256(args.control_benchmark_csv)
    control_dataset_sha256 = compute_sha256(args.control_dataset)
    route_qa_sha256 = compute_sha256(args.route_qa_json)
    thresholds_sha256 = (
        compute_sha256(args.thresholds_json)
        if args.thresholds_json and Path(args.thresholds_json).is_file()
        else None
    )
    supplemental_annotations_sha256 = args.supplemental_annotations_sha256
    supplemental_benchmark_sha256 = args.supplemental_benchmark_sha256
    handoff_sha256 = compute_sha256(handoff_path)

    # Count expanded val support & check threshold
    min_val_samples = 5
    if thresholds and isinstance(thresholds, dict):
        raw_val = thresholds.get("min_expanded_validation_samples")
        if isinstance(raw_val, int) and raw_val > 0 and not isinstance(raw_val, bool):
            min_val_samples = raw_val

    validation_min_slice_samples = 5
    if thresholds and isinstance(thresholds, dict):
        raw_slice = thresholds.get("min_supported_slice_samples")
        if (
            isinstance(raw_slice, int)
            and raw_slice > 0
            and not isinstance(raw_slice, bool)
        ):
            validation_min_slice_samples = raw_slice

    observed_val_samples = count_expanded_val_samples(args.expanded_benchmark_csv)
    observed_test_samples = _count_expanded_human_samples(
        args.expanded_benchmark_csv, split="test"
    )
    is_data_limited = observed_val_samples < min_val_samples

    manifest_path = run_dir / "run_manifest.json"
    experiments_jsonl = run_dir / "experiments.jsonl"
    decision_path = run_dir / "promotion_decision.json"
    summary_path = run_dir / "final_summary.json"

    # Handle --status mode
    if args.status:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Run status requested but manifest missing at {manifest_path}"
            )
        try:
            status_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Run manifest at {manifest_path} is invalid JSON: {err}")

        if (
            not isinstance(status_manifest, dict)
            or "baseline_checkpoint_sha256" not in status_manifest
        ):
            raise ValueError(
                f"Run manifest at {manifest_path} is malformed or lacks baseline_checkpoint_sha256"
            )

        if args.supplemental_annotations is not None:
            supplemental_status_checks = [
                (
                    "supplemental_annotations_sha256",
                    supplemental_annotations_sha256,
                ),
                ("supplemental_benchmark_sha256", supplemental_benchmark_sha256),
                (
                    "supplemental_annotations_path",
                    str(args.supplemental_annotations),
                ),
                ("supplemental_metrics", supp_metrics),
                ("expanded_val_support", observed_val_samples),
                ("expanded_test_support", observed_test_samples),
                ("data_limited", is_data_limited),
            ]
            status_mismatches = [
                f"{key}: manifest={status_manifest.get(key)}, current={current}"
                for key, current in supplemental_status_checks
                if status_manifest.get(key) != current
            ]
            if status_mismatches:
                raise ValueError(
                    "Run manifest validation failed on resume:\n"
                    + "\n".join(status_mismatches)
                )

        manifest_baseline_checkpoint_sha256 = status_manifest[
            "baseline_checkpoint_sha256"
        ]

        completed_map, all_records = load_existing_experiments(
            experiments_jsonl,
            expected_baseline_sha256=manifest_baseline_checkpoint_sha256,
        )
        print(f"Run status for '{args.run_name}':")
        print(f"  Directory: {run_dir}")
        print(
            f"  Recorded Run Baseline Checkpoint SHA256: {manifest_baseline_checkpoint_sha256}"
        )
        print(
            f"  Current Registry Active Checkpoint SHA256: {baseline_checkpoint_sha256}"
        )
        print(f"  Current Registry SHA256: {baseline_registry_sha256}")
        print(f"  Completed Experiments: {len(completed_map)}")
        for exp_id, rec in completed_map.items():
            metrics = rec.get("metrics") or rec.get("validation_metrics") or {}
            print(
                f"    - {exp_id}: MAE={metrics.get('mae')} RMSE={metrics.get('rmse')} "
                f"MSE={metrics.get('mse')} Corr={metrics.get('pearson_corr', metrics.get('corr'))}"
            )
        exposure_records = detect_heldout_test_exposure(all_records)
        if exposure_records:
            print(
                f"  HELDOUT TEST EXPOSURE: {len(exposure_records)} record(s) with "
                "full-eval evidence but no validation evidence (exp_ids: "
                + ", ".join(str(r.get("exp_id")) for r in exposure_records)
                + "). Run is quarantined: no promotion until >=20 fresh, untouched "
                "test rows replace the exposed held-out test set."
            )
        sys.exit(0)

    print(f"METRIC expanded_val_support={observed_val_samples}")
    print(f"METRIC min_expanded_validation_samples={min_val_samples}")
    print(f"METRIC data_limited={1 if is_data_limited else 0}")

    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_baseline_checkpoint_sha256 = baseline_checkpoint_sha256
    # Handle Manifest (Immutable run identity/config/input hashes)
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Resume requested but run manifest missing at {manifest_path}"
            )
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Run manifest at {manifest_path} is invalid JSON: {err}")
        validate_run_manifest(
            existing_manifest=existing_manifest,
            run_name=args.run_name,
            handoff_sha256=handoff_sha256,
            baseline_registry_sha256=baseline_registry_sha256,
            baseline_checkpoint_sha256=baseline_checkpoint_sha256,
            control_dataset_sha256=control_dataset_sha256,
            expanded_benchmark_sha256=expanded_benchmark_sha256,
            control_benchmark_sha256=control_benchmark_sha256,
            route_qa_sha256=route_qa_sha256,
            thresholds_sha256=thresholds_sha256,
            supplemental_annotations_sha256=supplemental_annotations_sha256,
            supplemental_benchmark_sha256=supplemental_benchmark_sha256,
            observed_val_samples=observed_val_samples,
            observed_test_samples=observed_test_samples,
            is_data_limited=is_data_limited,
            supp_metrics=supp_metrics,
            args=args,
        )
        if not args.dry_run and not args.status and not args.thresholds_json:
            raise ValueError("--thresholds-json is required for non-dry execution")
        if "baseline_checkpoint_sha256" in existing_manifest:
            manifest_baseline_checkpoint_sha256 = existing_manifest[
                "baseline_checkpoint_sha256"
            ]
    else:
        manifest = {
            "run_name": args.run_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage1_handoff_path": str(handoff_path),
            "stage1_handoff_sha256": handoff_sha256,
            "baseline_registry_path": str(registry_path),
            "baseline_registry_sha256": baseline_registry_sha256,
            "baseline_checkpoint_path": str(baseline_checkpoint_path),
            "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
            "expanded_benchmark_sha256": expanded_benchmark_sha256,
            "control_benchmark_sha256": control_benchmark_sha256,
            "control_dataset_path": str(args.control_dataset),
            "control_dataset_sha256": control_dataset_sha256,
            "route_qa_sha256": route_qa_sha256,
            "thresholds_sha256": thresholds_sha256,
            "supplemental_annotations_path": (
                str(args.supplemental_annotations)
                if args.supplemental_annotations
                else None
            ),
            "supplemental_annotations_sha256": supplemental_annotations_sha256,
            "supplemental_benchmark_sha256": supplemental_benchmark_sha256,
            "supplemental_metrics": supp_metrics,
            "expanded_val_support": observed_val_samples,
            "expanded_test_support": observed_test_samples,
            "min_expanded_validation_samples": min_val_samples,
            "data_limited": is_data_limited,
            "rejection_reason": (
                "insufficient_expanded_human_validation_support"
                if is_data_limited
                else None
            ),
            "dry_run": args.dry_run,
            "promote_requested": args.promote,
            "config": {
                "seed": args.seed,
                "device": args.device,
                "max_experiments": args.max_experiments,
                "max_steps": args.max_steps,
                "max_seconds": args.max_seconds,
                "expanded_benchmark_csv": str(args.expanded_benchmark_csv),
                "control_benchmark_csv": str(args.control_benchmark_csv),
                "control_dataset": str(args.control_dataset),
                "route_qa_json": str(args.route_qa_json),
                "thresholds_json": (
                    str(args.thresholds_json) if args.thresholds_json else None
                ),
                "supplemental_annotations": (
                    str(args.supplemental_annotations)
                    if args.supplemental_annotations
                    else None
                ),
                "supplemental_annotations_sha256": args.supplemental_annotations_sha256,
                "supplemental_benchmark_sha256": args.supplemental_benchmark_sha256,
                "supplemental_metrics": supp_metrics,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # 4. Build Experiment Ladder
    base_config = ActiveTrainingConfig(
        seed=args.seed,
        epochs=1 if is_data_limited else ActiveTrainingConfig().epochs,
        device=args.device,
        max_steps=args.max_steps,
    )
    ladder = build_candidate_ladder(base_config, args.max_experiments)

    if time.time() >= global_deadline:
        print(f"Time budget of {args.max_seconds}s reached before dataset preparation.")
        sys.exit(0)

    prepared_dataset_path = run_dir / "prepared_dataset.npz"
    dataset_sha256_dry = (
        compute_sha256(prepared_dataset_path)
        if prepared_dataset_path.is_file()
        else "pending_preparation"
    )

    expected_digests: Dict[str, str] = {}
    for exp in ladder:
        exp_id = exp["exp_id"]
        expected_digests[exp_id] = compute_experiment_digest(
            exp_id=exp_id,
            config=exp["config"],
            handoff_sha256=handoff_sha256,
            dataset_sha256=dataset_sha256_dry,
            control_dataset_sha256=control_dataset_sha256,
            expanded_benchmark_sha256=expanded_benchmark_sha256,
            control_benchmark_sha256=control_benchmark_sha256,
            route_qa_sha256=route_qa_sha256,
            baseline_checkpoint_sha256=baseline_checkpoint_sha256,
            thresholds_sha256=thresholds_sha256,
        )

    # Handle --dry-run mode
    if args.dry_run:
        completed_map = {}
        if args.resume and experiments_jsonl.exists():
            completed_map, _ = load_existing_experiments(
                experiments_jsonl,
                expected_baseline_sha256=manifest_baseline_checkpoint_sha256,
                expected_input_digests=expected_digests,
            )

        planned_records = []
        for exp in ladder:
            exp_id = exp["exp_id"]
            if args.resume and exp_id in completed_map:
                planned_records.append(completed_map[exp_id])
            else:
                exp_rec = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": "planned",
                    "command": sanitize_command(f"train_active_model --exp {exp_id}"),
                    "config": dataclasses.asdict(exp["config"])
                    if dataclasses.is_dataclass(exp["config"])
                    else str(exp["config"]),
                    "input_digest": expected_digests[exp_id],
                }
                planned_records.append(exp_rec)

        with experiments_jsonl.open("w", encoding="utf-8") as f:
            for rec in planned_records:
                f.write(json.dumps(rec) + "\n")

        print(f"METRIC dry_run=1 planned_experiments={len(planned_records)}")
        print("Dry run plan written successfully. No mutations performed.")
        sys.exit(0)

    prepared_split_path = prepared_dataset_path.with_suffix(".filtered_index.csv")
    if (
        args.resume
        and prepared_dataset_path.is_file()
        and prepared_split_path.is_file()
    ):
        dataset_info = {
            "dataset_path": str(prepared_dataset_path),
            "split_path": str(prepared_split_path),
        }
    else:
        remaining_prep = global_deadline - time.time()
        try:
            with deadline_guard(remaining_prep):
                dataset_info = prepare_active_dataset(
                    handoff_path, prepared_dataset_path
                )
        except DeadlineExceededError:
            print(
                f"Time budget of {args.max_seconds}s reached during dataset preparation.",
                file=sys.stderr,
            )
            sys.exit(0)

    dataset_path = dataset_info["dataset_path"]
    split_csv_path = dataset_info["split_path"]
    dataset_sha256 = compute_sha256(dataset_path)

    # Recompute expected digests with actual dataset sha256
    expected_digests = {}
    for exp in ladder:
        exp_id = exp["exp_id"]
        expected_digests[exp_id] = compute_experiment_digest(
            exp_id=exp_id,
            config=exp["config"],
            handoff_sha256=handoff_sha256,
            dataset_sha256=dataset_sha256,
            control_dataset_sha256=control_dataset_sha256,
            expanded_benchmark_sha256=expanded_benchmark_sha256,
            control_benchmark_sha256=control_benchmark_sha256,
            route_qa_sha256=route_qa_sha256,
            baseline_checkpoint_sha256=baseline_checkpoint_sha256,
            thresholds_sha256=thresholds_sha256,
        )

    # 5. Load truthful resume state
    completed_map, all_records = (
        load_existing_experiments(
            experiments_jsonl,
            expected_baseline_sha256=manifest_baseline_checkpoint_sha256,
            expected_input_digests=expected_digests,
        )
        if args.resume
        else ({}, [])
    )
    # 5b. Historical preselection test exposure quarantine. Any record with
    # full-eval evidence but no validation evidence means the held-out test
    # set was touched before finalist selection. Such runs are quarantined:
    # no retraining, no evaluation, no promotion, and the next-data request
    # demands fresh untouched test rows. Historical records are never deleted.
    exposure_records = detect_heldout_test_exposure(all_records)
    heldout_exposure = len(exposure_records) > 0
    exposure_exp_ids = [
        str(rec["exp_id"]) for rec in exposure_records if rec.get("exp_id")
    ]
    if heldout_exposure:
        print(
            f"METRIC heldout_test_exposure=1 exposure_count={len(exposure_records)} "
            f"exp_ids={','.join(exposure_exp_ids)}"
        )
        print(
            "ERROR: Held-out test exposure detected before finalist selection "
            f"({len(exposure_records)} record(s) with full-eval evidence but no "
            "validation evidence). Run quarantined: no training or evaluation "
            "will run and the registry stays unchanged.",
            file=sys.stderr,
        )
    # 6. Autoresearch Loop
    evaluated_records: List[Dict[str, Any]] = []

    loop_ladder: List[Dict[str, Any]] = [] if heldout_exposure else ladder
    for exp in loop_ladder:
        exp_id = exp["exp_id"]
        exp_digest = expected_digests[exp_id]

        now = time.time()
        remaining_seconds = global_deadline - now
        if remaining_seconds <= 0:
            print(
                f"Time budget of {args.max_seconds}s reached. "
                "Stopping autoresearch loop."
            )
            break

        # Reuse completed experiment if resuming
        if args.resume and exp_id in completed_map:
            print(f"METRIC exp_id={exp_id} status=reused")
            evaluated_records.append(completed_map[exp_id])
            continue

        print(f"METRIC exp_id={exp_id} status=starting")
        exp_dir = run_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # On resume, reuse an already-completed training checkpoint without
            # retraining (legacy records lacking validated evidence, or runs
            # interrupted between training completion and validation).
            completed_candidate_ckpt = (
                get_completed_candidate_checkpoint(exp_dir) if args.resume else None
            )
            if completed_candidate_ckpt is not None:
                candidate_ckpt = completed_candidate_ckpt
                train_result = {
                    "state": "completed",
                    "candidate_checkpoint": candidate_ckpt,
                }
                print(f"METRIC exp_id={exp_id} status=training_reused")
            else:
                experiment_config = dataclasses.replace(
                    exp["config"], max_seconds=remaining_seconds
                )
                try:
                    with deadline_guard(remaining_seconds):
                        train_result = train_active_model(
                            dataset_path=dataset_path,
                            split_csv=split_csv_path,
                            output_dir=exp_dir,
                            config=experiment_config,
                            resume=args.resume
                            and is_valid_resumable_experiment(exp_dir),
                        )
                except DeadlineExceededError:
                    print(
                        f"Time budget of {args.max_seconds}s reached during training of {exp_id}. "
                        "Stopping autoresearch loop.",
                        file=sys.stderr,
                    )
                    exp_record = {
                        "exp_id": exp_id,
                        "hypothesis": exp["hypothesis"],
                        "status": "stopped",
                        "reason": "deadline_exceeded_during_training",
                        "input_digest": exp_digest,
                    }
                    evaluated_records.append(exp_record)
                    with experiments_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(exp_record) + "\n")
                    break

            if train_result.get("state") != "completed":
                exp_record = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": train_result.get("state", "paused"),
                    "training": train_result,
                    "input_digest": exp_digest,
                }
                evaluated_records.append(exp_record)
                with experiments_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp_record) + "\n")
                print(f"METRIC exp_id={exp_id} status={exp_record['status']}")
                continue
            candidate_ckpt = train_result["candidate_checkpoint"]

            if is_data_limited:
                cand_sha256 = train_result.get(
                    "candidate_checkpoint_sha256"
                ) or compute_sha256(candidate_ckpt)
                training_metrics = train_result.get("metrics", {})
                exp_record = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": "rejected",
                    "all_gates_pass": False,
                    "candidate_checkpoint": candidate_ckpt,
                    "candidate_checkpoint_sha256": cand_sha256,
                    "metrics": training_metrics,
                    "training_metrics": training_metrics,
                    "rejection_reason": "insufficient_expanded_human_validation_support",
                    "expanded_val_support": observed_val_samples,
                    "min_expanded_validation_samples": min_val_samples,
                    "input_digest": exp_digest,
                    "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                evaluated_records.append(exp_record)
                with experiments_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp_record) + "\n")
                print(
                    f"METRIC exp_id={exp_id} status=rejected "
                    "reason=insufficient_expanded_human_validation_support gate_pass=0"
                )
                continue

            # Check global deadline before in-process non-preemptible validation
            remaining_eval = global_deadline - time.time()
            if remaining_eval <= 0:
                print(
                    f"Time budget of {args.max_seconds}s reached before validation of {exp_id}. "
                    "Stopping autoresearch loop."
                )
                exp_record = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": "stopped",
                    "reason": "deadline_exceeded_before_validation",
                    "candidate_checkpoint": candidate_ckpt,
                    "input_digest": exp_digest,
                }
                evaluated_records.append(exp_record)
                with experiments_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp_record) + "\n")
                break

            validation_decision_path = exp_dir / "validation_decision.json"
            try:
                with deadline_guard(remaining_eval):
                    validation_result = evaluate_candidate_validation(
                        dataset_path=dataset_path,
                        candidate_checkpoint=candidate_ckpt,
                        baseline_checkpoint=baseline_checkpoint_path,
                        expanded_benchmark_csv=args.expanded_benchmark_csv,
                        output_path=validation_decision_path,
                        min_supported_slice_samples=validation_min_slice_samples,
                        device=args.device,
                    )
            except DeadlineExceededError:
                print(
                    f"Time budget of {args.max_seconds}s reached during validation of {exp_id}. "
                    "Stopping autoresearch loop.",
                    file=sys.stderr,
                )
                exp_record = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": "stopped",
                    "reason": "deadline_exceeded_during_validation",
                    "candidate_checkpoint": candidate_ckpt,
                    "input_digest": exp_digest,
                }
                evaluated_records.append(exp_record)
                with experiments_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp_record) + "\n")
                break

            validation_metrics = validation_result["candidate_metrics"]
            val_mse = validation_metrics["mse"]
            cand_sha256 = validation_result["candidate"]["sha256"]

            exp_record = {
                "exp_id": exp_id,
                "hypothesis": exp["hypothesis"],
                "status": "validated",
                "candidate_checkpoint": candidate_ckpt,
                "candidate_checkpoint_sha256": cand_sha256,
                "validation_decision_path": str(validation_decision_path),
                "validation_metrics": validation_metrics,
                "mse_improvement": validation_result.get("mse_improvement"),
                "input_digest": exp_digest,
                "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            evaluated_records.append(exp_record)

            # Append to experiments.jsonl
            with experiments_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(exp_record) + "\n")

            print(
                f"METRIC exp_id={exp_id} status=validated val_mse={val_mse:.4f}"
            )

        except Exception as e:
            print(f"ERROR: Experiment {exp_id} failed: {e}", file=sys.stderr)
            exp_record = {
                "exp_id": exp_id,
                "hypothesis": exp["hypothesis"],
                "status": "failed",
                "error": str(e),
                "input_digest": exp_digest,
            }
            evaluated_records.append(exp_record)
            with experiments_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(exp_record) + "\n")
            print(f"METRIC exp_id={exp_id} status=failed")
    # 7. Select Validation Finalist and Run Single Full Stage-Two Evaluation
    # Only when every ladder candidate has a valid terminal validation record
    # (and none is paused/stopped/failed) is exactly one finalist chosen by
    # lowest validation MSE with deterministic ladder/exp_id tie-break; the
    # held-out test/control/route/compound gates run at most once for that
    # finalist, and only after every candidate finished.
    finalist = None
    all_gates_pass = False
    final_eval_pending = False
    if not is_data_limited and not heldout_exposure:
        ladder_exp_ids = [exp["exp_id"] for exp in ladder]
        finalist = select_validation_finalist(evaluated_records, ladder_exp_ids)
        if finalist is not None:
            if finalist.get("eval_decision_path") and isinstance(
                finalist.get("all_gates_pass"), bool
            ):
                # Full evaluation evidence already recorded (resume): reuse it.
                all_gates_pass = finalist["all_gates_pass"]
                print(
                    f"METRIC exp_id={finalist['exp_id']} status=full_eval_reused "
                    f"gate_pass={1 if all_gates_pass else 0}"
                )
            else:
                remaining_full = global_deadline - time.time()
                if remaining_full <= 0:
                    print(
                        f"Time budget of {args.max_seconds}s reached before final full "
                        f"evaluation of {finalist['exp_id']}. Full evaluation deferred "
                        "to a resumed run.",
                        file=sys.stderr,
                    )
                    final_eval_pending = True
                else:
                    finalist_dir = run_dir / finalist["exp_id"]
                    finalist_dir.mkdir(parents=True, exist_ok=True)
                    full_eval_path = finalist_dir / "eval_decision.json"
                    final_thresholds = None
                    if args.thresholds_json and Path(args.thresholds_json).exists():
                        final_thresholds = json.loads(
                            Path(args.thresholds_json).read_text(encoding="utf-8")
                        )
                    try:
                        with deadline_guard(remaining_full):
                            full_eval_result = evaluate_stage_two(
                                dataset_path=dataset_path,
                                control_dataset_path=args.control_dataset,
                                candidate_checkpoint=finalist["candidate_checkpoint"],
                                baseline_checkpoint=baseline_checkpoint_path,
                                expanded_benchmark_csv=args.expanded_benchmark_csv,
                                control_benchmark_csv=args.control_benchmark_csv,
                                route_qa_json=args.route_qa_json,
                                thresholds=final_thresholds,
                                output_path=full_eval_path,
                                device=args.device,
                            )
                    except DeadlineExceededError:
                        print(
                            f"Time budget of {args.max_seconds}s reached during final full "
                            f"evaluation of {finalist['exp_id']}. Full evaluation deferred "
                            "to a resumed run.",
                            file=sys.stderr,
                        )
                        final_eval_pending = True
                    else:
                        all_gates_pass = full_eval_result.get("all_gates_pass", False)
                        full_metrics = full_eval_result["expanded_human_benchmark"][
                            "candidate_metrics"
                        ]
                        finalist_record = {
                            "exp_id": finalist["exp_id"],
                            "hypothesis": finalist.get("hypothesis"),
                            "status": "completed",
                            "candidate_checkpoint": finalist["candidate_checkpoint"],
                            "candidate_checkpoint_sha256": finalist.get(
                                "candidate_checkpoint_sha256"
                            ),
                            "validation_decision_path": finalist.get(
                                "validation_decision_path"
                            ),
                            "validation_metrics": finalist.get("validation_metrics"),
                            "eval_decision_path": str(full_eval_path),
                            "metrics": full_metrics,
                            "all_gates_pass": all_gates_pass,
                            "input_digest": finalist.get("input_digest"),
                            "evaluated_at": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        }
                        evaluated_records.append(finalist_record)
                        with experiments_jsonl.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(finalist_record) + "\n")
                        # The finalist now carries the full evaluation evidence.
                        finalist = finalist_record
                        print(
                            f"METRIC exp_id={finalist['exp_id']} status=completed "
                            f"full_eval=1 gate_pass={1 if all_gates_pass else 0}"
                        )
        else:
            print("METRIC finalist_selected=0")
    if heldout_exposure:
        print("METRIC finalist_selected=0 heldout_test_exposure=1")
    print(f"METRIC all_gates_pass={1 if all_gates_pass else 0}")

    requested_annotation_batch = {
        "target_validation_human_rows": 5,
        "target_test_human_rows": 20,
        "min_expanded_validation_samples": min_val_samples,
        "observed_expanded_validation_samples": observed_val_samples,
        "observed_expanded_test_samples": observed_test_samples,
        "needed_validation_human_rows": max(0, min_val_samples - observed_val_samples),
        "needed_test_human_rows": max(0, 20 - observed_test_samples),
        "sampling_strategy": "qa_overlap_and_confidence_diversity",
        "description": (
            f"Requesting next annotation batch targeting >=5 validation (observed: {observed_val_samples}, "
            f"required: {min_val_samples}) and >=20 test human rows (observed: {observed_test_samples}) "
            "with QA overlap/confidence diversity."
        ),
    }
    if heldout_exposure:
        # Exposed test rows are contaminated; request only fresh untouched rows
        # while keeping the fixed 64-row validation selection unchanged.
        requested_annotation_batch = build_exposure_annotation_request(
            observed_test_samples=observed_test_samples,
            observed_val_samples=observed_val_samples,
        )
    decision_rejection_reason = None
    if heldout_exposure:
        decision_rejection_reason = EXPOSURE_REJECTION_REASON
    elif is_data_limited:
        decision_rejection_reason = (
            "insufficient_expanded_human_validation_support"
        )
    elif finalist is not None and not all_gates_pass and not final_eval_pending:
        decision_rejection_reason = "selected_finalist_failed_compound_gates"
    elif finalist is None and not final_eval_pending:
        decision_rejection_reason = "no_validation_finalist"


    promoted = False
    decision_summary = {
        "run_name": args.run_name,
        "all_gates_pass": (
            False if (is_data_limited or heldout_exposure) else all_gates_pass
        ),
        "data_limited": is_data_limited,
        "selected_finalist_exp_id": (
            finalist["exp_id"] if finalist is not None else None
        ),
        "validation_mse": (
            float(finalist["validation_metrics"]["mse"])
            if finalist is not None
            else None
        ),
        "promotion_evidence_valid": not heldout_exposure,
        "heldout_test_exposure": {
            "detected": heldout_exposure,
            "count": len(exposure_exp_ids),
            "exp_ids": exposure_exp_ids,
        },
        "retained_candidate": (
            None
            if (
                is_data_limited
                or heldout_exposure
                or finalist is None
                or not all_gates_pass
            )
            else finalist
        ),
        "evaluated_experiments": len(evaluated_records),
        "baseline_registry_sha256": baseline_registry_sha256,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "observed_expanded_val_samples": observed_val_samples,
        "min_expanded_validation_samples": min_val_samples,
        "rejection_reason": decision_rejection_reason,
        "registry_status": "unchanged"
        if (is_data_limited or heldout_exposure)
        else ("promoted" if promoted else "unchanged"),
        "requested_annotation_batch": requested_annotation_batch
        if (is_data_limited or heldout_exposure)
        else None,
        "decision_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 8. Registry Promotion Safety
    PROMOTION_RESERVE_SECONDS = 5.0
    if heldout_exposure:
        print(
            "Promotion blocked: held-out test exposure detected before finalist "
            "selection. Registry unchanged."
        )
        print("METRIC promoted=0 promotion_blocked=1 heldout_test_exposure=1")
    elif args.promote:
        remaining_promo = global_deadline - time.time()
        if is_data_limited:
            print(
                "Promotion requested but run is data limited (insufficient validation support). Registry unchanged."
            )
            print("METRIC promoted=0 promotion_blocked=1 data_limited=1")
        elif not all_gates_pass or finalist is None:
            print(
                "Promotion requested but compound decision gates failed or no passing candidate found. Registry unchanged."
            )
            print("METRIC promoted=0 promotion_blocked=1")
        elif remaining_promo < PROMOTION_RESERVE_SECONDS:
            print(
                "Promotion requested without enough time for an atomic registry update. Registry unchanged."
            )
            print("METRIC promoted=0 promotion_blocked=1 deadline=1")
        else:
            cand_ckpt = finalist["candidate_checkpoint"]
            print(f"Promoting candidate {finalist['exp_id']} to registry...")
            promo_res = promote_from_decision(
                decision_path=finalist["eval_decision_path"],
                candidate_checkpoint=cand_ckpt,
                registry_path=registry_path,
                expected_registry_sha256=baseline_registry_sha256,
                run_name=args.run_name,
            )
            promoted = True
            print(
                f"METRIC promoted=1 registry_sha256={promo_res['new_registry_sha256']}"
            )
    else:
        print("METRIC promoted=0 promote_flag_absent=1")
    decision_summary["registry_status"] = "promoted" if promoted else "unchanged"
    decision_path.write_text(json.dumps(decision_summary, indent=2), encoding="utf-8")

    # 9. Final Summary
    run_state = determine_aggregate_run_state(
        evaluated_records, len(ladder), timed_out=timed_out_flag
    )
    if heldout_exposure:
        run_state = "rejected"
    elif final_eval_pending:
        run_state = "paused"
    selected_finalist_summary = None
    if finalist is not None:
        selected_finalist_summary = {
            "exp_id": finalist["exp_id"],
            "validation_mse": float(finalist["validation_metrics"]["mse"]),
            "validation_decision_path": finalist.get("validation_decision_path"),
            "full_evaluation": (
                {
                    "eval_decision_path": finalist.get("eval_decision_path"),
                    "all_gates_pass": finalist.get("all_gates_pass"),
                }
                if finalist.get("eval_decision_path")
                else None
            ),
        }
    rejection_reason = decision_rejection_reason

    final_summary = {
        "run_name": args.run_name,
        "run_state": run_state,
        "total_experiments": len(ladder),
        "all_gates_pass": (
            False if (is_data_limited or heldout_exposure) else all_gates_pass
        ),
        "data_limited": is_data_limited,
        "selected_finalist": selected_finalist_summary,
        "promotion_evidence_valid": not heldout_exposure,
        "heldout_test_exposure": {
            "detected": heldout_exposure,
            "count": len(exposure_exp_ids),
            "exp_ids": exposure_exp_ids,
        },
        "retained_exp_id": (
            None
            if (
                is_data_limited
                or heldout_exposure
                or finalist is None
                or not all_gates_pass
            )
            else finalist["exp_id"]
        ),
        "retained_candidate": (
            None
            if (
                is_data_limited
                or heldout_exposure
                or finalist is None
                or not all_gates_pass
            )
            else finalist
        ),
        "promoted": False if (is_data_limited or heldout_exposure) else promoted,
        "pending_final_evaluation": final_eval_pending,
        "registry_status": "unchanged"
        if (is_data_limited or heldout_exposure)
        else ("promoted" if promoted else "unchanged"),
        "observed_expanded_val_samples": observed_val_samples,
        "min_expanded_validation_samples": min_val_samples,
        "rejection_reason": rejection_reason,
        "requested_annotation_batch": requested_annotation_batch
        if (is_data_limited or heldout_exposure)
        else None,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(
        f"METRIC run_state={run_state} retained_exp={final_summary['retained_exp_id']} promoted={1 if promoted else 0} data_limited={1 if is_data_limited else 0}"
    )


if __name__ == "__main__":
    main()
