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
    from scripts.modeling.validate_stage2_preflight import validate_handoff  # noqa: E402
except (ImportError, SyntaxError):
    import importlib.util

    _preflight_path = PROJECT_ROOT / "scripts" / "modeling" / "validate_stage2_preflight.py"
    _spec = importlib.util.spec_from_file_location(
        "scripts_modeling_validate_stage2_preflight", _preflight_path
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        validate_handoff = _mod.validate_handoff
    else:
        raise

from src.scenic_scorer.active_training import (  # noqa: E402
    ActiveTrainingConfig,
    prepare_active_dataset,
    train_active_model,
)
from src.scenic_scorer.active_evaluation import (  # noqa: E402
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
    expanded_benchmark_sha256: str,
    control_benchmark_sha256: str,
    route_qa_sha256: str,
    baseline_checkpoint_sha256: str,
    thresholds_sha256: Optional[str] = None,
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
        "expanded_benchmark_sha256": expanded_benchmark_sha256,
        "control_benchmark_sha256": control_benchmark_sha256,
        "route_qa_sha256": route_qa_sha256,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "thresholds_sha256": thresholds_sha256,
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


import csv


def count_expanded_val_samples(expanded_csv_path: str | Path) -> int:
    """
    Count expanded CSV split=val rows with finite human target in [0,10], unique image_path.
    """
    path = Path(expanded_csv_path)
    if not path.is_file():
        return 0
    unique_val_paths = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_val = str(row.get("split", "")).strip().lower()
            if split_val == "val":
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
                            unique_val_paths.add(img_path)
                    except (ValueError, TypeError):
                        pass
    return len(unique_val_paths)


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

    # Candidate 5: Higher capacity / extra steps
    if max_experiments >= 5:
        c5 = dataclasses.replace(base_config, epochs=15)
        ladder.append(
            {
                "exp_id": "exp_05_extended_epochs",
                "hypothesis": "Extended training epochs improves continuous scenic score ranking.",
                "config": c5,
            }
        )

    return ladder[:max_experiments]


def validate_experiment_record(
    rec: Dict[str, Any],
    expected_baseline_sha256: Optional[str] = None,
    expected_input_digest: Optional[str] = None,
) -> bool:
    """
    Validate a reused completed, retained, or rejected experiment record.
    Returns True if candidate checkpoint exists, hashes, gates, identities,
    and input digest align.
    """
    if not isinstance(rec, dict):
        return False

    status = rec.get("status")
    if status not in ("completed", "retained", "rejected"):
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
        if rec_gate is not False or reason != "insufficient_expanded_human_validation_support":
            return False
        return True

    eval_dec_str = rec.get("eval_decision_path")
    if not eval_dec_str:
        return False
    eval_dec_path = Path(eval_dec_str)
    if not eval_dec_path.is_file():
        return False
    try:
        cand_sha256 = compute_sha256(ckpt_path)
    except Exception:
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
        if status not in ("completed", "retained", "rejected"):
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


def validate_run_manifest(
    existing_manifest: Dict[str, Any],
    run_name: str,
    handoff_sha256: str,
    baseline_registry_sha256: str,
    baseline_checkpoint_sha256: str,
    expanded_benchmark_sha256: str,
    control_benchmark_sha256: str,
    route_qa_sha256: str,
    thresholds_sha256: Optional[str],
    args: argparse.Namespace,
) -> None:
    """Validate that existing run_manifest matches current invocation material config and input/baseline hashes."""
    mismatches = []

    if "run_name" not in existing_manifest or existing_manifest["run_name"] != run_name:
        mismatches.append(
            f"run_name mismatch: manifest={existing_manifest.get('run_name')}, current={run_name}"
        )
    # Material execution mode flags
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
        ("expanded_benchmark_sha256", expanded_benchmark_sha256),
        ("control_benchmark_sha256", control_benchmark_sha256),
        ("route_qa_sha256", route_qa_sha256),
        ("thresholds_sha256", thresholds_sha256),
    ]
    for key, current_val in hash_checks:
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
            ("route_qa_json", str(args.route_qa_json)),
            (
                "thresholds_json",
                str(args.thresholds_json) if args.thresholds_json else None,
            ),
        ]
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
                    f"WARNING: Stale or invalid experiment record {exp_id} in {experiments_jsonl}. Will rerun.",
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
    args = parser.parse_args()
    args.run_name = validate_run_name(args.run_name)
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if not (1 <= args.max_experiments <= 5):
        raise ValueError("--max-experiments must be between 1 and 5")
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
        "route QA evidence": args.route_qa_json,
    }
    if args.thresholds_json:
        required_inputs["thresholds"] = Path(args.thresholds_json)
    for label, path in required_inputs.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

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

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    experiments_jsonl = run_dir / "experiments.jsonl"
    decision_path = run_dir / "promotion_decision.json"
    summary_path = run_dir / "final_summary.json"

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
        if not isinstance(status_manifest, dict) or not status_manifest.get(
            "baseline_checkpoint_sha256"
        ):
            raise ValueError(
                f"Run manifest at {manifest_path} is malformed or lacks baseline_checkpoint_sha256"
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
            metrics = rec.get("metrics", {})
            print(
                f"    - {exp_id}: MAE={metrics.get('mae')} RMSE={metrics.get('rmse')} Corr={metrics.get('pearson_corr', metrics.get('corr'))}"
            )
        sys.exit(0)

    # Compute Input Hashes
    expanded_benchmark_sha256 = compute_sha256(args.expanded_benchmark_csv)
    control_benchmark_sha256 = compute_sha256(args.control_benchmark_csv)
    route_qa_sha256 = compute_sha256(args.route_qa_json)
    thresholds_sha256 = (
        compute_sha256(args.thresholds_json)
        if args.thresholds_json and Path(args.thresholds_json).is_file()
        else None
    )
    handoff_sha256 = compute_sha256(handoff_path)

    # Count expanded val support & check threshold
    min_val_samples = 5
    if thresholds and isinstance(thresholds, dict):
        raw_val = thresholds.get("min_expanded_validation_samples")
        if isinstance(raw_val, int) and raw_val > 0 and not isinstance(raw_val, bool):
            min_val_samples = raw_val

    observed_val_samples = count_expanded_val_samples(args.expanded_benchmark_csv)
    is_data_limited = observed_val_samples < min_val_samples

    print(f"METRIC expanded_val_support={observed_val_samples}")
    print(f"METRIC min_expanded_validation_samples={min_val_samples}")
    print(f"METRIC data_limited={1 if is_data_limited else 0}")

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
            expanded_benchmark_sha256=expanded_benchmark_sha256,
            control_benchmark_sha256=control_benchmark_sha256,
            route_qa_sha256=route_qa_sha256,
            thresholds_sha256=thresholds_sha256,
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
            "route_qa_sha256": route_qa_sha256,
            "thresholds_sha256": thresholds_sha256,
            "expanded_val_support": observed_val_samples,
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
                "route_qa_json": str(args.route_qa_json),
                "thresholds_json": (
                    str(args.thresholds_json) if args.thresholds_json else None
                ),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4. Build Experiment Ladder
    base_config = ActiveTrainingConfig(
        seed=args.seed,
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

    # Dataset Preparation with deadline guard (executed only in non-dry-run mode)
    remaining_prep = global_deadline - time.time()
    try:
        with deadline_guard(remaining_prep):
            dataset_info = prepare_active_dataset(handoff_path, prepared_dataset_path)
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
            expanded_benchmark_sha256=expanded_benchmark_sha256,
            control_benchmark_sha256=control_benchmark_sha256,
            route_qa_sha256=route_qa_sha256,
            baseline_checkpoint_sha256=baseline_checkpoint_sha256,
            thresholds_sha256=thresholds_sha256,
        )

    # 5. Load truthful resume state
    completed_map, _ = (
        load_existing_experiments(
            experiments_jsonl,
            expected_baseline_sha256=manifest_baseline_checkpoint_sha256,
            expected_input_digests=expected_digests,
        )
        if args.resume
        else ({}, [])
    )
    # 6. Autoresearch Loop
    evaluated_records: List[Dict[str, Any]] = []

    for exp in ladder:
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
                        resume=args.resume and is_valid_resumable_experiment(exp_dir),
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
                cand_sha256 = (
                    train_result.get("candidate_checkpoint_sha256")
                    or compute_sha256(candidate_ckpt)
                )
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

            # Check global deadline before in-process non-preemptible evaluation
            remaining_eval = global_deadline - time.time()
            if remaining_eval <= 0:
                print(
                    f"Time budget of {args.max_seconds}s reached before evaluation of {exp_id}. "
                    "Stopping autoresearch loop."
                )
                exp_record = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": "stopped",
                    "reason": "deadline_exceeded_before_eval",
                    "candidate_checkpoint": candidate_ckpt,
                    "input_digest": exp_digest,
                }
                evaluated_records.append(exp_record)
                with experiments_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp_record) + "\n")
                break

            thresholds = None
            if args.thresholds_json and Path(args.thresholds_json).exists():
                thresholds = json.loads(
                    Path(args.thresholds_json).read_text(encoding="utf-8")
                )

            eval_decision_path = exp_dir / "eval_decision.json"
            try:
                with deadline_guard(remaining_eval):
                    eval_result = evaluate_stage_two(
                        dataset_path=dataset_path,
                        candidate_checkpoint=candidate_ckpt,
                        baseline_checkpoint=baseline_checkpoint_path,
                        expanded_benchmark_csv=args.expanded_benchmark_csv,
                        control_benchmark_csv=args.control_benchmark_csv,
                        route_qa_json=args.route_qa_json,
                        thresholds=thresholds,
                        output_path=eval_decision_path,
                    )
            except DeadlineExceededError:
                print(
                    f"Time budget of {args.max_seconds}s reached during evaluation of {exp_id}. "
                    "Stopping autoresearch loop.",
                    file=sys.stderr,
                )
                exp_record = {
                    "exp_id": exp_id,
                    "hypothesis": exp["hypothesis"],
                    "status": "stopped",
                    "reason": "deadline_exceeded_during_eval",
                    "candidate_checkpoint": candidate_ckpt,
                    "input_digest": exp_digest,
                }
                evaluated_records.append(exp_record)
                with experiments_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(exp_record) + "\n")
                break

            all_pass = eval_result.get("all_gates_pass", False)
            metrics = eval_result["expanded_human_benchmark"]["candidate_metrics"]
            mae = metrics["mae"]
            rmse = metrics["rmse"]
            corr = metrics["pearson_corr"]

            exp_record = {
                "exp_id": exp_id,
                "hypothesis": exp["hypothesis"],
                "status": "completed",
                "candidate_checkpoint": candidate_ckpt,
                "eval_decision_path": str(eval_decision_path),
                "metrics": metrics,
                "all_gates_pass": all_pass,
                "input_digest": exp_digest,
                "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            evaluated_records.append(exp_record)

            # Append to experiments.jsonl
            with experiments_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(exp_record) + "\n")

            print(
                f"METRIC exp_id={exp_id} status=completed mae={mae:.4f} "
                f"rmse={rmse:.4f} corr={corr:.4f} gate_pass={1 if all_pass else 0}"
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
    # 7. Select Best Passing Candidate
    best_candidate = select_best_candidate(evaluated_records)
    all_gates_pass = best_candidate is not None if not is_data_limited else False
    print(f"METRIC all_gates_pass={1 if all_gates_pass else 0}")

    requested_annotation_batch = {
        "target_validation_human_rows": 5,
        "target_test_human_rows": 20,
        "min_expanded_validation_samples": min_val_samples,
        "observed_expanded_validation_samples": observed_val_samples,
        "needed_validation_human_rows": max(0, min_val_samples - observed_val_samples),
        "needed_test_human_rows": 20,
        "sampling_strategy": "qa_overlap_and_confidence_diversity",
        "description": (
            f"Requesting next annotation batch targeting >=5 validation (observed: {observed_val_samples}, "
            f"required: {min_val_samples}) and >=20 test human rows with QA overlap/confidence diversity."
        ),
    }

    decision_summary = {
        "run_name": args.run_name,
        "all_gates_pass": False if is_data_limited else all_gates_pass,
        "data_limited": is_data_limited,
        "retained_candidate": None if is_data_limited else best_candidate,
        "evaluated_experiments": len(evaluated_records),
        "baseline_registry_sha256": baseline_registry_sha256,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "observed_expanded_val_samples": observed_val_samples,
        "min_expanded_validation_samples": min_val_samples,
        "rejection_reason": (
            "insufficient_expanded_human_validation_support"
            if is_data_limited
            else None
        ),
        "registry_status": "unchanged" if is_data_limited else ("promoted" if promoted else "unchanged"),
        "requested_annotation_batch": requested_annotation_batch if is_data_limited else None,
        "decision_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    decision_path.write_text(json.dumps(decision_summary, indent=2), encoding="utf-8")

    # 8. Registry Promotion Safety
    promoted = False
    PROMOTION_RESERVE_SECONDS = 5.0
    if args.promote:
        remaining_promo = global_deadline - time.time()
        if is_data_limited:
            print(
                "Promotion requested but run is data limited (insufficient validation support). Registry unchanged."
            )
            print("METRIC promoted=0 promotion_blocked=1 data_limited=1")
        elif remaining_promo < PROMOTION_RESERVE_SECONDS:
            cand_ckpt = best_candidate["candidate_checkpoint"]
            print(f"Promoting candidate {best_candidate['exp_id']} to registry...")
            promo_res = promote_from_decision(
                decision_path=best_candidate["eval_decision_path"],
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
            print(
                "Promotion requested but compound decision gates failed or no passing candidate found. Registry unchanged."
            )
            print("METRIC promoted=0 promotion_blocked=1")
    else:
        print("METRIC promoted=0 promote_flag_absent=1")

    # 9. Final Summary
    run_state = determine_aggregate_run_state(
        evaluated_records, len(ladder), timed_out=timed_out_flag
    )
    final_summary = {
        "run_name": args.run_name,
        "run_state": run_state,
        "total_experiments": len(evaluated_records),
        "all_gates_pass": False if is_data_limited else all_gates_pass,
        "data_limited": is_data_limited,
        "retained_exp_id": None if is_data_limited else (best_candidate["exp_id"] if best_candidate else None),
        "retained_candidate": None if is_data_limited else best_candidate,
        "promoted": False if is_data_limited else promoted,
        "registry_status": "unchanged" if is_data_limited else ("promoted" if promoted else "unchanged"),
        "observed_expanded_val_samples": observed_val_samples,
        "min_expanded_validation_samples": min_val_samples,
        "rejection_reason": (
            "insufficient_expanded_human_validation_support"
            if is_data_limited
            else None
        ),
        "requested_annotation_batch": requested_annotation_batch if is_data_limited else None,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(
        f"METRIC run_state={run_state} retained_exp={final_summary['retained_exp_id']} promoted={1 if promoted else 0} data_limited={1 if is_data_limited else 0}"
    )


if __name__ == "__main__":
    main()
