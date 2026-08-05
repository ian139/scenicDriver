#!/usr/bin/env python3
"""
Stage-Two Active Scenic Autoresearch Orchestrator.

Orchestrates Stage-Two active scenic model training, evaluation, threshold gating,
and optional model registry promotion from a validated Stage-One handoff.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.scenic_scorer.active_training import (
    ActiveTrainingConfig,
    prepare_active_dataset,
    train_active_model,
)
from src.scenic_scorer.active_evaluation import (
    evaluate_stage_two,
    promote_from_decision,
    rollback_registry,
)


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


def sanitize_command(cmd: str | List[str]) -> str:
    """Sanitize command strings before persisting them in run artifacts."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    return re.sub(
        r"(?i)(--?(?:api[-_]?key|token|secret|password|auth))(?:=|\s+)(['\"]?)([^'\"\s]+)\2",
        r"\1=[REDACTED]",
        cmd_str,
    )


def resolve_stage_one_handoff(
    handoff_arg: Optional[str] = None
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
        raise ValueError(
            f"Handoff {handoff_path} has active blockers: {blockers}"
        )

    artifacts = handoff_data.get("artifacts", {})
    root_dir = handoff_path.parent
    for name, record in artifacts.items():
        if isinstance(record, dict):
            rel_path = record.get("path")
            required = record.get("required", True)
            expected_hash = record.get("sha256")
            if rel_path:
                art_p = root_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
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


def build_candidate_ladder(
    base_config: ActiveTrainingConfig, max_experiments: int
) -> List[Dict[str, Any]]:
    """Generate a deterministic experiment candidate ladder."""
    ladder = []
    # Candidate 1: Baseline configuration
    c1 = dataclasses.replace(base_config)
    ladder.append({
        "exp_id": "exp_01_baseline_control",
        "hypothesis": "Baseline training configuration with standard weights.",
        "config": c1,
    })

    # Candidate 2: Region-balanced sample weighting
    if max_experiments >= 2:
        c2 = dataclasses.replace(base_config, sample_weight_scheme="region_balanced")
        ladder.append({
            "exp_id": "exp_02_region_balanced",
            "hypothesis": "Region-balanced sample weighting improves low-support regional slices.",
            "config": c2,
        })

    # Candidate 3: Robust Huber loss
    if max_experiments >= 3:
        c3 = dataclasses.replace(base_config, loss_function="huber")
        ladder.append({
            "exp_id": "exp_03_robust_huber_loss",
            "hypothesis": "Huber loss reduces sensitivity to noisy human annotations.",
            "config": c3,
        })

    # Candidate 4: Lower learning rate with weight decay
    if max_experiments >= 4:
        c4 = dataclasses.replace(base_config, learning_rate=5e-5, weight_decay=1e-3)
        ladder.append({
            "exp_id": "exp_04_fine_learning_rate",
            "hypothesis": "Lower learning rate prevents overfitting on small active sets.",
            "config": c4,
        })

    # Candidate 5: Higher capacity / extra steps
    if max_experiments >= 5:
        c5 = dataclasses.replace(base_config, epochs=15)
        ladder.append({
            "exp_id": "exp_05_extended_epochs",
            "hypothesis": "Extended training epochs improves continuous scenic score ranking.",
            "config": c5,
        })

    return ladder[:max_experiments]


def load_existing_experiments(experiments_jsonl: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
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
            status = rec.get("status")
            if exp_id and status in ("completed", "retained"):
                completed_map[exp_id] = rec
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
        default=None,
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
        type=str,
        default=None,
        help="Path to expanded human benchmark CSV.",
    )
    parser.add_argument(
        "--control-benchmark-csv",
        type=str,
        default=None,
        help="Path to New England control benchmark CSV.",
    )
    parser.add_argument(
        "--route-qa-json",
        type=str,
        default=None,
        help="Path to route QA JSON specifications.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Handoff Resolution and Fast Failure before training
    print("METRIC handoff_validating=1")
    try:
        handoff_path, handoff_data = resolve_stage_one_handoff(args.handoff)
        validate_handoff_content(handoff_path, handoff_data)
        print(f"METRIC handoff_ready=1 handoff_path={handoff_path}")
    except Exception as err:
        print(f"ERROR: Stage-One handoff validation failed: {err}", file=sys.stderr)
        print("METRIC handoff_ready=0")
        sys.exit(1)

    # 2. Setup Run Directory
    run_dir = Path("data/processed/modeling_autoresearch") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    experiments_jsonl = run_dir / "experiments.jsonl"
    decision_path = run_dir / "promotion_decision.json"
    summary_path = run_dir / "final_summary.json"

    # 3. Capture Immutable Baseline Identity
    registry_path = Path("data/processed/regression/model_registry.json")
    if not registry_path.exists():
        print(f"ERROR: Active model registry missing at {registry_path}", file=sys.stderr)
        print("METRIC baseline_sha256=missing")
        sys.exit(1)

    baseline_registry_sha256 = compute_sha256(registry_path)
    print(f"METRIC baseline_sha256={baseline_registry_sha256}")

    registry_content = json.loads(registry_path.read_text(encoding="utf-8"))
    active_model = registry_content.get("active_model", {})
    baseline_checkpoint_path = active_model.get("checkpoint_path", "data/processed/regression/model.pt")

    # Handle --status mode
    if args.status:
        completed_map, all_records = load_existing_experiments(experiments_jsonl)
        print(f"Run status for '{args.run_name}':")
        print(f"  Directory: {run_dir}")
        print(f"  Baseline Registry SHA256: {baseline_registry_sha256}")
        print(f"  Completed Experiments: {len(completed_map)}")
        for exp_id, rec in completed_map.items():
            metrics = rec.get("metrics", {})
            print(f"    - {exp_id}: MAE={metrics.get('mae')} RMSE={metrics.get('rmse')} Corr={metrics.get('corr')}")
        sys.exit(0)

    # Initialize Manifest
    manifest = {
        "run_name": args.run_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage1_handoff_path": str(handoff_path),
        "stage1_handoff_sha256": compute_sha256(handoff_path),
        "baseline_registry_path": str(registry_path),
        "baseline_registry_sha256": baseline_registry_sha256,
        "baseline_checkpoint_path": baseline_checkpoint_path,
        "dry_run": args.dry_run,
        "promote_requested": args.promote,
        "config": {
            "seed": args.seed,
            "device": args.device,
            "max_experiments": args.max_experiments,
            "max_steps": args.max_steps,
            "max_seconds": args.max_seconds,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4. Build Experiment Ladder
    base_config = ActiveTrainingConfig(
        seed=args.seed,
        max_steps=args.max_steps,
    )
    ladder = build_candidate_ladder(base_config, args.max_experiments)

    # Handle --dry-run mode
    if args.dry_run:
        planned_records = []
        for exp in ladder:
            exp_rec = {
                "exp_id": exp["exp_id"],
                "hypothesis": exp["hypothesis"],
                "status": "planned",
                "command": sanitize_command(f"train_active_model --exp {exp['exp_id']}"),
                "config": dataclasses.asdict(exp["config"]) if dataclasses.is_dataclass(exp["config"]) else str(exp["config"]),
            }
            planned_records.append(exp_rec)

        with experiments_jsonl.open("w", encoding="utf-8") as f:
            for rec in planned_records:
                f.write(json.dumps(rec) + "\n")

        print(f"METRIC dry_run=1 planned_experiments={len(planned_records)}")
        print("Dry run plan written successfully. No mutations performed.")
        sys.exit(0)

    # 5. Load truthful resume state
    completed_map, _ = load_existing_experiments(experiments_jsonl) if args.resume else ({}, [])

    # 6. Autoresearch Loop
    start_time = time.time()
    evaluated_records: List[Dict[str, Any]] = []

    prepared_dataset_path = run_dir / "prepared_dataset.npz"
    split_csv_path = handoff_path.parent / "splits.csv"
    if not split_csv_path.exists():
        split_csv_path = handoff_path.parent / "benchmark_split.csv"

    # Dataset Preparation
    dataset_info = prepare_active_dataset(handoff_path, prepared_dataset_path)
    dataset_path = dataset_info.get("dataset_path", str(prepared_dataset_path))

    for exp in ladder:
        exp_id = exp["exp_id"]

        # Check time budget
        if args.max_seconds and (time.time() - start_time) > args.max_seconds:
            print(f"Time budget of {args.max_seconds}s reached. Stopping autoresearch loop.")
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
            train_result = train_active_model(
                dataset_path=dataset_path,
                split_csv=split_csv_path if split_csv_path.exists() else handoff_path,
                output_dir=exp_dir,
                config=exp["config"],
                resume=args.resume,
            )
            candidate_ckpt = train_result.get("checkpoint_path", str(exp_dir / "candidate.pt"))

            thresholds = None
            if args.thresholds_json and Path(args.thresholds_json).exists():
                thresholds = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))

            eval_result = evaluate_stage_two(
                dataset_path=dataset_path,
                candidate_checkpoint=candidate_ckpt,
                baseline_checkpoint=baseline_checkpoint_path,
                expanded_benchmark_csv=args.expanded_benchmark_csv,
                control_benchmark_csv=args.control_benchmark_csv,
                route_qa_json=args.route_qa_json,
                thresholds=thresholds,
                output_path=exp_dir / "eval_decision.json",
            )

            all_pass = eval_result.get("all_gates_pass", False)
            metrics = eval_result.get("metrics", {})
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            corr = metrics.get("corr", 0.0)

            exp_record = {
                "exp_id": exp_id,
                "hypothesis": exp["hypothesis"],
                "status": "completed",
                "candidate_checkpoint": candidate_ckpt,
                "metrics": metrics,
                "all_gates_pass": all_pass,
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
            }
            evaluated_records.append(exp_record)
            with experiments_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(exp_record) + "\n")
            print(f"METRIC exp_id={exp_id} status=failed")

    # 7. Select Best Passing Candidate
    passing_candidates = [r for r in evaluated_records if r.get("all_gates_pass")]
    best_candidate = None
    if passing_candidates:
        # Sort by highest correlation or lowest MAE
        passing_candidates.sort(
            key=lambda r: (r.get("metrics", {}).get("corr", 0.0), -r.get("metrics", {}).get("mae", 10.0)),
            reverse=True,
        )
        best_candidate = passing_candidates[0]

    all_gates_pass = best_candidate is not None
    print(f"METRIC all_gates_pass={1 if all_gates_pass else 0}")

    decision_summary = {
        "run_name": args.run_name,
        "all_gates_pass": all_gates_pass,
        "retained_candidate": best_candidate,
        "evaluated_experiments": len(evaluated_records),
        "baseline_registry_sha256": baseline_registry_sha256,
        "decision_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    decision_path.write_text(json.dumps(decision_summary, indent=2), encoding="utf-8")

    # 8. Registry Promotion Safety
    promoted = False
    if args.promote:
        if all_gates_pass and best_candidate:
            cand_ckpt = best_candidate["candidate_checkpoint"]
            print(f"Promoting candidate {best_candidate['exp_id']} to registry...")
            promo_res = promote_from_decision(
                decision_path=decision_path,
                candidate_checkpoint=cand_ckpt,
                registry_path=registry_path,
                expected_registry_sha256=baseline_registry_sha256,
                run_name=args.run_name,
            )
            promoted = True
            print(f"METRIC promoted=1 new_version={promo_res.get('promoted_version')}")
        else:
            print("Promotion requested but compound decision gates failed or no passing candidate found. Registry unchanged.")
            print("METRIC promoted=0 promotion_blocked=1")
    else:
        print("METRIC promoted=0 promote_flag_absent=1")

    # 9. Final Summary
    final_summary = {
        "run_name": args.run_name,
        "run_state": "completed",
        "total_experiments": len(evaluated_records),
        "all_gates_pass": all_gates_pass,
        "retained_exp_id": best_candidate["exp_id"] if best_candidate else None,
        "promoted": promoted,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"METRIC run_state=completed retained_exp={final_summary['retained_exp_id']} promoted={1 if promoted else 0}")


if __name__ == "__main__":
    main()
