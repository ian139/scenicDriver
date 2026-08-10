"""Hash-guarded promotion, explicit user override, and rollback for regression registries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.active_evaluation import (  # noqa: E402
    promote_from_decision,
    promote_with_user_override,
    rollback_registry,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Promote or rollback a regression model registry"
    )
    decision_group = p.add_mutually_exclusive_group()
    decision_group.add_argument("--decision", type=Path)
    decision_group.add_argument(
        "--override-decision",
        type=Path,
        help=(
            "user override JSON (schema_version 1, action "
            "activate_rejected_candidate) to explicitly activate a rejected candidate"
        ),
    )
    p.add_argument("--candidate-checkpoint", type=Path)
    p.add_argument(
        "--registry",
        "--registry-json",
        dest="registry",
        type=Path,
        default=Path("data/processed/regression/model_registry.json"),
    )
    p.add_argument("--expected-registry-sha256")
    p.add_argument("--expected-checkpoint-sha256")
    p.add_argument("--run-name", default="active_model")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--target-history-index", type=int)
    args = p.parse_args()
    if args.rollback:
        if args.target_history_index is None:
            p.error("--rollback requires --target-history-index")
    elif args.decision is None and args.override_decision is None:
        p.error("promotion requires --decision or --override-decision")
    elif args.candidate_checkpoint is None:
        p.error("promotion requires --candidate-checkpoint")
    if args.expected_registry_sha256 is None:
        p.error(
            "--expected-registry-sha256 is required (use the current registry SHA256)"
        )
    return args


def main() -> int:
    args = parse_args()
    if args.rollback:
        result = rollback_registry(
            args.registry,
            args.target_history_index,
            args.expected_registry_sha256,
            args.expected_checkpoint_sha256,
        )
    elif args.override_decision is not None:
        result = promote_with_user_override(
            args.override_decision,
            args.candidate_checkpoint,
            args.registry,
            args.expected_registry_sha256,
            args.run_name,
        )
    else:
        result = promote_from_decision(
            args.decision,
            args.candidate_checkpoint,
            args.registry,
            args.expected_registry_sha256,
            args.run_name,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
