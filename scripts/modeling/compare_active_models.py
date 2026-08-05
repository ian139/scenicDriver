"""CLI for strict Stage-Two active model comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.scenic_scorer.active_evaluation import evaluate_stage_two


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare candidate and baseline scenic models")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--candidate-checkpoint", required=True, type=Path)
    p.add_argument("--baseline-checkpoint", required=True, type=Path)
    p.add_argument("--expanded-benchmark", required=True, type=Path)
    p.add_argument("--control-benchmark", required=True, type=Path)
    p.add_argument("--route-qa", required=True, type=Path)
    p.add_argument("--thresholds", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    decision = evaluate_stage_two(
        args.dataset, args.candidate_checkpoint, args.baseline_checkpoint,
        args.expanded_benchmark, args.control_benchmark, args.route_qa,
        args.thresholds, args.output,
    )
    print(json.dumps({"all_gates_pass": decision["all_gates_pass"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
