"""CLI for strict Stage-Two active model comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.active_evaluation import evaluate_stage_two  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare candidate and baseline scenic models"
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--control-dataset", required=True, type=Path)
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
        dataset_path=args.dataset,
        control_dataset_path=args.control_dataset,
        candidate_checkpoint=args.candidate_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        expanded_benchmark_csv=args.expanded_benchmark,
        control_benchmark_csv=args.control_benchmark,
        route_qa_json=args.route_qa,
        thresholds=args.thresholds,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"all_gates_pass": decision["all_gates_pass"], "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
