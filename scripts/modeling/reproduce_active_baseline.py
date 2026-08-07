"""
CLI script to reproduce active baseline evaluation metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenic_scorer.active_evaluation import evaluate_active_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce active baseline evaluation metrics."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to exported .npz dataset"
    )
    parser.add_argument(
        "--control-dataset",
        type=Path,
        required=True,
        help="Path to canonical control feature dataset .npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to active baseline checkpoint",
    )
    parser.add_argument(
        "--expanded-benchmark",
        type=Path,
        required=True,
        help="Path to expanded human benchmark CSV",
    )
    parser.add_argument(
        "--control-benchmark",
        type=Path,
        required=True,
        help="Path to control benchmark CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write evaluation JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = evaluate_active_baseline(
        dataset_path=args.dataset,
        control_dataset_path=args.control_dataset,
        checkpoint_path=args.checkpoint,
        expanded_benchmark_csv=args.expanded_benchmark,
        control_benchmark_csv=args.control_benchmark,
        output_path=args.output,
    )

    exp_metrics = result["benchmarks"]["expanded_human_benchmark"]["metrics"]
    ctrl_metrics = result["benchmarks"]["control_benchmark"]["metrics"]
    counts = result["sample_counts"]
    det = result["deterministic_inference"]
    cal = result["calibration_distribution_summary"]["expanded_human_benchmark"]

    print(f"METRIC dataset_test_samples={counts['dataset_test']}")
    print(f"METRIC control_dataset_total_samples={counts['control_dataset_total']}")
    print(f"METRIC control_dataset_test_samples={counts['control_dataset_test']}")
    print(f"METRIC expanded_benchmark_test_samples={counts['expanded_benchmark_test']}")
    print(f"METRIC control_benchmark_test_samples={counts['control_benchmark_test']}")
    print(f"METRIC expanded_mae={exp_metrics['mae']:.6f}")
    print(f"METRIC expanded_rmse={exp_metrics['rmse']:.6f}")
    print(f"METRIC expanded_pearson_corr={exp_metrics['pearson_corr']:.6f}")
    print(f"METRIC expanded_spearman_corr={exp_metrics['spearman_corr']:.6f}")
    print(f"METRIC control_mae={ctrl_metrics['mae']:.6f}")
    print(f"METRIC control_rmse={ctrl_metrics['rmse']:.6f}")
    print(f"METRIC control_pearson_corr={ctrl_metrics['pearson_corr']:.6f}")
    print(f"METRIC control_spearman_corr={ctrl_metrics['spearman_corr']:.6f}")
    print(f"METRIC max_deterministic_delta={det['max_absolute_difference']:.9f}")
    print(f"METRIC calibration_error={cal['calibration_error']:.6f}")


if __name__ == "__main__":
    main()
