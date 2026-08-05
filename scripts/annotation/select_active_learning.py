"""CLI for deterministic active-learning selection and spatial split artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.active_learning.selection import SelectionConfig, run_selection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic active-learning annotation batch"
    )
    parser.add_argument(
        "--candidate-input",
        "--candidates",
        "--input",
        dest="candidate_input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/active_learning")
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-annotations", type=Path)
    parser.add_argument("--adjacency-radius", type=int, default=1)
    parser.add_argument("--min-separation-km", type=float, default=0.0)
    parser.add_argument("--qa-overlap-count", type=int, default=0)
    parser.add_argument("--qa-overlap-fraction", type=float, default=0.0)
    parser.add_argument("--random-control-count", type=int, default=0)
    parser.add_argument("--random-control-fraction", type=float, default=0.0)
    parser.add_argument("--weight", action="append", default=[], metavar="NAME=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights: dict[str, float] = {}
    for item in args.weight:
        if "=" not in item:
            raise SystemExit("--weight uses NAME=VALUE")
        key, value = item.split("=", 1)
        weights[key] = float(value)
    config = SelectionConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        run_name=args.run_name,
        adjacency_radius=args.adjacency_radius,
        min_separation_km=args.min_separation_km,
        qa_overlap_count=args.qa_overlap_count,
        qa_overlap_fraction=args.qa_overlap_fraction,
        random_control_count=args.random_control_count,
        random_control_fraction=args.random_control_fraction,
        weights=weights or SelectionConfig().weights,
    )
    artifacts = run_selection(
        args.candidate_input,
        output_dir=args.output_dir,
        run_name=args.run_name,
        config=config,
        prior_annotations=args.prior_annotations,
    )
    root = args.output_dir / args.run_name
    print(
        json.dumps(
            {
                "run_root": str(root),
                "batch": str(root / "annotation_batch.csv"),
                "rows": len(artifacts.selected),
                "batch_id": artifacts.selected.attrs.get("batch_id"),
                "leakage_audit_valid": artifacts.leakage_audit.get("valid", False),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
