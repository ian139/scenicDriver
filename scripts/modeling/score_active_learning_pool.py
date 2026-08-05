"""CLI for resumable active-learning tile-pool scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.active_learning.scoring import run_active_learning_scoring


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a canonical tile manifest for active learning")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/active_learning"))
    parser.add_argument("--run-name", default="active_learning")
    parser.add_argument("--registry", type=Path, default=Path("data/processed/regression/model_registry.json"))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path("models/classifier/best_model.pt"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--lsh-seed", type=int, default=0)
    parser.add_argument("--lsh-bits", type=int, default=16)
    parser.add_argument("--imagenet-stats", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_active_learning_scoring(
        args.manifest,
        output_dir=args.output_dir,
        run_name=args.run_name,
        registry_path=args.registry,
        classifier_checkpoint=args.classifier_checkpoint,
        device=args.device,
        classifier_use_resisc45_stats=not args.imagenet_stats,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        lsh_seed=args.lsh_seed,
        lsh_bits=args.lsh_bits,
    )
    print(json.dumps({
        "run_root": str(args.output_dir / args.run_name),
        "candidate_pool": "candidate_pool.csv",
        "feature_embeddings": "feature_embeddings.npz",
        "scoring_manifest": "scoring_manifest.json",
        "counts": result["counts"],
        "state": result["state"],
    }, sort_keys=True))
    if not result["state"]["ready_for_selection"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
