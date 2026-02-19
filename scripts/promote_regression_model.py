"""
Promotion gate for regression checkpoints.

Promotes candidate only if it beats baseline thresholds and updates
data/processed/regression/model_registry.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote candidate model based on baseline comparison")
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--registry-json",
        type=Path,
        default=Path("data/processed/regression/model_registry.json"),
    )
    parser.add_argument("--min-corr-delta", type=float, default=0.0)
    parser.add_argument("--min-mae-improvement", type=float, default=0.0)
    parser.add_argument("--min-rmse-improvement", type=float, default=0.0)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing json file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_record(
    *,
    checkpoint: Path,
    metrics: dict,
    run_name: str | None,
    source: str,
) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "metrics": {
            "corr": float(metrics["corr"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "samples": int(metrics.get("samples", 0)),
        },
        "run_name": run_name,
        "source_metrics": source,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    candidate = _read_json(args.candidate_metrics)
    baseline = _read_json(args.baseline_metrics)

    corr_delta = float(candidate["corr"]) - float(baseline["corr"])
    mae_improvement = float(baseline["mae"]) - float(candidate["mae"])
    rmse_improvement = float(baseline["rmse"]) - float(candidate["rmse"])

    pass_gate = (
        corr_delta >= args.min_corr_delta
        and mae_improvement >= args.min_mae_improvement
        and rmse_improvement >= args.min_rmse_improvement
    )

    registry = {"active": None, "history": []}
    if args.registry_json.exists():
        registry = _read_json(args.registry_json)
        if "history" not in registry or not isinstance(registry["history"], list):
            registry["history"] = []

    decision = {
        "candidate_metrics": str(args.candidate_metrics),
        "baseline_metrics": str(args.baseline_metrics),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "thresholds": {
            "min_corr_delta": args.min_corr_delta,
            "min_mae_improvement": args.min_mae_improvement,
            "min_rmse_improvement": args.min_rmse_improvement,
        },
        "actual": {
            "corr_delta": corr_delta,
            "mae_improvement": mae_improvement,
            "rmse_improvement": rmse_improvement,
        },
        "promoted": pass_gate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if pass_gate:
        active = _build_record(
            checkpoint=args.candidate_checkpoint,
            metrics=candidate,
            run_name=args.run_name,
            source=str(args.candidate_metrics),
        )
        registry["active"] = active
        registry["history"].append(
            {
                "event": "promote",
                "record": active,
                "decision": decision,
            }
        )
    else:
        registry["history"].append(
            {
                "event": "reject",
                "record": _build_record(
                    checkpoint=args.candidate_checkpoint,
                    metrics=candidate,
                    run_name=args.run_name,
                    source=str(args.candidate_metrics),
                ),
                "decision": decision,
            }
        )

    args.registry_json.parent.mkdir(parents=True, exist_ok=True)
    args.registry_json.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"Wrote registry: {args.registry_json}")


if __name__ == "__main__":
    main()
