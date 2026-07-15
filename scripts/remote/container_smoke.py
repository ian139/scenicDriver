#!/usr/bin/env python3
"""Container smoke check — verifies imports and device availability.

Does NOT download models, touch S3, create data files, or train.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            print(
                json.dumps({"ok": False, "error": "cuda requested but unavailable"}),
                flush=True,
            )
            sys.exit(2)
        return "cuda"

    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_imports() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    import_checks: list[tuple[str, str, str | None]] = [
        ("torch", "torch", None),
        ("torchvision", "torchvision", None),
        ("timm", "timm", None),
        ("boto3", "boto3", None),
        ("src.classifier.model", "src.classifier.model", "LandscapeClassifier"),
        (
            "src.scenic_scorer.regression",
            "src.scenic_scorer.regression",
            "ScenicRegressionModel",
        ),
    ]
    for name, module_path, attr_name in import_checks:
        try:
            module = importlib.import_module(module_path)
            if attr_name is not None:
                getattr(module, attr_name)
            checks[name] = True
        except Exception:
            checks[name] = False
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Container smoke check")
    parser.add_argument(
        "--check-imports",
        action="store_true",
        default=False,
        help="Verify training imports resolve",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Target device (default: auto)",
    )
    parser.add_argument(
        "--json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Output JSON (default: true)",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)

    import torch

    checks = _check_imports() if args.check_imports else {}
    result: dict = {
        "ok": all(checks.values()) if args.check_imports else True,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "checks": checks,
        "cwd": os.getcwd(),
    }

    if args.json:
        print(json.dumps(result), flush=True)
    else:
        for key, value in result.items():
            print(f"{key}: {value}")

    if args.check_imports and not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
