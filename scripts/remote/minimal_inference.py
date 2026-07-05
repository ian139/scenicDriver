#!/usr/bin/env python3
"""Minimal inference smoke for a Vast GPU runtime.

Default behavior with ``--checkpoint`` loads the ScenicRegressionModel checkpoint
pulled from S3 and runs one forward pass. If ``--dataset`` is supplied, the first
sample from that downloaded .npz dataset is used; otherwise deterministic
synthetic features with the checkpoint's recorded dimensions are used.

Without ``--checkpoint`` the script runs a classifier graph smoke with
``LandscapeClassifier(pretrained=False)``. That fallback is intended for local or
``SCENIC_ALLOW_MISSING_ARTIFACTS=1`` smoke-only runs; it performs no downloads.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            print(json.dumps({"ok": False, "error": "cuda requested but unavailable"}), flush=True)
            sys.exit(2)
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_info() -> dict[str, Any]:
    import torch

    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count() if available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if available else None,
    }


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a dict payload: {path}")
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload
    else:
        state_dict = payload
        metadata = {}
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint model_state_dict must be a dict: {path}")
    return state_dict, metadata


def _sample_features(dataset_path: Path | None, dims: tuple[int, int, int], device: str) -> tuple[Any, Any, Any, float | None]:
    import torch

    if dataset_path is None:
        vit_dim, terrain_dim, num_classes = dims
        return (
            torch.randn(1, vit_dim, device=device),
            torch.randn(1, terrain_dim, device=device),
            torch.randn(1, num_classes, device=device),
            None,
        )

    from src.scenic_scorer.regression import ScenicScoreDataset

    dataset = ScenicScoreDataset(dataset_path)
    vit_embedding, terrain_features, class_logits, scenic_score, _sample_weight = dataset[0]
    return (
        vit_embedding.unsqueeze(0).to(device),
        terrain_features.unsqueeze(0).to(device),
        class_logits.unsqueeze(0).to(device),
        float(scenic_score.reshape(-1)[0].item()),
    )


def _run_regression_smoke(device: str, checkpoint_path: Path, dataset_path: Path | None) -> dict[str, Any]:
    import torch

    from src.scenic_scorer.regression import ScenicRegressionModel

    state_dict, metadata = _load_checkpoint(checkpoint_path)
    vit_dim = int(metadata.get("vit_dim", 768))
    terrain_dim = int(metadata.get("terrain_dim", 6))
    num_classes = int(metadata.get("num_classes", 45))

    model = ScenicRegressionModel(vit_dim=vit_dim, terrain_dim=terrain_dim, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    vit_embedding, terrain_features, class_logits, target_score = _sample_features(
        dataset_path, (vit_dim, terrain_dim, num_classes), device
    )

    with torch.no_grad():
        _ = model(vit_embedding, terrain_features, class_logits)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        scores = model(vit_embedding, terrain_features, class_logits)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = {
        "ok": True,
        "mode": "regression-inference",
        "device": device,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path) if dataset_path else None,
        "forward_pass_sec": round(elapsed, 6),
        "batch_size": 1,
        "vit_embedding_shape": list(vit_embedding.shape),
        "terrain_features_shape": list(terrain_features.shape),
        "class_logits_shape": list(class_logits.shape),
        "output_shape": list(scores.shape),
        "scenic_score": round(float(scores.reshape(-1)[0].item()), 6),
        "target_score": target_score,
    }
    result.update(_cuda_info())
    return result


def _run_classifier_smoke(device: str) -> dict[str, Any]:
    import torch

    from src.classifier.model import LandscapeClassifier, TERRAIN_CLASSES, get_scenic_weight

    model = LandscapeClassifier(num_classes=45, pretrained=False).to(device)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    probs = torch.softmax(logits, dim=-1)
    top_prob, top_idx = probs.max(dim=-1)
    top_class = TERRAIN_CLASSES[top_idx.item()]

    result = {
        "ok": True,
        "mode": "classifier-smoke",
        "device": device,
        "checkpoint": None,
        "dataset": None,
        "forward_pass_sec": round(elapsed, 6),
        "batch_size": 1,
        "input_shape": [1, 3, 224, 224],
        "output_shape": list(logits.shape),
        "top_class": top_class,
        "top_probability": round(float(top_prob.item()), 6),
        "scenic_weight": get_scenic_weight(top_class),
    }
    result.update(_cuda_info())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Scenic Drive inference smoke test")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for inference")
    parser.add_argument("--checkpoint", type=Path, help="ScenicRegressionModel checkpoint pulled from S3")
    parser.add_argument("--dataset", type=Path, help="Optional downloaded .npz dataset; uses the first sample")
    parser.add_argument("--output", type=Path, help="Write JSON result to this file")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for synthetic fallback")
    args = parser.parse_args()

    random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    if args.dataset is not None and not args.dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
        result = _run_regression_smoke(device=device, checkpoint_path=args.checkpoint, dataset_path=args.dataset)
    else:
        result = _run_classifier_smoke(device=device)

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Result written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
