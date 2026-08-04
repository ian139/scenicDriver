"""
Export regression training features from labels.csv.

Outputs an .npz with:
- vit_embeddings [N, 768]
- terrain_features [N, 6]
- class_logits [N, 45]
- class_probs [N, 45]
- scenic_scores [N]
- class_ids [N]
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.inference import get_inference_transform  # noqa: E402
from src.classifier.model import LandscapeClassifier  # noqa: E402
from src.terrain.features import compute_terrain_features  # noqa: E402
from src.scenic_scorer.regression import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scenic regression dataset (.npz)")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier-ckpt", type=Path, default=Path("models/classifier/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--use-resisc45-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RESISC45 normalization stats for classifier transforms",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip rows whose satellite/terrain files are missing instead of failing fast",
    )
    parser.add_argument(
        "--sample-weight-column",
        type=str,
        default=None,
        help="Optional labels.csv numeric column to use as sample_weight",
    )
    parser.add_argument(
        "--label-source-column",
        type=str,
        default="label_source",
        help="Optional labels.csv column indicating source (human/heuristic)",
    )
    parser.add_argument(
        "--human-weight",
        type=float,
        default=4.0,
        help="Sample weight for human/manual labels when using source-based weighting",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=1.0,
        help="Sample weight for heuristic labels when using source-based weighting",
    )
    parser.add_argument(
        "--default-weight",
        type=float,
        default=1.0,
        help="Fallback sample weight when source is unknown/missing",
    )
    return parser.parse_args()


def _parse_s3_raw(raw_dir: str) -> tuple[str, str] | None:
    if not raw_dir.startswith("s3://"):
        return None
    rest = raw_dir.replace("s3://", "", 1)
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _open_image(path: str, raw_dir: str, s3_client: Any | None) -> Image.Image:
    s3_info = _parse_s3_raw(raw_dir)
    if s3_info is None:
        return Image.open(Path(raw_dir) / path).convert("RGB")
    bucket, prefix = s3_info
    if s3_client is None:
        raise RuntimeError("S3 client required for s3:// raw_dir")
    key = f"{prefix}/{path}" if prefix else path
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return Image.open(BytesIO(obj["Body"].read())).convert("RGB")


def _terrain_features_from_images(terrain_img: Image.Image, sat_img: Image.Image) -> np.ndarray:
    """Return the canonical terrain vector for regression export."""
    return compute_terrain_features(terrain_img, sat_img).features.to_array().astype(np.float32)


def _terrain_path_from_sat(image_path: str) -> str:
    return image_path.replace("images/satellite/", "images/terrain/")


def _validate_positive_weight(value: float, name: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"{name} must be a positive finite float, got: {value}")
    return weight


def _sample_weight_for_row(
    row: pd.Series,
    *,
    sample_weight_column: str | None,
    label_source_column: str,
    has_label_source: bool,
    has_scenic_human: bool,
    human_weight: float,
    heuristic_weight: float,
    default_weight: float,
) -> float:
    if sample_weight_column is not None:
        raw_weight = row.get(sample_weight_column)
        return _validate_positive_weight(raw_weight, f"labels column '{sample_weight_column}'")

    if has_label_source:
        source_raw = row.get(label_source_column)
        source = "" if pd.isna(source_raw) else str(source_raw).strip().lower()
        if source == "human_override":
            return human_weight
        if source == "heuristic":
            return heuristic_weight
        return default_weight

    if has_scenic_human:
        scenic_human = row.get("scenic_human")
        if not pd.isna(scenic_human):
            return human_weight
        return heuristic_weight

    return default_weight


def main() -> None:
    args = parse_args()
    if not args.labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {args.labels_csv}")
    if not args.classifier_ckpt.exists():
        raise FileNotFoundError(f"classifier checkpoint not found: {args.classifier_ckpt}")

    device = resolve_device(args.device)

    df = pd.read_csv(args.labels_csv)
    required = {"image_path", "scenic_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {sorted(missing)}")
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    if df.empty:
        raise ValueError("No rows found in labels.csv")
    if args.sample_weight_column is not None and args.sample_weight_column not in df.columns:
        raise ValueError(
            f"Requested --sample-weight-column '{args.sample_weight_column}' not found in labels.csv"
        )

    human_weight = _validate_positive_weight(args.human_weight, "--human-weight")
    heuristic_weight = _validate_positive_weight(args.heuristic_weight, "--heuristic-weight")
    default_weight = _validate_positive_weight(args.default_weight, "--default-weight")
    has_label_source = args.label_source_column in df.columns
    has_scenic_human = "scenic_human" in df.columns

    s3_info = _parse_s3_raw(args.raw_dir)
    s3_client = None
    if s3_info is not None:
        import boto3
        s3_client = boto3.client("s3")

    model = LandscapeClassifier(
        num_classes=45,
        pretrained=False,
        pretrained_path=args.classifier_ckpt,
        device=device,
    ).to(device)
    model.eval()
    transform = get_inference_transform(use_resisc45_stats=args.use_resisc45_stats)

    vit_embeddings = []
    terrain_features = []
    class_logits = []
    class_probs = []
    scenic_scores = []
    class_ids = []
    sample_weights = []
    image_paths = []

    batch_imgs = []
    batch_meta = []
    missing_rows = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            sat_rel = str(row["image_path"])
            terr_rel = _terrain_path_from_sat(sat_rel)
            try:
                sat_img = _open_image(sat_rel, args.raw_dir, s3_client)
                terr_img = _open_image(terr_rel, args.raw_dir, s3_client)
            except FileNotFoundError as exc:
                if not args.skip_missing:
                    raise FileNotFoundError(
                        f"Missing tile for labels row: sat='{sat_rel}', terrain='{terr_rel}', raw_dir='{args.raw_dir}'. "
                        "Use --skip-missing to continue or point --raw-dir to the correct local/S3 root."
                    ) from exc
                missing_rows += 1
                if missing_rows <= 5:
                    print(f"[skip-missing] {exc}")
                continue

            tensor = transform(sat_img)
            batch_imgs.append(tensor)
            batch_meta.append(
                (
                    _terrain_features_from_images(terr_img, sat_img),
                    float(row["scenic_score"]),
                    int(row["class_id"]) if "class_id" in df.columns else -1,
                    _sample_weight_for_row(
                        row,
                        sample_weight_column=args.sample_weight_column,
                        label_source_column=args.label_source_column,
                        has_label_source=has_label_source,
                        has_scenic_human=has_scenic_human,
                        human_weight=human_weight,
                        heuristic_weight=heuristic_weight,
                        default_weight=default_weight,
                    ),
                    sat_rel,
                )
            )

            if len(batch_imgs) < args.batch_size:
                continue

            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            feats = model.get_features(x)
            probs = torch.softmax(logits, dim=1)

            for i in range(x.shape[0]):
                terrain, score, class_id, sample_weight, image_path = batch_meta[i]
                vit_embeddings.append(feats[i].detach().cpu().numpy().astype(np.float32))
                terrain_features.append(terrain)
                class_logits.append(logits[i].detach().cpu().numpy().astype(np.float32))
                class_probs.append(probs[i].detach().cpu().numpy().astype(np.float32))
                scenic_scores.append(np.float32(score))
                class_ids.append(np.int64(class_id))
                sample_weights.append(np.float32(sample_weight))
                image_paths.append(str(image_path))

            batch_imgs = []
            batch_meta = []

        if batch_imgs:
            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            feats = model.get_features(x)
            probs = torch.softmax(logits, dim=1)
            for i in range(x.shape[0]):
                terrain, score, class_id, sample_weight, image_path = batch_meta[i]
                vit_embeddings.append(feats[i].detach().cpu().numpy().astype(np.float32))
                terrain_features.append(terrain)
                class_logits.append(logits[i].detach().cpu().numpy().astype(np.float32))
                class_probs.append(probs[i].detach().cpu().numpy().astype(np.float32))
                scenic_scores.append(np.float32(score))
                class_ids.append(np.int64(class_id))
                sample_weights.append(np.float32(sample_weight))
                image_paths.append(str(image_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vit_embeddings=np.array(vit_embeddings, dtype=np.float32),
        terrain_features=np.array(terrain_features, dtype=np.float32),
        class_logits=np.array(class_logits, dtype=np.float32),
        class_probs=np.array(class_probs, dtype=np.float32),
        scenic_scores=np.array(scenic_scores, dtype=np.float32),
        class_ids=np.array(class_ids, dtype=np.int64),
        sample_weights=np.array(sample_weights, dtype=np.float32),
        image_paths=np.array(image_paths, dtype=str),
    )
    print(f"Wrote dataset: {args.output}")
    print(f"Samples: {len(vit_embeddings)}")
    if sample_weights:
        weights_arr = np.asarray(sample_weights, dtype=np.float32)
        unique_weights = np.unique(np.round(weights_arr, 6))
        if len(unique_weights) <= 10:
            printable = ", ".join(str(float(w)) for w in unique_weights.tolist())
            print(f"Sample weights used: [{printable}]")
        print(
            "Weight stats: "
            f"min={weights_arr.min():.3f}, max={weights_arr.max():.3f}, mean={weights_arr.mean():.3f}"
        )
    if missing_rows:
        print(f"Skipped rows with missing files: {missing_rows}")
    if len(vit_embeddings) == 0:
        raise ValueError("No samples exported. Check labels paths and --raw-dir.")


if __name__ == "__main__":
    main()
