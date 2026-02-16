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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.inference import get_inference_transform
from src.classifier.model import LandscapeClassifier
from src.terrain.features import TerrainFeatures, repair_terrain_zero_seam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scenic regression dataset (.npz)")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier-ckpt", type=Path, default=Path("models/classifier/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def _decode_terrain_rgb(terrain_img: Image.Image) -> np.ndarray:
    arr = np.array(terrain_img).astype(np.float32)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1


def _terrain_features_from_images(terrain_img: Image.Image, sat_img: Image.Image) -> np.ndarray:
    elev = repair_terrain_zero_seam(_decode_terrain_rgb(terrain_img))
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx ** 2 + gy ** 2)

    relief = float(elev.max() - elev.min())
    slope_variation = float(min(slope.std() / 15.0, 1.0))
    low_elev = elev < np.percentile(elev, 10)
    flat = slope < np.percentile(slope, 10)
    water_proximity = float((low_elev & flat).mean())

    sat_arr = np.array(sat_img).astype(np.float32)
    r = sat_arr[..., 0]
    g = sat_arr[..., 1]
    b = sat_arr[..., 2]
    vegetation_density = float((g / np.maximum(r + g + b, 1e-6)).mean())

    terrain = TerrainFeatures(
        slope_variation=slope_variation,
        elevation_change=relief,
        water_proximity=water_proximity,
        vegetation_density=vegetation_density,
        coastal=False,
        has_lake=False,
        has_river=False,
    )
    return terrain.to_array().astype(np.float32)


def _terrain_path_from_sat(image_path: str) -> str:
    return image_path.replace("images/satellite/", "images/terrain/")


def main() -> None:
    args = parse_args()
    if not args.labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {args.labels_csv}")
    if not args.classifier_ckpt.exists():
        raise FileNotFoundError(f"classifier checkpoint not found: {args.classifier_ckpt}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.labels_csv)
    required = {"image_path", "scenic_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {sorted(missing)}")
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    if df.empty:
        raise ValueError("No rows found in labels.csv")

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
                )
            )

            if len(batch_imgs) < args.batch_size:
                continue

            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            feats = model.get_features(x)
            probs = torch.softmax(logits, dim=1)

            for i in range(x.shape[0]):
                terrain, score, class_id = batch_meta[i]
                vit_embeddings.append(feats[i].detach().cpu().numpy().astype(np.float32))
                terrain_features.append(terrain)
                class_logits.append(logits[i].detach().cpu().numpy().astype(np.float32))
                class_probs.append(probs[i].detach().cpu().numpy().astype(np.float32))
                scenic_scores.append(np.float32(score))
                class_ids.append(np.int64(class_id))

            batch_imgs = []
            batch_meta = []

        if batch_imgs:
            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            feats = model.get_features(x)
            probs = torch.softmax(logits, dim=1)
            for i in range(x.shape[0]):
                terrain, score, class_id = batch_meta[i]
                vit_embeddings.append(feats[i].detach().cpu().numpy().astype(np.float32))
                terrain_features.append(terrain)
                class_logits.append(logits[i].detach().cpu().numpy().astype(np.float32))
                class_probs.append(probs[i].detach().cpu().numpy().astype(np.float32))
                scenic_scores.append(np.float32(score))
                class_ids.append(np.int64(class_id))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vit_embeddings=np.array(vit_embeddings, dtype=np.float32),
        terrain_features=np.array(terrain_features, dtype=np.float32),
        class_logits=np.array(class_logits, dtype=np.float32),
        class_probs=np.array(class_probs, dtype=np.float32),
        scenic_scores=np.array(scenic_scores, dtype=np.float32),
        class_ids=np.array(class_ids, dtype=np.int64),
    )
    print(f"Wrote dataset: {args.output}")
    print(f"Samples: {len(vit_embeddings)}")
    if missing_rows:
        print(f"Skipped rows with missing files: {missing_rows}")
    if len(vit_embeddings) == 0:
        raise ValueError("No samples exported. Check labels paths and --raw-dir.")


if __name__ == "__main__":
    main()
