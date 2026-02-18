from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional
import re
import os
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from src.terrain.features import TerrainFeatures, repair_terrain_zero_seam


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_COORD_RE = re.compile(r"z(\d+)[/_-]*x(\d+)[/_-]*y(\d+)", re.IGNORECASE)


@dataclass
class HeuristicLabelerConfig:
    raw_dir: str = "data/raw"
    satellite_dir: str = "data/raw/images/satellite/z16"
    terrain_dir: str = "data/raw/images/terrain/z16"
    processed_dir: str = "data/processed"
    labels_csv: str = "data/raw/labels.csv"
    classifier_best_ckpt: str = "models/classifier/best_model.pt"
    classifier_use_resisc45_stats: bool = True
    heuristic_max_tiles: int = 2000
    heuristic_use_classifier: bool = True
    heuristic_default_lat: float = 0.0
    heuristic_default_lon: float = 0.0
    seed: int = 42


def parse_tile_coords(path: Path | PurePosixPath | str) -> Optional[tuple[int, int, int]]:
    posix_path = PurePosixPath(path) if isinstance(path, str) else path
    match = _COORD_RE.search(posix_path.as_posix())
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Handle layout: .../z16/19073_24793.png (x_y in filename, zoom in parent dir)
    try:
        zoom = None
        if posix_path.parent.name.lower().startswith("z"):
            zoom = int(posix_path.parent.name[1:])
        elif posix_path.parent.parent.name.lower().startswith("z"):
            zoom = int(posix_path.parent.parent.name[1:])
        stem_parts = posix_path.stem.split("_")
        if zoom is not None and len(stem_parts) == 2:
            x, y = int(stem_parts[0]), int(stem_parts[1])
            return zoom, x, y
    except ValueError:
        return None

    return None


def run_heuristic_labeling(
    *,
    satellite_dir: str | Path,
    terrain_dir: str | Path,
    raw_dir: str | Path,
    max_tiles: int,
    use_classifier: bool,
    classifier_best_ckpt: str | Path,
    classifier_use_resisc45_stats: bool,
    default_lat: float,
    default_lon: float,
    seed: int,
    device: str = "auto",
    learned_regression_ckpt: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from math import ceil

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for heuristic labeling.") from exc

    np.random.seed(seed)
    torch.manual_seed(seed)

    sat_dir = Path(satellite_dir)
    terrain_dir = Path(terrain_dir)
    raw_root = Path(raw_dir)

    s3_bucket = os.getenv("SCENIC_S3_BUCKET")
    s3_only = bool(s3_bucket) and os.getenv("SCENIC_S3_ONLY", "1").lower() not in ("0", "false", "no")
    s3_client = _make_s3_client(s3_bucket)

    sat_files: list[Path | S3ImageRef]
    terrain_files: list[Path | S3ImageRef]

    if s3_only:
        sat_files = _list_images_s3(s3_bucket, _s3_prefix_from_local_dir(sat_dir), s3_client=s3_client)
        terrain_files = _list_images_s3(
            s3_bucket, _s3_prefix_from_local_dir(terrain_dir), s3_client=s3_client
        )
    else:
        if not sat_dir.exists():
            raise FileNotFoundError(f"Satellite dir not found: {sat_dir}")
        if not terrain_dir.exists():
            raise FileNotFoundError(f"Terrain dir not found: {terrain_dir}")

        sat_files = _list_images(sat_dir)
        terrain_files = _list_images(terrain_dir)

        if (not sat_files or not terrain_files) and s3_bucket:
            if not sat_files:
                sat_files = _list_images_s3(
                    s3_bucket, _s3_prefix_from_local_dir(sat_dir), s3_client=s3_client
                )
            if not terrain_files:
                terrain_files = _list_images_s3(
                    s3_bucket, _s3_prefix_from_local_dir(terrain_dir), s3_client=s3_client
                )

    if not sat_files:
        raise ValueError(f"No satellite images found in: {sat_dir}")
    if not terrain_files:
        raise ValueError(f"No terrain images found in: {terrain_dir}")

    sat_coords = [parse_tile_coords(_path_for_coords(p)) for p in sat_files]
    terrain_coords = [parse_tile_coords(_path_for_coords(p)) for p in terrain_files]
    coords_ok = all(c is not None for c in sat_coords) and all(c is not None for c in terrain_coords)

    warnings: list[str] = []
    if not coords_ok:
        warnings.append(
            "Tile coordinate parsing failed for some files; spatial heatmap will be disabled."
        )

    if coords_ok:
        terrain_index: dict[tuple[int, int, int], Path | S3ImageRef] = {
            coords: path for coords, path in zip(terrain_coords, terrain_files)
        }
    else:
        terrain_index = {Path(_path_for_coords(path)).stem: path for path in terrain_files}

    classifier, clf_transform, clf_device, class_names = _load_classifier(
        use_classifier=use_classifier,
        classifier_best_ckpt=classifier_best_ckpt,
        classifier_use_resisc45_stats=classifier_use_resisc45_stats,
        device=device,
        warnings=warnings,
    )
    use_learned_scoring = learned_regression_ckpt is not None
    if use_learned_scoring and classifier is None:
        raise ValueError(
            "Learned scoring requires classifier outputs. "
            "Ensure classifier checkpoint is available and do not disable classifier."
        )
    learned_model = _load_learned_regressor(
        learned_regression_ckpt=learned_regression_ckpt,
        device=clf_device,
        warnings=warnings,
    )

    rows: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    processed = 0
    missing_pairs = 0
    max_tiles = max(1, int(ceil(max_tiles)))

    for sat_path, coords in zip(sat_files, sat_coords):
        if processed >= max_tiles:
            break

        key = coords if coords_ok else Path(_path_for_coords(sat_path)).stem
        terrain_path = terrain_index.get(key)
        if terrain_path is None:
            missing_pairs += 1
            continue

        sat_rel, terrain_rel = _relative_paths_or_raise(
            sat_path,
            terrain_path,
            raw_root,
            s3_bucket=s3_bucket,
            s3_only=s3_only,
        )

        sat_img = _open_image(sat_path, s3_bucket, s3_client=s3_client).convert("RGB")
        terrain_img = _open_image(terrain_path, s3_bucket, s3_client=s3_client).convert("RGB")

        elev = repair_terrain_zero_seam(_decode_terrain_rgb(terrain_img))
        gy, gx = np.gradient(elev)
        slope = np.sqrt(gx ** 2 + gy ** 2)

        relief = float(elev.max() - elev.min())
        roughness = float(elev.std())
        slope_mean = float(slope.mean())

        low_elev = elev < np.percentile(elev, 10)
        flat = slope < np.percentile(slope, 10)
        water_proxy = float((low_elev & flat).mean())

        sat_arr = np.array(sat_img).astype(np.float32)
        r = sat_arr[..., 0] / 255.0
        g = sat_arr[..., 1] / 255.0
        b = sat_arr[..., 2] / 255.0
        veg_proxy = float(_safe_div(g, r + g + b).mean())
        brightness = (r + g + b) / 3.0
        maxc = np.maximum(r, np.maximum(g, b))
        minc = np.minimum(r, np.minimum(g, b))
        saturation = _safe_div(maxc - minc, maxc + 1e-6)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        texture = float(gray.std())

        water_mask = (
            (b > r * 1.2)
            & (b > g * 1.15)
            & (brightness < 0.65)
            & (saturation > 0.18)
            & (texture < 0.12)
        )
        water_fraction = float(water_mask.mean())

        terrain_features = _build_regression_terrain_features(
            slope=slope,
            relief=relief,
            water_proxy=water_proxy,
            veg_proxy=veg_proxy,
        )
        slope_variation = float(terrain_features[0])

        if learned_model is not None:
            class_id, class_name, class_score, class_logits, vit_embedding = _infer_classifier_outputs(
                sat_img=sat_img,
                classifier=classifier,
                clf_transform=clf_transform,
                device=clf_device,
                class_names=class_names,
            )
            score = _predict_learned_score(
                model=learned_model,
                vit_embedding=vit_embedding,
                terrain_features=terrain_features,
                class_logits=class_logits,
                device=clf_device,
            )
        else:
            class_id, class_name, class_score = _infer_class(
                sat_img=sat_img,
                classifier=classifier,
                clf_transform=clf_transform,
                device=clf_device,
                class_names=class_names,
            )
            class_weight = 2.5 * max(0.3, 1.0 - 0.9 * water_fraction)
            score = (
                class_weight * class_score
                + 2.0 * np.tanh(relief / 500.0)
                + 1.5 * np.tanh(roughness / 200.0)
                + 1.5 * np.tanh(slope_mean / 15.0)
                + 1.5 * water_proxy
                + 1.0 * veg_proxy
                - 1.2 * water_fraction
            )
            score = float(np.clip(score, 0.0, 10.0))

        rows.append(
            {
                "image_path": sat_rel,
                "scenic_score": score,
                "lat": default_lat,
                "lon": default_lon,
                "class_id": class_id,
            }
        )

        tile_info = {
            "image_path": sat_rel,
            "terrain_path": terrain_rel,
            "scenic_score": score,
            "class_id": class_id,
            "class_name": class_name,
            "class_score": class_score,
            "relief": relief,
            "roughness": roughness,
            "slope_variation": slope_variation,
            "slope_mean": slope_mean,
            "water_proxy": water_proxy,
            "veg_proxy": veg_proxy,
            "water_fraction": water_fraction,
            "texture": texture,
        }
        if coords_ok and coords is not None:
            tile_info.update({"z": coords[0], "x": coords[1], "y": coords[2]})
        tiles.append(tile_info)
        processed += 1

    if not rows:
        raise ValueError("No paired satellite/terrain tiles found to label.")

    labels_df = pd.DataFrame(rows)

    raw_dir_config = raw_dir if isinstance(raw_dir, str) and raw_dir.startswith("s3://") else str(raw_root)

    run_info = {
        "counts": {
            "satellite_total": len(sat_files),
            "terrain_total": len(terrain_files),
            "paired": len(rows),
            "missing_pairs": missing_pairs,
        },
        "used_classifier": classifier is not None,
        "device": clf_device,
        "coords_available": coords_ok,
        "warnings": warnings,
        "seed": seed,
        "scoring_mode": "learned" if learned_model is not None else "heuristic",
        "config": {
            "satellite_dir": str(sat_dir),
            "terrain_dir": str(terrain_dir),
            "raw_dir": raw_dir_config,
            "s3_bucket": s3_bucket,
            "s3_only": s3_only,
            "max_tiles": max_tiles,
            "use_classifier": use_classifier,
            "classifier_best_ckpt": str(classifier_best_ckpt),
            "classifier_use_resisc45_stats": classifier_use_resisc45_stats,
            "learned_regression_ckpt": str(learned_regression_ckpt) if learned_regression_ckpt else None,
            "default_lat": default_lat,
            "default_lon": default_lon,
        },
    }

    return labels_df, tiles, run_info


def _list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("**/*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def _decode_terrain_rgb(terrain_img: Image.Image) -> np.ndarray:
    arr = np.array(terrain_img).astype(np.float32)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    return -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / (np.maximum(den, 1e-6))


@dataclass(frozen=True)
class S3ImageRef:
    key: str


def _path_for_coords(path: Path | S3ImageRef) -> PurePosixPath:
    if isinstance(path, S3ImageRef):
        return PurePosixPath(path.key)
    return path


def _s3_prefix_from_local_dir(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path("data/raw").resolve())
        return f"raw/{rel.as_posix()}"
    except ValueError:
        return path.as_posix().lstrip("/")


def _make_s3_client(bucket: str | None) -> Any | None:
    if not bucket:
        return None
    import boto3

    return boto3.client("s3")


def _list_images_s3(bucket: str, prefix: str, *, s3_client: Any | None = None) -> list[S3ImageRef]:
    s3 = s3_client if s3_client is not None else _make_s3_client(bucket)
    if s3 is None:
        return []
    paginator = s3.get_paginator("list_objects_v2")
    images: list[S3ImageRef] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            suffix = Path(key).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                images.append(S3ImageRef(key=key))
    return sorted(images, key=lambda ref: ref.key)


def _open_image(path: Path | S3ImageRef, bucket: str | None, *, s3_client: Any | None = None) -> Image.Image:
    if isinstance(path, S3ImageRef):
        if not bucket:
            raise ValueError("SCENIC_S3_BUCKET is required to load S3 tiles.")

        s3 = s3_client if s3_client is not None else _make_s3_client(bucket)
        if s3 is None:
            raise ValueError("Failed to create S3 client.")
        resp = s3.get_object(Bucket=bucket, Key=path.key)
        return Image.open(BytesIO(resp["Body"].read()))
    return Image.open(path)


def _relative_paths_or_raise(
    sat_path: Path | S3ImageRef,
    terrain_path: Path | S3ImageRef,
    raw_root: Path,
    *,
    s3_bucket: str | None,
    s3_only: bool,
) -> tuple[str, str]:
    if s3_only:
        if not s3_bucket:
            raise ValueError("SCENIC_S3_BUCKET is required for S3-only mode.")
        raw_prefix = "raw/"
        sat_key = sat_path.key if isinstance(sat_path, S3ImageRef) else sat_path.as_posix()
        terr_key = (
            terrain_path.key if isinstance(terrain_path, S3ImageRef) else terrain_path.as_posix()
        )
        if not sat_key.startswith(raw_prefix) or not terr_key.startswith(raw_prefix):
            raise ValueError("S3 tile keys must live under raw/ in the bucket.")
        return sat_key[len(raw_prefix):], terr_key[len(raw_prefix):]

    raw_root = raw_root.resolve()
    try:
        sat_rel = Path(sat_path).resolve().relative_to(raw_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Satellite path {sat_path} is not under raw_dir {raw_root}"
        ) from exc
    try:
        terrain_rel = Path(terrain_path).resolve().relative_to(raw_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Terrain path {terrain_path} is not under raw_dir {raw_root}"
        ) from exc
    return sat_rel, terrain_rel


def _load_classifier(
    *,
    use_classifier: bool,
    classifier_best_ckpt: str | Path,
    classifier_use_resisc45_stats: bool,
    device: str,
    warnings: list[str],
) -> tuple[Any, Any, str, Optional[list[str]]]:
    try:
        import torch
    except ImportError:
        torch = None

    if device == "auto":
        if torch is None:
            device = "cpu"
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
    elif device == "cuda" and torch is not None and not torch.cuda.is_available():
        warnings.append("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    elif (
        device == "mps"
        and torch is not None
        and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available())
    ):
        warnings.append("MPS requested but unavailable; falling back to CPU.")
        device = "cpu"

    if not use_classifier:
        return None, None, device, None

    ckpt = Path(classifier_best_ckpt)
    if not ckpt.exists():
        warnings.append(f"Classifier checkpoint not found: {ckpt}")
        return None, None, device, None

    try:
        from torchvision import transforms as tv
        from src.classifier.model import LandscapeClassifier, TERRAIN_CLASSES, get_scenic_weight
        from src.classifier.inference import RESISC45_MEAN, RESISC45_STD, IMAGENET_MEAN, IMAGENET_STD
    except ImportError as exc:
        warnings.append(f"Classifier dependencies missing: {exc}")
        return None, None, device, None

    mean = RESISC45_MEAN if classifier_use_resisc45_stats else IMAGENET_MEAN
    std = RESISC45_STD if classifier_use_resisc45_stats else IMAGENET_STD

    classifier = LandscapeClassifier(
        pretrained=False,
        pretrained_path=ckpt,
        device=device,
    )
    classifier.to(device)
    classifier.eval()

    transform = tv.Compose(
        [
            tv.Resize((224, 224)),
            tv.ToTensor(),
            tv.Normalize(mean=mean, std=std),
        ]
    )

    class_names = list(TERRAIN_CLASSES)
    _ = get_scenic_weight  # keep for type reference
    return classifier, transform, device, class_names


def _load_learned_regressor(
    *,
    learned_regression_ckpt: str | Path | None,
    device: str,
    warnings: list[str],
) -> Any | None:
    if learned_regression_ckpt is None:
        return None

    ckpt_path = Path(learned_regression_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Learned regression checkpoint not found: {ckpt_path}")

    try:
        import torch
        from src.scenic_scorer.regression import ScenicRegressionModel
    except ImportError as exc:
        raise ImportError("Learned scoring requires scenic regression dependencies.") from exc

    ckpt = torch.load(ckpt_path, map_location=device)
    required = {"model_state_dict", "vit_dim", "terrain_dim", "num_classes"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(
            f"Invalid learned regression checkpoint '{ckpt_path}': missing keys {sorted(missing)}"
        )

    model = ScenicRegressionModel(
        vit_dim=int(ckpt["vit_dim"]),
        terrain_dim=int(ckpt["terrain_dim"]),
        num_classes=int(ckpt["num_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    warnings.append(f"Using learned regression checkpoint: {ckpt_path}")
    return model


def _build_regression_terrain_features(
    *,
    slope: np.ndarray,
    relief: float,
    water_proxy: float,
    veg_proxy: float,
) -> np.ndarray:
    slope_variation = float(min(slope.std() / 15.0, 1.0))
    terrain = TerrainFeatures(
        slope_variation=slope_variation,
        elevation_change=float(relief),
        water_proximity=float(water_proxy),
        vegetation_density=float(veg_proxy),
        coastal=False,
        has_lake=False,
        has_river=False,
    )
    return terrain.to_array().astype(np.float32)


def _infer_classifier_outputs(
    *,
    sat_img: Image.Image,
    classifier: Any,
    clf_transform: Any,
    device: str,
    class_names: Optional[list[str]],
) -> tuple[int, str, float, np.ndarray, np.ndarray]:
    if classifier is None or clf_transform is None or class_names is None:
        raise ValueError("Classifier is required for learned scoring outputs.")

    import torch
    from src.classifier.model import get_scenic_weight

    input_tensor = clf_transform(sat_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_t = classifier(input_tensor)
        features_t = classifier.get_features(input_tensor)
        probs = torch.softmax(logits_t, dim=-1).cpu().numpy()[0]

    class_id = int(np.argmax(probs))
    class_name = class_names[class_id]
    class_score = float(get_scenic_weight(class_name))

    class_logits = logits_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    vit_embedding = features_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return class_id, class_name, class_score, class_logits, vit_embedding


def _predict_learned_score(
    *,
    model: Any,
    vit_embedding: np.ndarray,
    terrain_features: np.ndarray,
    class_logits: np.ndarray,
    device: str,
) -> float:
    import torch

    vit_t = torch.from_numpy(vit_embedding).float().unsqueeze(0).to(device)
    terrain_t = torch.from_numpy(terrain_features).float().unsqueeze(0).to(device)
    logits_t = torch.from_numpy(class_logits).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(vit_t, terrain_t, logits_t).squeeze().item()
    return float(np.clip(pred, 0.0, 10.0))


def _infer_class(
    *,
    sat_img: Image.Image,
    classifier: Any,
    clf_transform: Any,
    device: str,
    class_names: Optional[list[str]],
) -> tuple[int, str, float]:
    if classifier is None or clf_transform is None or class_names is None:
        return 0, "unknown", 0.3

    import torch
    from src.classifier.model import get_scenic_weight

    input_tensor = clf_transform(sat_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(input_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    class_id = int(np.argmax(probs))
    class_name = class_names[class_id]
    class_score = float(get_scenic_weight(class_name))
    return class_id, class_name, class_score
