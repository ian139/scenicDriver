from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional
import re
import os
from io import BytesIO
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image

from scenicdriver.active_learning.scoring import TERRAIN_FEATURE_CONTRACT
from scenicdriver.active_learning.water import (
    WATER_METRIC,
    compute_satellite_water_fraction,
    weak_score_water_components,
)
from scenicdriver.data_pipeline.web_mercator import tile_pixel_ground_spacing_m
from scenicdriver.terrain.features import compute_terrain_features, compute_terrain_valid_mask


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
    max_tiles: int | None,
    use_classifier: bool,
    classifier_best_ckpt: str | Path,
    default_lat: float,
    default_lon: float,
    seed: int,
    device: str = "auto",
    learned_regression_ckpt: str | Path | None = None,
    learned_run_json: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Score paired tiles; learned identities must be attested by checkpoint metadata
    or learned_run_json. Checkpoint hashes always identify the bytes actually loaded.
    """
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
    learned_identity = (
        _learned_report_identity(
            model=learned_model,
            classifier_checkpoint=classifier_best_ckpt,
            regression_checkpoint=learned_regression_ckpt,
            learned_run_json=learned_run_json,
        )
        if learned_model is not None else {}
    )

    rows: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    processed = 0
    missing_pairs = 0
    if max_tiles is not None:
        max_tiles = max(1, int(ceil(max_tiles)))

    for sat_path, coords in zip(sat_files, sat_coords):
        if max_tiles is not None and processed >= max_tiles:
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
        if coords is None:
            raise ValueError(
                f"Tile z/x/y required for terrain pixel spacing: {_path_for_coords(sat_path)}"
            )
        if sat_img.size != terrain_img.size:
            raise ValueError(f"Satellite and terrain dimensions differ: {sat_rel}")

        # Canonical PNGs reserve black DEM for outside-land sea fill.
        valid_mask = compute_terrain_valid_mask(terrain_img)
        terrain_result = compute_terrain_features(
            terrain_img,
            sat_img,
            valid_mask=valid_mask,
            pixel_spacing_m=tile_pixel_ground_spacing_m(
                coords[1], coords[2], coords[0], *terrain_img.size
            ),
        )
        terrain_features = terrain_result.features.to_array().astype(np.float32)
        relief = terrain_result.relief
        roughness = terrain_result.roughness
        slope_mean = terrain_result.slope_mean
        slope_variation = terrain_result.features.slope_variation
        water_proxy = terrain_result.features.water_proximity
        veg_proxy = terrain_result.features.vegetation_density
        water_fraction = compute_satellite_water_fraction(sat_img, land_mask=valid_mask)

        if learned_model is not None:
            class_id, class_name, class_score, class_logits, vit_embedding = _infer_class(
                sat_img=sat_img,
                classifier=classifier,
                clf_transform=clf_transform,
                device=clf_device,
                class_names=class_names,
                with_features=True,
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
            water = weak_score_water_components(
                class_score, water_fraction=water_fraction, water_proximity=water_proxy
            )
            score = (
                water["heuristic_class_component"]
                + 2.0 * np.tanh(relief / 500.0)
                + 1.5 * np.tanh(roughness / 200.0)
                + 1.5 * np.tanh(slope_mean / 15.0)
                + water["heuristic_water_component"]
                + 1.0 * veg_proxy
                + water["heuristic_water_fraction_penalty"]
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
        **learned_identity,
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
        "water_metric": WATER_METRIC,
        "terrain_feature_contract": TERRAIN_FEATURE_CONTRACT,
        "config": {
            "satellite_dir": str(sat_dir),
            "terrain_dir": str(terrain_dir),
            "raw_dir": raw_dir_config,
            "s3_bucket": s3_bucket,
            "s3_only": s3_only,
            "max_tiles": max_tiles,
            "use_classifier": use_classifier,
            "classifier_best_ckpt": str(classifier_best_ckpt),
            "learned_regression_ckpt": str(learned_regression_ckpt) if learned_regression_ckpt else None,
            "default_lat": default_lat,
            "default_lon": default_lon,
        },
    }
    if learned_run_json is not None:
        run_info["config"]["learned_run_json"] = str(learned_run_json)

    return labels_df, tiles, run_info


def _learned_report_identity(
    *, model: Any, classifier_checkpoint: str | Path,
    regression_checkpoint: str | Path, learned_run_json: str | Path | None,
) -> dict[str, Any]:
    from scenicdriver.heuristics.report import (
        REQUIRED_LEARNED_FIELDS, _extract_report_identities,
        validate_report_union_compatibility,
    )

    identity = dict(model.report_identity)
    guidance = (
        "Supply the exact identity in the regression checkpoint metadata or "
        "pass --learned-run-json <path> to an attested run record; "
        "legacy artifacts without recoverable identity must be re-scored."
    )
    if identity.get("scoring_mode", "learned") != "learned":
        raise ValueError(f"scoring_mode in regression checkpoint metadata is not learned; {guidance}")
    if learned_run_json is not None:
        try:
            content = json.loads(Path(learned_run_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read learned identity from {learned_run_json}; {guidance}") from exc
        if not isinstance(content, dict):
            raise ValueError(f"Invalid learned identity in {learned_run_json}; {guidance}")
        run_info = content.get("run_info") or {}
        if not isinstance(run_info, dict):
            raise ValueError(f"Invalid run_info in {learned_run_json}; {guidance}")
        mode = run_info.get("scoring_mode") or content.get("scoring_mode")
        if mode is not None and mode != "learned":
            raise ValueError(f"scoring_mode in {learned_run_json} is not learned; {guidance}")
        _, supplied = _extract_report_identities(content)
        for field, value in supplied.items():
            if field in identity and identity[field] != value:
                raise ValueError(f"{field} in --learned-run-json does not match regression checkpoint metadata")
            identity[field] = value

    for field, path in (
        ("classifier_checkpoint_sha256", classifier_checkpoint),
        ("regression_checkpoint_sha256", regression_checkpoint),
    ):
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if field in identity and identity[field] != actual:
            raise ValueError(f"{field} does not match actual checkpoint bytes at {path}; {guidance}")
        identity[field] = actual
    for field in REQUIRED_LEARNED_FIELDS.values():
        if field not in identity:
            raise ValueError(f"Missing required learned identity {field!r}; {guidance}")
    identity["scoring_mode"] = "learned"
    validate_report_union_compatibility([{"run_info": identity}])
    return identity


def _list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("**/*") if p.suffix.lower() in IMAGE_EXTENSIONS])


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
    """Preserve raw-relative local paths and exact S3 keys in either storage mode."""
    if s3_only and not s3_bucket:
        raise ValueError("SCENIC_S3_BUCKET is required for S3-only mode.")

    raw_root = raw_root.resolve()

    def relative_path(path: Path | S3ImageRef, kind: str) -> str:
        if isinstance(path, S3ImageRef):
            if not s3_bucket:
                raise ValueError("SCENIC_S3_BUCKET is required to load S3 tiles.")
            key_path = PurePosixPath(path.key)
            if not path.key.startswith("raw/") or path.key.startswith("raw//") or ".." in key_path.parts:
                raise ValueError("S3 tile keys must live under raw/ in the bucket.")
            return path.key[len("raw/"):]
        try:
            return path.resolve().relative_to(raw_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"{kind} path {path} is not under raw_dir {raw_root}"
            ) from exc

    return relative_path(sat_path, "Satellite"), relative_path(terrain_path, "Terrain")


def _load_classifier(
    *,
    use_classifier: bool,
    classifier_best_ckpt: str | Path,
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
            from scenicdriver.scenic_scorer.regression import resolve_device

            device = resolve_device(device)
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
        from scenicdriver.classifier.model import LandscapeClassifier
        from scenicdriver.classifier.inference import get_inference_transform
    except ImportError as exc:
        warnings.append(f"Classifier dependencies missing: {exc}")
        return None, None, device, None

    classifier = LandscapeClassifier(
        pretrained=False,
        pretrained_path=ckpt,
        device=device,
    )
    classifier.to(device)
    classifier.eval()

    transform = get_inference_transform(classifier.preprocessing)
    class_names = list(classifier.preprocessing.class_names)
    return classifier, transform, device, class_names


def _load_learned_regressor(
    *,
    learned_regression_ckpt: str | Path | None,
    device: str,
    warnings: list[str],
) -> Any | None:
    """Restore the regression architecture declared by the checkpoint."""
    if learned_regression_ckpt is None:
        return None

    ckpt_path = Path(learned_regression_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Learned regression checkpoint not found: {ckpt_path}")

    try:
        import torch
        from scenicdriver.scenic_scorer.regression import ScenicRegressionModel
    except ImportError as exc:
        raise ImportError("Learned scoring requires scenic regression dependencies.") from exc

    numpy_safe_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(numpy_safe_globals):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    required = {"model_state_dict", "vit_dim", "terrain_dim", "num_classes"}
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(
            f"Invalid learned regression checkpoint '{ckpt_path}': missing keys {sorted(missing)}"
        )

    model_kwargs = {
        name: int(ckpt[name]) for name in ("vit_dim", "terrain_dim", "num_classes")
    }
    if "hidden_dim" in ckpt:
        model_kwargs["hidden_dim"] = int(ckpt["hidden_dim"])

    model = ScenicRegressionModel(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    from scenicdriver.heuristics.report import _extract_report_identities

    _, model.report_identity = _extract_report_identities(ckpt)
    warnings.append(f"Using learned regression checkpoint: {ckpt_path}")
    return model


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
    with_features: bool = False,
) -> tuple[int, str, float] | tuple[int, str, float, np.ndarray, np.ndarray]:
    """Return a class prediction, or class ID -1 when no classifier is available.

    With ``with_features=True`` the classifier is required and a single
    ``forward_with_features`` pass additionally yields class logits and the ViT
    embedding (both float32) appended to the returned tuple.
    """
    if classifier is None or clf_transform is None or class_names is None:
        if with_features:
            raise ValueError("Classifier is required for learned scoring outputs.")
        return -1, "unknown", 0.3

    import torch
    from scenicdriver.classifier.model import get_scenic_weight

    input_tensor = clf_transform(sat_img).unsqueeze(0).to(device)
    with torch.no_grad():
        if with_features:
            logits, features = classifier.forward_with_features(input_tensor)
        else:
            logits = classifier(input_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    class_id = int(np.argmax(probs))
    class_name = class_names[class_id]
    class_score = float(get_scenic_weight(class_name))
    if not with_features:
        return class_id, class_name, class_score

    class_logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
    vit_embedding = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return class_id, class_name, class_score, class_logits, vit_embedding
