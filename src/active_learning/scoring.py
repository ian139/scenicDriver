"""Resumable active-learning pool scoring.

The scorer keeps a scalar, selector-ready candidate table and stores dense
model features in a compressed NPZ.  It preserves every manifest row (including
missing/error rows) while setting ``selector_eligible`` and paired availability
flags only for successfully scored rows.  Production model loading follows
the repository's classifier, terrain, regression, and active-registry
contracts; dependency injection keeps focused tests free of real weights and
network access.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image

from .common import atomic_write_json, atomic_write_text, jsonable, sha256_bytes, sha256_file
from src.terrain.features import compute_terrain_features

SCORING_SCHEMA_VERSION = 1
SCHEMA_VERSION = SCORING_SCHEMA_VERSION
CANONICAL_TILE_COLUMNS = (
    "region", "z", "x", "y", "lat", "lon", "satellite_path", "terrain_path",
    "satellite_present", "terrain_present",
)
OPTIONAL_TILE_COLUMNS = ("satellite_s3_uri", "terrain_s3_uri")

# Scalar-only columns.  Vectors are never expanded into CSV or JSON.
CANDIDATE_POOL_COLUMNS = (
    "image_path", "tile_identity", "source_identity", "region", "z", "x", "y", "lat", "lon",
    "satellite_path", "terrain_path", "satellite_s3_uri", "terrain_s3_uri",
    "manifest_satellite_present", "manifest_terrain_present", "satellite_present", "terrain_present",
    "availability_state", "score_status", "selector_eligible", "error",
    "satellite_content_sha256", "terrain_content_sha256",
    "heuristic_score", "scenic_score", "scenic_score_heuristic",
    "heuristic_class_component", "heuristic_relief_component", "heuristic_roughness_component",
    "heuristic_slope_component", "heuristic_water_component", "heuristic_vegetation_component",
    "heuristic_water_fraction_penalty", "relief", "roughness", "slope_mean", "slope_variation",
    "water_proximity", "vegetation_density", "water_fraction",
    "regression_prediction", "model_prediction", "label_source",
    "classifier_checkpoint_sha256", "regression_checkpoint_sha256",
    "class_id", "class_name", "class_probability", "class_count",
    "normalized_class_entropy", "class_uncertainty", "uncertainty",
    "embedding_identity", "embedding_dimension", "embedding_row_index", "embedding_cluster_id", "cluster_id",
    "cache_hit",
)

UNCERTAINTY_NAME = "normalized_class_entropy"
UNCERTAINTY_DEFINITION = (
    "class uncertainty = -sum(p_i * ln(p_i)) / ln(K), where p is the softmax "
    "distribution over K RESISC45 classes; zero terms are zero and K <= 1 is zero"
)


@dataclass
class ScoringDependencies:
    """Injected models/callables for tests, or loaded production objects.

    ``classifier_predictor`` returns ``(logits, embeddings)`` or a mapping with
    those keys for one batch.  ``regression_predictor`` accepts float32 NumPy
    arrays ``(embeddings, terrain_features, class_logits)`` and returns one
    prediction per row.
    """

    classifier: Any | None = None
    classifier_transform: Callable[[Image.Image], Any] | None = None
    regression_model: Any | None = None
    class_names: Sequence[str] | None = None
    classifier_predictor: Callable[..., Any] | None = None
    regression_predictor: Callable[..., Any] | None = None
    terrain_feature_fn: Callable[..., Any] = compute_terrain_features
    image_loader: Callable[..., Image.Image] | None = None
    s3_client: Any | None = None
    device: str = "cpu"
    classifier_hash: str | None = None
    regression_hash: str | None = None
    classifier_checkpoint: Path | None = None
    regression_checkpoint: Path | None = None
    classifier_preprocess_identity: str | None = None


@dataclass(frozen=True)
class LoadedScoringModels:
    classifier: Any
    classifier_transform: Callable[[Image.Image], Any]
    regression_model: Any
    class_names: tuple[str, ...]
    device: str
    classifier_checkpoint: Path
    regression_checkpoint: Path
    classifier_hash: str
    regression_hash: str


@dataclass
class _SourceData:
    value: str
    uri: str
    content_hash: str | None = None
    payload: bytes | None = None
    resolved_path: Path | None = None


@dataclass
class _PendingRow:
    index: int
    satellite: Image.Image
    terrain_features: np.ndarray
    terrain_metrics: dict[str, float]
    satellite_hash: str
    terrain_hash: str


@dataclass
class _CacheData:
    rows: dict[str, dict[str, Any]]
    arrays: dict[str, np.ndarray]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return _clean_text(value).lower() in {
        "1", "true", "yes", "y", "t", "on", "available", "complete", "completed"
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stable_row_text(frame: pd.DataFrame) -> bytes:
    copy = frame.copy().reindex(sorted(str(c) for c in frame.columns), axis=1)
    return copy.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _tile_identity(row: Mapping[str, Any]) -> str:
    region = _clean_text(row.get("region")) or "unknown"
    parts = []
    for name in ("z", "x", "y"):
        number = _finite(row.get(name))
        parts.append(str(int(number)) if number is not None else "unknown")
    return f"{region}/z{parts[0]}/x{parts[1]}/y{parts[2]}"


def _source_identity(row: Mapping[str, Any]) -> str:
    return (
        f"{_tile_identity(row)}|satellite={_clean_text(row.get('satellite_path'))}"
        f"|terrain={_clean_text(row.get('terrain_path'))}"
        f"|satellite_s3={_clean_text(row.get('satellite_s3_uri'))}"
        f"|terrain_s3={_clean_text(row.get('terrain_s3_uri'))}"
    )


def _resolve_local_path(value: str, *, manifest_dir: Path, run_root: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            manifest_dir / raw,
            run_root / raw,
            run_root.parent / raw,
            Path.cwd() / raw,
        ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            pass
    return None


def _s3_parts(uri: str) -> tuple[str, str] | None:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        return None
    return parsed.netloc, parsed.path.lstrip("/")


def _default_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise ValueError("boto3 is required for S3 tile sources") from exc
    try:
        return boto3.client("s3")
    except Exception as exc:
        raise ValueError("unable to create an S3 client for tile sources") from exc


def _read_source(
    *, value: str, uri: str, manifest_dir: Path, run_root: Path,
    s3_client: Any | None, style: str,
) -> _SourceData:
    source = _SourceData(value=value, uri=uri)
    resolved = _resolve_local_path(value, manifest_dir=manifest_dir, run_root=run_root)
    if resolved is not None:
        source.resolved_path = resolved
        source.payload = resolved.read_bytes()
    else:
        s3_ref = _s3_parts(uri) if uri else None
        if s3_ref:
            if s3_client is None:
                raise ValueError(f"{style} source requires an injected or configured S3 client")
            bucket, key = s3_ref
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content_type = _clean_text(response.get("ContentType")).lower()
            if content_type and content_type not in {
                "image/png",
                "image/x-png",
                "application/octet-stream",
            }:
                raise ValueError(
                    f"{style} S3 object has unexpected content type: {content_type}"
                )
            body = response.get("Body")
            payload = body.read() if hasattr(body, "read") else body
            if not isinstance(payload, (bytes, bytearray)):
                raise ValueError(f"{style} S3 object did not return bytes")
            source.payload = bytes(payload)
        else:
            raise FileNotFoundError(f"{style} image not found: {value or uri or '<empty>'}")
    source.content_hash = hashlib.sha256(source.payload).hexdigest()
    return source


def _open_rgb(
    source: _SourceData,
    *,
    image_loader: Callable[..., Image.Image] | None = None,
    style: str = "image",
) -> Image.Image:
    if image_loader is not None:
        try:
            image = image_loader(source.value or source.uri, style)
        except TypeError:
            image = image_loader(source.value or source.uri)
        if not isinstance(image, Image.Image):
            raise TypeError(f"injected {style} image loader must return PIL.Image.Image")
        return image.convert("RGB")
    if source.payload is None:
        raise ValueError("image source has no bytes")
    with Image.open(io.BytesIO(source.payload)) as image:
        return image.convert("RGB").copy()


def _torch_module(value: Any) -> bool:
    if value is None:
        return False
    module = type(value).__module__
    return module.startswith("torch") or (hasattr(value, "state_dict") and hasattr(value, "parameters"))


def _to_numpy(value: Any, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _stack_transforms(values: list[Any], *, use_torch: bool) -> Any:
    if not values:
        raise ValueError("cannot stack an empty transformed batch")
    if use_torch:
        import torch
        tensors = [value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value), dtype=torch.float32) for value in values]
        return torch.stack(tensors, dim=0)
    return np.stack([_to_numpy(value) for value in values], axis=0).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(values, -745.0, 80.0))
    denominator = exponent.sum(axis=1, keepdims=True)
    probabilities = np.divide(exponent, denominator, out=np.full_like(exponent, 1.0 / max(1, exponent.shape[1])), where=denominator > 0)
    return probabilities.astype(np.float32)


def normalized_class_entropy(probabilities: Sequence[float] | np.ndarray) -> float:
    """Return normalized class entropy in [0, 1] (not scenic confidence)."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return 0.0
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    values /= total
    positive = values[values > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    normalizer = math.log(float(values.size))
    return float(np.clip(entropy / normalizer if normalizer > 0 else 0.0, 0.0, 1.0))


def _terrain_values(result: Any) -> tuple[np.ndarray, dict[str, float]]:
    features_obj = result.get("features") if isinstance(result, Mapping) else getattr(result, "features", None)
    if features_obj is None:
        raise ValueError("terrain feature result has no features")
    terrain_array = np.asarray(features_obj.to_array() if hasattr(features_obj, "to_array") else features_obj, dtype=np.float32).reshape(-1)
    if terrain_array.size == 0 or not np.isfinite(terrain_array).all():
        raise ValueError("terrain feature result contains no finite features")

    def metric(name: str) -> float:
        raw = result.get(name, 0.0) if isinstance(result, Mapping) else getattr(result, name, 0.0)
        number = _finite(raw)
        return float(number if number is not None else 0.0)

    metrics = {name: metric(name) for name in ("relief", "roughness", "slope_mean")}
    for name in ("slope_variation", "water_proximity", "vegetation_density"):
        raw = features_obj.get(name, 0.0) if isinstance(features_obj, Mapping) else getattr(features_obj, name, 0.0)
        number = _finite(raw)
        metrics[name] = float(number if number is not None else 0.0)
    return terrain_array, metrics


def _satellite_metrics(image: Image.Image) -> dict[str, float]:
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    brightness = (r + g + b) / 3.0
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    saturation = (maxc - minc) / np.maximum(maxc + 1e-6, 1e-6)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    texture = float(gray.std())
    water_mask = ((b > r * 1.2) & (b > g * 1.15) & (brightness < 0.65) & (saturation > 0.18) & (texture < 0.12))
    return {"water_fraction": float(water_mask.mean())}


def _heuristic_components(class_score: float, terrain: Mapping[str, float], sat: Mapping[str, float]) -> dict[str, float]:
    water_fraction = float(sat.get("water_fraction", 0.0))
    class_weight = 2.5 * max(0.3, 1.0 - 0.9 * water_fraction)
    components = {
        "heuristic_class_component": class_weight * float(class_score),
        "heuristic_relief_component": 2.0 * math.tanh(float(terrain.get("relief", 0.0)) / 500.0),
        "heuristic_roughness_component": 1.5 * math.tanh(float(terrain.get("roughness", 0.0)) / 200.0),
        "heuristic_slope_component": 1.5 * math.tanh(float(terrain.get("slope_mean", 0.0)) / 15.0),
        "heuristic_water_component": 1.5 * float(terrain.get("water_proximity", 0.0)),
        "heuristic_vegetation_component": float(terrain.get("vegetation_density", 0.0)),
        "heuristic_water_fraction_penalty": -1.2 * water_fraction,
    }
    components["heuristic_score"] = float(np.clip(sum(components.values()), 0.0, 10.0))
    components["scenic_score"] = components["heuristic_score"]
    components["scenic_score_heuristic"] = components["heuristic_score"]
    return components


def _model_identity(label: str, value: Any, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    if value is None:
        return f"{label}:none"
    if isinstance(value, (str, Path)) and Path(value).exists():
        return sha256_file(value)
    cls = type(value)
    return sha256_bytes(f"{label}:{cls.__module__}.{cls.__qualname__}".encode("utf-8"))


def _device_name(device: str) -> str:
    requested = str(device or "auto").lower()
    try:
        import torch
    except ImportError:
        return "cpu" if requested == "auto" else requested
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        return "cpu"
    return requested


def _torch_load(path: Path, *, map_location: str) -> Any:
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_active_regression_checkpoint(registry_path: str | Path = "data/processed/regression/model_registry.json") -> Path:
    """Resolve active registry checkpoint and fail closed on malformed data."""
    registry_file = Path(registry_path)
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"active regression registry not found: {registry_file}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed active regression registry: {registry_file}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("active"), Mapping):
        raise ValueError("malformed active regression registry: active record is missing")
    raw_checkpoint = payload["active"].get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
        raise ValueError("malformed active regression registry: active checkpoint is missing")
    checkpoint_value = Path(raw_checkpoint).expanduser()
    candidates = [checkpoint_value]
    if not checkpoint_value.is_absolute():
        candidates += [registry_file.parent / checkpoint_value, Path.cwd() / checkpoint_value]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"active regression checkpoint not found: {raw_checkpoint}")


def _validate_regression_checkpoint(path: Path, *, device: str) -> tuple[Any, dict[str, int]]:
    payload = _torch_load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"malformed regression checkpoint: {path}")
    required = {"model_state_dict", "vit_dim", "terrain_dim", "num_classes"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"malformed regression checkpoint {path}: missing {missing}")
    try:
        dims = {name: int(payload[name]) for name in ("vit_dim", "terrain_dim", "num_classes")}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed regression checkpoint {path}: invalid dimensions") from exc
    if any(value <= 0 for value in dims.values()) or not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError(f"malformed regression checkpoint {path}: invalid state/dimensions")
    from src.scenic_scorer.regression import ScenicRegressionModel
    model = ScenicRegressionModel(**dims).to(device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed regression checkpoint {path}: state dict mismatch") from exc
    model.eval()
    return model, dims


def load_scoring_models(
    *, registry_path: str | Path = "data/processed/regression/model_registry.json",
    classifier_checkpoint: str | Path = "models/classifier/best_model.pt",
    device: str = "auto", classifier_use_resisc45_stats: bool = True,
) -> LoadedScoringModels:
    """Load classifier and active regression exactly once, read-only."""
    resolved_device = _device_name(device)
    regression_path = resolve_active_regression_checkpoint(registry_path)
    classifier_path = Path(classifier_checkpoint)
    if not classifier_path.exists() or not classifier_path.is_file():
        raise FileNotFoundError(f"classifier checkpoint not found: {classifier_path}")
    try:
        payload = _torch_load(classifier_path, map_location=resolved_device)
    except Exception as exc:
        raise ValueError(f"malformed classifier checkpoint: {classifier_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"malformed classifier checkpoint: {classifier_path}")
    if any(key in payload for key in ("model_state_dict", "state_dict")):
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if not isinstance(state, Mapping):
            raise ValueError(f"malformed classifier checkpoint: {classifier_path}")
    elif not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"malformed classifier checkpoint: {classifier_path}")
    try:
        from src.classifier.inference import get_inference_transform
        from src.classifier.model import LandscapeClassifier, TERRAIN_CLASSES
        classifier = LandscapeClassifier(pretrained=False, pretrained_path=classifier_path, device=resolved_device)
        classifier.to(resolved_device)
        classifier.eval()
    except ImportError as exc:
        raise ImportError("classifier dependencies are required for active-learning scoring") from exc
    except Exception as exc:
        raise ValueError(f"malformed classifier checkpoint: {classifier_path}") from exc
    regression, _dims = _validate_regression_checkpoint(regression_path, device=resolved_device)
    return LoadedScoringModels(
        classifier=classifier, classifier_transform=get_inference_transform(use_resisc45_stats=classifier_use_resisc45_stats),
        regression_model=regression, class_names=tuple(TERRAIN_CLASSES), device=resolved_device,
        classifier_checkpoint=classifier_path, regression_checkpoint=regression_path,
        classifier_hash=sha256_file(classifier_path), regression_hash=sha256_file(regression_path),
    )


def _dependencies_from_loaded(models: LoadedScoringModels) -> ScoringDependencies:
    return ScoringDependencies(
        classifier=models.classifier, classifier_transform=models.classifier_transform,
        regression_model=models.regression_model, class_names=models.class_names, device=models.device,
        classifier_hash=models.classifier_hash, regression_hash=models.regression_hash,
        classifier_checkpoint=models.classifier_checkpoint, regression_checkpoint=models.regression_checkpoint,
    )


def _read_cache(run_root: Path) -> _CacheData | None:
    candidate_path, embedding_path = run_root / "candidate_pool.csv", run_root / "feature_embeddings.npz"
    if not candidate_path.exists() or not embedding_path.exists():
        return None
    try:
        frame = pd.read_csv(candidate_path, low_memory=False)
        if frame.empty or "source_identity" not in frame or "score_status" not in frame:
            return None
        with np.load(embedding_path, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
        if arrays.get("embeddings") is None or arrays["embeddings"].ndim != 2:
            return None
        rows: dict[str, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            key = _clean_text(row.get("source_identity"))
            if key and key not in rows:
                rows[key] = row.to_dict()
        return _CacheData(rows=rows, arrays=arrays)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _cache_match(row: Mapping[str, Any], cached: Mapping[str, Any], *, classifier_hash: str, regression_hash: str, arrays: Mapping[str, np.ndarray]) -> int | None:
    if _clean_text(cached.get("score_status")) != "scored":
        return None
    if _clean_text(cached.get("satellite_content_sha256")) != _clean_text(row.get("satellite_content_sha256")) or _clean_text(cached.get("terrain_content_sha256")) != _clean_text(row.get("terrain_content_sha256")):
        return None
    if _clean_text(cached.get("classifier_checkpoint_sha256")) != classifier_hash or _clean_text(cached.get("regression_checkpoint_sha256")) != regression_hash:
        return None
    raw_index = _finite(cached.get("embedding_row_index"))
    if raw_index is None or int(raw_index) < 0:
        return None
    index = int(raw_index)
    embeddings = arrays.get("embeddings")
    if embeddings is None or embeddings.ndim != 2 or index >= len(embeddings):
        return None
    for name in ("embeddings", "class_logits", "class_probs", "terrain_features"):
        array = arrays.get(name)
        if (
            array is None
            or array.ndim != 2
            or len(array) != len(embeddings)
            or index >= len(array)
            or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array[index]).all()
        ):
            return None
    return index


def _copy_cached_fields(result: dict[str, Any], cached: Mapping[str, Any]) -> None:
    identity_fields = {"image_path", "tile_identity", "source_identity", "region", "z", "x", "y", "lat", "lon", "satellite_path", "terrain_path", "satellite_s3_uri", "terrain_s3_uri", "manifest_satellite_present", "manifest_terrain_present", "satellite_content_sha256", "terrain_content_sha256", "cache_hit", "embedding_row_index", "embedding_cluster_id", "cluster_id"}
    for key in CANDIDATE_POOL_COLUMNS:
        if key not in identity_fields and key in cached:
            result[key] = _json_scalar(cached[key])


def _call_classifier(dependencies: ScoringDependencies, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    transformed = [dependencies.classifier_transform(image) if dependencies.classifier_transform else np.asarray(image, dtype=np.float32) for image in images]
    if dependencies.classifier_predictor is not None:
        try:
            output = dependencies.classifier_predictor(_stack_transforms(transformed, use_torch=False))
        except (TypeError, AttributeError):
            output = dependencies.classifier_predictor(images)
    else:
        if dependencies.classifier is None:
            raise ValueError("classifier is required for active-model scoring")
        use_torch = _torch_module(dependencies.classifier)
        batch = _stack_transforms(transformed, use_torch=use_torch)
        if use_torch:
            batch = batch.to(dependencies.device)
            import torch
            with torch.no_grad():
                logits_value = dependencies.classifier(batch)
                features_value = dependencies.classifier.get_features(batch)
        else:
            logits_value = dependencies.classifier(batch)
            if not hasattr(dependencies.classifier, "get_features"):
                raise ValueError("classifier must expose get_features for embeddings")
            features_value = dependencies.classifier.get_features(batch)
        output = {"logits": logits_value, "embeddings": features_value}
    if isinstance(output, Mapping):
        logits_value = output.get("logits", output.get("class_logits"))
        features_value = output.get("embeddings", output.get("features", output.get("vit_embeddings")))
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        logits_value, features_value = output[0], output[1]
    else:
        raise ValueError("classifier predictor must return logits and embeddings")
    if logits_value is None or features_value is None:
        raise ValueError("classifier predictor returned incomplete outputs")
    logits, embeddings = _to_numpy(logits_value), _to_numpy(features_value)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if logits.ndim != 2 or embeddings.ndim != 2 or len(logits) != len(images) or len(embeddings) != len(images) or logits.shape[1] <= 0 or embeddings.shape[1] <= 0:
        raise ValueError("classifier outputs do not match batch shape")
    return logits.astype(np.float32), embeddings.astype(np.float32)


def _call_regression(dependencies: ScoringDependencies, embeddings: np.ndarray, terrain_features: np.ndarray, logits: np.ndarray) -> np.ndarray:
    if dependencies.regression_predictor is not None:
        output = dependencies.regression_predictor(embeddings, terrain_features, logits)
    else:
        if dependencies.regression_model is None:
            raise ValueError("active regression model is required for scoring")
        if _torch_module(dependencies.regression_model):
            import torch
            with torch.no_grad():
                output = dependencies.regression_model(torch.from_numpy(embeddings).float().to(dependencies.device), torch.from_numpy(terrain_features).float().to(dependencies.device), torch.from_numpy(logits).float().to(dependencies.device))
        else:
            output = dependencies.regression_model(embeddings, terrain_features, logits)
    values = _to_numpy(output).reshape(-1)
    if len(values) != len(embeddings) or not np.isfinite(values).all():
        raise ValueError("regression predictions do not match batch or contain non-finite values")
    return np.clip(values.astype(np.float32), 0.0, 10.0)


def _lsh_clusters(embeddings: np.ndarray, *, seed: int, bits: int) -> list[str]:
    if embeddings.ndim != 2 or not len(embeddings):
        return []
    if bits <= 0 or bits > 63:
        raise ValueError("LSH bits must be between 1 and 63")
    rng = np.random.default_rng(int(seed))
    hyperplanes = rng.standard_normal((embeddings.shape[1], int(bits))).astype(np.float32)
    signs = (embeddings @ hyperplanes) >= 0.0
    values = np.zeros(len(embeddings), dtype=np.uint64)
    for bit in range(int(bits)):
        values |= signs[:, bit].astype(np.uint64) << np.uint64(bit)
    width = max(1, math.ceil(int(bits) / 4))
    return [f"lsh-{value:0{width}x}" for value in values]


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def _base_result(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: None for key in CANDIDATE_POOL_COLUMNS}
    for name in CANONICAL_TILE_COLUMNS + OPTIONAL_TILE_COLUMNS:
        result[name] = _json_scalar(row.get(name))
    satellite_path = _clean_text(row.get("satellite_path"))
    image_path = _clean_text(row.get("image_path")) or satellite_path
    result.update({
        "image_path": image_path or _clean_text(row.get("satellite_s3_uri")),
        "tile_identity": _tile_identity(row),
        "source_identity": _source_identity(row),
        "region": _clean_text(row.get("region")) or "unknown",
        "manifest_satellite_present": _bool(row.get("satellite_present"))
        or _bool(row.get("satellite_s3_present")),
        "manifest_terrain_present": _bool(row.get("terrain_present"))
        or _bool(row.get("terrain_s3_present")),
        "satellite_present": False, "terrain_present": False,
        "availability_state": "missing", "score_status": "missing", "selector_eligible": False,
        "error": None, "embedding_row_index": -1, "cache_hit": False,
    })
    return result


def _mark_error(result: dict[str, Any], message: str, *, missing: bool = False) -> None:
    result.update({"availability_state": "missing" if missing else "error", "score_status": "missing" if missing else "error", "selector_eligible": False, "satellite_present": False, "terrain_present": False, "error": str(message)})


def score_tile_manifest(
    manifest: str | Path | pd.DataFrame,
    *, run_root: str | Path,
    dependencies: ScoringDependencies | None = None,
    registry_path: str | Path = "data/processed/regression/model_registry.json",
    classifier_checkpoint: str | Path = "models/classifier/best_model.pt",
    device: str = "auto", classifier_use_resisc45_stats: bool = True,
    batch_size: int = 32, lsh_seed: int = 0, lsh_bits: int = 16,
    max_rows: int | None = None, write: bool = True,
) -> dict[str, Any]:
    """Score manifest rows and atomically publish candidate, NPZ, and manifest.

    Cache reuse is keyed by source content SHA-256 plus both model identities;
    ``max_rows`` is a deterministic prefix in the input order.  Models are
    loaded once and classifier inference is batched by ``batch_size``.
    """
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(manifest, pd.DataFrame):
        frame = manifest.copy()
        source_hash = sha256_bytes(_stable_row_text(frame))
        manifest_label, manifest_dir = "<dataframe>", Path.cwd()
    else:
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"tile manifest not found: {manifest_path}")
        try:
            frame = pd.read_csv(manifest_path, low_memory=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid tile manifest: {manifest_path}") from exc
        source_hash, manifest_label, manifest_dir = sha256_file(manifest_path), str(manifest_path), manifest_path.parent
    missing_columns = sorted(set(CANONICAL_TILE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"tile manifest missing columns: {missing_columns}")
    if frame.empty:
        raise ValueError("tile manifest is empty")
    source_rows = int(len(frame))
    if max_rows is not None:
        frame = frame.iloc[: max(0, int(max_rows))].copy()
        if frame.empty:
            raise ValueError("tile manifest has no rows after max_rows")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if dependencies is None:
        dependencies = _dependencies_from_loaded(load_scoring_models(registry_path=registry_path, classifier_checkpoint=classifier_checkpoint, device=device, classifier_use_resisc45_stats=classifier_use_resisc45_stats))
    else:
        dependencies = ScoringDependencies(**dependencies.__dict__)
        dependencies.device = _device_name(device if device != "auto" else dependencies.device)
    if dependencies.classifier_predictor is not None and not dependencies.classifier_hash:
        raise ValueError("injected classifier_predictor requires classifier_hash")
    if dependencies.regression_predictor is not None and not dependencies.regression_hash:
        raise ValueError("injected regression_predictor requires regression_hash")
    if dependencies.s3_client is None and any(
        _s3_parts(_clean_text(value))
        for name in OPTIONAL_TILE_COLUMNS
        if name in frame.columns
        for value in frame[name].tolist()
    ):
        dependencies.s3_client = _default_s3_client()
    classifier_hash = _model_identity("classifier", dependencies.classifier, dependencies.classifier_hash)
    regression_hash = _model_identity("regression", dependencies.regression_model, dependencies.regression_hash)
    cache = _read_cache(root)
    results = [_base_result(row) for _, row in frame.iterrows()]
    cache_hits, pending = 0, []
    vector_by_index, logits_by_index, probs_by_index, terrain_by_index = {}, {}, {}, {}
    errors: list[dict[str, Any]] = []
    def score_chunk(chunk: list[_PendingRow]) -> None:
        try:
            logits, embeddings = _call_classifier(
                dependencies, [item.satellite for item in chunk]
            )
            terrain_features = np.stack(
                [item.terrain_features for item in chunk], axis=0
            ).astype(np.float32)
            predictions = _call_regression(
                dependencies, embeddings, terrain_features, logits
            )
            probabilities = _softmax(logits)
            class_names = tuple(dependencies.class_names or ())
            if class_names and len(class_names) != logits.shape[1]:
                class_names = tuple(class_names[: logits.shape[1]])
            for offset, item in enumerate(chunk):
                result = results[item.index]
                row_probs = probabilities[offset]
                class_id = int(np.argmax(row_probs))
                class_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"class_{class_id}"
                )
                try:
                    from src.classifier.model import get_scenic_weight

                    class_score = float(get_scenic_weight(class_name))
                except Exception:
                    class_score = 0.3
                sat_metrics = _satellite_metrics(item.satellite)
                components, entropy = (
                    _heuristic_components(
                        class_score, item.terrain_metrics, sat_metrics
                    ),
                    normalized_class_entropy(row_probs),
                )
                result.update(
                    {
                        **components,
                        **item.terrain_metrics,
                        "water_fraction": sat_metrics["water_fraction"],
                        "regression_prediction": float(predictions[offset]),
                        "model_prediction": float(predictions[offset]),
                        "label_source": "active_regression_prediction",
                        "classifier_checkpoint_sha256": classifier_hash,
                        "regression_checkpoint_sha256": regression_hash,
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_probability": float(row_probs[class_id]),
                        "class_count": int(len(row_probs)),
                        "normalized_class_entropy": entropy,
                        "class_uncertainty": entropy,
                        "uncertainty": entropy,
                        "satellite_present": True,
                        "terrain_present": True,
                        "availability_state": "available",
                        "score_status": "scored",
                        "selector_eligible": True,
                        "error": None,
                    }
                )
                vector_by_index[item.index] = embeddings[offset].astype(
                    np.float32, copy=True
                )
                logits_by_index[item.index] = logits[offset].astype(
                    np.float32, copy=True
                )
                probs_by_index[item.index] = row_probs.astype(
                    np.float32, copy=True
                )
                terrain_by_index[item.index] = item.terrain_features.astype(
                    np.float32, copy=True
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for item in chunk:
                _mark_error(results[item.index], message)
                errors.append(
                    {
                        "source_identity": results[item.index]["source_identity"],
                        "error": message,
                        "status": "error",
                    }
                )


    for index, (_, source_row) in enumerate(frame.iterrows()):
        row, result = source_row.to_dict(), results[index]
        if not result["manifest_satellite_present"] or not result["manifest_terrain_present"]:
            missing_style = "satellite" if not result["manifest_satellite_present"] else "terrain"
            _mark_error(result, f"manifest marks {missing_style} imagery unavailable", missing=True)
            errors.append({"source_identity": result["source_identity"], "error": result["error"], "status": "missing"})
            continue
        try:
            satellite = _read_source(value=_clean_text(row.get("satellite_path")), uri=_clean_text(row.get("satellite_s3_uri")), manifest_dir=manifest_dir, run_root=root, s3_client=dependencies.s3_client, style="satellite")
            terrain = _read_source(value=_clean_text(row.get("terrain_path")), uri=_clean_text(row.get("terrain_s3_uri")), manifest_dir=manifest_dir, run_root=root, s3_client=dependencies.s3_client, style="terrain")
            assert satellite.content_hash is not None and terrain.content_hash is not None
            result["satellite_content_sha256"] = satellite.content_hash
            result["terrain_content_sha256"] = terrain.content_hash
            cached_row = cache.rows.get(result["source_identity"]) if cache else None
            cached_index = _cache_match(result, cached_row, classifier_hash=classifier_hash, regression_hash=regression_hash, arrays=cache.arrays) if cache and cached_row else None
            if cached_index is not None:
                _copy_cached_fields(result, cached_row)
                result.update({"satellite_present": True, "terrain_present": True, "availability_state": "available", "score_status": "scored", "selector_eligible": True, "cache_hit": True})
                cache_hits += 1
                vector_by_index[index] = np.asarray(cache.arrays["embeddings"][cached_index], dtype=np.float32).copy()
                for name, target in (
                    ("class_logits", logits_by_index),
                    ("class_probs", probs_by_index),
                    ("terrain_features", terrain_by_index),
                ):
                    target[index] = np.asarray(
                        cache.arrays[name][cached_index], dtype=np.float32
                    ).copy()
                continue
            sat_img = _open_rgb(
                satellite,
                image_loader=dependencies.image_loader,
                style="satellite",
            )
            terrain_img = _open_rgb(
                terrain,
                image_loader=dependencies.image_loader,
                style="terrain",
            )
            if sat_img.size != terrain_img.size or min(*sat_img.size) <= 0:
                raise ValueError(
                    "satellite and terrain imagery must have matching positive dimensions"
                )
            terrain_result = dependencies.terrain_feature_fn(terrain_img, sat_img)
            terrain_features, terrain_metrics = _terrain_values(terrain_result)
            pending.append(_PendingRow(index=index, satellite=sat_img, terrain_features=terrain_features, terrain_metrics=terrain_metrics, satellite_hash=satellite.content_hash, terrain_hash=terrain.content_hash))
            if len(pending) >= batch_size:
                score_chunk(pending)
                pending.clear()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            _mark_error(result, message, missing=isinstance(exc, FileNotFoundError))
            errors.append({"source_identity": result["source_identity"], "error": message, "status": result["score_status"]})

    if pending:
        score_chunk(pending)
        pending.clear()

    scored_indices = [i for i, result in enumerate(results) if result.get("score_status") == "scored" and i in vector_by_index]
    embedding_dimension = 0
    if scored_indices:
        dimensions = {int(vector_by_index[i].shape[-1]) for i in scored_indices}
        if len(dimensions) != 1:
            message = "embedding dimensions differ across scored rows"
            for i in scored_indices:
                _mark_error(results[i], message)
                errors.append({"source_identity": results[i]["source_identity"], "error": message, "status": "error"})
            scored_indices = []
        else:
            embedding_dimension = next(iter(dimensions))
    embedding_matrix = np.stack([vector_by_index[i] for i in scored_indices], axis=0).astype(np.float32) if scored_indices else np.empty((0, 0), dtype=np.float32)
    clusters = _lsh_clusters(embedding_matrix, seed=lsh_seed, bits=lsh_bits) if scored_indices else []
    embedding_identity = f"classifier-vit:{classifier_hash}:float32:{embedding_dimension}" if embedding_dimension else None
    row_to_embedding = {i: offset for offset, i in enumerate(scored_indices)}
    for i, result in enumerate(results):
        if i in row_to_embedding:
            row_index = row_to_embedding[i]
            result.update({"embedding_row_index": row_index, "embedding_dimension": embedding_dimension, "embedding_identity": embedding_identity, "embedding_cluster_id": clusters[row_index], "cluster_id": clusters[row_index]})
        elif result.get("score_status") != "scored":
            result.update({"embedding_row_index": -1, "embedding_dimension": None, "embedding_identity": None, "embedding_cluster_id": None, "cluster_id": None})

    candidate_frame = pd.DataFrame(results).reindex(columns=CANDIDATE_POOL_COLUMNS)
    for column in CANDIDATE_POOL_COLUMNS:
        candidate_frame[column] = candidate_frame[column].map(_json_scalar)
    for column in ("selector_eligible", "satellite_present", "terrain_present", "cache_hit"):
        candidate_frame[column] = candidate_frame[column].map(_bool)
    arrays = {
        "embeddings": embedding_matrix,
        "class_logits": np.stack([logits_by_index[i] for i in scored_indices], axis=0).astype(np.float32) if scored_indices else np.empty((0, 0), dtype=np.float32),
        "class_probs": np.stack([probs_by_index[i] for i in scored_indices], axis=0).astype(np.float32) if scored_indices else np.empty((0, 0), dtype=np.float32),
        "terrain_features": np.stack([terrain_by_index[i] for i in scored_indices], axis=0).astype(np.float32) if scored_indices else np.empty((0, 0), dtype=np.float32),
        "row_indices": np.asarray(scored_indices, dtype=np.int64),
    }
    candidate_path, embeddings_path, scoring_path = root / "candidate_pool.csv", root / "feature_embeddings.npz", root / "scoring_manifest.json"
    artifact_hashes = {}
    if write:
        atomic_write_text(candidate_path, candidate_frame.to_csv(index=False, lineterminator="\n"))
        _atomic_write_npz(embeddings_path, arrays)
        artifact_hashes = {"candidate_pool.csv": sha256_file(candidate_path), "feature_embeddings.npz": sha256_file(embeddings_path)}
    scored_count = len(scored_indices)
    missing_count = sum(result.get("score_status") == "missing" for result in results)
    error_count = sum(result.get("score_status") == "error" for result in results)
    payload: dict[str, Any] = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "source": {"tile_manifest": {"path": manifest_label, "sha256": source_hash, "rows": source_rows, "selected_rows": int(len(frame))}, "identity_definition": "region/z/x/y plus satellite_path, terrain_path, and optional S3 URIs", "source_content_definition": "SHA-256 of exact source bytes before PIL conversion"},
        "models": {"classifier_checkpoint": str(dependencies.classifier_checkpoint) if dependencies.classifier_checkpoint else None, "classifier_checkpoint_sha256": classifier_hash, "regression_checkpoint": str(dependencies.regression_checkpoint) if dependencies.regression_checkpoint else None, "regression_checkpoint_sha256": regression_hash, "label_semantics": "regression_prediction/model_prediction are active-model outputs; no human label is inferred"},
        "preprocessing": {"classifier_transform": dependencies.classifier_preprocess_identity or (f"{type(dependencies.classifier_transform).__module__}.{type(dependencies.classifier_transform).__qualname__}" if dependencies.classifier_transform else "injected_or_identity"), "classifier_normalization": "RESISC45" if classifier_use_resisc45_stats else "ImageNet", "classifier_input": "RGB; project inference transform targets 224x224", "terrain_features": "src.terrain.features.compute_terrain_features", "device": dependencies.device, "batch_size": int(batch_size)},
        "uncertainty": {"name": UNCERTAINTY_NAME, "definition": UNCERTAINTY_DEFINITION, "range": [0.0, 1.0], "class_count": 45, "selector_column": "uncertainty"},
        "embedding": {"identity": embedding_identity, "array_key": "embeddings", "dtype": "float32", "dimension": int(embedding_dimension), "row_index_column": "embedding_row_index", "row_indices_array_key": "row_indices", "cluster_column": "embedding_cluster_id", "lsh": {"method": "sign random-hyperplane locality-sensitive hash", "seed": int(lsh_seed), "bits": int(lsh_bits), "definition": "hex bit signature of embedding dot products against standard-normal fixed-seed hyperplanes"}},
        "counts": {"manifest_rows": int(len(frame)), "scored_rows": int(scored_count), "selector_eligible_rows": int(candidate_frame["selector_eligible"].sum()), "missing_rows": int(missing_count), "error_rows": int(error_count), "cache_hits": int(cache_hits), "cache_misses": int(max(0, len(frame) - cache_hits))},
        "errors": errors,
        "artifacts": {"candidate_pool.csv": {"path": "candidate_pool.csv", "sha256": artifact_hashes.get("candidate_pool.csv"), "rows": int(len(candidate_frame)), "columns": list(CANDIDATE_POOL_COLUMNS)}, "feature_embeddings.npz": {"path": "feature_embeddings.npz", "sha256": artifact_hashes.get("feature_embeddings.npz"), "arrays": {key: list(value.shape) for key, value in arrays.items()}}},
        "state": {"complete": bool(scored_count + missing_count + error_count == len(frame)), "ready_for_selection": bool(scored_count > 0), "readiness": "ready_for_selection" if scored_count > 0 else "blocked_no_successful_scored_rows"},
    }
    if write:
        atomic_write_json(scoring_path, jsonable(payload))
        payload["artifacts"]["scoring_manifest.json"] = {"path": "scoring_manifest.json", "sha256": sha256_file(scoring_path)}
    else:
        payload["artifacts"]["scoring_manifest.json"] = {"path": "scoring_manifest.json", "sha256": None}
    return jsonable(payload)


def run_active_learning_scoring(manifest_path: str | Path, *, output_dir: str | Path = "data/processed/active_learning", run_name: str = "active_learning", **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper using the canonical ignored active-learning root."""
    return score_tile_manifest(manifest_path, run_root=Path(output_dir) / run_name, **kwargs)


score_manifest = score_tile_manifest
score_active_learning_pool = run_active_learning_scoring


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a canonical active-learning tile pool")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/active_learning"))
    parser.add_argument("--run-name", default="active_learning")
    parser.add_argument("--registry", type=Path, default=Path("data/processed/regression/model_registry.json"))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path("models/classifier/best_model.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lsh-seed", type=int, default=0)
    parser.add_argument("--lsh-bits", type=int, default=16)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--imagenet-stats", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_active_learning_scoring(args.manifest, output_dir=args.output_dir, run_name=args.run_name, registry_path=args.registry, classifier_checkpoint=args.classifier_checkpoint, device=args.device, classifier_use_resisc45_stats=not args.imagenet_stats, batch_size=args.batch_size, lsh_seed=args.lsh_seed, lsh_bits=args.lsh_bits, max_rows=args.max_rows)
    print(json.dumps({"run_root": str(Path(args.output_dir) / args.run_name), "candidate_pool": "candidate_pool.csv", "feature_embeddings": "feature_embeddings.npz", "scoring_manifest": "scoring_manifest.json", "counts": result["counts"], "state": result["state"]}, sort_keys=True))
    if not result["state"]["ready_for_selection"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
