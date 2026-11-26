"""Resumable active-learning pool scoring.

The scorer keeps a scalar, selector-ready candidate table and stores dense
model features in a compressed NPZ.  It preserves every manifest row (including
missing/error rows) while setting ``selector_eligible`` and paired availability
flags only for successfully scored rows.  Production model loading follows
the repository's classifier, terrain, regression, and active-registry
contracts; dependency injection keeps focused tests free of real weights and
network access.

Canonical acquisition claims must match inventory provenance and both PNG
hashes before decoding or cache reuse. Persisted canonical terrain PNGs reserve
black for outside-land sea: that mask excludes fill from terrain/vegetation
features and counts known sea in water coverage. Schema 4 invalidates earlier
unmasked feature caches without changing the six-value terrain vector.

Each invocation owns one streaming loader: a bounded source queue overlaps
acquisition with worker decoding and inference. The queue and loader prefetch
windows bound payload residency independently of manifest length. Torch head
outputs remain on device until regression finishes; artifacts cross to NumPy
once per batch.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import tempfile
from threading import Event, Thread
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import IterableDataset
from .common import validate_run_name


from .common import (
    atomic_write_json,
    atomic_write_text,
    jsonable,
    sha256_bytes,
    sha256_file,
)
from .water import (
    MAX_SELECTABLE_WATER_FRACTION,
    UNUSABLE_REASON_EXCESSIVE_WATER,
    compute_satellite_water_fraction,
    evaluate_water_status,
    is_excessive_water,
    weak_score_water_components,
)
from scenicdriver.data_pipeline.web_mercator import tile_pixel_ground_spacing_m
from scenicdriver.terrain.features import compute_terrain_features, compute_terrain_valid_mask

SCORING_SCHEMA_VERSION = 4
SCHEMA_VERSION = SCORING_SCHEMA_VERSION
CANONICAL_TILE_COLUMNS = (
    "region",
    "z",
    "x",
    "y",
    "lat",
    "lon",
    "satellite_path",
    "terrain_path",
    "satellite_present",
    "terrain_present",
)
OPTIONAL_TILE_COLUMNS = ("satellite_s3_uri", "terrain_s3_uri")

# Scalar-only columns.  Vectors are never expanded into CSV or JSON.
CANDIDATE_POOL_COLUMNS = (
    "image_path",
    "tile_identity",
    "source_identity",
    "region",
    "z",
    "x",
    "y",
    "lat",
    "lon",
    "provider",
    "satellite_source",
    "terrain_source",
    "source_contract_sha256",
    "preprocessing_contract_sha256",
    "acquisition_tile_sha256",
    "embedding_feature_sha256",
    "prediction_report_sha256",
    "satellite_path",
    "terrain_path",
    "satellite_s3_uri",
    "terrain_s3_uri",
    "manifest_satellite_present",
    "manifest_terrain_present",
    "satellite_present",
    "terrain_present",
    "availability_state",
    "score_status",
    "selector_eligible",
    "error",
    "satellite_content_sha256",
    "terrain_content_sha256",
    "heuristic_score",
    "scenic_score",
    "scenic_score_heuristic",
    "heuristic_class_component",
    "heuristic_relief_component",
    "heuristic_roughness_component",
    "heuristic_slope_component",
    "heuristic_water_component",
    "heuristic_vegetation_component",
    "heuristic_water_fraction_penalty",
    "relief",
    "roughness",
    "slope_mean",
    "slope_variation",
    "water_proximity",
    "vegetation_density",
    "water_fraction",
    "terrain_sea_level_fraction",
    "effective_water_fraction",
    "water_filter_status",
    "unusable_reason",
    "regression_prediction",
    "model_prediction",
    "label_source",
    "classifier_checkpoint_sha256",
    "regression_checkpoint_sha256",
    "class_id",
    "class_name",
    "class_probability",
    "class_count",
    "normalized_class_entropy",
    "class_uncertainty",
    "uncertainty",
    "embedding_identity",
    "embedding_dimension",
    "embedding_row_index",
    "embedding_cluster_id",
    "cluster_id",
    "cache_hit",
)

UNCERTAINTY_NAME = "normalized_class_entropy"
UNCERTAINTY_DEFINITION = (
    "class uncertainty = -sum(p_i * ln(p_i)) / ln(K), where p is the softmax "
    "distribution over K RESISC45 classes; zero terms are zero and K <= 1 is zero"
)
DEFAULT_SCORING_BATCH_SIZE = 256
DEFAULT_SCORING_NUM_WORKERS = 4
SCORING_PREFETCH_FACTOR = 2
_TERRAIN_METRIC_NAMES = (
    "relief",
    "roughness",
    "slope_mean",
    "slope_variation",
    "water_proximity",
    "vegetation_density",
)


@dataclass
class ScoringDependencies:
    """Injected models/callables for tests, or loaded production objects.

    ``classifier_predictor`` returns ``(logits, embeddings)`` or a mapping with
    those keys for one batch.  ``regression_predictor`` accepts float32 NumPy
    arrays ``(embeddings, terrain_features, class_logits)`` and returns one
    prediction per row.
    Injected model objects with ``state_dict`` are fingerprinted from their
    current state, including when an explicit identity is supplied. Opaque
    objects and predictor callables require a content-bound ``*_hash`` (or
    an object checkpoint); callers must change it when prediction state changes.
    Injected preprocessors are bound the same way: a ``classifier_transform``
    requires ``classifier_preprocess_identity`` and a non-default
    ``terrain_feature_fn`` requires ``terrain_feature_identity``; both must
    change whenever the preprocessing configuration or implementation does.
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
    terrain_feature_identity: str | None = None


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
    classifier_preprocess_identity: str


@dataclass
class _SourceData:
    value: str
    uri: str
    content_hash: str | None = None
    payload: bytes | None = None
    resolved_path: Path | None = None


@dataclass
class _ScoringInput:
    index: int
    satellite: _SourceData
    terrain: _SourceData
    pixel_spacing_m: float


@dataclass
class _ScoringError:
    index: int
    message: str
    missing: bool = False


@dataclass
class _PendingRow:
    index: int
    satellite: Image.Image
    transformed: Any
    terrain_features: np.ndarray
    terrain_metrics: dict[str, float]
    satellite_metrics: dict[str, float]
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
        "1",
        "true",
        "yes",
        "y",
        "t",
        "on",
        "available",
        "complete",
        "completed",
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
    tile_id = _tile_identity(row)
    sat_path = _clean_text(row.get("satellite_path"))
    terr_path = _clean_text(row.get("terrain_path"))
    sat_s3 = _clean_text(row.get("satellite_s3_uri"))
    terr_s3 = _clean_text(row.get("terrain_s3_uri"))
    provider = _clean_text(row.get("satellite_provider") or row.get("provider"))
    sat_src = _clean_text(
        row.get("satellite_collection") or row.get("satellite_source")
    )
    terr_src = _clean_text(row.get("terrain_collection") or row.get("terrain_source"))
    src_contract = _clean_text(row.get("source_contract_sha256"))
    preproc = _clean_text(row.get("preprocessing_contract_sha256"))
    acq_id = _clean_text(
        row.get("acquisition_tile_identity_sha256")
        or row.get("acquisition_tile_sha256")
    )

    parts = [
        f"{tile_id}|satellite={sat_path}|terrain={terr_path}|satellite_s3={sat_s3}|terrain_s3={terr_s3}"
    ]
    if provider:
        parts.append(f"provider={provider}")
    if sat_src:
        parts.append(f"sat_src={sat_src}")
    if terr_src:
        parts.append(f"terr_src={terr_src}")
    if src_contract:
        parts.append(f"src_contract={src_contract}")
    if preproc:
        parts.append(f"preproc={preproc}")
    if acq_id:
        parts.append(f"acq_id={acq_id}")
    return "|".join(parts)


def _resolve_local_path(
    value: str, *, manifest_dir: Path, run_root: Path
) -> Path | None:
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
    *,
    value: str,
    uri: str,
    manifest_dir: Path,
    run_root: Path,
    s3_client: Any | None,
    style: str,
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
                raise ValueError(
                    f"{style} source requires an injected or configured S3 client"
                )
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
            raise FileNotFoundError(
                f"{style} image not found: {value or uri or '<empty>'}"
            )
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
            raise TypeError(
                f"injected {style} image loader must return PIL.Image.Image"
            )
        return image.convert("RGB")
    if source.payload is None:
        raise ValueError("image source has no bytes")
    with Image.open(io.BytesIO(source.payload)) as image:
        return image.convert("RGB").copy()


def _torch_module(value: Any) -> bool:
    if value is None:
        return False
    module = type(value).__module__
    return module.startswith("torch") or (
        hasattr(value, "state_dict") and hasattr(value, "parameters")
    )


def _cuda_device(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def _autocast_context(device: str) -> Any:
    if not _cuda_device(device):
        return nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _pin_tensor(value: Any, *, device: str) -> Any:
    if not _cuda_device(device):
        return value
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if value.is_cuda or value.is_pinned():
        return value
    if not torch.cuda.is_available():
        return value
    return value.pin_memory()


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

        tensors = [
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(np.asarray(value), dtype=torch.float32)
            for value in values
        ]
        return torch.stack(tensors, dim=0)
    return np.stack([_to_numpy(value) for value in values], axis=0).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(values, -745.0, 80.0))
    denominator = exponent.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        exponent,
        denominator,
        out=np.full_like(exponent, 1.0 / max(1, exponent.shape[1])),
        where=denominator > 0,
    )
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
    features_obj = (
        result.get("features")
        if isinstance(result, Mapping)
        else getattr(result, "features", None)
    )
    if features_obj is None:
        raise ValueError("terrain feature result has no features")
    terrain_array = np.asarray(
        features_obj.to_array() if hasattr(features_obj, "to_array") else features_obj,
        dtype=np.float32,
    ).reshape(-1)
    if terrain_array.size == 0 or not np.isfinite(terrain_array).all():
        raise ValueError("terrain feature result contains no finite features")

    def metric(name: str) -> float:
        raw = (
            result.get(name, 0.0)
            if isinstance(result, Mapping)
            else getattr(result, name, 0.0)
        )
        number = _finite(raw)
        return float(number if number is not None else 0.0)

    metrics = {name: metric(name) for name in ("relief", "roughness", "slope_mean")}
    for name in ("slope_variation", "water_proximity", "vegetation_density"):
        raw = (
            features_obj.get(name, 0.0)
            if isinstance(features_obj, Mapping)
            else getattr(features_obj, name, 0.0)
        )
        number = _finite(raw)
        metrics[name] = float(number if number is not None else 0.0)
    twf_raw = (
        result.get("terrain_sea_level_fraction")
        if isinstance(result, Mapping)
        else getattr(result, "terrain_sea_level_fraction", None)
    )
    if twf_raw is None:
        twf_raw = (
            features_obj.get("terrain_sea_level_fraction")
            if isinstance(features_obj, Mapping)
            else getattr(features_obj, "terrain_sea_level_fraction", None)
        )
    twf_num = _finite(twf_raw)
    metrics["terrain_sea_level_fraction"] = (
        float(twf_num) if twf_num is not None else 0.0
    )
    return terrain_array, metrics


def _satellite_metrics(image: Image.Image, *, land_mask: np.ndarray) -> dict[str, float]:
    return {"water_fraction": compute_satellite_water_fraction(image, land_mask=land_mask)}


def _requires_acquisition_provenance(row: Mapping[str, Any]) -> bool:
    """Require attributable sources; only explicit Mapbox markers stay legacy."""
    sources = [
        _clean_text(row.get(key)).lower()
        for key in (
            "provider", "satellite_provider", "terrain_provider",
            "satellite_collection", "satellite_source",
            "terrain_collection", "terrain_source",
        )
    ]
    acquisition_claim = any(
        _clean_text(row.get(key))
        for key in (
            "grid_sha256", "preprocessing_contract_sha256",
            "satellite_output_sha256", "terrain_output_sha256",
            "acquisition_tile_identity_sha256", "acquisition_tile_sha256",
        )
    )
    if acquisition_claim or any("naip" in source or "3dep" in source for source in sources):
        return True
    attributed = [source for source in sources if source]
    if attributed and all(source.startswith("mapbox") for source in attributed):
        return False
    if _clean_text(row.get("source_contract_sha256")):
        return True
    if attributed:
        raise ValueError("unrecognized_source_attribution: tile source cannot be attributed")
    raise ValueError("missing_source_attribution: tile source cannot be attributed")


def _acquisition_digest(value: Any) -> str:
    """Match the acquisition producer's canonical JSON identity."""
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8"))


def _acquisition_preprocessing(
    *, manifest_dir: Path, run_root: Path
) -> tuple[dict[str, Any], str, str]:
    """Read the existing execution-plan contract; verify bytes and metadata hashes."""
    plan_path = _resolve_local_path(
        "execution_plan.json", manifest_dir=manifest_dir, run_root=run_root
    )
    if plan_path is None:
        raise ValueError("missing_acquisition_metadata: execution_plan.json is required")
    try:
        plan = json.loads(plan_path.read_bytes())
        contract_path = _resolve_local_path(
            plan["inputs"]["source_contract_path"],
            manifest_dir=plan_path.parent, run_root=run_root,
        )
        if contract_path is None:
            raise ValueError("missing source contract")
        raw = contract_path.read_bytes()
        contract = json.loads(raw)
        preprocessing = contract["preprocessing"]
        source_hash = sha256_bytes(raw)
        preprocessing_hash = _acquisition_digest(preprocessing)
        if (
            plan["hashes"]["source_contract_sha256"] != source_hash
            or plan["hashes"]["preprocessing_contract_sha256"] != preprocessing_hash
        ):
            raise ValueError("source/preprocessing contract hash mismatch")
        for field in ("zoom", "tile_width", "tile_height"):
            value = preprocessing[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an integer")
        if not 0 <= preprocessing["zoom"] <= 22:
            raise ValueError("zoom must be between 0 and 22")
        if min(preprocessing["tile_width"], preprocessing["tile_height"]) <= 0:
            raise ValueError("tile dimensions must be positive")
        if preprocessing["target_crs"] != "EPSG:3857":
            raise ValueError("target_crs must be EPSG:3857 for slippy tiles")
        for field in ("satellite_resampling", "terrain_resampling"):
            if not isinstance(preprocessing[field], str) or not preprocessing[field].strip():
                raise ValueError(f"{field} is required")
        grid = plan["grid_contract"]
        if (
            plan["zoom"] != preprocessing["zoom"]
            or grid["tile_width"] != preprocessing["tile_width"]
            or grid["tile_height"] != preprocessing["tile_height"]
            or grid["crs"] != preprocessing["target_crs"]
        ):
            raise ValueError("execution-plan grid differs from source contract")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid_acquisition_metadata: {exc}") from exc
    return preprocessing, source_hash, preprocessing_hash


class _ScoringDataset(IterableDataset):
    """Consume bounded source inputs; decode and transform in persistent workers."""

    def __init__(
        self,
        inputs: Any,
        *,
        classifier_transform: Callable[[Image.Image], Any] | None,
        terrain_feature_fn: Callable[..., Any],
        image_loader: Callable[..., Image.Image] | None,
        stop_event: Any,
    ) -> None:
        super().__init__()
        self._inputs = inputs
        self._classifier_transform = classifier_transform
        self._terrain_feature_fn = terrain_feature_fn
        self._image_loader = image_loader
        self._stop_event = stop_event

    def __iter__(self):
        while not self._stop_event.is_set():
            try:
                item = self._inputs.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                return
            yield self._prepare(item)

    def _prepare(self, item: _ScoringInput) -> _PendingRow | _ScoringError:
        try:
            satellite = _open_rgb(
                item.satellite,
                image_loader=self._image_loader,
                style="satellite",
            )
            terrain = _open_rgb(
                item.terrain,
                image_loader=self._image_loader,
                style="terrain",
            )
            if satellite.size != terrain.size or min(*satellite.size) <= 0:
                raise ValueError(
                    "satellite and terrain imagery must have matching positive dimensions"
                )
            # Scoring reads persisted canonical PNGs; black DEM is reserved sea fill.
            valid_mask = compute_terrain_valid_mask(terrain)
            terrain_result = self._terrain_feature_fn(
                terrain, satellite, valid_mask=valid_mask,
                pixel_spacing_m=item.pixel_spacing_m,
            )
            terrain_features, terrain_metrics = _terrain_values(terrain_result)
            transformed = (
                self._classifier_transform(satellite)
                if self._classifier_transform is not None
                else np.asarray(satellite, dtype=np.float32)
            )
            return _PendingRow(
                index=item.index,
                satellite=satellite,
                transformed=transformed,
                terrain_features=terrain_features,
                terrain_metrics=terrain_metrics,
                satellite_metrics=_satellite_metrics(satellite, land_mask=valid_mask),
                satellite_hash=str(item.satellite.content_hash),
                terrain_hash=str(item.terrain.content_hash),
            )
        except Exception as exc:
            return _ScoringError(
                index=item.index,
                message=f"{type(exc).__name__}: {exc}",
                missing=isinstance(exc, FileNotFoundError),
            )


def _collate_scoring_items(
    items: list[_PendingRow | _ScoringError],
) -> list[_PendingRow | _ScoringError]:
    """Keep worker results ordered while allowing per-row failures."""
    return items


def _loader_options(
    *,
    device: str,
    num_workers: int,
    dependencies: ScoringDependencies,
) -> tuple[dict[str, Any], int]:
    requested = int(num_workers)
    if requested < 0:
        raise ValueError("num_workers must be non-negative")
    cuda = _cuda_device(device)
    if cuda and requested <= 0:
        raise ValueError("CUDA scoring requires num_workers > 0")
    injected = bool(
        dependencies.classifier_predictor is not None
        or dependencies.regression_predictor is not None
        or dependencies.image_loader is not None
        or dependencies.classifier_checkpoint is None
        or dependencies.regression_checkpoint is None
    )
    effective = requested if cuda or not injected else 0
    options: dict[str, Any] = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": effective,
        "pin_memory": cuda,
        "collate_fn": _collate_scoring_items,
    }
    if effective > 0:
        options.update(
            {
                "prefetch_factor": SCORING_PREFETCH_FACTOR,
                "persistent_workers": True,
            }
        )
    return options, effective


def _heuristic_components(
    class_score: float, terrain: Mapping[str, float], sat: Mapping[str, float]
) -> dict[str, float]:
    water_fraction = float(sat.get("water_fraction", 0.0))
    water = weak_score_water_components(
        class_score,
        water_fraction=water_fraction,
        water_proximity=float(terrain.get("water_proximity", 0.0)),
    )
    components = {
        "heuristic_class_component": water["heuristic_class_component"],
        "heuristic_relief_component": 2.0
        * math.tanh(float(terrain.get("relief", 0.0)) / 500.0),
        "heuristic_roughness_component": 1.5
        * math.tanh(float(terrain.get("roughness", 0.0)) / 200.0),
        "heuristic_slope_component": 1.5
        * math.tanh(float(terrain.get("slope_mean", 0.0)) / 15.0),
        "heuristic_water_component": water["heuristic_water_component"],
        "heuristic_vegetation_component": float(terrain.get("vegetation_density", 0.0)),
        "heuristic_water_fraction_penalty": water["heuristic_water_fraction_penalty"],
    }
    components["heuristic_score"] = float(np.clip(sum(components.values()), 0.0, 10.0))
    components["scenic_score"] = components["heuristic_score"]
    components["scenic_score_heuristic"] = components["heuristic_score"]
    return components


def _model_identity(
    label: str,
    value: Any,
    explicit: str | None = None,
    checkpoint: Path | None = None,
) -> str:
    """Bind injected identities to current state, never merely the model class."""
    state_dict = getattr(value, "state_dict", None)
    if callable(state_dict):
        state = state_dict()
        if not isinstance(state, Mapping):
            raise ValueError(f"injected {label} state_dict must be a mapping")
        digest = hashlib.sha256()
        cls = type(value)
        digest.update(f"{label}:{cls.__module__}.{cls.__qualname__}".encode())
        digest.update(str(explicit or "").encode())
        for key in sorted(state):
            item = state[key]
            if not isinstance(key, str):
                raise ValueError(f"injected {label} state keys must be strings")
            if isinstance(item, np.ndarray):
                array = np.ascontiguousarray(item)
                if array.dtype.hasobject:
                    raise ValueError(f"injected {label} requires a content hash")
                header = [key, str(array.dtype), list(array.shape)]
                data = memoryview(array).cast("B")
            else:
                import torch

                if not isinstance(item, torch.Tensor) or item.layout != torch.strided:
                    raise ValueError(f"injected {label} requires a content hash")
                tensor = item.detach().cpu().contiguous()
                header = [key, str(tensor.dtype), list(tensor.shape)]
                data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
            digest.update(json.dumps(header, separators=(",", ":")).encode())
            digest.update(data)
        return digest.hexdigest()
    if explicit:
        return str(explicit)
    if checkpoint is not None:
        return sha256_file(checkpoint)
    if isinstance(value, (str, Path)) and Path(value).is_file():
        return sha256_file(value)
    if value is None:
        return f"{label}:none"
    raise ValueError(f"injected {label} requires a content hash or state_dict")


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
    if requested == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        return "cpu"
    return requested


def _torch_load(path: Path, *, map_location: str) -> Any:
    from scenicdriver.scenic_scorer.regression import load_checkpoint

    return load_checkpoint(path, map_location=map_location)


def resolve_active_regression_checkpoint(
    registry_path: str | Path = "data/processed/regression/model_registry.json",
) -> Path:
    """Resolve relative checkpoints beside the registry before project/CWD paths."""
    registry_file = Path(registry_path)
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"active regression registry not found: {registry_file}"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"malformed active regression registry: {registry_file}"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("active"), Mapping
    ):
        raise ValueError(
            "malformed active regression registry: active record is missing"
        )
    raw_checkpoint = payload["active"].get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
        raise ValueError(
            "malformed active regression registry: active checkpoint is missing"
        )
    checkpoint_value = Path(raw_checkpoint).expanduser()
    candidates = (
        [checkpoint_value]
        if checkpoint_value.is_absolute()
        else [registry_file.parent / checkpoint_value, Path.cwd() / checkpoint_value]
    )
    found_candidate: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            found_candidate = candidate
            break
    if found_candidate is None:
        raise FileNotFoundError(
            f"active regression checkpoint not found: {raw_checkpoint}"
        )

    actual_sha256 = sha256_file(found_candidate)
    active_record = payload["active"]
    expected_sha256 = active_record.get("sha256")
    if expected_sha256 is not None:
        expected_str = str(expected_sha256).strip()
        if expected_str != actual_sha256:
            raise ValueError(
                f"active regression checkpoint sha256 mismatch for {found_candidate}: expected {expected_str}, got {actual_sha256}"
            )

    return found_candidate


def _validate_regression_checkpoint(
    path: Path, *, device: str
) -> tuple[Any, dict[str, int]]:
    payload = _torch_load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"malformed regression checkpoint: {path}")
    if "model_state_dict" not in payload or not isinstance(
        payload["model_state_dict"], Mapping
    ):
        raise ValueError(
            f"malformed regression checkpoint {path}: missing model_state_dict"
        )

    source = payload
    required = ("vit_dim", "terrain_dim", "num_classes")
    missing = sorted(set(required) - set(source.keys()))
    if missing:
        raise ValueError(f"malformed regression checkpoint {path}: missing {missing}")

    dims: dict[str, int] = {}
    for key in required:
        val = source.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"malformed regression checkpoint {path}: invalid {key}")
        parsed = int(val)
        if parsed <= 0:
            raise ValueError(f"malformed regression checkpoint {path}: invalid {key}")
        dims[key] = parsed

    hidden_val = source.get("hidden_dim", payload.get("hidden_dim", 256))
    if isinstance(hidden_val, bool) or not isinstance(hidden_val, (int, np.integer)):
        raise ValueError(f"malformed regression checkpoint {path}: invalid hidden_dim")
    hidden_dim = int(hidden_val)
    if hidden_dim <= 0:
        raise ValueError(f"malformed regression checkpoint {path}: invalid hidden_dim")
    dims["hidden_dim"] = hidden_dim

    from scenicdriver.scenic_scorer.regression import ScenicRegressionModel

    try:
        model = ScenicRegressionModel(**dims).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"malformed regression checkpoint {path}: state dict mismatch"
        ) from exc
    model.eval()
    return model, dims


def classifier_preprocess_identity(preprocessing: Any) -> str:
    """Canonical identity of the checkpoint-derived classifier transform config."""
    return json.dumps(
        {
            "pipeline": "Resize(image_size,image_size)+ToTensor+Normalize(mean,std)",
            "preprocessing": preprocessing.as_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def load_scoring_models(
    *,
    registry_path: str | Path = "data/processed/regression/model_registry.json",
    classifier_checkpoint: str | Path = "models/classifier/best_model.pt",
    device: str = "auto",
    classifier_normalization: str | None = None,
) -> LoadedScoringModels:
    """Load classifier and active regression exactly once, read-only.

    The classifier transform and class order come from the checkpoint's
    preprocessing contract; ``classifier_normalization`` is only an assertion
    and a conflicting value raises ``ValueError``.
    """
    resolved_device = _device_name(device)
    regression_path = resolve_active_regression_checkpoint(registry_path)
    classifier_path = Path(classifier_checkpoint)
    if not classifier_path.exists() or not classifier_path.is_file():
        raise FileNotFoundError(f"classifier checkpoint not found: {classifier_path}")
    try:
        from scenicdriver.classifier.inference import get_inference_transform
        from scenicdriver.classifier.model import LandscapeClassifier

        classifier = LandscapeClassifier(
            pretrained=False, pretrained_path=classifier_path, device=resolved_device
        )
        classifier.to(resolved_device)
        classifier.eval()
    except ImportError as exc:
        raise ImportError(
            "classifier dependencies are required for active-learning scoring"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"malformed classifier checkpoint: {classifier_path}: {exc}"
        ) from exc
    preprocessing = classifier.preprocessing
    preprocessing.require_normalization(classifier_normalization)
    regression, _dims = _validate_regression_checkpoint(
        regression_path, device=resolved_device
    )
    return LoadedScoringModels(
        classifier=classifier,
        classifier_transform=get_inference_transform(preprocessing),
        regression_model=regression,
        class_names=preprocessing.class_names,
        device=resolved_device,
        classifier_checkpoint=classifier_path,
        regression_checkpoint=regression_path,
        classifier_hash=sha256_file(classifier_path),
        regression_hash=sha256_file(regression_path),
        classifier_preprocess_identity=classifier_preprocess_identity(preprocessing),
    )


def _dependencies_from_loaded(models: LoadedScoringModels) -> ScoringDependencies:
    return ScoringDependencies(
        classifier=models.classifier,
        classifier_transform=models.classifier_transform,
        regression_model=models.regression_model,
        class_names=models.class_names,
        device=models.device,
        classifier_hash=models.classifier_hash,
        regression_hash=models.regression_hash,
        classifier_checkpoint=models.classifier_checkpoint,
        regression_checkpoint=models.regression_checkpoint,
        classifier_preprocess_identity=models.classifier_preprocess_identity,
    )


# Versioned contract of the default terrain feature implementation. Bump
# ``version`` whenever compute_terrain_features / compute_terrain_valid_mask
# change semantics so cached features are invalidated.
# v2: slope_mean is degrees from horizontal and slope_variation is the slope
# std in degrees / 15 (clipped to 1), both derived with the tile's ground
# pixel spacing; v1 used unscaled per-pixel elevation differences.
TERRAIN_FEATURE_CONTRACT = {
    "version": 2,
    "function": "scenicdriver.terrain.features.compute_terrain_features",
    "valid_mask": "scenicdriver.terrain.features.compute_terrain_valid_mask",
    "pixel_spacing": "scenicdriver.data_pipeline.web_mercator.tile_pixel_ground_spacing_m",
    "slope_units": "degrees",
    "features": list(_TERRAIN_METRIC_NAMES),
}


def _preprocessing_contract(dependencies: ScoringDependencies) -> dict[str, str]:
    if dependencies.classifier_preprocess_identity:
        classifier_transform = dependencies.classifier_preprocess_identity
    elif dependencies.classifier_transform is None:
        classifier_transform = "none:rgb-float32-array"
    else:
        raise ValueError(
            "injected classifier_transform requires classifier_preprocess_identity"
        )
    if dependencies.terrain_feature_identity:
        terrain_features = dependencies.terrain_feature_identity
    elif dependencies.terrain_feature_fn is compute_terrain_features:
        terrain_features = json.dumps(
            TERRAIN_FEATURE_CONTRACT, sort_keys=True, separators=(",", ":")
        )
    else:
        raise ValueError("injected terrain_feature_fn requires terrain_feature_identity")
    contract = {
        "classifier_transform": classifier_transform,
        "water_filter": f"schema-{SCORING_SCHEMA_VERSION}:inclusive-{MAX_SELECTABLE_WATER_FRACTION}",
        "terrain_features": terrain_features,
        "device": dependencies.device,
    }
    contract["pipeline_sha256"] = sha256_bytes(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    )
    return contract


def _read_cache(run_root: Path, *, pipeline_identity: str) -> _CacheData | None:
    candidate_path = run_root / "candidate_pool.csv"
    embedding_path = run_root / "feature_embeddings.npz"
    scoring_path = run_root / "scoring_manifest.json"
    if (
        not candidate_path.exists()
        or not embedding_path.exists()
        or not scoring_path.exists()
    ):
        return None
    try:
        manifest = json.loads(scoring_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCORING_SCHEMA_VERSION:
            return None
        preprocessing = manifest.get("preprocessing", {})
        if (
            not isinstance(preprocessing, Mapping)
            or preprocessing.get("pipeline_sha256") != pipeline_identity
        ):
            return None
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            return None
        for name, path in (
            ("candidate_pool.csv", candidate_path),
            ("feature_embeddings.npz", embedding_path),
        ):
            entry = artifacts.get(name)
            expected = entry.get("sha256") if isinstance(entry, Mapping) else None
            if (
                not isinstance(expected, str)
                or sha256_file(path).lower() != expected.lower()
            ):
                return None
    except (OSError, json.JSONDecodeError, TypeError):
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


def _cache_match(
    row: Mapping[str, Any],
    cached: Mapping[str, Any],
    *,
    classifier_hash: str,
    regression_hash: str,
    arrays: Mapping[str, np.ndarray],
) -> int | None:
    if _clean_text(cached.get("score_status")) != "scored":
        return None
    if _clean_text(cached.get("satellite_content_sha256")) != _clean_text(
        row.get("satellite_content_sha256")
    ) or _clean_text(cached.get("terrain_content_sha256")) != _clean_text(
        row.get("terrain_content_sha256")
    ):
        return None
    if (
        _clean_text(cached.get("classifier_checkpoint_sha256")) != classifier_hash
        or _clean_text(cached.get("regression_checkpoint_sha256")) != regression_hash
    ):
        return None

    for identity_key in (
        "provider",
        "satellite_source",
        "terrain_source",
        "source_contract_sha256",
        "preprocessing_contract_sha256",
        "acquisition_tile_sha256",
    ):
        row_val = _clean_text(row.get(identity_key))
        cached_val = _clean_text(cached.get(identity_key))
        if row_val != cached_val:
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
    identity_fields = {
        "image_path",
        "tile_identity",
        "source_identity",
        "region",
        "z",
        "x",
        "y",
        "lat",
        "lon",
        "provider",
        "satellite_source",
        "terrain_source",
        "source_contract_sha256",
        "preprocessing_contract_sha256",
        "acquisition_tile_sha256",
        "embedding_feature_sha256",
        "prediction_report_sha256",
        "satellite_path",
        "terrain_path",
        "satellite_s3_uri",
        "terrain_s3_uri",
        "manifest_satellite_present",
        "manifest_terrain_present",
        "satellite_content_sha256",
        "terrain_content_sha256",
        "cache_hit",
        "embedding_row_index",
        "embedding_cluster_id",
        "cluster_id",
    }
    for key in CANDIDATE_POOL_COLUMNS:
        if key not in identity_fields and key in cached:
            result[key] = _json_scalar(cached[key])


def _call_classifier(
    dependencies: ScoringDependencies,
    images: list[Image.Image],
    *,
    transformed_batch: Any | None = None,
) -> tuple[Any, Any]:
    use_torch = (
        dependencies.classifier_predictor is None
        and _torch_module(dependencies.classifier)
    )
    transformed: Any
    if transformed_batch is None:
        transformed = [
            dependencies.classifier_transform(image)
            if dependencies.classifier_transform
            else np.asarray(image, dtype=np.float32)
            for image in images
        ]
    else:
        transformed = transformed_batch
    if dependencies.classifier_predictor is not None:
        if isinstance(transformed, list):
            transformed = _stack_transforms(transformed, use_torch=False)
        try:
            output = dependencies.classifier_predictor(transformed)
        except (TypeError, AttributeError):
            output = dependencies.classifier_predictor(images)
    else:
        if dependencies.classifier is None:
            raise ValueError("classifier is required for active-model scoring")
        batch = (
            _stack_transforms(transformed, use_torch=use_torch)
            if isinstance(transformed, list)
            else transformed
        )
        if use_torch:
            import torch

            if not isinstance(batch, torch.Tensor):
                batch = torch.as_tensor(batch, dtype=torch.float32)
            batch = _pin_tensor(batch, device=dependencies.device)
            batch = batch.to(
                dependencies.device,
                non_blocking=_cuda_device(dependencies.device),
            )
            with torch.no_grad(), _autocast_context(dependencies.device):
                combined = getattr(
                    dependencies.classifier, "forward_with_features", None
                )
                if combined is not None:
                    output = combined(batch)
                else:
                    logits_value = dependencies.classifier(batch)
                    if not hasattr(dependencies.classifier, "get_features"):
                        raise ValueError(
                            "classifier must expose get_features for embeddings"
                        )
                    features_value = dependencies.classifier.get_features(batch)
                    output = {"logits": logits_value, "embeddings": features_value}
        else:
            logits_value = dependencies.classifier(batch)
            if not hasattr(dependencies.classifier, "get_features"):
                raise ValueError("classifier must expose get_features for embeddings")
            features_value = dependencies.classifier.get_features(batch)
            output = {"logits": logits_value, "embeddings": features_value}
    if isinstance(output, Mapping):
        logits_value = output.get("logits", output.get("class_logits"))
        features_value = output.get(
            "embeddings", output.get("features", output.get("vit_embeddings"))
        )
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        logits_value, features_value = output[0], output[1]
    else:
        raise ValueError("classifier predictor must return logits and embeddings")
    if logits_value is None or features_value is None:
        raise ValueError("classifier predictor returned incomplete outputs")
    # The Torch model path keeps both head inputs on device. Injected predictors
    # and non-Torch models have an explicit NumPy boundary instead.
    if use_torch:
        import torch

        if not isinstance(logits_value, torch.Tensor) or not isinstance(
            features_value, torch.Tensor
        ):
            raise ValueError("Torch classifier must return tensor logits and embeddings")
        logits, embeddings = logits_value, features_value
    else:
        logits, embeddings = _to_numpy(logits_value), _to_numpy(features_value)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if (
        logits.ndim != 2
        or embeddings.ndim != 2
        or len(logits) != len(images)
        or len(embeddings) != len(images)
        or logits.shape[1] <= 0
        or embeddings.shape[1] <= 0
    ):
        raise ValueError("classifier outputs do not match batch shape")
    if use_torch:
        return logits, embeddings
    return logits.astype(np.float32), embeddings.astype(np.float32)


def _call_regression(
    dependencies: ScoringDependencies,
    embeddings: Any,
    terrain_features: np.ndarray,
    logits: Any,
) -> np.ndarray:
    if dependencies.regression_predictor is not None:
        output = dependencies.regression_predictor(
            _to_numpy(embeddings), terrain_features, _to_numpy(logits)
        )
    else:
        if dependencies.regression_model is None:
            raise ValueError("active regression model is required for scoring")
        if _torch_module(dependencies.regression_model):
            import torch

            def on_device(value: Any) -> torch.Tensor:
                if isinstance(value, torch.Tensor):
                    # Classifier tensors already reside on the inference device.
                    return value
                tensor = torch.as_tensor(value, dtype=torch.float32).contiguous()
                tensor = _pin_tensor(tensor, device=dependencies.device)
                return tensor.to(
                    dependencies.device,
                    non_blocking=_cuda_device(dependencies.device),
                )

            with torch.no_grad(), _autocast_context(dependencies.device):
                output = dependencies.regression_model(
                    on_device(embeddings),
                    on_device(terrain_features),
                    on_device(logits),
                )
        else:
            output = dependencies.regression_model(
                _to_numpy(embeddings), terrain_features, _to_numpy(logits)
            )
    values = _to_numpy(output).reshape(-1)
    if len(values) != len(embeddings) or not np.isfinite(values).all():
        raise ValueError(
            "regression predictions do not match batch or contain non-finite values"
        )
    return np.clip(values.astype(np.float32), 0.0, 10.0)


def _lsh_clusters(embeddings: np.ndarray, *, seed: int, bits: int) -> list[str]:
    if embeddings.ndim != 2 or not len(embeddings):
        return []
    if bits <= 0 or bits > 63:
        raise ValueError("LSH bits must be between 1 and 63")
    rng = np.random.default_rng(int(seed))
    hyperplanes = rng.standard_normal((embeddings.shape[1], int(bits))).astype(
        np.float32
    )
    signs = (embeddings @ hyperplanes) >= 0.0
    values = np.zeros(len(embeddings), dtype=np.uint64)
    for bit in range(int(bits)):
        values |= signs[:, bit].astype(np.uint64) << np.uint64(bit)
    width = max(1, math.ceil(int(bits) / 4))
    return [f"lsh-{value:0{width}x}" for value in values]


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
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
    result.update(
        {
            "image_path": image_path or _clean_text(row.get("satellite_s3_uri")),
            "tile_identity": _tile_identity(row),
            "source_identity": _source_identity(row),
            "region": _clean_text(row.get("region")) or "unknown",
            "manifest_satellite_present": _bool(row.get("satellite_present"))
            or _bool(row.get("satellite_s3_present")),
            "manifest_terrain_present": _bool(row.get("terrain_present"))
            or _bool(row.get("terrain_s3_present")),
            "satellite_present": False,
            "terrain_present": False,
            "availability_state": "missing",
            "score_status": "missing",
            "selector_eligible": False,
            "error": None,
            "embedding_row_index": -1,
            "cache_hit": False,
        }
    )
    result["provider"] = _json_scalar(
        row.get("satellite_provider") or row.get("provider")
    )
    result["satellite_source"] = _json_scalar(
        row.get("satellite_collection") or row.get("satellite_source")
    )
    result["terrain_source"] = _json_scalar(
        row.get("terrain_collection") or row.get("terrain_source")
    )
    result["acquisition_tile_sha256"] = _json_scalar(
        row.get("acquisition_tile_identity_sha256")
        or row.get("acquisition_tile_sha256")
    )
    for identity_col in (
        "provider",
        "satellite_source",
        "terrain_source",
        "source_contract_sha256",
        "preprocessing_contract_sha256",
        "acquisition_tile_sha256",
        "embedding_feature_sha256",
        "prediction_report_sha256",
    ):
        if identity_col in row:
            result[identity_col] = _json_scalar(row.get(identity_col))
    return result


def _mark_error(result: dict[str, Any], message: str, *, missing: bool = False) -> None:
    result.update(
        {
            "availability_state": "missing" if missing else "error",
            "score_status": "missing" if missing else "error",
            "selector_eligible": False,
            "satellite_present": False,
            "terrain_present": False,
            "error": str(message),
        }
    )


def _validate_device_request(device: str) -> str:
    requested = str(device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")
    return requested


def _manifest_frame(
    manifest: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, str, Path, str]:
    if isinstance(manifest, pd.DataFrame):
        frame = manifest.copy()
        return frame, "<dataframe>", Path.cwd(), sha256_bytes(_stable_row_text(frame))
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"tile manifest not found: {manifest_path}")
    try:
        frame = pd.read_csv(manifest_path, low_memory=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid tile manifest: {manifest_path}") from exc
    return frame, str(manifest_path), manifest_path.parent, sha256_file(manifest_path)


def validate_scoring_inputs(
    manifest: str | Path | pd.DataFrame,
    *,
    run_name: str = "active_learning",
    batch_size: int = DEFAULT_SCORING_BATCH_SIZE,
    num_workers: int = DEFAULT_SCORING_NUM_WORKERS,
    device: str = "auto",
    max_rows: int | None = None,
    lsh_bits: int = 16,
) -> dict[str, Any]:
    """Validate dry-run inputs without loading models or touching output paths."""
    validated_name = validate_run_name(run_name)
    requested_device = _validate_device_request(device)
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(num_workers, bool) or int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if isinstance(lsh_bits, bool) or not 1 <= int(lsh_bits) <= 63:
        raise ValueError("LSH bits must be between 1 and 63")
    if requested_device == "cuda" and int(num_workers) <= 0:
        raise ValueError("CUDA scoring requires num_workers > 0")
    frame, manifest_label, _manifest_dir, source_hash = _manifest_frame(manifest)
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
    return {
        "dry_run": True,
        "valid": True,
        "run_name": validated_name,
        "manifest": {
            "path": manifest_label,
            "sha256": source_hash,
            "rows": source_rows,
            "selected_rows": int(len(frame)),
        },
        "configuration": {
            "device": requested_device,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": requested_device == "cuda",
            "prefetch_factor": SCORING_PREFETCH_FACTOR
            if int(num_workers) > 0
            else None,
            "persistent_workers": int(num_workers) > 0,
            "max_rows": int(max_rows) if max_rows is not None else None,
        },
        "artifacts": {},
    }


def score_tile_manifest(
    manifest: str | Path | pd.DataFrame,
    *,
    run_root: str | Path,
    dependencies: ScoringDependencies | None = None,
    registry_path: str | Path = "data/processed/regression/model_registry.json",
    classifier_checkpoint: str | Path = "models/classifier/best_model.pt",
    device: str = "auto",
    classifier_normalization: str | None = None,
    batch_size: int = DEFAULT_SCORING_BATCH_SIZE,
    num_workers: int = DEFAULT_SCORING_NUM_WORKERS,
    lsh_seed: int = 0,
    lsh_bits: int = 16,
    max_rows: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Score manifest rows and atomically publish candidate, NPZ, and manifest.

    Cache reuse is keyed by source content SHA-256 plus both model identities;
    ``max_rows`` is a deterministic prefix in the input order.  Models are
    loaded once and classifier inference is batched by ``batch_size``.
    Canonical acquisition rows require the existing execution-plan/source-contract
    sidecars. Grid metadata and separate resamplers are hash-verified, and PNG
    dimensions are checked before inference or cache reuse. Legacy Mapbox inputs
    report measured dimensions and unknown acquisition resampling.
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
        source_hash, manifest_label, manifest_dir = (
            sha256_file(manifest_path),
            str(manifest_path),
            manifest_path.parent,
        )
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
    if isinstance(num_workers, bool) or int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")

    models_loaded = dependencies is None
    if dependencies is None:
        dependencies = _dependencies_from_loaded(
            load_scoring_models(
                registry_path=registry_path,
                classifier_checkpoint=classifier_checkpoint,
                device=device,
                classifier_normalization=classifier_normalization,
            )
        )
    else:
        dependencies = ScoringDependencies(**dependencies.__dict__)
        dependencies.device = _device_name(
            device if device != "auto" else dependencies.device
        )
    if (
        dependencies.classifier_predictor is not None
        and not dependencies.classifier_hash
    ):
        raise ValueError("injected classifier_predictor requires classifier_hash")
    if (
        dependencies.regression_predictor is not None
        and not dependencies.regression_hash
    ):
        raise ValueError("injected regression_predictor requires regression_hash")
    if dependencies.s3_client is None and any(
        _s3_parts(_clean_text(value))
        for name in OPTIONAL_TILE_COLUMNS
        if name in frame.columns
        for value in frame[name].tolist()
    ):
        dependencies.s3_client = _default_s3_client()
    classifier_hash = _model_identity(
        "classifier",
        None if models_loaded else dependencies.classifier,
        dependencies.classifier_hash,
        dependencies.classifier_checkpoint,
    )
    regression_hash = _model_identity(
        "regression",
        None if models_loaded else dependencies.regression_model,
        dependencies.regression_hash,
        dependencies.regression_checkpoint,
    )
    preprocessing = _preprocessing_contract(dependencies)
    cache = _read_cache(root, pipeline_identity=preprocessing["pipeline_sha256"])
    results = [_base_result(row) for _, row in frame.iterrows()]
    loader_options, effective_num_workers = _loader_options(
        device=dependencies.device,
        num_workers=num_workers,
        dependencies=dependencies,
    )
    loader_options["batch_size"] = batch_size
    cache_hits = 0
    vector_by_index, logits_by_index, probs_by_index, terrain_by_index = {}, {}, {}, {}
    errors: list[dict[str, Any]] = []
    acquisition_contract: tuple[dict[str, Any], str, str] | None = None
    acquisition_metadata: dict[str, Any] | None = None

    def record_error(index: int, message: str, *, missing: bool = False) -> None:
        result = results[index]
        _mark_error(result, message, missing=missing)
        errors.append(
            {
                "source_identity": result["source_identity"],
                "error": message,
                "status": result["score_status"],
            }
        )

    def score_chunk(chunk: list[_PendingRow]) -> None:
        try:
            use_torch = (
                dependencies.classifier_predictor is None
                and _torch_module(dependencies.classifier)
            )
            transformed = _stack_transforms(
                [item.transformed for item in chunk], use_torch=use_torch
            )
            transformed = _pin_tensor(transformed, device=dependencies.device)
            logits, embeddings = _call_classifier(
                dependencies,
                [item.satellite for item in chunk],
                transformed_batch=transformed,
            )
            terrain_features = np.stack(
                [item.terrain_features for item in chunk], axis=0
            ).astype(np.float32)
            predictions = _call_regression(
                dependencies, embeddings, terrain_features, logits
            )
            # Export whole batches only after both model forwards have finished.
            logits, embeddings = _to_numpy(logits), _to_numpy(embeddings)
            probabilities = _softmax(logits)
            class_names = tuple(dependencies.class_names or ())
            if class_names and len(class_names) != logits.shape[1]:
                class_names = tuple(class_names[: logits.shape[1]])
            for offset, item in enumerate(chunk):
                result = results[item.index]
                sat_wf = item.satellite_metrics.get("water_fraction")
                terr_wf = item.terrain_metrics.get("terrain_sea_level_fraction")
                effective_wf, water_status, unusable_reason = evaluate_water_status(
                    sat_wf, terr_wf
                )
                if is_excessive_water(effective_wf):
                    result.update(
                        {
                            "water_fraction": sat_wf,
                            "terrain_sea_level_fraction": terr_wf,
                            "effective_water_fraction": effective_wf,
                            "water_filter_status": water_status,
                            "unusable_reason": unusable_reason,
                            "availability_state": "unusable",
                            "score_status": "unusable",
                            "selector_eligible": False,
                            "error": None,
                            "satellite_present": True,
                            "terrain_present": True,
                            "relief": item.terrain_metrics.get("relief"),
                            "roughness": item.terrain_metrics.get("roughness"),
                            "slope_mean": item.terrain_metrics.get("slope_mean"),
                            "slope_variation": item.terrain_metrics.get(
                                "slope_variation"
                            ),
                            "water_proximity": item.terrain_metrics.get(
                                "water_proximity"
                            ),
                            "vegetation_density": item.terrain_metrics.get(
                                "vegetation_density"
                            ),
                            "heuristic_score": None,
                            "scenic_score": None,
                            "scenic_score_heuristic": None,
                            "heuristic_class_component": None,
                            "heuristic_relief_component": None,
                            "heuristic_roughness_component": None,
                            "heuristic_slope_component": None,
                            "heuristic_water_component": None,
                            "heuristic_vegetation_component": None,
                            "heuristic_water_fraction_penalty": None,
                            "regression_prediction": None,
                            "model_prediction": None,
                            "label_source": None,
                            "class_id": None,
                            "class_name": None,
                            "class_probability": None,
                            "class_count": None,
                            "normalized_class_entropy": None,
                            "class_uncertainty": None,
                            "uncertainty": None,
                        }
                    )
                else:
                    row_probs = probabilities[offset]
                    class_id = int(np.argmax(row_probs))
                    class_name = (
                        class_names[class_id]
                        if class_id < len(class_names)
                        else f"class_{class_id}"
                    )
                    try:
                        from scenicdriver.classifier.model import get_scenic_weight

                        class_score = float(get_scenic_weight(class_name))
                    except Exception:
                        class_score = 0.3
                    components, entropy = (
                        _heuristic_components(
                            class_score, item.terrain_metrics, item.satellite_metrics
                        ),
                        normalized_class_entropy(row_probs),
                    )
                    result.update(
                        {
                            **components,
                            **item.terrain_metrics,
                            "water_fraction": sat_wf,
                            "terrain_sea_level_fraction": terr_wf,
                            "effective_water_fraction": effective_wf,
                            "water_filter_status": water_status,
                            "unusable_reason": None,
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
                    probs_by_index[item.index] = row_probs.astype(np.float32, copy=True)
                    terrain_by_index[item.index] = item.terrain_features.astype(
                        np.float32, copy=True
                    )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for item in chunk:
                record_error(item.index, message)


    def iter_inputs():
        nonlocal cache_hits, acquisition_contract, acquisition_metadata
        for index, (_, source_row) in enumerate(frame.iterrows()):
            row, result = source_row.to_dict(), results[index]
            if stop_event.is_set():
                return
            try:
                requires_provenance = _requires_acquisition_provenance(row)
                if requires_provenance:
                    from scenicdriver.data_pipeline.tile_inventory import acquisition_provenance_reason
    
                    reason = acquisition_provenance_reason(row)
                    if reason is not None:
                        raise ValueError(reason)
                    if acquisition_contract is None:
                        acquisition_contract = _acquisition_preprocessing(
                            manifest_dir=manifest_dir, run_root=root
                        )
                    contract, contract_hash, preprocessing_hash = acquisition_contract
                    if (
                        _clean_text(row.get("source_contract_sha256")) != contract_hash
                        or _clean_text(row.get("preprocessing_contract_sha256")) != preprocessing_hash
                    ):
                        raise ValueError("acquisition_contract_hash_mismatch")
                    if _finite(row.get("z")) != contract["zoom"]:
                        raise ValueError("acquisition_zoom_mismatch")
                    grid_hash = _acquisition_digest({
                        "z": int(row["z"]), "x": int(row["x"]), "y": int(row["y"]),
                        "width": contract["tile_width"], "height": contract["tile_height"],
                        "crs": contract["target_crs"],
                    })
                    if _clean_text(row.get("grid_sha256")) != grid_hash:
                        raise ValueError("acquisition_grid_mismatch")
            except Exception as exc:
                record_error(index, f"{type(exc).__name__}: {exc}")
                continue
            if (
                not result["manifest_satellite_present"]
                or not result["manifest_terrain_present"]
            ):
                missing_style = (
                    "satellite" if not result["manifest_satellite_present"] else "terrain"
                )
                record_error(
                    index,
                    f"manifest marks {missing_style} imagery unavailable",
                    missing=True,
                )
                continue
            try:
                satellite = _read_source(
                    value=_clean_text(row.get("satellite_path")),
                    uri=_clean_text(row.get("satellite_s3_uri")),
                    manifest_dir=manifest_dir,
                    run_root=root,
                    s3_client=dependencies.s3_client,
                    style="satellite",
                )
                terrain = _read_source(
                    value=_clean_text(row.get("terrain_path")),
                    uri=_clean_text(row.get("terrain_s3_uri")),
                    manifest_dir=manifest_dir,
                    run_root=root,
                    s3_client=dependencies.s3_client,
                    style="terrain",
                )
                assert (
                    satellite.content_hash is not None and terrain.content_hash is not None
                )
                if requires_provenance:
                    for style, source in (("satellite", satellite), ("terrain", terrain)):
                        if source.content_hash != _clean_text(row.get(f"{style}_output_sha256")):
                            raise ValueError(f"{style}_output_sha256_mismatch")
                dimensions = []
                for source in (satellite, terrain):
                    with Image.open(io.BytesIO(source.payload)) as image:
                        dimensions.append(image.size)
                if dimensions[0] != dimensions[1]:
                    raise ValueError("satellite and terrain imagery must have matching dimensions")
                width, height = dimensions[0]
                if requires_provenance:
                    assert acquisition_contract is not None
                    contract = acquisition_contract[0]
                    if (width, height) != (contract["tile_width"], contract["tile_height"]):
                        raise ValueError("acquisition_dimensions_mismatch")
                else:
                    contract = {}
                zoom = _finite(row.get("z"))
                if zoom is None or zoom != int(zoom) or not 0 <= zoom <= 22:
                    raise ValueError("invalid_acquisition_zoom")
                tile_x, tile_y = _finite(row.get("x")), _finite(row.get("y"))
                if tile_x is None or tile_y is None or tile_x != int(tile_x) or tile_y != int(tile_y):
                    raise ValueError("invalid_acquisition_tile: x/y required for pixel spacing")
                pixel_spacing_m = tile_pixel_ground_spacing_m(
                    int(tile_x), int(tile_y), int(zoom), width, height
                )
                metadata = {
                    "zoom": int(zoom), "width": width, "height": height,
                    "crs": contract.get("target_crs", "EPSG:3857"),
                    "satellite_resampling": contract.get("satellite_resampling"),
                    "terrain_resampling": contract.get("terrain_resampling"),
                }
                for field, expected in metadata.items():
                    if field in row and _clean_text(row[field]) and _clean_text(row[field]) != _clean_text(expected):
                        raise ValueError(f"acquisition_{field}_mismatch")
                if acquisition_metadata is not None and metadata != acquisition_metadata:
                    raise ValueError("mixed_acquisition_metadata")
                acquisition_metadata = metadata
                result["satellite_content_sha256"] = satellite.content_hash
                result["terrain_content_sha256"] = terrain.content_hash
                cached_row = cache.rows.get(result["source_identity"]) if cache else None
                cached_index = (
                    _cache_match(
                        result,
                        cached_row,
                        classifier_hash=classifier_hash,
                        regression_hash=regression_hash,
                        arrays=cache.arrays,
                    )
                    if cache and cached_row
                    else None
                )
                if cached_index is not None:
                    assert cached_row is not None and cache is not None
                    _copy_cached_fields(result, cached_row)
                    sat_wf = result.get("water_fraction")
                    terr_wf = result.get("terrain_sea_level_fraction")
                    effective_wf, water_status, _ = evaluate_water_status(sat_wf, terr_wf)
                    if (
                        is_excessive_water(effective_wf)
                        or _clean_text(cached_row.get("unusable_reason"))
                        == UNUSABLE_REASON_EXCESSIVE_WATER
                    ):
                        result.update(
                            {
                                "satellite_present": True,
                                "terrain_present": True,
                                "terrain_sea_level_fraction": terr_wf,
                                "effective_water_fraction": effective_wf,
                                "water_filter_status": water_status,
                                "unusable_reason": UNUSABLE_REASON_EXCESSIVE_WATER,
                                "availability_state": "unusable",
                                "score_status": "unusable",
                                "selector_eligible": False,
                                "cache_hit": True,
                                "heuristic_score": None,
                                "scenic_score": None,
                                "scenic_score_heuristic": None,
                                "heuristic_class_component": None,
                                "heuristic_relief_component": None,
                                "heuristic_roughness_component": None,
                                "heuristic_slope_component": None,
                                "heuristic_water_component": None,
                                "heuristic_vegetation_component": None,
                                "heuristic_water_fraction_penalty": None,
                                "regression_prediction": None,
                                "model_prediction": None,
                                "label_source": None,
                                "class_id": None,
                                "class_name": None,
                                "class_probability": None,
                                "class_count": None,
                                "normalized_class_entropy": None,
                                "class_uncertainty": None,
                                "uncertainty": None,
                            }
                        )
                    else:
                        result.update(
                            {
                                "satellite_present": True,
                                "terrain_present": True,
                                "terrain_sea_level_fraction": terr_wf,
                                "effective_water_fraction": effective_wf,
                                "water_filter_status": water_status,
                                "unusable_reason": None,
                                "availability_state": "available",
                                "score_status": "scored",
                                "selector_eligible": True,
                                "cache_hit": True,
                            }
                        )
                        vector_by_index[index] = np.asarray(
                            cache.arrays["embeddings"][cached_index], dtype=np.float32
                        ).copy()
                        for name, target in (
                            ("class_logits", logits_by_index),
                            ("class_probs", probs_by_index),
                            ("terrain_features", terrain_by_index),
                        ):
                            target[index] = np.asarray(
                                cache.arrays[name][cached_index], dtype=np.float32
                            ).copy()
                    cache_hits += 1
                    continue
                yield _ScoringInput(
                    index=index, satellite=satellite, terrain=terrain,
                    pixel_spacing_m=pixel_spacing_m,
                )
            except Exception as exc:
                record_error(
                    index,
                    f"{type(exc).__name__}: {exc}",
                    missing=isinstance(exc, FileNotFoundError),
                )

    import torch
    import torch.multiprocessing as mp

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    # Bound encoded inputs separately from DataLoader's decoded prefetch window.
    # At most one additional source pair is being acquired by the producer.
    queue_capacity = batch_size * max(1, effective_num_workers) * SCORING_PREFETCH_FACTOR
    if effective_num_workers:
        context = mp.get_context("spawn")
        inputs = context.Queue(maxsize=queue_capacity)
        stop_event = context.Event()
        loader_options["multiprocessing_context"] = context
    else:
        inputs = Queue(maxsize=queue_capacity)
        stop_event = Event()
    producer_errors: list[BaseException] = []

    def enqueue(item: _ScoringInput | None) -> bool:
        while not stop_event.is_set():
            try:
                inputs.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce_inputs() -> None:
        try:
            for item in iter_inputs():
                if not enqueue(item):
                    break
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            for _ in range(max(1, effective_num_workers)):
                if not enqueue(None):
                    break

    loader = torch.utils.data.DataLoader(
        _ScoringDataset(
            inputs,
            classifier_transform=dependencies.classifier_transform,
            terrain_feature_fn=dependencies.terrain_feature_fn,
            image_loader=dependencies.image_loader,
            stop_event=stop_event,
        ),
        **loader_options,
    )
    producer = Thread(target=produce_inputs, name="scoring-source-producer", daemon=True)
    iterator = None
    try:
        iterator = iter(loader)
        producer.start()
        for batch in iterator:
            chunk: list[_PendingRow] = []
            for item in batch:
                if isinstance(item, _ScoringError):
                    record_error(item.index, item.message, missing=item.missing)
                else:
                    chunk.append(item)
            if chunk:
                score_chunk(chunk)
        if producer_errors:
            raise producer_errors[0]
    finally:
        stop_event.set()
        if producer.ident is not None:
            producer.join()
        # Release the single loader's workers only when the invocation finishes.
        del iterator
        del loader
        if effective_num_workers:
            inputs.cancel_join_thread()
            inputs.close()

    scored_indices = [
        i
        for i, result in enumerate(results)
        if result.get("score_status") == "scored" and i in vector_by_index
    ]
    embedding_dimension = 0
    if scored_indices:
        dimensions = {int(vector_by_index[i].shape[-1]) for i in scored_indices}
        if len(dimensions) != 1:
            message = "embedding dimensions differ across scored rows"
            for i in scored_indices:
                _mark_error(results[i], message)
                errors.append(
                    {
                        "source_identity": results[i]["source_identity"],
                        "error": message,
                        "status": "error",
                    }
                )
            scored_indices = []
        else:
            embedding_dimension = next(iter(dimensions))
    embedding_matrix = (
        np.stack([vector_by_index[i] for i in scored_indices], axis=0).astype(
            np.float32
        )
        if scored_indices
        else np.empty((0, 0), dtype=np.float32)
    )
    clusters = (
        _lsh_clusters(embedding_matrix, seed=lsh_seed, bits=lsh_bits)
        if scored_indices
        else []
    )
    embedding_identity = (
        f"classifier-vit:{classifier_hash}:float32:{embedding_dimension}"
        if embedding_dimension
        else None
    )
    row_to_embedding = {i: offset for offset, i in enumerate(scored_indices)}
    for i, result in enumerate(results):
        if i in row_to_embedding:
            row_index = row_to_embedding[i]
            result.update(
                {
                    "embedding_row_index": row_index,
                    "embedding_dimension": embedding_dimension,
                    "embedding_identity": embedding_identity,
                    "embedding_cluster_id": clusters[row_index],
                    "cluster_id": clusters[row_index],
                }
            )
        elif result.get("score_status") != "scored":
            result.update(
                {
                    "embedding_row_index": -1,
                    "embedding_dimension": None,
                    "embedding_identity": None,
                    "embedding_cluster_id": None,
                    "cluster_id": None,
                }
            )

    candidate_frame = pd.DataFrame(results).reindex(columns=CANDIDATE_POOL_COLUMNS)
    for column in CANDIDATE_POOL_COLUMNS:
        candidate_frame[column] = candidate_frame[column].map(_json_scalar)
    for column in (
        "selector_eligible",
        "satellite_present",
        "terrain_present",
        "cache_hit",
    ):
        candidate_frame[column] = candidate_frame[column].map(_bool)
    arrays = {
        "embeddings": embedding_matrix,
        "class_logits": np.stack(
            [logits_by_index[i] for i in scored_indices], axis=0
        ).astype(np.float32)
        if scored_indices
        else np.empty((0, 0), dtype=np.float32),
        "class_probs": np.stack(
            [probs_by_index[i] for i in scored_indices], axis=0
        ).astype(np.float32)
        if scored_indices
        else np.empty((0, 0), dtype=np.float32),
        "terrain_features": np.stack(
            [terrain_by_index[i] for i in scored_indices], axis=0
        ).astype(np.float32)
        if scored_indices
        else np.empty((0, 0), dtype=np.float32),
        "row_indices": np.asarray(scored_indices, dtype=np.int64),
    }
    candidate_path, embeddings_path, scoring_path = (
        root / "candidate_pool.csv",
        root / "feature_embeddings.npz",
        root / "scoring_manifest.json",
    )
    artifact_hashes = {}
    if write:
        atomic_write_text(
            candidate_path, candidate_frame.to_csv(index=False, lineterminator="\n")
        )
        _atomic_write_npz(embeddings_path, arrays)
        artifact_hashes = {
            "candidate_pool.csv": sha256_file(candidate_path),
            "feature_embeddings.npz": sha256_file(embeddings_path),
        }
    scored_count = len(scored_indices)
    missing_count = sum(result.get("score_status") == "missing" for result in results)
    error_count = sum(result.get("score_status") == "error" for result in results)
    unusable_count = sum(result.get("score_status") == "unusable" for result in results)
    provider = _clean_text(
        frame["satellite_provider"].iloc[0]
        if "satellite_provider" in frame.columns
        else frame["provider"].iloc[0]
        if "provider" in frame.columns
        else None
    )
    satellite_source = _clean_text(
        frame["satellite_collection"].iloc[0]
        if "satellite_collection" in frame.columns
        else frame["satellite_source"].iloc[0]
        if "satellite_source" in frame.columns
        else None
    )
    terrain_source = _clean_text(
        frame["terrain_collection"].iloc[0]
        if "terrain_collection" in frame.columns
        else frame["terrain_source"].iloc[0]
        if "terrain_source" in frame.columns
        else None
    )
    source_contract_sha256 = _clean_text(
        frame["source_contract_sha256"].iloc[0]
        if "source_contract_sha256" in frame.columns
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "source": {
            "tile_manifest": {
                "path": manifest_label,
                "sha256": source_hash,
                "rows": source_rows,
                "selected_rows": int(len(frame)),
            },
            "provider": provider,
            "satellite_source": satellite_source,
            "terrain_source": terrain_source,
            "source_contract_sha256": source_contract_sha256,
            "identity_definition": "region/z/x/y plus satellite_path, terrain_path, provider, source contracts, and content SHA-256s",
            "source_content_definition": "SHA-256 of exact source bytes before PIL conversion",
            "object_hashes": artifact_hashes,
        },
        "grid": {
            **{
                field: acquisition_metadata.get(field) if acquisition_metadata else None
                for field in ("zoom", "width", "height", "crs")
            },
            "has_coords": all(
                _finite(row.get("lat")) is not None and _finite(row.get("lon")) is not None
                for row in results
            ),
        },
        "mosaic": {
            "mosaic_order": "deterministic",
        },
        "datum": {
            "horizontal_crs": acquisition_metadata.get("crs") if acquisition_metadata else None,
            "vertical_datum": "NAVD88",
        },
        "nodata": {
            "allowed_land_fraction": 0.0,
            "nodata_value": None,
        },
        "QC": {
            "land_admission_threshold": 0.05,
            "water_fraction_max": MAX_SELECTABLE_WATER_FRACTION,
        },
        "object_hashes": artifact_hashes,
        "models": {
            "classifier_checkpoint": str(dependencies.classifier_checkpoint)
            if dependencies.classifier_checkpoint
            else None,
            "classifier_checkpoint_sha256": classifier_hash,
            "regression_checkpoint": str(dependencies.regression_checkpoint)
            if dependencies.regression_checkpoint
            else None,
            "regression_checkpoint_sha256": regression_hash,
            "label_semantics": "regression_prediction/model_prediction are active-model outputs; no human label is inferred",
        },
        "preprocessing": {
            **preprocessing,
            "preprocess_sha256": preprocessing.get("pipeline_sha256"),
            "satellite_resampling": acquisition_metadata.get("satellite_resampling") if acquisition_metadata else None,
            "terrain_resampling": acquisition_metadata.get("terrain_resampling") if acquisition_metadata else None,
            "resample_alg": (
                acquisition_metadata["satellite_resampling"]
                if acquisition_metadata
                and acquisition_metadata["satellite_resampling"] == acquisition_metadata["terrain_resampling"]
                else None
            ),
            "batch_size": int(batch_size),
            "num_workers": int(effective_num_workers),
            "requested_num_workers": int(num_workers),
            "pin_memory": bool(loader_options["pin_memory"]),
            "prefetch_factor": (
                int(loader_options["prefetch_factor"])
                if "prefetch_factor" in loader_options
                else None
            ),
            "persistent_workers": bool(loader_options.get("persistent_workers", False)),
        },
        "uncertainty": {
            "name": UNCERTAINTY_NAME,
            "definition": UNCERTAINTY_DEFINITION,
            "range": [0.0, 1.0],
            "class_count": 45,
            "selector_column": "uncertainty",
        },
        "embedding": {
            "identity": embedding_identity,
            "array_key": "embeddings",
            "dtype": "float32",
            "dimension": int(embedding_dimension),
            "row_index_column": "embedding_row_index",
            "row_indices_array_key": "row_indices",
            "cluster_column": "embedding_cluster_id",
            "lsh": {
                "method": "sign random-hyperplane locality-sensitive hash",
                "seed": int(lsh_seed),
                "bits": int(lsh_bits),
                "definition": "hex bit signature of embedding dot products against standard-normal fixed-seed hyperplanes",
            },
        },
        "counts": {
            "manifest_rows": int(len(frame)),
            "scored_rows": int(scored_count),
            "selector_eligible_rows": int(candidate_frame["selector_eligible"].sum()),
            "missing_rows": int(missing_count),
            "error_rows": int(error_count),
            "unusable_rows": int(unusable_count),
            "cache_hits": int(cache_hits),
            "cache_misses": int(max(0, len(frame) - cache_hits)),
        },
        "errors": errors,
        "artifacts": {
            "candidate_pool.csv": {
                "path": "candidate_pool.csv",
                "sha256": artifact_hashes.get("candidate_pool.csv"),
                "rows": int(len(candidate_frame)),
                "columns": list(CANDIDATE_POOL_COLUMNS),
            },
            "feature_embeddings.npz": {
                "path": "feature_embeddings.npz",
                "sha256": artifact_hashes.get("feature_embeddings.npz"),
                "arrays": {key: list(value.shape) for key, value in arrays.items()},
            },
        },
        "state": {
            "complete": bool(
                scored_count + missing_count + error_count + unusable_count
                == len(frame)
            ),
            "ready_for_selection": bool(scored_count > 0),
            "readiness": "ready_for_selection"
            if scored_count > 0
            else "blocked_no_successful_scored_rows",
        },
    }
    if write:
        atomic_write_json(scoring_path, jsonable(payload))
        payload["artifacts"]["scoring_manifest.json"] = {
            "path": "scoring_manifest.json",
            "sha256": sha256_file(scoring_path),
        }
    else:
        payload["artifacts"]["scoring_manifest.json"] = {
            "path": "scoring_manifest.json",
            "sha256": None,
        }
    return jsonable(payload)


def run_active_learning_scoring(
    manifest_path: str | Path,
    *,
    output_dir: str | Path = "data/processed/active_learning",
    run_name: str = "active_learning",
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run scoring or validate its inputs without creating a run."""
    validated_name = validate_run_name(run_name)
    if dry_run:
        return validate_scoring_inputs(
            manifest_path,
            run_name=validated_name,
            batch_size=kwargs.get("batch_size", DEFAULT_SCORING_BATCH_SIZE),
            num_workers=kwargs.get("num_workers", DEFAULT_SCORING_NUM_WORKERS),
            device=kwargs.get("device", "auto"),
            max_rows=kwargs.get("max_rows"),
            lsh_bits=kwargs.get("lsh_bits", 16),
        )
    return score_tile_manifest(
        manifest_path, run_root=Path(output_dir) / validated_name, **kwargs
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a canonical active-learning tile pool"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/active_learning")
    )
    parser.add_argument("--run-name", default="active_learning")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/processed/regression/model_registry.json"),
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("models/classifier/best_model.pt"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SCORING_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_SCORING_NUM_WORKERS)
    parser.add_argument("--lsh-seed", type=int, default=0)
    parser.add_argument("--lsh-bits", type=int, default=16)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--normalization",
        choices=("imagenet", "resisc45"),
        default=None,
        help="Optional assertion against the classifier checkpoint's preprocessing contract",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_active_learning_scoring(
        args.manifest,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dry_run=args.dry_run,
        registry_path=args.registry,
        classifier_checkpoint=args.classifier_checkpoint,
        device=args.device,
        classifier_normalization=args.normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lsh_seed=args.lsh_seed,
        lsh_bits=args.lsh_bits,
        max_rows=args.max_rows,
    )
    if args.dry_run:
        print(json.dumps(result, sort_keys=True))
        return
    print(
        json.dumps(
            {
                "run_root": str(
                    Path(args.output_dir) / validate_run_name(args.run_name)
                ),
                "candidate_pool": "candidate_pool.csv",
                "feature_embeddings": "feature_embeddings.npz",
                "scoring_manifest": "scoring_manifest.json",
                "counts": result["counts"],
                "state": result["state"],
            },
            sort_keys=True,
        )
    )
    if not result["state"]["ready_for_selection"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
