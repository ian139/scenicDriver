"""Stage-two active scenic dataset preparation and resumable training.

This module is strict at the stage-one boundary. It consumes the scorer's
ordered candidate pool and feature cache, overlays absolute human labels, and
trains the existing :class:`ScenicRegressionModel` on fixed geographic
assignments supplied by stage one. No random split or implicit row filter is
performed here. An explicit ``human_only_training`` config option narrows the
train rows to human-labeled rows only and never changes validation or test
membership.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import torch

from .regression import (
    ScenicRegressionModel,
    atomic_torch_save,
    atomic_write,
    capture_rng_state,
    load_checkpoint,
    resolve_device,
    restore_rng_state,
)


HANDOFF_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
_ALLOWED_SPLITS = {"train", "val", "validation", "test"}
_REQUIRED_READINESS = (
    "data_complete",
    "annotations_valid",
    "splits_valid",
    "benchmark_valid",
)
_REQUIRED_FEATURE_ARRAYS = (
    "embeddings",
    "terrain_features",
    "class_logits",
    "class_probs",
    "row_indices",
)
_FILTERED_INDEX_NAMES = (
    "filtered_index",
    "filtered_indices",
    "filtered_index_artifact",
    "filtered_index_csv",
)
_TILE_IDENTITY_COORDS = re.compile(r"(?:^|/)z(-?\d+)/x(-?\d+)/y(-?\d+)$")


@dataclass(frozen=True)
class ActiveTrainingConfig:
    """Material and bounded controls for active stage-two training.

    ``max_steps`` and ``max_seconds`` are per-invocation continuation budgets;
    the checkpoint's global step remains cumulative. They are excluded from
    ``config_hash`` so a paused run can continue with a larger budget.
    ``human_only_training`` narrows the train index array to rows whose
    ``label_source`` is exactly ``human``; it is included in ``config_hash``
    and never changes validation or test membership.
    """

    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    hidden_dim: int = 256
    use_sample_weights: bool = True
    sample_weight_scheme: str = "standard"
    loss_function: str = "mse"
    max_steps: int | None = None
    max_seconds: float | None = None
    initial_checkpoint_path: str | None = None
    initial_checkpoint_sha256: str | None = None
    human_sample_weight_multiplier: float = 1.0
    evaluate_test_during_training: bool = False
    human_only_training: bool = False

    def validate(self) -> None:
        if isinstance(self.epochs, bool) or int(self.epochs) < 1:
            raise ValueError("epochs must be a positive integer")
        if isinstance(self.batch_size, bool) or int(self.batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        learning_rate = float(self.learning_rate)
        weight_decay = float(self.weight_decay)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if isinstance(self.hidden_dim, bool) or int(self.hidden_dim) < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or int(self.max_steps) < 0
        ):
            raise ValueError("max_steps must be a non-negative integer or null")
        if self.max_seconds is not None:
            seconds = float(self.max_seconds)
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("max_seconds must be finite and non-negative or null")
        if not isinstance(self.use_sample_weights, bool):
            raise ValueError("use_sample_weights must be boolean")
        if self.sample_weight_scheme not in {"standard", "region_balanced"}:
            raise ValueError(
                "sample_weight_scheme must be 'standard' or 'region_balanced'"
            )
        if self.loss_function not in {"mse", "huber"}:
            raise ValueError("loss_function must be 'mse' or 'huber'")
        if (self.initial_checkpoint_path is None) != (
            self.initial_checkpoint_sha256 is None
        ):
            raise ValueError(
                "initial_checkpoint_path and initial_checkpoint_sha256 "
                "must be provided together"
            )
        if self.initial_checkpoint_path is not None:
            if not str(self.initial_checkpoint_path).strip():
                raise ValueError("initial_checkpoint_path must be a non-empty path")
            expected = str(self.initial_checkpoint_sha256).lower()
            if len(expected) != 64 or any(
                char not in "0123456789abcdef" for char in expected
            ):
                raise ValueError(
                    "initial_checkpoint_sha256 must be a 64-character hex digest"
                )
        multiplier = float(self.human_sample_weight_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(
                "human_sample_weight_multiplier must be finite and positive"
            )
        if not isinstance(self.evaluate_test_during_training, bool):
            raise ValueError("evaluate_test_during_training must be boolean")
        if not isinstance(self.human_only_training, bool):
            raise ValueError("human_only_training must be boolean")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("epochs", "batch_size", "seed", "hidden_dim"):
            value[name] = int(value[name])
        for name in ("learning_rate", "weight_decay"):
            value[name] = float(value[name])
        if value["max_steps"] is not None:
            value["max_steps"] = int(value["max_steps"])
        if value["max_seconds"] is not None:
            value["max_seconds"] = float(value["max_seconds"])
        if value["initial_checkpoint_path"] is not None:
            value["initial_checkpoint_path"] = str(value["initial_checkpoint_path"])
        if value["initial_checkpoint_sha256"] is not None:
            value["initial_checkpoint_sha256"] = str(
                value["initial_checkpoint_sha256"]
            ).lower()
        value["human_sample_weight_multiplier"] = float(
            value["human_sample_weight_multiplier"]
        )
        return value


class ActiveTrainingError(ValueError):
    """Raised when an immutable stage-one or training artifact is unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    atomic_write(path, lambda handle: handle.write(payload.encode("utf-8")))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n")
    atomic_write(path, lambda handle: handle.write(payload.encode("utf-8")))


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write(path, lambda handle: np.savez_compressed(handle, **arrays))


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    atomic_torch_save(path, value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


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


def _as_bool(value: Any) -> bool:
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
        "completed",
        "skipped",
    }


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ActiveTrainingError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ActiveTrainingError(f"{field} must be finite")
    return parsed


def _optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = _clean_text(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ActiveTrainingError(f"{field} must be null or a 64-character hex digest")
    return text


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActiveTrainingError(f"invalid {label} JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActiveTrainingError(f"{label} must be a JSON object: {path}")
    return value


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError) as exc:
        raise ActiveTrainingError(f"invalid {label} CSV: {path}: {exc}") from exc


def _handoff_root(handoff: Mapping[str, Any], handoff_path: Path) -> Path:
    raw = handoff.get("run_root")
    if raw is None:
        return handoff_path.parent
    candidate = Path(str(raw)).expanduser()
    return (
        candidate
        if candidate.is_absolute()
        else (handoff_path.parent / candidate).resolve()
    )


def _artifact_entry(
    handoff: Mapping[str, Any], names: Sequence[str]
) -> tuple[str, Any] | None:
    artifacts = handoff.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    for name in names:
        if name in artifacts:
            return name, artifacts[name]
    return None


def _artifact_path_value(entry: Any) -> str | None:
    raw = entry.get("path") if isinstance(entry, Mapping) else entry
    text = _clean_text(raw)
    return text or None


def _artifact_expected_hash(
    handoff: Mapping[str, Any], key: str, entry: Any
) -> str | None:
    if isinstance(entry, Mapping):
        raw = entry.get("sha256")
        if isinstance(raw, str) and raw:
            return raw.lower()
    hashes = handoff.get("artifact_hashes")
    if isinstance(hashes, Mapping):
        for candidate in (key, key.removesuffix(".csv"), key.removesuffix(".npz")):
            raw = hashes.get(candidate)
            if isinstance(raw, str) and raw:
                return raw.lower()
    return None


def _resolve_artifact(
    handoff: Mapping[str, Any],
    handoff_path: Path,
    names: Sequence[str],
    *,
    required: bool = True,
) -> tuple[Path | None, str | None]:
    found = _artifact_entry(handoff, names)
    if found is None:
        if required:
            raise ActiveTrainingError(
                f"handoff is missing required artifact reference: {names[0]}"
            )
        return None, None
    key, entry = found
    raw_path = _artifact_path_value(entry)
    if raw_path is None:
        if required or (isinstance(entry, Mapping) and entry.get("required") is True):
            raise ActiveTrainingError(f"handoff artifact has no path: {key}")
        return None, None
    path_value = Path(raw_path).expanduser()
    root = _handoff_root(handoff, handoff_path)
    candidates = (
        [path_value]
        if path_value.is_absolute()
        else [
            root / path_value,
            handoff_path.parent / path_value,
            Path.cwd() / path_value,
        ]
    )
    path = next((item for item in candidates if item.exists() and item.is_file()), None)
    if path is None:
        raise ActiveTrainingError(f"handoff artifact is missing: {key}: {raw_path}")
    expected = _artifact_expected_hash(handoff, key, entry)
    if expected is None or len(expected) != 64:
        raise ActiveTrainingError(f"handoff artifact lacks a valid SHA-256: {key}")
    actual = _sha256_file(path)
    if actual.lower() != expected.lower():
        raise ActiveTrainingError(f"handoff artifact hash mismatch: {key}")
    return path.resolve(), actual


def _resolve_baseline_checkpoint(
    handoff: Mapping[str, Any], handoff_path: Path
) -> tuple[Path, str]:
    baseline = handoff.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ActiveTrainingError("stage-one handoff lacks baseline identity")
    required = ("checkpoint_path", "checkpoint_sha256", "registry_sha256", "active")
    missing = [key for key in required if key not in baseline]
    if missing:
        raise ActiveTrainingError(f"baseline identity is incomplete: {missing}")
    checkpoint_raw = _clean_text(baseline.get("checkpoint_path"))
    expected = _clean_text(baseline.get("checkpoint_sha256")).lower()
    registry_expected = _clean_text(baseline.get("registry_sha256")).lower()
    if (
        not checkpoint_raw
        or len(expected) != 64
        or len(registry_expected) != 64
        or not isinstance(baseline.get("active"), Mapping)
    ):
        raise ActiveTrainingError(
            "baseline identity has invalid path, hashes, or active record"
        )
    root = _handoff_root(handoff, handoff_path)
    raw_path = Path(checkpoint_raw).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [root / raw_path, handoff_path.parent / raw_path, Path.cwd() / raw_path]
    )
    checkpoint = next(
        (item for item in candidates if item.exists() and item.is_file()), None
    )
    if checkpoint is None:
        raise ActiveTrainingError(f"baseline checkpoint is missing: {checkpoint_raw}")
    actual = _sha256_file(checkpoint)
    if actual.lower() != expected:
        raise ActiveTrainingError("baseline checkpoint hash mismatch")
    return checkpoint.resolve(), actual


def _bind_registry_active_checkpoint(
    registry_path: Path,
    active: Mapping[str, Any],
    baseline_checkpoint: Path,
    baseline_hash: str,
) -> None:
    """Require the handoff baseline to be the registry's active checkpoint.

    ``active["checkpoint"]`` follows the registry convention: absolute paths
    are used as-is; relative paths resolve against the registry directory, then
    the working directory. The resolved file must be the handoff baseline file,
    its bytes must hash to the baseline hash, and any recorded
    ``active["sha256"]`` must equal that hash.
    """

    raw = _clean_text(active.get("checkpoint"))
    if not raw:
        raise ActiveTrainingError("registry-active checkpoint is not declared")
    raw_path = Path(raw).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [registry_path.parent / raw_path, Path.cwd() / raw_path]
    )
    active_checkpoint = next(
        (item for item in candidates if item.exists() and item.is_file()), None
    )
    if active_checkpoint is None:
        raise ActiveTrainingError(f"registry-active checkpoint is missing: {raw}")
    active_hash = _sha256_file(active_checkpoint).lower()
    recorded = active.get("sha256")
    if recorded is not None and _clean_text(recorded).lower() != active_hash:
        raise ActiveTrainingError("registry-active checkpoint hash mismatch")
    if (
        active_checkpoint.resolve() != baseline_checkpoint.resolve()
        or active_hash != baseline_hash.lower()
    ):
        raise ActiveTrainingError(
            "baseline checkpoint is not the registry-active checkpoint"
        )


def _validate_split_registry_binding(
    splits_path: Path, registry_path: Path
) -> None:
    """Fail closed when handoff splits disagree with their registry snapshot."""
    registry = _load_json(registry_path, "geographic split registry snapshot")
    regions = registry.get("regions")
    if not isinstance(regions, Mapping):
        raise ActiveTrainingError(
            "geographic split registry snapshot lacks regions mapping"
        )
    splits = _load_csv(splits_path, "geographic splits")
    for column in ("geographic_block", "region", "split"):
        if column not in splits.columns:
            raise ActiveTrainingError(
                f"geographic splits lacks registry column: {column}"
            )
    for row in splits.to_dict(orient="records"):
        region = str(row["region"])
        block = str(row["geographic_block"])
        assignments = regions.get(region)
        assigned = (
            assignments.get(block) if isinstance(assignments, Mapping) else None
        )
        if not isinstance(assigned, str) or not assigned:
            raise ActiveTrainingError(
                "geographic block missing from registry snapshot: "
                f"{region}/{block}"
            )
        if str(row["split"]) != assigned:
            raise ActiveTrainingError(
                "geographic split disagrees with registry snapshot: "
                f"{region}/{block}: {row['split']} != {assigned}"
            )


def _validate_handoff(
    handoff_path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, tuple[Path, str]]]:
    source = Path(handoff_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise ActiveTrainingError(f"stage-one handoff not found: {source}")
    handoff = _load_json(source, "stage-one handoff")
    if handoff.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ActiveTrainingError("stage-one handoff schema version mismatch")
    if handoff.get("ready_for_stage2") is not True:
        raise ActiveTrainingError("stage-one handoff is not ready_for_stage2")
    readiness = handoff.get("readiness")
    for key in _REQUIRED_READINESS:
        value = handoff.get(key)
        if value is not True and (
            not isinstance(readiness, Mapping) or readiness.get(key) is not True
        ):
            raise ActiveTrainingError(
                f"stage-one handoff readiness is false or missing: {key}"
            )
    artifacts = handoff.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ActiveTrainingError("stage-one handoff lacks artifacts object")

    # Validate every required artifact explicitly declared by stage one.
    for key, entry in artifacts.items():
        if not isinstance(entry, Mapping) or entry.get("required") is not True:
            continue
        _resolve_artifact(handoff, source, (str(key),), required=True)

    resolved: dict[str, tuple[Path, str]] = {}
    for canonical, aliases in {
        "candidate_pool": ("candidate_pool", "candidate_pool.csv"),
        "feature_embeddings": ("feature_embeddings", "feature_embeddings.npz"),
        "annotations": (
            "absolute_annotations",
            "annotations",
            "human_annotations",
            "absolute_annotations.csv",
        ),
        "mixed_labels": (
            "mixed_labels",
            "labels_mixed",
            "mixed_labels.csv",
            "labels_mixed.csv",
        ),
        "splits": ("geographic_splits", "splits", "geographic_splits.csv"),
        "split_registry": (
            "geographic_split_registry",
            "geographic_split_registry.json",
        ),
        "baseline_registry": (
            "baseline_registry",
            "registry",
            "model_registry",
            "model_registry.json",
        ),
    }.items():
        path, digest = _resolve_artifact(handoff, source, aliases, required=True)
        assert path is not None and digest is not None
        resolved[canonical] = (path, digest)
    _validate_split_registry_binding(
        resolved["splits"][0], resolved["split_registry"][0]
    )

    baseline_checkpoint, checkpoint_hash = _resolve_baseline_checkpoint(handoff, source)
    baseline = handoff["baseline"]
    assert isinstance(baseline, Mapping)
    registry_path, registry_hash = resolved["baseline_registry"]
    expected_registry_hash = _clean_text(baseline.get("registry_sha256")).lower()
    if registry_hash.lower() != expected_registry_hash:
        raise ActiveTrainingError("baseline registry hash mismatch")
    registry = _load_json(registry_path, "baseline registry")
    active = registry.get("active")
    if not isinstance(active, Mapping) or dict(active) != dict(baseline["active"]):
        raise ActiveTrainingError("baseline active record does not match registry")
    _bind_registry_active_checkpoint(
        registry_path, active, baseline_checkpoint, checkpoint_hash
    )
    resolved["baseline_checkpoint"] = (baseline_checkpoint, checkpoint_hash)

    leakage = _artifact_entry(handoff, ("leakage_audit", "leakage_audit.json"))
    if leakage is not None:
        path, _ = _resolve_artifact(handoff, source, (leakage[0],), required=True)
        assert path is not None
        audit = _load_json(path, "leakage audit")
        if audit.get("valid") is not True:
            raise ActiveTrainingError("stage-one leakage audit is not valid")
    return handoff, source, resolved


def _read_feature_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = [key for key in _REQUIRED_FEATURE_ARRAYS if key not in loaded]
            if missing:
                raise ActiveTrainingError(
                    f"feature NPZ is missing required arrays: {missing}"
                )
            arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    except ActiveTrainingError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ActiveTrainingError(f"invalid feature NPZ: {path}: {exc}") from exc
    n = len(arrays["embeddings"])
    for key in ("embeddings", "terrain_features", "class_logits", "class_probs"):
        array = arrays[key]
        if array.ndim != 2 or len(array) != n or array.shape[1] <= 0:
            raise ActiveTrainingError(f"feature NPZ array has invalid shape: {key}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ActiveTrainingError(
                f"feature NPZ array is non-finite or non-numeric: {key}"
            )
    row_indices = arrays["row_indices"]
    if row_indices.ndim != 1 or len(row_indices) != n:
        raise ActiveTrainingError("feature NPZ row_indices shape mismatch")
    if not np.issubdtype(row_indices.dtype, np.integer):
        converted = row_indices.astype(np.int64)
        if not np.array_equal(converted, row_indices):
            raise ActiveTrainingError("feature NPZ row_indices must contain integers")
        arrays["row_indices"] = converted
    if len(np.unique(arrays["row_indices"])) != n:
        raise ActiveTrainingError("feature NPZ row_indices contain duplicates")
    return arrays


def _candidate_identity(frame: pd.DataFrame) -> list[str]:
    if "image_path" not in frame.columns:
        raise ActiveTrainingError("candidate pool lacks image_path")
    values = [_clean_text(value) for value in frame["image_path"].tolist()]
    if any(not value for value in values):
        raise ActiveTrainingError("candidate pool contains empty image_path")
    if len(set(values)) != len(values):
        raise ActiveTrainingError("candidate pool contains duplicate image IDs")
    for column in ("source_identity", "tile_identity"):
        if column in frame.columns:
            values_for_column = [_clean_text(value) for value in frame[column].tolist()]
            nonempty = [value for value in values_for_column if value]
            if len(set(nonempty)) != len(nonempty):
                raise ActiveTrainingError(
                    f"candidate pool contains duplicate {column} IDs"
                )
    return values


def _read_filtered_indices(path: Path, candidate: pd.DataFrame) -> list[int]:
    suffix = path.suffix.lower()
    values: Any = None
    if suffix == ".json":
        payload = _load_json(path, "filtered-index")
        for key in ("candidate_row_indices", "row_indices", "indices"):
            if key in payload:
                values = payload[key]
                break
        if values is None and "image_paths" in payload:
            values = {"image_paths": payload["image_paths"]}
    elif suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as arrays:
                for key in ("candidate_row_indices", "row_indices", "indices"):
                    if key in arrays:
                        values = np.asarray(arrays[key]).tolist()
                        break
                if values is None and "image_paths" in arrays:
                    values = {"image_paths": np.asarray(arrays["image_paths"]).tolist()}
        except (OSError, ValueError, TypeError) as exc:
            raise ActiveTrainingError(f"invalid filtered-index NPZ: {path}") from exc
    else:
        frame = _load_csv(path, "filtered-index")
        for key in ("candidate_row_index", "row_index", "index"):
            if key in frame.columns:
                values = frame[key].tolist()
                break
        if values is None and "image_path" in frame.columns:
            values = {"image_paths": frame["image_path"].tolist()}
    if isinstance(values, Mapping):
        raw_paths = values.get("image_paths")
        if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)):
            raise ActiveTrainingError(
                "filtered-index image_paths must be an ordered list"
            )
        positions = {
            value: index for index, value in enumerate(_candidate_identity(candidate))
        }
        try:
            indices = [positions[_clean_text(value)] for value in raw_paths]
        except KeyError as exc:
            raise ActiveTrainingError(
                "filtered-index references unknown image_path"
            ) from exc
    else:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ActiveTrainingError("filtered-index must contain ordered row indices")
        indices = []
        for value in values:
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise ActiveTrainingError(
                    "filtered-index row index is not an integer"
                ) from exc
            if number != value:
                raise ActiveTrainingError("filtered-index row index is not an integer")
            indices.append(number)
    if not indices or len(set(indices)) != len(indices):
        raise ActiveTrainingError("filtered-index must contain unique non-empty rows")
    if min(indices) < 0 or max(indices) >= len(candidate):
        raise ActiveTrainingError("filtered-index row is outside candidate pool")
    return indices


def _split_values(frame: pd.DataFrame) -> dict[str, str]:
    if "image_path" not in frame.columns or "split" not in frame.columns:
        raise ActiveTrainingError(
            "geographic splits require image_path and split columns"
        )
    paths = [_clean_text(value) for value in frame["image_path"].tolist()]
    if any(not path for path in paths):
        raise ActiveTrainingError("geographic splits contain empty image_path")
    if len(set(paths)) != len(paths):
        raise ActiveTrainingError("geographic splits contain duplicate image IDs")
    result: dict[str, str] = {}
    for image_path, raw_split in zip(paths, frame["split"].tolist(), strict=True):
        split = _clean_text(raw_split).lower()
        if split not in _ALLOWED_SPLITS:
            raise ActiveTrainingError(f"invalid geographic split: {raw_split}")
        result[image_path] = "val" if split == "validation" else split
    return result


def _physical_tile(row: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Return a row's region-independent physical ``(z, x, y)`` tile.

    Integer ``z``/``x``/``y`` columns are authoritative. A supplied
    region-qualified ``tile_identity`` (``<region>/z<z>/x<x>/y<y>``) only
    contributes its trailing coordinates and must agree with the columns; the
    region prefix never distinguishes physical tiles.
    """

    texts = [_clean_text(row.get(name)) for name in ("z", "x", "y")]
    coords: tuple[int, int, int] | None = None
    if all(texts):
        try:
            numbers = [float(text) for text in texts]
        except ValueError as exc:
            raise ActiveTrainingError("candidate z/x/y must be integers") from exc
        if any(not math.isfinite(value) or value != int(value) for value in numbers):
            raise ActiveTrainingError("candidate z/x/y must be integers")
        coords = (int(numbers[0]), int(numbers[1]), int(numbers[2]))
    match = _TILE_IDENTITY_COORDS.search(_clean_text(row.get("tile_identity")))
    parsed = (
        (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if match
        else None
    )
    if coords is not None and parsed is not None and coords != parsed:
        raise ActiveTrainingError(
            f"tile_identity disagrees with z/x/y for {_clean_text(row.get('image_path'))}"
        )
    return coords if coords is not None else parsed


def _validate_geographic_leakage(
    candidate: pd.DataFrame, selected: Sequence[int], splits: Mapping[str, str]
) -> None:
    """Reject split leakage by physical tile, adjacency, or declared group.

    Physical identity ignores region names, so the same or an adjacent z/x/y
    tile cannot appear in different splits under different regions. Rows
    without coordinates fall back to their exact ``tile_identity`` text.
    """

    rows = candidate.iloc[list(selected)]
    coords: dict[tuple[int, int, int], str] = {}
    by_identity: dict[str, str] = {}
    for _, row in rows.iterrows():
        image_path = _clean_text(row.get("image_path"))
        split = splits[image_path]
        physical = _physical_tile(row)
        if physical is not None:
            prior = coords.get(physical)
            if prior is not None and prior != split:
                raise ActiveTrainingError(
                    "geographic leakage across splits for physical tile "
                    f"z{physical[0]}/x{physical[1]}/y{physical[2]}"
                )
            coords[physical] = split
            continue
        identity = _clean_text(row.get("tile_identity"))
        if identity:
            prior = by_identity.get(identity)
            if prior is not None and prior != split:
                raise ActiveTrainingError(
                    f"geographic leakage across splits for tile:{identity}"
                )
            by_identity[identity] = split
    for column in ("geographic_group", "geo_group", "block", "group"):
        if column not in rows.columns:
            continue
        groups: dict[str, str] = {}
        for _, row in rows.iterrows():
            group = _clean_text(row.get(column))
            if not group:
                continue
            image_path = _clean_text(row.get("image_path"))
            split = splits[image_path]
            prior = groups.get(group)
            if prior is not None and prior != split:
                raise ActiveTrainingError(
                    f"geographic leakage across splits for {column}={group}"
                )
            groups[group] = split
    for (z, x, y), split in coords.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                adjacent_split = coords.get((z, x + dx, y + dy))
                if adjacent_split is not None and adjacent_split != split:
                    raise ActiveTrainingError(
                        "adjacent geographic tiles leak across splits"
                    )


def _deduplicate_physical_tiles(
    candidate: pd.DataFrame, selected: Sequence[int], labels: Sequence[str]
) -> list[int]:
    """Return positions keeping one row per physical tile.

    Leakage validation has already proven duplicates share a split. A human
    label wins over a weak one; otherwise the first row in prepared order is
    kept. Rows without physical coordinates are always kept.
    """

    tiles = [_physical_tile(candidate.iloc[index]) for index in selected]
    chosen: dict[tuple[int, int, int], int] = {}
    for position, physical in enumerate(tiles):
        if physical is None:
            continue
        prior = chosen.get(physical)
        if prior is None or (labels[prior] != "human" and labels[position] == "human"):
            chosen[physical] = position
    return [
        position
        for position, physical in enumerate(tiles)
        if physical is None or chosen[physical] == position
    ]


def _validate_splits_for_selected(
    split_frame: pd.DataFrame, candidate: pd.DataFrame, selected: Sequence[int]
) -> dict[str, str]:
    values = _split_values(split_frame)
    selected_paths = [
        _clean_text(candidate.iloc[index]["image_path"]) for index in selected
    ]
    if set(values) != set(selected_paths):
        missing = sorted(set(selected_paths) - set(values))
        extra = sorted(set(values) - set(selected_paths))
        raise ActiveTrainingError(
            f"geographic splits do not exactly match prepared identities; missing={missing}, extra={extra}"
        )
    counts = {
        split: sum(value == split for value in values.values())
        for split in ("train", "val", "test")
    }
    if any(counts[split] == 0 for split in counts):
        raise ActiveTrainingError(
            "geographic splits must contain train, val, and test rows"
        )
    _validate_geographic_leakage(candidate, selected, values)
    return values


def _annotation_labels(
    annotation: pd.DataFrame, candidate_paths: Sequence[str]
) -> dict[str, float | None]:
    if (
        "image_path" not in annotation.columns
        or "scenic_human" not in annotation.columns
    ):
        raise ActiveTrainingError(
            "human annotations require image_path and scenic_human"
        )
    paths = [_clean_text(value) for value in annotation["image_path"].tolist()]
    if any(not path for path in paths):
        raise ActiveTrainingError("human annotations contain empty image_path")
    unknown = sorted(set(paths) - set(candidate_paths))
    if unknown:
        raise ActiveTrainingError(
            f"human annotations reference unknown image IDs: {unknown}"
        )
    grouped: dict[str, list[float]] = {}
    for _, row in annotation.iterrows():
        image_path = _clean_text(row["image_path"])
        skipped = (
            _as_bool(row.get("skip", False)) if "skip" in annotation.columns else False
        )
        value = row.get("scenic_human")
        if skipped or value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        score = _finite_float(value, "scenic_human")
        if not 0 <= score <= 10:
            raise ActiveTrainingError("scenic_human must be in [0, 10]")
        grouped.setdefault(image_path, []).append(score)
    return {
        image_path: float(sum(scores) / len(scores))
        for image_path, scores in grouped.items()
    }


def _mixed_label_targets(
    mixed_path: Path,
    annotation_path: Path,
    candidate: pd.DataFrame,
    selected: Sequence[int],
    candidate_paths: Sequence[str],
    all_candidate_paths: Sequence[str],
) -> tuple[list[float], list[float], list[str]]:
    mixed = _load_csv(mixed_path, "mixed labels")
    required_cols = {"image_path", "scenic_score", "label_source"}
    missing_cols = required_cols - set(mixed.columns)
    if missing_cols:
        raise ActiveTrainingError(
            f"mixed labels CSV missing required columns: {sorted(missing_cols)}"
        )

    raw_mixed_paths = mixed["image_path"].tolist()
    cleaned_mixed_paths = [_clean_text(p) for p in raw_mixed_paths]
    if any(not p for p in cleaned_mixed_paths):
        raise ActiveTrainingError("mixed labels contain empty image_path")
    if len(cleaned_mixed_paths) != len(set(cleaned_mixed_paths)):
        raise ActiveTrainingError(
            "mixed labels contain duplicate image_path identities"
        )

    mixed_path_set = set(cleaned_mixed_paths)
    all_candidate_set = set(all_candidate_paths)
    unknown = sorted(mixed_path_set - all_candidate_set)
    if unknown:
        raise ActiveTrainingError(
            f"mixed labels reference unknown image IDs: {unknown}"
        )

    candidate_selected_set = set(candidate_paths)
    missing_candidates = sorted(candidate_selected_set - mixed_path_set)
    if missing_candidates:
        raise ActiveTrainingError(
            f"mixed labels missing candidate image IDs: {missing_candidates}"
        )

    annotation = _load_csv(annotation_path, "human annotations")
    annotation = annotation.loc[
        annotation["image_path"].astype(str).isin(candidate_paths)
    ].copy()
    current_human_labels = _annotation_labels(annotation, candidate_paths)

    mixed_rows: dict[str, dict[str, Any]] = {}
    for idx, path in enumerate(cleaned_mixed_paths):
        mixed_rows[path] = mixed.iloc[idx].to_dict()

    targets: list[float] = []
    weights: list[float] = []
    labels: list[str] = []

    for selected_position, image_path in enumerate(candidate_paths):
        candidate_row = candidate.iloc[selected[selected_position]]
        weak_candidate = _finite_float(
            candidate_row.get("scenic_score"), "candidate scenic_score"
        )
        if not 0 <= weak_candidate <= 10:
            raise ActiveTrainingError("candidate scenic_score must be in [0, 10]")

        mixed_row = mixed_rows[image_path]
        mixed_score = _finite_float(
            mixed_row.get("scenic_score"), "mixed labels scenic_score"
        )
        if not 0 <= mixed_score <= 10:
            raise ActiveTrainingError("mixed labels scenic_score must be in [0, 10]")

        label_source_raw = _clean_text(mixed_row.get("label_source")).lower()

        human_score_val: float | None = None
        for human_col in ("scenic_human", "scenic_human_mean"):
            if human_col in mixed_row:
                raw_sh = mixed_row.get(human_col)
                if (
                    raw_sh is not None
                    and not (isinstance(raw_sh, float) and math.isnan(raw_sh))
                    and _clean_text(raw_sh) != ""
                ):
                    sh_parsed = _finite_float(raw_sh, f"mixed labels {human_col}")
                    if not 0 <= sh_parsed <= 10:
                        raise ActiveTrainingError(f"{human_col} must be in [0, 10]")
                    human_score_val = sh_parsed
                    break

        is_human_in_mixed = (
            label_source_raw in {"human", "human_override", "human_annotation"}
            or label_source_raw.startswith("human")
            or human_score_val is not None
        )

        current_human = current_human_labels.get(image_path)

        if current_human is not None:
            targets.append(float(current_human))
            weights.append(4.0)
            labels.append("human")
        elif is_human_in_mixed:
            score = human_score_val if human_score_val is not None else mixed_score
            targets.append(float(score))
            weights.append(4.0)
            labels.append("human")
        else:
            if mixed_score != weak_candidate:
                raise ActiveTrainingError(
                    f"mixed labels weak scenic_score mismatch against candidate pool for {image_path}: {mixed_score} != {weak_candidate}"
                )
            targets.append(float(weak_candidate))
            weights.append(1.0)
            labels.append("weak")

    return targets, weights, labels


def _candidate_embedding_rows(
    handoff: Mapping[str, Any],
    handoff_path: Path,
    candidate: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[int], list[int]]:
    n_features = len(arrays["embeddings"])
    row_indices = np.asarray(arrays["row_indices"], dtype=np.int64)
    if np.any(row_indices < 0) or np.any(row_indices >= len(candidate)):
        raise ActiveTrainingError(
            "feature NPZ row_indices reference outside candidate pool"
        )
    candidate_values = candidate.get("embedding_row_index")
    if candidate_values is None:
        raise ActiveTrainingError("candidate pool lacks embedding_row_index")
    mapped: dict[int, int] = {}
    used_feature_rows: set[int] = set()
    for candidate_row, raw in enumerate(candidate_values.tolist()):
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ActiveTrainingError(
                "candidate embedding_row_index must be integer"
            ) from exc
        if value != raw or value < 0 or value >= n_features:
            continue
        if value in used_feature_rows or int(row_indices[value]) != candidate_row:
            continue
        mapped[candidate_row] = value
        used_feature_rows.add(value)
    found = _artifact_entry(handoff, _FILTERED_INDEX_NAMES)
    if found is not None:
        filtered_path, _ = _resolve_artifact(
            handoff, handoff_path, (found[0],), required=True
        )
        assert filtered_path is not None
        selected = _read_filtered_indices(filtered_path, candidate)
        if any(index not in mapped for index in selected):
            raise ActiveTrainingError(
                "filtered-index references rows without aligned feature embeddings"
            )
        return selected, [mapped[index] for index in selected]
    selected = list(range(len(candidate)))
    if len(mapped) == len(candidate) and set(mapped) == set(row_indices.tolist()):
        return selected, [mapped[index] for index in selected]
    raise ActiveTrainingError(
        "candidate pool and feature NPZ are filtered without a declared "
        "filtered-index artifact"
    )


def prepare_active_dataset(
    handoff_path: str | Path,
    output_path: str | Path,
    supplemental_benchmark_path: str | Path | None = None,
    supplemental_benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a stage-one handoff and materialize its ordered training NPZ.

    When ``supplemental_benchmark_path`` and ``supplemental_benchmark_sha256``
    are both supplied, the canonical supplemental benchmark CSV is hash-verified
    before use and cross-validated: unique nonempty identities, only
    train/val/test split names, identities known to the candidate pool, split
    assignments exactly equal to the immutable geographic splits, finite human
    means in [0, 10], and nonempty train rows. Only ``split=train`` identities
    merge into the output targets with base weight 4 and ``label_source``
    ``"human"``; val/test output rows are never rewritten.

    Geographic leakage is checked on region-independent physical z/x/y tiles,
    including adjacent tiles. Rows sharing one physical tile (necessarily in
    the same split) collapse to one prepared row, preferring a human label and
    then prepared order; ``dropped_physical_duplicates`` counts the removals.
    """

    handoff, source, resolved = _validate_handoff(handoff_path)
    candidate_path, candidate_hash = resolved["candidate_pool"]
    feature_path, feature_hash = resolved["feature_embeddings"]
    annotation_path, annotation_hash = resolved["annotations"]
    mixed_path, mixed_hash = resolved["mixed_labels"]
    split_path, split_hash = resolved["splits"]
    candidate = _load_csv(candidate_path, "candidate pool")
    all_candidate_paths = _candidate_identity(candidate)
    arrays = _read_feature_arrays(feature_path)
    selected, feature_rows = _candidate_embedding_rows(
        handoff, source, candidate, arrays
    )
    candidate_paths = [all_candidate_paths[index] for index in selected]
    split_frame = _load_csv(split_path, "geographic splits")
    declared_splits = _split_values(split_frame)
    unknown_split_paths = sorted(set(declared_splits) - set(all_candidate_paths))
    if unknown_split_paths:
        raise ActiveTrainingError(
            "geographic splits reference identities absent from the candidate pool: "
            f"{unknown_split_paths}"
        )
    supplemental: dict[str, Any] | None = None
    if (
        supplemental_benchmark_path is not None
        or supplemental_benchmark_sha256 is not None
    ):
        if supplemental_benchmark_path is None or supplemental_benchmark_sha256 is None:
            raise ActiveTrainingError(
                "supplemental benchmark path and expected SHA must be provided together"
            )
        benchmark_path = Path(supplemental_benchmark_path).expanduser().resolve()
        if not benchmark_path.exists() or not benchmark_path.is_file():
            raise ActiveTrainingError(
                f"supplemental benchmark not found: {benchmark_path}"
            )
        expected_benchmark_sha = _optional_sha256(
            supplemental_benchmark_sha256, "supplemental benchmark expected SHA"
        )
        assert expected_benchmark_sha is not None
        actual_benchmark_sha = _sha256_file(benchmark_path).lower()
        if actual_benchmark_sha != expected_benchmark_sha:
            raise ActiveTrainingError("supplemental benchmark hash mismatch")
        benchmark = _load_csv(benchmark_path, "supplemental benchmark")
        required_benchmark = {"image_path", "scenic_human_mean", "split"}
        missing_benchmark = required_benchmark - set(benchmark.columns)
        if missing_benchmark:
            raise ActiveTrainingError(
                "supplemental benchmark CSV missing required columns: "
                f"{sorted(missing_benchmark)}"
            )
        benchmark_paths = [
            _clean_text(value) for value in benchmark["image_path"].tolist()
        ]
        if any(not value for value in benchmark_paths):
            raise ActiveTrainingError(
                "supplemental benchmark contains empty image_path"
            )
        if len(set(benchmark_paths)) != len(benchmark_paths):
            raise ActiveTrainingError(
                "supplemental benchmark contains duplicate image IDs"
            )
        benchmark_splits: dict[str, str] = {}
        for image_path, raw_split in zip(
            benchmark_paths, benchmark["split"].tolist(), strict=True
        ):
            split = _clean_text(raw_split).lower()
            if split not in _ALLOWED_SPLITS:
                raise ActiveTrainingError(
                    f"invalid supplemental benchmark split: {raw_split}"
                )
            benchmark_splits[image_path] = "val" if split == "validation" else split
        unknown_benchmark = sorted(set(benchmark_paths) - set(all_candidate_paths))
        if unknown_benchmark:
            raise ActiveTrainingError(
                "supplemental benchmark references unknown image IDs: "
                f"{unknown_benchmark}"
            )
        for image_path in benchmark_paths:
            declared = declared_splits.get(image_path)
            if declared is None:
                raise ActiveTrainingError(
                    "supplemental benchmark identity is absent from the geographic "
                    f"splits: {image_path}"
                )
            if declared != benchmark_splits[image_path]:
                raise ActiveTrainingError(
                    f"supplemental benchmark split mismatch for {image_path}: "
                    f"{benchmark_splits[image_path]} != {declared}"
                )
        benchmark_means: dict[str, float] = {}
        for image_path, raw_mean in zip(
            benchmark_paths, benchmark["scenic_human_mean"].tolist(), strict=True
        ):
            mean = _finite_float(raw_mean, "supplemental benchmark scenic_human_mean")
            if not 0 <= mean <= 10:
                raise ActiveTrainingError(
                    "supplemental benchmark scenic_human_mean must be in [0, 10]"
                )
            benchmark_means[image_path] = mean
        benchmark_train_paths = [
            image_path
            for image_path in benchmark_paths
            if benchmark_splits[image_path] == "train"
        ]
        if not benchmark_train_paths:
            raise ActiveTrainingError("supplemental benchmark must contain train rows")
        supplemental = {
            "path": benchmark_path,
            "actual": actual_benchmark_sha,
            "expected": expected_benchmark_sha,
            "means": benchmark_means,
            "train_paths": benchmark_train_paths,
            "rows": len(benchmark_paths),
            "train": len(benchmark_train_paths),
            "val": sum(
                benchmark_splits[image_path] == "val" for image_path in benchmark_paths
            ),
            "test": sum(
                benchmark_splits[image_path] == "test" for image_path in benchmark_paths
            ),
        }
    retained_positions = [
        position
        for position, image_path in enumerate(candidate_paths)
        if image_path in declared_splits
    ]
    if not retained_positions:
        raise ActiveTrainingError(
            "geographic splits have no identities in the filtered candidate set"
        )
    dropped_without_split = len(selected) - len(retained_positions)
    selected = [selected[position] for position in retained_positions]
    feature_rows = [feature_rows[position] for position in retained_positions]
    candidate_paths = [candidate_paths[position] for position in retained_positions]
    retained_path_set = set(candidate_paths)
    split_frame = split_frame.loc[
        split_frame["image_path"].map(_clean_text).isin(retained_path_set)
    ].copy()
    split_values = _validate_splits_for_selected(split_frame, candidate, selected)
    targets, weights, labels = _mixed_label_targets(
        mixed_path,
        annotation_path,
        candidate,
        selected,
        candidate_paths,
        all_candidate_paths,
    )
    changed_rows = 0
    if supplemental is not None:
        positions = {path: index for index, path in enumerate(candidate_paths)}
        missing_train = sorted(
            path for path in supplemental["train_paths"] if path not in positions
        )
        if missing_train:
            raise ActiveTrainingError(
                "supplemental benchmark train identities absent from prepared "
                f"dataset: {missing_train}"
            )
        for image_path in supplemental["train_paths"]:
            index = positions[image_path]
            if split_values[image_path] != "train":
                raise ActiveTrainingError(
                    f"supplemental benchmark identity is not a train row: {image_path}"
                )
            targets[index] = supplemental["means"][image_path]
            weights[index] = 4.0
            labels[index] = "human"
            changed_rows += 1
    kept_positions = _deduplicate_physical_tiles(candidate, selected, labels)
    dropped_physical_duplicates = len(selected) - len(kept_positions)
    selected = [selected[position] for position in kept_positions]
    feature_rows = [feature_rows[position] for position in kept_positions]
    candidate_paths = [candidate_paths[position] for position in kept_positions]
    targets = [targets[position] for position in kept_positions]
    weights = [weights[position] for position in kept_positions]
    labels = [labels[position] for position in kept_positions]
    regions = [
        _clean_text(candidate.iloc[index].get("region")) or "unknown"
        for index in selected
    ]
    selected_feature_rows = np.asarray(feature_rows, dtype=np.int64)
    output = Path(output_path).expanduser().resolve()
    filtered_index_path = output.with_suffix(".filtered_index.csv")
    filtered_labels_path = output.with_suffix(".filtered_labels.csv")
    filtered_index = pd.DataFrame(
        {
            "candidate_row_index": selected,
            "feature_row_index": feature_rows,
            "image_path": candidate_paths,
            "split": [split_values[path] for path in candidate_paths],
        }
    )
    filtered_labels = filtered_index.loc[
        :, ["candidate_row_index", "image_path", "split"]
    ].copy()
    filtered_labels["scenic_score"] = targets
    filtered_labels["sample_weight"] = weights
    filtered_labels["label_source"] = labels
    _atomic_csv(filtered_index_path, filtered_index)
    _atomic_csv(filtered_labels_path, filtered_labels)
    output_arrays = {
        "vit_embeddings": np.asarray(arrays["embeddings"])[
            selected_feature_rows
        ].astype(np.float32, copy=False),
        "terrain_features": np.asarray(arrays["terrain_features"])[
            selected_feature_rows
        ].astype(np.float32, copy=False),
        "class_logits": np.asarray(arrays["class_logits"])[
            selected_feature_rows
        ].astype(np.float32, copy=False),
        "scenic_scores": np.asarray(targets, dtype=np.float32),
        "sample_weights": np.asarray(weights, dtype=np.float32),
        "image_paths": np.asarray(candidate_paths, dtype=str),
        "label_sources": np.asarray(labels, dtype=str),
        "regions": np.asarray(regions, dtype=str),
        "splits": np.asarray(
            [split_values[path] for path in candidate_paths], dtype=str
        ),
    }
    _atomic_npz(output, output_arrays)
    dataset_hash = _sha256_file(output)
    filtered_index_hash = _sha256_file(filtered_index_path)
    filtered_labels_hash = _sha256_file(filtered_labels_path)
    counts = {
        "rows": len(candidate_paths),
        "train": sum(split_values[path] == "train" for path in candidate_paths),
        "val": sum(split_values[path] == "val" for path in candidate_paths),
        "test": sum(split_values[path] == "test" for path in candidate_paths),
        "human": labels.count("human"),
        "weak": labels.count("weak"),
    }
    return _jsonable(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "state": "prepared",
            "dataset_path": str(output),
            "dataset_sha256": dataset_hash,
            "split_path": str(filtered_index_path),
            "split_sha256": filtered_index_hash,
            "source_split_path": str(split_path),
            "source_split_sha256": split_hash,
            "filtered_index_path": str(filtered_index_path),
            "filtered_index_sha256": filtered_index_hash,
            "filtered_labels_path": str(filtered_labels_path),
            "filtered_labels_sha256": filtered_labels_hash,
            "handoff_path": str(Path(handoff_path).expanduser().resolve()),
            "handoff_sha256": _sha256_file(source),
            "source": handoff.get("source"),
            "preprocessing": handoff.get("preprocessing"),
            "grid": handoff.get("grid"),
            "mosaic": handoff.get("mosaic"),
            "datum": handoff.get("datum"),
            "nodata": handoff.get("nodata"),
            "QC": handoff.get("QC"),
            "object_hashes": handoff.get("object_hashes"),
            "counts": counts,
            "dropped_without_split": dropped_without_split,
            "dropped_physical_duplicates": dropped_physical_duplicates,
            "changed_rows": changed_rows,
            "supplemental_benchmark": (
                {
                    "path": str(supplemental["path"]),
                    "sha256": supplemental["actual"],
                    "expected_sha256": supplemental["expected"],
                    "rows": supplemental["rows"],
                    "train": supplemental["train"],
                    "val": supplemental["val"],
                    "test": supplemental["test"],
                }
                if supplemental is not None
                else None
            ),
            "sample_count": counts["rows"],
            "hashes": {
                "candidate_pool": candidate_hash,
                "feature_embeddings": feature_hash,
                "annotations": annotation_hash,
                "mixed_labels": mixed_hash,
                "splits": filtered_index_hash,
                "source_splits": split_hash,
                "dataset": dataset_hash,
                "filtered_index": filtered_index_hash,
                "filtered_labels": filtered_labels_hash,
                **(
                    {"supplemental_benchmark": supplemental["actual"]}
                    if supplemental is not None
                    else {}
                ),
            },
            "ordered_image_paths_sha256": _sha256_bytes(
                "\n".join(candidate_paths).encode("utf-8")
            ),
            "label_sources": {"human": counts["human"], "weak": counts["weak"]},
            "deterministic_limitations": [
                "Training order is deterministic and explicit; accelerator kernels may remain nondeterministic.",
                "The compressed NPZ container is content-addressed after materialization.",
            ],
        }
    )


def _load_prepared_dataset(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            required = (
                "vit_embeddings",
                "terrain_features",
                "class_logits",
                "scenic_scores",
                "sample_weights",
                "image_paths",
                "label_sources",
                "regions",
                "splits",
            )
            missing = [key for key in required if key not in loaded]
            if missing:
                raise ActiveTrainingError(
                    f"prepared dataset is missing arrays: {missing}"
                )
            arrays = {key: np.asarray(loaded[key]).copy() for key in required}
    except ActiveTrainingError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ActiveTrainingError(f"invalid prepared dataset: {path}: {exc}") from exc
    n = len(arrays["scenic_scores"])
    for key in ("vit_embeddings", "terrain_features", "class_logits"):
        if arrays[key].ndim != 2 or len(arrays[key]) != n or arrays[key].shape[1] <= 0:
            raise ActiveTrainingError(f"prepared dataset array shape mismatch: {key}")
        if (
            not np.issubdtype(arrays[key].dtype, np.number)
            or not np.isfinite(arrays[key]).all()
        ):
            raise ActiveTrainingError(
                f"prepared dataset has non-finite features: {key}"
            )
    scores = np.asarray(arrays["scenic_scores"], dtype=np.float32).reshape(-1)
    weights = np.asarray(arrays["sample_weights"], dtype=np.float32).reshape(-1)
    if len(scores) != n or len(weights) != n or not np.isfinite(scores).all():
        raise ActiveTrainingError(
            "prepared dataset scenic_scores/sample_weights length or finiteness mismatch"
        )
    if np.any(scores < 0) or np.any(scores > 10):
        raise ActiveTrainingError("prepared dataset scenic_scores must be in [0, 10]")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ActiveTrainingError(
            "prepared dataset sample_weights must be finite and strictly positive"
        )
    paths = [_clean_text(value) for value in arrays["image_paths"].reshape(-1).tolist()]
    if len(paths) != n or any(not value for value in paths) or len(set(paths)) != n:
        raise ActiveTrainingError(
            "prepared dataset image_paths must be unique and aligned"
        )
    sources = [
        _clean_text(value) for value in arrays["label_sources"].reshape(-1).tolist()
    ]
    if len(sources) != n or any(value not in {"human", "weak"} for value in sources):
        raise ActiveTrainingError("prepared dataset label_sources are invalid")
    regions = [_clean_text(value) for value in arrays["regions"].reshape(-1).tolist()]
    if len(regions) != n or any(not value for value in regions):
        raise ActiveTrainingError("prepared dataset regions are invalid")
    splits = [
        _clean_text(value).lower() for value in arrays["splits"].reshape(-1).tolist()
    ]
    if len(splits) != n or any(not value for value in splits):
        raise ActiveTrainingError("prepared dataset splits are invalid")
    normalized_splits = ["val" if split == "validation" else split for split in splits]
    if any(split not in {"train", "val", "test"} for split in normalized_splits):
        raise ActiveTrainingError("prepared dataset splits are invalid")
    if any(normalized_splits.count(split) == 0 for split in ("train", "val", "test")):
        raise ActiveTrainingError(
            "prepared dataset splits must contain train, val, and test rows"
        )
    arrays["scenic_scores"] = scores
    arrays["sample_weights"] = weights
    arrays["image_paths"] = np.asarray(paths, dtype=str)
    arrays["label_sources"] = np.asarray(sources, dtype=str)
    arrays["regions"] = np.asarray(regions, dtype=str)
    arrays["splits"] = np.asarray(normalized_splits, dtype=str)
    return arrays


def _load_training_splits(
    path: Path, image_paths: Sequence[str], prepared_splits: Sequence[str]
) -> dict[str, np.ndarray]:
    frame = _load_csv(path, "training split")
    values = _split_values(frame)
    expected = set(image_paths)
    if set(values) != expected:
        raise ActiveTrainingError(
            "training split identities do not exactly match prepared dataset"
        )
    if len(values) != len(frame):
        raise ActiveTrainingError("training split contains duplicate image IDs")
    if len(prepared_splits) != len(image_paths):
        raise ActiveTrainingError("prepared dataset splits are not aligned")
    prepared_values = [_clean_text(value).lower() for value in prepared_splits]
    prepared_values = [
        "val" if split == "validation" else split for split in prepared_values
    ]
    if any(split not in {"train", "val", "test"} for split in prepared_values):
        raise ActiveTrainingError("prepared dataset splits are invalid")
    prepared_by_path = dict(zip(image_paths, prepared_values, strict=True))
    if any(values[path] != prepared_by_path[path] for path in image_paths):
        raise ActiveTrainingError(
            "training split assignments do not exactly match prepared dataset"
        )
    positions = {path: index for index, path in enumerate(image_paths)}
    result = {
        split: np.asarray(
            [positions[path] for path in image_paths if values[path] == split],
            dtype=np.int64,
        )
        for split in ("train", "val", "test")
    }
    if any(len(result[split]) == 0 for split in result):
        raise ActiveTrainingError(
            "training split must contain train, val, and test rows"
        )
    return result


def _material_config(
    config: ActiveTrainingConfig, resolved_device: str
) -> dict[str, Any]:
    value = config.as_dict()
    value.pop("max_steps", None)
    value.pop("max_seconds", None)
    value["device"] = resolved_device
    return value


def _config_hash(config: ActiveTrainingConfig, resolved_device: str) -> str:
    payload = json.dumps(
        _material_config(config, resolved_device),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _restore_rng_state(state: Mapping[str, Any], device: str) -> None:
    try:
        restore_rng_state(state, device)
    except ValueError as exc:
        raise ActiveTrainingError(str(exc)) from exc


def _torch_load(path: Path, device: str) -> Mapping[str, Any]:
    try:
        value = load_checkpoint(path, map_location=device)
    except Exception as exc:
        raise ActiveTrainingError(f"invalid checkpoint: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ActiveTrainingError(f"checkpoint is not a mapping: {path}")
    return value


def _checkpoint_payload(
    model: ScenicRegressionModel,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None = None,
    state: str,
    next_epoch: int,
    next_batch: int,
    global_step: int,
    dataset_hash: str,
    split_hash: str,
    config_hash: str,
    architecture: Mapping[str, int],
    counts: Mapping[str, int],
    history: Sequence[Mapping[str, Any]],
    initial_checkpoint_sha256: str | None,
    human_sample_weight_multiplier: float,
    human_only_training: bool,
    device: str,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_state": state,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "next_epoch": int(next_epoch),
        "next_batch": int(next_batch),
        "epoch": int(next_epoch),
        "global_step": int(global_step),
        "dataset_sha256": dataset_hash,
        "split_sha256": split_hash,
        "split_artifact_sha256": split_hash,
        "config_hash": config_hash,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "human_sample_weight_multiplier": float(human_sample_weight_multiplier),
        "human_only_training": bool(human_only_training),
        **{key: int(value) for key, value in architecture.items()},
        "counts": dict(counts),
        "history": list(history),
        "rng_state": capture_rng_state(device),
    }


def _metrics(
    model: ScenicRegressionModel,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    use_sample_weights: bool,
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate in bounded device batches without changing the caller's mode."""
    was_training = model.training
    use_cuda = str(device).startswith("cuda")

    def move(values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(values).float()
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=use_cuda)

    target = np.asarray(arrays["scenic_scores"])[indices].astype(
        np.float32, copy=False
    )
    weight = np.asarray(arrays["sample_weights"])[indices].astype(
        np.float32, copy=False
    )
    prediction = np.empty(len(indices), dtype=np.float32)
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_cuda):
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                vit = move(np.asarray(arrays["vit_embeddings"])[batch_indices])
                terrain = move(np.asarray(arrays["terrain_features"])[batch_indices])
                logits = move(np.asarray(arrays["class_logits"])[batch_indices])
                prediction[start : start + len(batch_indices)] = (
                    model(vit, terrain, logits).float().cpu().numpy().reshape(-1)
                )
    finally:
        model.train(was_training)
    error = prediction - target
    metric_weight = weight if use_sample_weights else np.ones_like(weight)
    denominator = float(metric_weight.sum())
    if denominator <= 0:
        raise ActiveTrainingError("metric denominator is not positive")
    mse = float(np.sum(metric_weight * error * error) / denominator)
    mae = float(np.sum(metric_weight * np.abs(error)) / denominator)
    pearson = (
        float(np.corrcoef(prediction, target)[0, 1])
        if len(prediction) > 1 and np.std(prediction) > 0 and np.std(target) > 0
        else 0.0
    )
    if not math.isfinite(pearson):
        pearson = 0.0
    return {
        "mse": mse,
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "mae": mae,
        "pearson": pearson,
        "count": int(len(indices)),
    }


def _load_completed_summary(
    summary_path: Path,
    candidate_path: Path,
    dataset_hash: str,
    split_hash: str,
    config_hash: str,
    architecture: Mapping[str, int],
    *,
    initial_checkpoint_sha256: str | None,
    human_sample_weight_multiplier: float,
    human_only_training: bool,
) -> dict[str, Any]:
    summary = _load_json(summary_path, "training summary")
    required_summary = {
        "schema_version",
        "state",
        "dataset_sha256",
        "split_sha256",
        "config_hash",
        "candidate_checkpoint_sha256",
        "architecture",
        "human_sample_weight_multiplier",
        "human_only_training",
    }
    missing_summary = sorted(required_summary - set(summary))
    if missing_summary:
        raise ActiveTrainingError(
            f"completed summary is missing required fields: {missing_summary}"
        )
    if summary.get("schema_version") != DATASET_SCHEMA_VERSION or isinstance(
        summary.get("schema_version"), bool
    ):
        raise ActiveTrainingError("completed summary schema version is invalid")
    if summary.get("state") != "completed":
        raise ActiveTrainingError("training summary is not completed")
    if (
        summary.get("dataset_sha256") != dataset_hash
        or summary.get("split_sha256") != split_hash
        or summary.get("config_hash") != config_hash
    ):
        raise ActiveTrainingError(
            "completed summary hashes do not match requested inputs"
        )
    summary_initial = summary.get("initial_checkpoint")
    if initial_checkpoint_sha256 is None:
        if summary_initial is not None:
            raise ActiveTrainingError(
                "completed summary initialization identity is invalid"
            )
    else:
        if (
            not isinstance(summary_initial, Mapping)
            or _optional_sha256(
                summary_initial.get("sha256"),
                "completed summary initial_checkpoint sha256",
            )
            != initial_checkpoint_sha256
        ):
            raise ActiveTrainingError(
                "completed summary initialization identity is invalid"
            )
    summary_multiplier = summary.get("human_sample_weight_multiplier")
    if (
        not isinstance(summary_multiplier, (int, float, np.integer, np.floating))
        or not math.isfinite(float(summary_multiplier))
        or float(summary_multiplier) != float(human_sample_weight_multiplier)
    ):
        raise ActiveTrainingError(
            "completed summary human weight multiplier is invalid"
        )
    recorded_human_only = summary.get("human_only_training")
    if not isinstance(recorded_human_only, bool) or (
        recorded_human_only != human_only_training
    ):
        raise ActiveTrainingError("completed summary human_only_training is invalid")
    summary_architecture = summary.get("architecture")
    if not isinstance(summary_architecture, Mapping) or dict(
        summary_architecture
    ) != dict(architecture):
        raise ActiveTrainingError("completed summary architecture is invalid")
    if not candidate_path.exists():
        raise ActiveTrainingError("completed summary has no candidate checkpoint")
    actual = _sha256_file(candidate_path)
    expected = summary.get("candidate_checkpoint_sha256")
    if not isinstance(expected, str) or actual.lower() != expected.lower():
        raise ActiveTrainingError("completed candidate checkpoint hash mismatch")
    try:
        checkpoint = _torch_load(candidate_path, "cpu")
    except ActiveTrainingError:
        raise
    except Exception as exc:
        raise ActiveTrainingError(
            "completed candidate checkpoint could not be loaded"
        ) from exc
    required_checkpoint = {
        "checkpoint_schema_version",
        "checkpoint_state",
        "model_state_dict",
        "optimizer_state_dict",
        "next_epoch",
        "next_batch",
        "global_step",
        "dataset_sha256",
        "split_sha256",
        "config_hash",
        "rng_state",
        "initial_checkpoint_sha256",
        "human_sample_weight_multiplier",
        "human_only_training",
        *architecture,
    }
    missing_checkpoint = sorted(required_checkpoint - set(checkpoint))
    if missing_checkpoint:
        raise ActiveTrainingError("completed candidate checkpoint is incomplete")
    if checkpoint.get(
        "checkpoint_schema_version"
    ) != CHECKPOINT_SCHEMA_VERSION or isinstance(
        checkpoint.get("checkpoint_schema_version"), bool
    ):
        raise ActiveTrainingError("completed candidate checkpoint schema is invalid")
    if checkpoint.get("checkpoint_state") != "completed":
        raise ActiveTrainingError("completed candidate checkpoint is not completed")
    if (
        checkpoint.get("dataset_sha256") != dataset_hash
        or checkpoint.get("split_sha256") != split_hash
        or checkpoint.get("config_hash") != config_hash
    ):
        raise ActiveTrainingError("completed candidate checkpoint hashes are invalid")
    if (
        _optional_sha256(
            checkpoint.get("initial_checkpoint_sha256"),
            "completed candidate checkpoint initial_checkpoint_sha256",
        )
        != initial_checkpoint_sha256
    ):
        raise ActiveTrainingError(
            "completed candidate checkpoint initialization identity is invalid"
        )
    recorded_multiplier = checkpoint.get("human_sample_weight_multiplier")
    if (
        not isinstance(recorded_multiplier, (int, float, np.integer, np.floating))
        or not math.isfinite(float(recorded_multiplier))
        or float(recorded_multiplier) <= 0
        or float(recorded_multiplier) != float(human_sample_weight_multiplier)
    ):
        raise ActiveTrainingError(
            "completed candidate checkpoint human weight multiplier is invalid"
        )
    recorded_human_only = checkpoint.get("human_only_training")
    if not isinstance(recorded_human_only, bool) or (
        recorded_human_only != human_only_training
    ):
        raise ActiveTrainingError(
            "completed candidate checkpoint human_only_training is invalid"
        )
    for key, expected_dimension in architecture.items():
        value = checkpoint.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or int(value) < 1
            or int(value) != int(expected_dimension)
        ):
            raise ActiveTrainingError(
                f"completed candidate checkpoint architecture is invalid: {key}"
            )
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ActiveTrainingError(
            "completed candidate checkpoint model state is invalid"
        )
    try:
        model = ScenicRegressionModel(**dict(architecture))
        model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise ActiveTrainingError(
            "completed candidate checkpoint model state is invalid"
        ) from exc
    result = dict(summary)
    result["reused"] = True
    return _jsonable(result)


def _resume_counter(
    checkpoint: Mapping[str, Any],
    name: str,
    upper_bound: int | None,
    *,
    inclusive: bool = False,
) -> int:
    value = checkpoint.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ActiveTrainingError(
            f"resume checkpoint {name} must be a non-bool integer"
        )
    parsed = int(value)
    too_large = upper_bound is not None and (
        parsed > upper_bound if inclusive else parsed >= upper_bound
    )
    if parsed < 0 or too_large:
        bound = "<=" if inclusive else "<"
        raise ActiveTrainingError(
            f"resume checkpoint {name} must be nonnegative"
            + (f" and {bound} {upper_bound}" if upper_bound is not None else "")
        )
    return parsed


def _resume_bundle_identity(bundle: Mapping[str, Any]) -> tuple[str, int]:
    """Return the run id and monotonically increasing generation of ``resume.pt``."""

    run_id = bundle.get("run_id")
    generation = bundle.get("resume_generation")
    if (
        not isinstance(run_id, str)
        or not run_id
        or isinstance(generation, (bool, np.bool_))
        or not isinstance(generation, Integral)
        or int(generation) < 1
    ):
        raise ActiveTrainingError("resume checkpoint generation identity is invalid")
    return run_id, int(generation)


def _bundle_summary_record(
    bundle: Mapping[str, Any], state: str
) -> Mapping[str, Any]:
    run_id, generation = _resume_bundle_identity(bundle)
    record = bundle.get("summary_record")
    if (
        not isinstance(record, Mapping)
        or record.get("run_id") != run_id
        or record.get("resume_generation") != generation
        or record.get("state") != state
        or bundle.get("checkpoint_state") != state
    ):
        raise ActiveTrainingError("resume checkpoint summary record is invalid")
    return record


def _publish_summary(
    summary_path: Path, resume_path: Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish the summary sidecar for the generation committed in ``resume_path``.

    ``resume.pt`` is the single commit point. The sidecar is derived from the
    record embedded in that generation plus the hash of the committed file, so
    a crash between the two writes is repaired by republishing on resume.
    """

    resume_sha = _sha256_file(resume_path)
    summary = dict(record)
    hashes = summary.get("hashes")
    summary["hashes"] = {
        **(dict(hashes) if isinstance(hashes, Mapping) else {}),
        "resume_checkpoint": resume_sha,
    }
    summary["resume_checkpoint_sha256"] = resume_sha
    published = _jsonable(summary)
    _atomic_json(summary_path, published)
    return published


def _admit_resume_generation(
    summary_path: Path, resume_path: Path, device: str
) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None]:
    """Load the authoritative resume generation and its matching summary.

    Returns ``(bundle, summary)``. ``summary`` is ``None`` when it is absent,
    belongs to a superseded run, or publishes an older generation of the same
    run (a crash after the generation commit and before sidecar publication).
    When ``resume.pt`` is absent the raw summary is returned for completed
    reuse. A same-run summary that is ahead of the checkpoint, or a failed
    same-run summary, fails closed.
    """

    summary = (
        _load_json(summary_path, "training summary") if summary_path.exists() else None
    )
    if not resume_path.exists():
        return None, summary
    bundle = _torch_load(resume_path, device)
    run_id, generation = _resume_bundle_identity(bundle)
    if summary is None or summary.get("run_id") != run_id:
        return bundle, None
    if summary.get("state") == "failed":
        raise ActiveTrainingError("failed training run is not resumable")
    summary_generation = summary.get("resume_generation")
    if isinstance(summary_generation, (bool, np.bool_)) or not isinstance(
        summary_generation, Integral
    ):
        raise ActiveTrainingError("training summary resume_generation is invalid")
    if int(summary_generation) > generation:
        raise ActiveTrainingError(
            "training summary is ahead of the resume checkpoint generation"
        )
    if int(summary_generation) < generation:
        return bundle, None
    return bundle, summary


def _recover_completed_publication(
    summary_path: Path,
    resume_path: Path,
    candidate_path: Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    record = _bundle_summary_record(bundle, "completed")
    expected = bundle.get("candidate_checkpoint_sha256")
    if (
        not isinstance(expected, str)
        or record.get("candidate_checkpoint_sha256") != expected
        or not candidate_path.exists()
        or _sha256_file(candidate_path) != expected
    ):
        raise ActiveTrainingError(
            "completed resume checkpoint does not match the candidate checkpoint"
        )
    return _publish_summary(summary_path, resume_path, record)


def train_active_model(
    dataset_path: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    config: ActiveTrainingConfig,
    resume: bool = False,
) -> dict[str, Any]:
    """Train the configured bounded loss on fixed geographic splits.

    When ``human_only_training`` is enabled, the train index array is narrowed
    to rows whose ``label_source`` is exactly ``human`` after the fixed splits
    load and before counts, batching, and resume-cursor validation. Validation
    and test membership is never altered, and an empty filtered train set
    fails closed.
    """

    if not isinstance(config, ActiveTrainingConfig):
        raise TypeError("config must be an ActiveTrainingConfig")
    config.validate()
    expected_initial_sha = _optional_sha256(
        config.initial_checkpoint_sha256, "initial_checkpoint_sha256"
    )
    initial_checkpoint_sha256: str | None = None
    initial_evidence_path: str | None = None
    human_weight_multiplier = float(config.human_sample_weight_multiplier)
    dataset = Path(dataset_path).expanduser().resolve()
    split_path = Path(split_csv).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        raise ActiveTrainingError(f"prepared dataset not found: {dataset}")
    if not split_path.exists() or not split_path.is_file():
        raise ActiveTrainingError(f"split CSV not found: {split_path}")
    arrays = _load_prepared_dataset(dataset)
    splits = _load_training_splits(
        split_path, arrays["image_paths"].tolist(), arrays["splits"].tolist()
    )
    human_only_training = bool(config.human_only_training)
    total_train_rows = int(len(splits["train"]))
    if human_only_training:
        human_train_mask = (
            np.asarray(arrays["label_sources"])[splits["train"]] == "human"
        )
        if not human_train_mask.any():
            raise ActiveTrainingError(
                "human_only_training requires at least one human-labeled train row"
            )
        splits["train"] = splits["train"][human_train_mask]
    human_train_rows = int(
        (np.asarray(arrays["label_sources"])[splits["train"]] == "human").sum()
    )
    dataset_hash = _sha256_file(dataset)
    split_hash = _sha256_file(split_path)
    device = resolve_device(config.device)
    config_hash = _config_hash(config, device)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_path, resume_path, summary_path = (
        root / "candidate.pt",
        root / "resume.pt",
        root / "training_summary.json",
    )
    architecture = {
        "vit_dim": int(arrays["vit_embeddings"].shape[1]),
        "terrain_dim": int(arrays["terrain_features"].shape[1]),
        "num_classes": int(arrays["class_logits"].shape[1]),
        "hidden_dim": int(config.hidden_dim),
    }

    resume_bundle: Mapping[str, Any] | None = None
    summary_data: dict[str, Any] | None = None
    if resume:
        resume_bundle, summary_data = _admit_resume_generation(
            summary_path, resume_path, device
        )
        if (
            summary_data is None
            and resume_bundle is not None
            and resume_bundle.get("checkpoint_state") == "completed"
        ):
            summary_data = _recover_completed_publication(
                summary_path, resume_path, candidate_path, resume_bundle
            )
        summary_state = summary_data.get("state") if summary_data else None
        if summary_state == "completed" and candidate_path.exists():
            return _load_completed_summary(
                summary_path,
                candidate_path,
                dataset_hash,
                split_hash,
                config_hash,
                architecture,
                initial_checkpoint_sha256=expected_initial_sha,
                human_sample_weight_multiplier=human_weight_multiplier,
                human_only_training=human_only_training,
            )
        if resume_bundle is None:
            raise ActiveTrainingError(
                "resume requested but no continuation checkpoint exists"
            )
        if summary_data is not None:
            if summary_state != "paused":
                raise ActiveTrainingError("training summary is not in paused state")
            if summary_data.get("schema_version") != DATASET_SCHEMA_VERSION:
                raise ActiveTrainingError("training summary schema version mismatch")
            if (
                summary_data.get("dataset_sha256") != dataset_hash
                or summary_data.get("split_sha256") != split_hash
                or summary_data.get("config_hash") != config_hash
            ):
                raise ActiveTrainingError("training summary input hash mismatch")
            hashes_obj = summary_data.get("hashes")
            expected_resume_hash = _clean_text(
                summary_data.get("resume_checkpoint_sha256")
                or (
                    hashes_obj.get("resume_checkpoint")
                    if isinstance(hashes_obj, Mapping)
                    else None
                )
            ).lower()
            actual_resume_hash = _sha256_file(resume_path).lower()
            if not expected_resume_hash or actual_resume_hash != expected_resume_hash:
                raise ActiveTrainingError(
                    "resume checkpoint hash mismatch against training summary"
                )
    splits_counts = {key: int(len(value)) for key, value in splits.items()}
    metric_splits = (
        ("train", "val", "test")
        if config.evaluate_test_during_training
        else ("train", "val")
    )
    training_weights = np.asarray(arrays["sample_weights"]).astype(
        np.float32, copy=True
    )
    human_rows = np.asarray(arrays["label_sources"]) == "human"
    training_weights[human_rows] *= human_weight_multiplier
    if config.sample_weight_scheme == "region_balanced":
        train_regions = np.asarray(arrays["regions"])[splits["train"]]
        unique_regions, region_counts = np.unique(train_regions, return_counts=True)
        multipliers = {
            str(region): len(train_regions) / (len(unique_regions) * int(count))
            for region, count in zip(unique_regions, region_counts, strict=True)
        }
        for index in splits["train"]:
            training_weights[index] *= multipliers[str(arrays["regions"][index])]
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    random.seed(int(config.seed))
    model = ScenicRegressionModel(**architecture).to(device)
    if not resume and config.initial_checkpoint_path is not None:
        initial_path = Path(config.initial_checkpoint_path).expanduser().resolve()
        if not initial_path.exists() or not initial_path.is_file():
            raise ActiveTrainingError(f"initial checkpoint not found: {initial_path}")
        actual_initial = _sha256_file(initial_path).lower()
        if actual_initial != expected_initial_sha:
            raise ActiveTrainingError("initial checkpoint hash mismatch")
        initial_checkpoint_sha256 = actual_initial
        initial_evidence_path = str(initial_path)
        initial_checkpoint = _torch_load(initial_path, device)
        initial_state = initial_checkpoint.get("model_state_dict")
        if not isinstance(initial_state, Mapping):
            raise ActiveTrainingError("initial checkpoint lacks a model state dict")
        required_declared = ("vit_dim", "terrain_dim", "num_classes")
        invalid_initial = [
            key
            for key in required_declared
            if isinstance(initial_checkpoint.get(key), (bool, np.bool_))
            or not isinstance(initial_checkpoint.get(key), Integral)
            or int(initial_checkpoint[key]) != int(architecture[key])
        ]
        declared_hidden = initial_checkpoint.get("hidden_dim")
        if declared_hidden is not None and (
            isinstance(declared_hidden, (bool, np.bool_))
            or not isinstance(declared_hidden, Integral)
            or int(declared_hidden) != int(architecture["hidden_dim"])
        ):
            invalid_initial.append("hidden_dim")
        if invalid_initial:
            raise ActiveTrainingError(
                f"initial checkpoint architecture is incompatible: {invalid_initial}"
            )
        try:
            model.load_state_dict(initial_state, strict=True)
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise ActiveTrainingError(
                "initial checkpoint model state is incompatible"
            ) from exc
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    use_cuda = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    history: list[dict[str, Any]] = []
    next_epoch = next_batch = global_step = 0
    num_train_batches = math.ceil(len(splits["train"]) / int(config.batch_size))
    run_id, generation = os.urandom(16).hex(), 0

    if resume_bundle is not None:
        checkpoint = resume_bundle
        required = {
            "checkpoint_schema_version",
            "checkpoint_state",
            "model_state_dict",
            "optimizer_state_dict",
            "next_epoch",
            "next_batch",
            "global_step",
            "dataset_sha256",
            "split_sha256",
            "config_hash",
            "rng_state",
            "initial_checkpoint_sha256",
            "human_sample_weight_multiplier",
            "human_only_training",
            *architecture,
        }
        if (
            not required.issubset(checkpoint)
            or checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ActiveTrainingError(
                "resume checkpoint is incomplete and cannot be resumed"
            )
        if checkpoint.get("checkpoint_state") != "paused":
            raise ActiveTrainingError("resume checkpoint is not a paused continuation")
        if (
            checkpoint.get("dataset_sha256") != dataset_hash
            or checkpoint.get("split_sha256") != split_hash
            or checkpoint.get("config_hash") != config_hash
        ):
            raise ActiveTrainingError("resume checkpoint hash mismatch")
        checkpoint_architecture = {key: checkpoint.get(key) for key in architecture}
        if checkpoint_architecture != architecture:
            raise ActiveTrainingError("resume checkpoint architecture mismatch")
        recorded_initial_sha = _optional_sha256(
            checkpoint.get("initial_checkpoint_sha256"),
            "resume checkpoint initial_checkpoint_sha256",
        )
        if recorded_initial_sha != expected_initial_sha:
            raise ActiveTrainingError(
                "resume checkpoint initialization identity mismatch"
            )
        recorded_multiplier = checkpoint.get("human_sample_weight_multiplier")
        if (
            not isinstance(recorded_multiplier, (int, float, np.integer, np.floating))
            or not math.isfinite(float(recorded_multiplier))
            or float(recorded_multiplier) <= 0
            or float(recorded_multiplier) != human_weight_multiplier
        ):
            raise ActiveTrainingError(
                "resume checkpoint human weight multiplier mismatch"
            )
        recorded_human_only = checkpoint.get("human_only_training")
        if not isinstance(recorded_human_only, bool) or (
            recorded_human_only != human_only_training
        ):
            raise ActiveTrainingError("resume checkpoint human_only_training mismatch")
        initial_checkpoint_sha256 = recorded_initial_sha
        human_weight_multiplier = float(recorded_multiplier)
        if config.initial_checkpoint_path is not None:
            initial_evidence_path = str(
                Path(config.initial_checkpoint_path).expanduser().resolve()
            )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            scaler_state = checkpoint.get("scaler_state_dict")
            if isinstance(scaler_state, Mapping):
                scaler.load_state_dict(scaler_state)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (RuntimeError, ValueError, KeyError) as exc:
            raise ActiveTrainingError(
                "resume checkpoint model or optimizer state is invalid"
            ) from exc
        next_epoch = _resume_counter(
            checkpoint, "next_epoch", int(config.epochs), inclusive=True
        )
        next_batch = _resume_counter(
            checkpoint, "next_batch", num_train_batches, inclusive=True
        )
        if next_epoch == int(config.epochs) and next_batch != 0:
            raise ActiveTrainingError(
                "resume checkpoint final epoch cursor must have next_batch 0"
            )
        global_step = _resume_counter(checkpoint, "global_step", None)
        expected_global_step = next_epoch * num_train_batches + next_batch
        if global_step != expected_global_step:
            raise ActiveTrainingError(
                "resume checkpoint global_step is inconsistent with epoch and batch cursors"
            )
        raw_history = checkpoint.get("history")
        if (
            not isinstance(raw_history, list)
            or len(raw_history) != next_epoch
            or not all(isinstance(item, Mapping) for item in raw_history)
        ):
            raise ActiveTrainingError(
                "resume checkpoint history is inconsistent with the epoch cursor"
            )
        history = [dict(item) for item in raw_history]
        _restore_rng_state(checkpoint["rng_state"], device)
        run_id, generation = _resume_bundle_identity(checkpoint)
        if summary_data is None and checkpoint.get("summary_record") is not None:
            _publish_summary(
                summary_path,
                resume_path,
                _bundle_summary_record(checkpoint, "paused"),
            )
    starting_global_step = global_step

    def current_payload(checkpoint_state: str) -> dict[str, Any]:
        return _checkpoint_payload(
            model,
            optimizer,
            scaler=scaler,
            state=checkpoint_state,
            next_epoch=next_epoch,
            next_batch=next_batch,
            global_step=global_step,
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            config_hash=config_hash,
            architecture=architecture,
            counts=splits_counts,
            history=history,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            human_sample_weight_multiplier=human_weight_multiplier,
            human_only_training=human_only_training,
            device=device,
        )

    def commit(
        payload: Mapping[str, Any],
        *,
        summary_record: Mapping[str, Any] | None = None,
        candidate_sha256: str | None = None,
    ) -> None:
        """Atomically publish the next resume generation as the commit point."""

        nonlocal generation
        _atomic_torch(
            resume_path,
            {
                **payload,
                "run_id": run_id,
                "resume_generation": generation + 1,
                "summary_record": summary_record,
                "candidate_checkpoint_sha256": candidate_sha256,
            },
        )
        generation += 1

    if resume_bundle is None:
        commit(current_payload("paused"))

    started = time.monotonic()
    steps_this_invocation = 0
    state, stop_reason = "completed", None
    try:
        while next_epoch < int(config.epochs):
            train_indices = splits["train"]
            while next_batch < num_train_batches:
                if config.max_steps is not None and steps_this_invocation >= int(
                    config.max_steps
                ):
                    state, stop_reason = "paused", "max_steps"
                    break
                if config.max_seconds is not None and (
                    time.monotonic() - started
                ) >= float(config.max_seconds):
                    state, stop_reason = "paused", "max_seconds"
                    break
                batch_start = next_batch * int(config.batch_size)
                batch_indices = train_indices[
                    batch_start : batch_start + int(config.batch_size)
                ]

                def move(values: np.ndarray) -> torch.Tensor:
                    tensor = torch.from_numpy(values).float()
                    if use_cuda:
                        tensor = tensor.pin_memory()
                    return tensor.to(device, non_blocking=use_cuda)

                vit = move(np.asarray(arrays["vit_embeddings"])[batch_indices])
                terrain = move(np.asarray(arrays["terrain_features"])[batch_indices])
                logits = move(np.asarray(arrays["class_logits"])[batch_indices])
                target = move(
                    np.asarray(arrays["scenic_scores"])[batch_indices]
                ).reshape(-1, 1)
                weights = move(training_weights[batch_indices]).reshape(-1, 1)
                # Resumed runs rebuild the model in train mode; pin the same mode
                # on every step so continuous and resumed runs draw identical
                # dropout masks from the restored RNG state.
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_cuda):
                    prediction = model(vit, terrain, logits)
                    if config.loss_function == "huber":
                        per_sample = torch.nn.functional.smooth_l1_loss(
                            prediction, target, reduction="none"
                        )
                    else:
                        per_sample = (prediction - target) ** 2
                    loss = (
                        (per_sample * weights).sum()
                        / torch.clamp(weights.sum(), min=1e-8)
                        if config.use_sample_weights
                        else per_sample.mean()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                global_step, steps_this_invocation, next_batch = (
                    global_step + 1,
                    steps_this_invocation + 1,
                    next_batch + 1,
                )
                commit(current_payload("paused"))
            if state == "paused":
                break
            if config.max_seconds is not None and (time.monotonic() - started) >= float(
                config.max_seconds
            ):
                state, stop_reason = "paused", "max_seconds"
                break
            history.append(
                {
                    "epoch": int(next_epoch),
                    "metrics": {
                        split: _metrics(
                            model,
                            arrays,
                            splits[split],
                            device,
                            config.use_sample_weights,
                            batch_size=int(config.batch_size),
                        )
                        for split in metric_splits
                    },
                }
            )
            next_epoch, next_batch = next_epoch + 1, 0
            commit(current_payload("paused"))
        if next_epoch < int(config.epochs):
            state = "paused"
            stop_reason = stop_reason or "budget"
        metrics = (
            {
                split: _metrics(
                    model, arrays, splits[split], device, config.use_sample_weights,
                    batch_size=int(config.batch_size),
                )
                for split in metric_splits
            }
            if state == "completed"
            else {}
        )
    except (TimeoutError, KeyboardInterrupt) as exc:
        committed = _torch_load(resume_path, device)
        try:
            model.load_state_dict(committed["model_state_dict"])
            optimizer.load_state_dict(committed["optimizer_state_dict"])
            next_epoch = int(committed["next_epoch"])
            next_batch = int(committed["next_batch"])
            global_step = int(committed["global_step"])
            history = [dict(item) for item in committed.get("history", [])]
            _restore_rng_state(committed["rng_state"], device)
            committed_scaler = committed.get("scaler_state_dict")
            if isinstance(committed_scaler, Mapping):
                scaler.load_state_dict(committed_scaler)
        except (KeyError, TypeError, ValueError, RuntimeError) as restore_exc:
            raise ActiveTrainingError(
                "last committed resume checkpoint cannot be restored after interruption"
            ) from restore_exc
        _, generation = _resume_bundle_identity(committed)
        steps_this_invocation = global_step - starting_global_step
        state = "paused"
        stop_reason = f"interrupted: {type(exc).__name__}"
        metrics = {}
    except Exception as exc:
        state, stop_reason = "failed", f"{type(exc).__name__}: {exc}"
        _atomic_json(
            summary_path,
            _jsonable(
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "state": state,
                    "failure_reason": stop_reason,
                    "dataset_sha256": dataset_hash,
                    "split_sha256": split_hash,
                    "config_hash": config_hash,
                    "human_only_training": human_only_training,
                    "counts": splits_counts,
                    "global_step": global_step,
                    "run_id": run_id,
                    "resume_generation": generation,
                    "hashes": {"dataset": dataset_hash, "split": split_hash},
                }
            ),
        )
        raise
    summary: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "state": state,
        "stop_reason": stop_reason,
        "dataset_path": str(dataset),
        "split_path": str(split_path),
        "output_dir": str(root),
        "dataset_sha256": dataset_hash,
        "split_sha256": split_hash,
        "config_hash": config_hash,
        "config": config.as_dict(),
        "resolved_device": device,
        "initial_checkpoint": (
            {
                "path": initial_evidence_path,
                "sha256": initial_checkpoint_sha256,
                "requested_sha256": expected_initial_sha,
            }
            if initial_checkpoint_sha256 is not None
            else None
        ),
        "human_sample_weight_multiplier": human_weight_multiplier,
        "human_only_training": human_only_training,
        "train_scope": {
            "mode": "human_only" if human_only_training else "all",
            "total_train_rows": total_train_rows,
            "human_train_rows": human_train_rows,
        },
        "evaluated_splits": list(metric_splits),
        "architecture": architecture,
        "counts": splits_counts,
        "global_step": int(global_step),
        "epochs_completed": int(next_epoch),
        "steps_this_invocation": int(steps_this_invocation),
        "metrics": metrics,
        "history": history,
        "candidate_checkpoint": str(candidate_path),
        "resume_checkpoint": str(resume_path),
        "deterministic_limitations": [
            "CPU execution uses fixed row and batch order with restored Python/NumPy/Torch RNG state.",
            "GPU/MPS kernels may be nondeterministic; no bitwise accelerator claim is made.",
            "Bound limits are per invocation; global_step is cumulative across resumes.",
        ],
    }
    if state == "completed":
        payload = current_payload("completed")
        _atomic_torch(candidate_path, payload)
        candidate_sha256: str | None = _sha256_file(candidate_path)
        summary["hashes"] = {
            "dataset": dataset_hash,
            "split": split_hash,
            "candidate_checkpoint": candidate_sha256,
        }
    else:
        payload = current_payload("paused")
        candidate_sha256 = None
        summary["hashes"] = {"dataset": dataset_hash, "split": split_hash}
    summary["candidate_checkpoint_sha256"] = candidate_sha256
    summary["run_id"], summary["resume_generation"] = run_id, generation + 1
    record = _jsonable(summary)
    commit(payload, summary_record=record, candidate_sha256=candidate_sha256)
    return _publish_summary(summary_path, resume_path, record)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train active scenic regression data"
    )
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/active_training")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--sample-weight-scheme",
        choices=["standard", "region_balanced"],
        default="standard",
    )
    parser.add_argument("--loss-function", choices=["mse", "huber"], default="mse")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-sample-weights", action="store_true")
    parser.add_argument("--supplemental-benchmark", type=Path, default=None)
    parser.add_argument("--supplemental-benchmark-sha256", type=str, default=None)
    parser.add_argument(
        "--initial-checkpoint",
        dest="initial_checkpoint_path",
        type=Path,
        default=None,
    )
    parser.add_argument("--initial-checkpoint-sha256", type=str, default=None)
    parser.add_argument("--human-sample-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--evaluate-test-during-training", action="store_true")
    parser.add_argument("--human-only-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset or (args.output_dir / "active_dataset.npz")
    prepared = prepare_active_dataset(
        args.handoff,
        dataset_path,
        supplemental_benchmark_path=args.supplemental_benchmark,
        supplemental_benchmark_sha256=args.supplemental_benchmark_sha256,
    )
    split_csv = args.split_csv or Path(prepared["split_path"])
    config = ActiveTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        max_seconds=args.max_seconds,
        use_sample_weights=not args.no_sample_weights,
        sample_weight_scheme=args.sample_weight_scheme,
        loss_function=args.loss_function,
        initial_checkpoint_path=(
            str(args.initial_checkpoint_path)
            if args.initial_checkpoint_path is not None
            else None
        ),
        initial_checkpoint_sha256=args.initial_checkpoint_sha256,
        human_sample_weight_multiplier=args.human_sample_weight_multiplier,
        evaluate_test_during_training=args.evaluate_test_during_training,
        human_only_training=args.human_only_training,
    )
    result = train_active_model(
        dataset_path, split_csv, args.output_dir, config, resume=args.resume
    )
    print(
        json.dumps(
            _jsonable({"prepared": prepared, "training": result}),
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(f"METRIC state={result.get('state')}")
    print(f"METRIC global_step={result.get('global_step', 0)}")
    metrics = result.get("metrics", {})
    if isinstance(metrics, Mapping) and isinstance(metrics.get("val"), Mapping):
        print(f"METRIC val_rmse={metrics['val'].get('rmse', 0.0)}")
        print(f"METRIC val_mae={metrics['val'].get('mae', 0.0)}")


if __name__ == "__main__":
    main()
