"""Stage-two active scenic dataset preparation and resumable training.

This module is strict at the stage-one boundary. It consumes the scorer's
ordered candidate pool and feature cache, overlays absolute human labels, and
trains the existing :class:`ScenicRegressionModel` on fixed geographic
assignments supplied by stage one. No random split or implicit row filter is
performed here.
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
import tempfile
import time
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import torch

from .regression import ScenicRegressionModel, resolve_device


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


@dataclass(frozen=True)
class ActiveTrainingConfig:
    """Material and bounded controls for active stage-two training.

    ``max_steps`` and ``max_seconds`` are per-invocation continuation budgets;
    the checkpoint's global step remains cumulative. They are excluded from
    ``config_hash`` so a paused run can continue with a larger budget.
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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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


def _geo_key(row: Mapping[str, Any]) -> str | None:
    tile = _clean_text(row.get("tile_identity"))
    if tile:
        return f"tile:{tile}"
    parts = [_clean_text(row.get(name)) for name in ("region", "z", "x", "y")]
    if all(parts):
        try:
            parts[1:] = [str(int(float(value))) for value in parts[1:]]
        except (TypeError, ValueError):
            pass
        return "tile:" + "/".join(parts)
    return None


def _validate_geographic_leakage(
    candidate: pd.DataFrame, selected: Sequence[int], splits: Mapping[str, str]
) -> None:
    rows = candidate.iloc[list(selected)]
    by_key: dict[str, str] = {}
    for _, row in rows.iterrows():
        image_path = _clean_text(row.get("image_path"))
        split = splits[image_path]
        key = _geo_key(row)
        if key is not None:
            prior = by_key.get(key)
            if prior is not None and prior != split:
                raise ActiveTrainingError(f"geographic leakage across splits for {key}")
            by_key[key] = split
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
    if not {"region", "z", "x", "y"}.issubset(rows.columns):
        return
    coords: dict[tuple[str, int, int, int], str] = {}
    for _, row in rows.iterrows():
        try:
            key = (
                _clean_text(row["region"]),
                int(float(row["z"])),
                int(float(row["x"])),
                int(float(row["y"])),
            )
        except (TypeError, ValueError):
            continue
        image_path = _clean_text(row["image_path"])
        split = splits[image_path]
        prior = coords.get(key)
        if prior is not None and prior != split:
            raise ActiveTrainingError(f"geographic leakage across splits for {key}")
        coords[key] = split
    for (region, z, x, y), split in coords.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                adjacent_split = coords.get((region, z, x + dx, y + dy))
                if adjacent_split is not None and adjacent_split != split:
                    raise ActiveTrainingError(
                        "adjacent geographic tiles leak across splits"
                    )


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
    handoff_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Validate a stage-one handoff and materialize its ordered training NPZ."""

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
            "counts": counts,
            "dropped_without_split": dropped_without_split,
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


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or not {"python", "numpy", "torch"}.issubset(state):
        raise ActiveTrainingError("checkpoint RNG state is incomplete")

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor) or torch_state.dtype != torch.uint8:
        raise ActiveTrainingError("checkpoint torch RNG state must be a uint8 ByteTensor")

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(torch_state.cpu())
    except (TypeError, ValueError, RuntimeError, SystemError, AttributeError) as exc:
        raise ActiveTrainingError(f"invalid RNG state: {exc}") from exc

    if torch.cuda.is_available() and "cuda" in state:
        cuda_state = state["cuda"]
        if not isinstance(cuda_state, (list, tuple)) or not all(
            isinstance(t, torch.Tensor) and t.dtype == torch.uint8 for t in cuda_state
        ):
            raise ActiveTrainingError(
                "checkpoint CUDA RNG state must be a sequence of uint8 ByteTensors"
            )
        try:
            torch.cuda.set_rng_state_all([t.cpu() for t in cuda_state])
        except (TypeError, ValueError, RuntimeError, SystemError, AttributeError) as exc:
            raise ActiveTrainingError(f"invalid CUDA RNG state: {exc}") from exc


def _torch_load(path: Path, device: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
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
        "config_hash": config_hash,
        **{key: int(value) for key, value in architecture.items()},
        "counts": dict(counts),
        "history": list(history),
        "rng_state": _rng_state(),
    }


def _metrics(
    model: ScenicRegressionModel,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    use_sample_weights: bool,
) -> dict[str, float]:
    model.eval()
    use_cuda = str(device).startswith("cuda")

    def move(values: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(values).float()
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=use_cuda)

    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_cuda):
        vit = move(np.asarray(arrays["vit_embeddings"])[indices])
        terrain = move(np.asarray(arrays["terrain_features"])[indices])
        logits = move(np.asarray(arrays["class_logits"])[indices])
        target = np.asarray(arrays["scenic_scores"])[indices].astype(
            np.float32, copy=False
        )
        weight = np.asarray(arrays["sample_weights"])[indices].astype(
            np.float32, copy=False
        )
        prediction = model(vit, terrain, logits).float().cpu().numpy().reshape(-1)
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


def train_active_model(
    dataset_path: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    config: ActiveTrainingConfig,
    resume: bool = False,
) -> dict[str, Any]:
    """Train the configured bounded loss on fixed geographic splits."""

    if not isinstance(config, ActiveTrainingConfig):
        raise TypeError("config must be an ActiveTrainingConfig")
    config.validate()
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

    if resume:
        if not summary_path.exists():
            raise ActiveTrainingError(
                "resume requested but training summary does not exist"
            )
        summary_data = _load_json(summary_path, "training summary")
        summary_state = summary_data.get("state")
        if summary_state == "completed" and candidate_path.exists():
            return _load_completed_summary(
                summary_path,
                candidate_path,
                dataset_hash,
                split_hash,
                config_hash,
                architecture,
            )
        if summary_state == "failed":
            raise ActiveTrainingError("failed training run is not resumable")
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
        if not resume_path.exists():
            raise ActiveTrainingError(
                "resume requested but no continuation checkpoint exists"
            )
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
    training_weights = np.asarray(arrays["sample_weights"]).astype(
        np.float32, copy=True
    )
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    use_cuda = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    history: list[dict[str, Any]] = []
    next_epoch = next_batch = global_step = 0

    if resume:
        checkpoint = _torch_load(resume_path, device)
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
        num_train_batches = math.ceil(len(splits["train"]) / int(config.batch_size))
        next_epoch = _resume_counter(checkpoint, "next_epoch", int(config.epochs))
        next_batch = _resume_counter(
            checkpoint, "next_batch", num_train_batches, inclusive=True
        )
        global_step = _resume_counter(checkpoint, "global_step", None)
        expected_global_step = next_epoch * num_train_batches + next_batch
        if global_step != expected_global_step:
            raise ActiveTrainingError(
                "resume checkpoint global_step is inconsistent with epoch and batch cursors"
            )
        history = [dict(item) for item in checkpoint.get("history", [])]
        _restore_rng_state(checkpoint["rng_state"])
    starting_global_step = global_step
    if not resume:
        _atomic_torch(
            resume_path,
            _checkpoint_payload(
                model,
                optimizer,
                scaler=scaler,
                state="paused",
                next_epoch=next_epoch,
                next_batch=next_batch,
                global_step=global_step,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                config_hash=config_hash,
                architecture=architecture,
                counts=splits_counts,
                history=history,
            ),
        )

    started = time.monotonic()
    steps_this_invocation = 0
    state, stop_reason = "completed", None
    try:
        while next_epoch < int(config.epochs):
            train_indices = splits["train"]
            batch_start = next_batch * int(config.batch_size)
            if batch_start >= len(train_indices):
                next_epoch, next_batch = next_epoch + 1, 0
                continue
            while batch_start < len(train_indices):
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
                batch_start += len(batch_indices)
                _atomic_torch(
                    resume_path,
                    _checkpoint_payload(
                        model,
                        optimizer,
                        scaler=scaler,
                        state="paused",
                        next_epoch=next_epoch,
                        next_batch=next_batch,
                        global_step=global_step,
                        dataset_hash=dataset_hash,
                        split_hash=split_hash,
                        config_hash=config_hash,
                        architecture=architecture,
                        counts=splits_counts,
                        history=history,
                    ),
                )
                if next_batch * int(config.batch_size) >= len(train_indices):
                    next_epoch, next_batch = next_epoch + 1, 0
                    break
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
                        )
                        for split in ("train", "val", "test")
                    },
                }
            )
        if next_epoch < int(config.epochs):
            state = "paused"
            stop_reason = stop_reason or "budget"
    except (TimeoutError, KeyboardInterrupt) as exc:
        committed = _torch_load(resume_path, device)
        try:
            model.load_state_dict(committed["model_state_dict"])
            optimizer.load_state_dict(committed["optimizer_state_dict"])
            next_epoch = int(committed["next_epoch"])
            next_batch = int(committed["next_batch"])
            global_step = int(committed["global_step"])
            history = [dict(item) for item in committed.get("history", [])]
            _restore_rng_state(committed["rng_state"])
            committed_scaler = committed.get("scaler_state_dict")
            if isinstance(committed_scaler, Mapping):
                scaler.load_state_dict(committed_scaler)
        except (KeyError, TypeError, ValueError, RuntimeError) as restore_exc:
            raise ActiveTrainingError(
                "last committed resume checkpoint cannot be restored after interruption"
            ) from restore_exc
        steps_this_invocation = global_step - starting_global_step
        state = "paused"
        stop_reason = f"interrupted: {type(exc).__name__}"
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
                    "counts": splits_counts,
                    "global_step": global_step,
                    "hashes": {"dataset": dataset_hash, "split": split_hash},
                }
            ),
        )
        raise
    metrics = (
        {
            split: _metrics(
                model, arrays, splits[split], device, config.use_sample_weights
            )
            for split in ("train", "val", "test")
        }
        if state == "completed"
        else {}
    )
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
        payload = _checkpoint_payload(
            model,
            optimizer,
            scaler=scaler,
            state="completed",
            next_epoch=next_epoch,
            next_batch=next_batch,
            global_step=global_step,
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            config_hash=config_hash,
            architecture=architecture,
            counts=splits_counts,
            history=history,
        )
        _atomic_torch(candidate_path, payload)
        _atomic_torch(resume_path, payload)
        summary["candidate_checkpoint_sha256"], summary["resume_checkpoint_sha256"] = (
            _sha256_file(candidate_path),
            _sha256_file(resume_path),
        )
        summary["hashes"] = {
            "dataset": dataset_hash,
            "split": split_hash,
            "candidate_checkpoint": summary["candidate_checkpoint_sha256"],
            "resume_checkpoint": summary["resume_checkpoint_sha256"],
        }
    else:
        _atomic_torch(
            resume_path,
            _checkpoint_payload(
                model,
                optimizer,
                scaler=scaler,
                state="paused",
                next_epoch=next_epoch,
                next_batch=next_batch,
                global_step=global_step,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                config_hash=config_hash,
                architecture=architecture,
                counts=splits_counts,
                history=history,
            ),
        )
        summary["candidate_checkpoint_sha256"] = None
        summary["resume_checkpoint_sha256"] = _sha256_file(resume_path)
        summary["hashes"] = {
            "dataset": dataset_hash,
            "split": split_hash,
            "resume_checkpoint": summary["resume_checkpoint_sha256"],
        }
    _atomic_json(summary_path, _jsonable(summary))
    return _jsonable(summary)


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset or (args.output_dir / "active_dataset.npz")
    prepared = prepare_active_dataset(args.handoff, dataset_path)
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
