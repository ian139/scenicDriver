"""Geographic train/validation membership and run provenance for baseline regression.

Training and evaluation must agree exactly on which exported rows form the
model's validation set, and evaluation must only admit a run whose inputs are
the exact bytes training used. This module owns both agreements:

- Physical identity: every exported row resolves to a region-independent
  ``(z, x, y)`` tile from its labels-CSV ``z``/``x``/``y`` columns, its
  ``tile_identity`` and its exported ``image_path``; disagreeing sources are
  an error, region aliases never distinguish tiles.
- ``holdout_population_mask``: rows whose tile lies within the declared buffer
  (Chebyshev tiles, compared at the coarser of the two zooms) of any fixed
  benchmark val/test tile or embedded held-out export row are excluded.
- ``derive_validation_membership``: the remaining rows are grouped into
  connected components of physical blocks and declared geographic groups;
  each component is assigned train or validation by a seeded SHA-256
  threshold, so membership never depends on pool sizes. Training rows adjacent
  to (or overlapping) a validation tile are then dropped, so no one-tile
  adjacency leaks across the split.
- the split artifact (``write_split_artifact``) records the rule, the input
  content digests and the resulting identities; the run manifest
  (``write_run_manifest``) binds the checkpoint, split artifact, feature NPZ,
  labels CSV, benchmark split CSV, classifier checkpoint and preprocessing
  identity by content digest.
- ``load_validation_membership`` admits a run only when every supplied file
  matches its bound digest (paths are never compared, so relocated identical
  bytes are accepted) and the shared derivation reproduces the artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from scripts.modeling.compare_regression_on_benchmark import (  # noqa: E402
    load_benchmark_split,
    load_exported_image_paths,
)
from scenicdriver.active_learning.common import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    stable_json_bytes,
)
from scenicdriver.classifier.model import load_classifier_checkpoint  # noqa: E402
from scenicdriver.heuristics.labeler import parse_tile_coords  # noqa: E402
from scenicdriver.terrain.features import compute_terrain_features  # noqa: E402

SPLIT_ARTIFACT_SCHEMA = "scenic-regression-split/2"
RUN_MANIFEST_SCHEMA = "scenic-regression-run/1"
GEOGRAPHIC_SPLIT_RULE = "sha256-physical-block-component-threshold/1"
# Declared geographic groups honoured from the labels CSV; rows sharing a value
# in any of these columns always land in the same split.
GROUP_COLUMNS = ("geographic_group", "geographic_block", "geo_group", "block", "group")
BOUND_FILES = (
    "checkpoint",
    "split_artifact",
    "dataset",
    "labels_csv",
    "benchmark_split_csv",
    "classifier_checkpoint",
)
_INPUT_DIGESTS = ("dataset_sha256", "labels_csv_sha256", "benchmark_split_csv_sha256")
_TILE_IDENTITY_COORDS = re.compile(r"(?:^|/)z(-?\d+)/x(-?\d+)/y(-?\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

Tile = tuple[int, int, int]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _coordinate(text: str, name: str, identity: str) -> int:
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer tile coordinate for {identity}") from exc
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be an integer tile coordinate for {identity}")
    return int(number)


def physical_tile(row: Mapping[str, Any]) -> Tile | None:
    """Return a row's region-independent physical ``(z, x, y)`` tile.

    ``z``/``x``/``y`` columns, a ``tile_identity`` (``<region>/z<z>/x<x>/y<y>``)
    and the exported ``image_path`` (``.../z<z>/<region>/<x>_<y>.png`` and the
    compact ``z<z>/x<x>/y<y>`` forms) are all consulted and must agree; region
    names never contribute. ``None`` means no source carries coordinates.
    """
    identity = _text(row.get("image_path"))
    sources: list[tuple[str, Tile]] = []
    texts = [_text(row.get(name)) for name in ("z", "x", "y")]
    if any(texts):
        if not all(texts):
            raise ValueError(f"partial z/x/y tile coordinates for {identity}")
        sources.append(
            (
                "z/x/y",
                tuple(
                    _coordinate(text, name, identity)
                    for text, name in zip(texts, ("z", "x", "y"), strict=True)
                ),
            )
        )
    match = _TILE_IDENTITY_COORDS.search(_text(row.get("tile_identity")))
    if match:
        sources.append(("tile_identity", tuple(int(group) for group in match.groups())))
    parsed = parse_tile_coords(identity) if identity else None
    if parsed is not None:
        sources.append(("image_path", tuple(int(value) for value in parsed)))
    if not sources:
        return None
    first_source, tile = sources[0]
    for source, other in sources[1:]:
        if other != tile:
            raise ValueError(f"{source} disagrees with {first_source} tile for {identity}")
    return tile


class _TileIndex:
    """Neighbourhood lookups between tiles of possibly different zooms.

    Two tiles are compared on the coarser of their zoom levels, so a fine tile
    is near a coarse tile whenever its coarse ancestor is.
    """

    def __init__(self, tiles: Iterable[Tile]) -> None:
        self._by_zoom: dict[int, set[tuple[int, int]]] = {}
        for z, x, y in tiles:
            self._by_zoom.setdefault(z, set()).add((x, y))
        self._projected: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def _cells(self, zoom: int, coarse: int) -> set[tuple[int, int]]:
        key = (zoom, coarse)
        cells = self._projected.get(key)
        if cells is None:
            shift = zoom - coarse
            cells = {(x >> shift, y >> shift) for x, y in self._by_zoom[zoom]}
            self._projected[key] = cells
        return cells

    def within(self, tile: Tile, radius: int) -> bool:
        z, x, y = tile
        for zoom in self._by_zoom:
            coarse = min(z, zoom)
            shift = z - coarse
            px, py = x >> shift, y >> shift
            cells = self._cells(zoom, coarse)
            if any(
                (px + dx, py + dy) in cells
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
            ):
                return True
        return False


def _load_labels(labels_csv: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(labels_csv)
    if "image_path" not in frame.columns:
        raise ValueError("labels CSV must contain image_path column")
    records: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        identity = _text(record.get("image_path"))
        if not identity:
            raise ValueError("labels CSV contains an empty image_path")
        if identity in records:
            raise ValueError(f"labels CSV contains duplicate image_path {identity}")
        records[identity] = record
    return records


def _embedded_splits(
    data: np.lib.npyio.NpzFile, paths: list[str], benchmark: pd.DataFrame
) -> list[str] | None:
    if "splits" not in data.files:
        return None
    assignments = np.asarray(data["splits"])
    if assignments.ndim != 1 or len(assignments) != len(paths):
        raise ValueError("dataset splits must align with image_paths")
    normalized = []
    for path, assignment in zip(paths, assignments, strict=True):
        assignment = assignment.decode("utf-8") if isinstance(assignment, bytes) else str(assignment)
        assignment = assignment.strip().lower().replace("validation", "val")
        if assignment not in {"train", "val", "test"}:
            raise ValueError("dataset has invalid split assignments")
        if path in benchmark.index and assignment != benchmark.at[path, "split"]:
            raise ValueError(f"benchmark split conflicts with dataset split for {path}")
        normalized.append(assignment)
    return normalized


def _holdout_tile(
    identity: str, record: Mapping[str, Any], labels: Mapping[str, Mapping[str, Any]]
) -> Tile:
    tile = physical_tile(record)
    labelled = labels.get(identity)
    labelled_tile = physical_tile(labelled) if labelled is not None else None
    if tile is not None and labelled_tile is not None and tile != labelled_tile:
        raise ValueError(f"benchmark and labels CSV disagree on the tile of {identity}")
    tile = tile if tile is not None else labelled_tile
    if tile is None:
        raise ValueError(f"benchmark held-out identity {identity} has no physical z/x/y tile")
    return tile


@dataclass(frozen=True)
class _Population:
    paths: list[str]
    tiles: list[Tile]
    labels: dict[str, dict[str, Any]]
    keep: np.ndarray


def _resolve_population(
    dataset_path: Path, labels_csv: Path, benchmark_split_csv: Path, holdout_buffer_tiles: int
) -> _Population:
    if int(holdout_buffer_tiles) < 1:
        raise ValueError("holdout buffer must be at least one tile to prevent adjacency leakage")
    benchmark = load_benchmark_split(benchmark_split_csv)
    labels = _load_labels(labels_csv)
    with np.load(dataset_path, allow_pickle=False) as data:
        paths = load_exported_image_paths(data)
        embedded = _embedded_splits(data, paths, benchmark)
    tiles: list[Tile] = []
    for path in paths:
        record = labels.get(path)
        if record is None:
            raise ValueError(f"dataset image_path {path} is absent from labels CSV")
        tile = physical_tile(record)
        if tile is None:
            raise ValueError(f"exported row {path} has no physical z/x/y tile")
        tiles.append(tile)
    holdout = [
        _holdout_tile(identity, record, labels)
        for identity, record in benchmark.to_dict(orient="index").items()
        if record["split"] != "train"
    ]
    if embedded is not None:
        holdout.extend(tile for tile, split in zip(tiles, embedded, strict=True) if split != "train")
    if not holdout:
        raise ValueError("benchmark split declares no held-out val/test identities")
    index = _TileIndex(holdout)
    keep = np.array([not index.within(tile, int(holdout_buffer_tiles)) for tile in tiles], dtype=bool)
    return _Population(paths=paths, tiles=tiles, labels=labels, keep=keep)


def holdout_population_mask(
    dataset_path: Path, labels_csv: Path, benchmark_split_csv: Path, *, holdout_buffer_tiles: int
) -> np.ndarray:
    """Keep-mask over exported rows outside every held-out tile plus its buffer."""
    return _resolve_population(
        dataset_path, labels_csv, benchmark_split_csv, holdout_buffer_tiles
    ).keep


@dataclass(frozen=True)
class GeographicSplitConfig:
    """The declared geographic train/validation rule."""

    val_split: float
    seed: int
    block_size: int = 32
    holdout_buffer_tiles: int = 1

    def __post_init__(self) -> None:
        if not 0 < float(self.val_split) < 1:
            raise ValueError("val_split must be in (0, 1)")
        if int(self.block_size) < 1:
            raise ValueError("block_size must be at least one tile")
        if int(self.holdout_buffer_tiles) < 1:
            raise ValueError("holdout buffer must be at least one tile to prevent adjacency leakage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": GEOGRAPHIC_SPLIT_RULE,
            "val_split": float(self.val_split),
            "seed": int(self.seed),
            "block_size": int(self.block_size),
            "holdout_buffer_tiles": int(self.holdout_buffer_tiles),
            "adjacency_radius_tiles": 1,
            "group_columns": list(GROUP_COLUMNS),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> GeographicSplitConfig:
        if not isinstance(payload, dict) or payload.get("name") != GEOGRAPHIC_SPLIT_RULE:
            raise ValueError("split artifact has an unsupported geographic split rule")
        config = cls(
            val_split=float(payload["val_split"]),
            seed=int(payload["seed"]),
            block_size=int(payload["block_size"]),
            holdout_buffer_tiles=int(payload["holdout_buffer_tiles"]),
        )
        if config.to_dict() != payload:
            raise ValueError("split artifact geographic split rule does not match this implementation")
        return config


@dataclass(frozen=True)
class ValidationMembership:
    """Explicit geographic train/validation membership over one exported dataset.

    ``train_indices``/``val_indices`` are positions into the training
    population (the order the trainer sees); ``*_dataset_indices`` map them
    back to rows of the full exported dataset.
    """

    config: GeographicSplitConfig
    input_digests: dict[str, str]
    population_indices: np.ndarray  # full-dataset rows kept, in dataset order
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_paths: tuple[str, ...]
    val_paths: tuple[str, ...]
    holdout_excluded_paths: tuple[str, ...]
    boundary_excluded_paths: tuple[str, ...]

    @property
    def val_split(self) -> float:
        return float(self.config.val_split)

    @property
    def seed(self) -> int:
        return int(self.config.seed)

    @property
    def train_dataset_indices(self) -> np.ndarray:
        return np.asarray(self.population_indices)[list(self.train_indices)]

    @property
    def val_dataset_indices(self) -> np.ndarray:
        return np.asarray(self.population_indices)[list(self.val_indices)]


def _component_draw(seed: int, component: str) -> float:
    payload = f"{int(seed)}\0split-component\0{component}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def derive_validation_membership(
    dataset_path: Path,
    labels_csv: Path,
    benchmark_split_csv: Path,
    config: GeographicSplitConfig,
) -> ValidationMembership:
    """Derive the model's exact geographic train/validation membership.

    This is the one derivation used by both training and evaluation.
    """
    input_digests = {
        "dataset_sha256": sha256_file(dataset_path),
        "labels_csv_sha256": sha256_file(labels_csv),
        "benchmark_split_csv_sha256": sha256_file(benchmark_split_csv),
    }
    population = _resolve_population(
        dataset_path, labels_csv, benchmark_split_csv, config.holdout_buffer_tiles
    )
    candidates = [int(row) for row in np.flatnonzero(population.keep)]

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    block_of: dict[int, str] = {}
    for row in candidates:
        z, x, y = population.tiles[row]
        size = int(config.block_size)
        nodes = [f"block:z{z}|b{x // size}|{y // size}"]
        record = population.labels[population.paths[row]]
        nodes.extend(
            f"{column}:{value}"
            for column in GROUP_COLUMNS
            if (value := _text(record.get(column)))
        )
        for node in nodes:
            parent.setdefault(node, node)
        for node in nodes[1:]:
            first, other = find(nodes[0]), find(node)
            if first != other:
                low, high = sorted((first, other))
                parent[high] = low
        block_of[row] = nodes[0]

    split_of = {
        row: (
            "val"
            if _component_draw(config.seed, find(block_of[row])) < float(config.val_split)
            else "train"
        )
        for row in candidates
    }
    val_rows = [row for row in candidates if split_of[row] == "val"]
    val_index = _TileIndex(population.tiles[row] for row in val_rows)
    boundary = {
        row
        for row in candidates
        if split_of[row] == "train" and val_rows and val_index.within(population.tiles[row], 1)
    }
    train_rows = [row for row in candidates if split_of[row] == "train" and row not in boundary]
    if not train_rows or not val_rows:
        raise ValueError(
            "geographic split produced an empty training or validation set; "
            "adjust --val-split, --seed or --block-size"
        )
    kept = sorted(train_rows + val_rows)
    position = {row: index for index, row in enumerate(kept)}
    paths = population.paths
    return ValidationMembership(
        config=config,
        input_digests=input_digests,
        population_indices=np.asarray(kept, dtype=np.int64),
        train_indices=tuple(position[row] for row in train_rows),
        val_indices=tuple(position[row] for row in val_rows),
        train_paths=tuple(paths[row] for row in train_rows),
        val_paths=tuple(paths[row] for row in val_rows),
        holdout_excluded_paths=tuple(
            path for path, keep in zip(paths, population.keep, strict=True) if not keep
        ),
        boundary_excluded_paths=tuple(paths[row] for row in sorted(boundary)),
    )


def training_arrays(dataset_path: Path, membership: ValidationMembership) -> dict[str, np.ndarray]:
    """Return every row-aligned export array restricted to the training population."""
    with np.load(dataset_path, allow_pickle=False) as data:
        paths = load_exported_image_paths(data)
        arrays = {}
        for key in data.files:
            values = data[key]
            if values.ndim < 1 or len(values) != len(paths):
                raise ValueError(f"dataset array {key} must align with image_paths")
            arrays[key] = values[membership.population_indices]
    population_paths = arrays["image_paths"].tolist()
    expected = {position: path for position, path in zip(membership.train_indices, membership.train_paths)}
    expected.update(zip(membership.val_indices, membership.val_paths))
    if any(population_paths[position] != path for position, path in expected.items()):
        raise ValueError("dataset rows do not match the derived validation membership")
    return arrays


def preprocessing_identity(
    classifier_checkpoint: Path, *, normalization: str | None = None
) -> dict[str, Any]:
    """Identity of the feature-export preprocessing, read from the classifier checkpoint.

    The classifier input contract is the one the checkpoint carries (and the
    exporter derives its transform from); ``normalization`` is only an
    assertion that must agree with it. The terrain feature implementation is
    bound by qualified name.
    """
    _, contract = load_classifier_checkpoint(classifier_checkpoint, map_location="cpu")
    contract.require_normalization(normalization)
    descriptor = {
        "classifier": contract.as_dict(),
        "terrain_features": (
            f"{compute_terrain_features.__module__}.{compute_terrain_features.__qualname__}"
        ),
    }
    return {**descriptor, "sha256": _canonical_sha256(descriptor)}


def _read_json(path: Path, name: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def write_split_artifact(path: Path, membership: ValidationMembership) -> None:
    """Persist the exact geographic membership produced during training."""
    atomic_write_json(
        path,
        {
            "schema": SPLIT_ARTIFACT_SCHEMA,
            "rule": membership.config.to_dict(),
            "inputs": dict(membership.input_digests),
            "population_size": int(len(membership.population_indices)),
            "train_image_paths": list(membership.train_paths),
            "val_image_paths": list(membership.val_paths),
            "holdout_excluded_image_paths": list(membership.holdout_excluded_paths),
            "boundary_excluded_image_paths": list(membership.boundary_excluded_paths),
        },
    )


def write_run_manifest(
    path: Path,
    membership: ValidationMembership,
    *,
    checkpoint: Path,
    split_artifact: Path,
    dataset: Path,
    labels_csv: Path,
    benchmark_split_csv: Path,
    classifier_checkpoint: Path,
    preprocessing: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every run input and output by content digest."""
    files = {
        "checkpoint": checkpoint,
        "split_artifact": split_artifact,
        "dataset": dataset,
        "labels_csv": labels_csv,
        "benchmark_split_csv": benchmark_split_csv,
        "classifier_checkpoint": classifier_checkpoint,
    }
    artifacts = {
        role: {"sha256": sha256_file(file), "bytes": Path(file).stat().st_size}
        for role, file in files.items()
    }
    for role, key in zip(("dataset", "labels_csv", "benchmark_split_csv"), _INPUT_DIGESTS, strict=True):
        if artifacts[role]["sha256"] != membership.input_digests[key]:
            raise ValueError(f"{role} changed during training; refusing to bind the run")
    payload = {
        "schema": RUN_MANIFEST_SCHEMA,
        "artifacts": artifacts,
        "preprocessing": dict(preprocessing),
        "split_rule": membership.config.to_dict(),
        "training": dict(training),
    }
    atomic_write_json(path, payload)
    return payload


def load_validation_membership(
    dataset_path: Path,
    split_artifact_path: Path,
    *,
    run_manifest: Path,
    checkpoint: Path,
    labels_csv: Path,
    benchmark_split_csv: Path,
    classifier_checkpoint: Path,
) -> ValidationMembership:
    """Admit a baseline run and reproduce its exact validation membership.

    Every supplied file must match the content digest the run manifest binds
    (any location is accepted for identical bytes); the recorded preprocessing
    identity must match the current implementation; and the shared derivation
    over the admitted inputs must reproduce the split artifact exactly. Any
    missing file, tampering or drift is a hard error — never a guessed split.
    """
    manifest = _read_json(run_manifest, "run manifest")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError(f"run manifest has unsupported schema: {run_manifest}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(BOUND_FILES):
        raise ValueError(f"run manifest must bind exactly {list(BOUND_FILES)}: {run_manifest}")
    files = {
        "checkpoint": checkpoint,
        "split_artifact": split_artifact_path,
        "dataset": dataset_path,
        "labels_csv": labels_csv,
        "benchmark_split_csv": benchmark_split_csv,
        "classifier_checkpoint": classifier_checkpoint,
    }
    for role, file in files.items():
        file = Path(file)
        if not file.is_file():
            raise FileNotFoundError(f"Missing {role} for run admission: {file}")
        entry = artifacts[role]
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError(f"run manifest has no valid sha256 for {role}")
        if sha256_file(file) != expected:
            raise ValueError(f"run manifest {role} content digest mismatch: {file}")

    preprocessing = manifest.get("preprocessing")
    if preprocessing != preprocessing_identity(classifier_checkpoint):
        raise ValueError(
            "run manifest preprocessing identity does not match the classifier checkpoint contract"
        )

    payload = _read_json(split_artifact_path, "split artifact")
    if payload.get("schema") != SPLIT_ARTIFACT_SCHEMA:
        raise ValueError(f"split artifact has unsupported schema: {split_artifact_path}")
    required = (
        "rule",
        "inputs",
        "train_image_paths",
        "val_image_paths",
        "holdout_excluded_image_paths",
        "boundary_excluded_image_paths",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"split artifact missing fields {missing}: {split_artifact_path}")
    if payload["rule"] != manifest.get("split_rule"):
        raise ValueError("split artifact rule does not match the run manifest")
    bound_inputs = {
        key: artifacts[role]["sha256"]
        for role, key in zip(("dataset", "labels_csv", "benchmark_split_csv"), _INPUT_DIGESTS, strict=True)
    }
    if payload["inputs"] != bound_inputs:
        raise ValueError("split artifact inputs do not match the run manifest dataset digests")
    config = GeographicSplitConfig.from_dict(payload["rule"])

    derived = derive_validation_membership(dataset_path, labels_csv, benchmark_split_csv, config)
    if derived.input_digests != bound_inputs:
        raise ValueError("run inputs changed during admission")
    if (
        list(derived.train_paths) != list(payload["train_image_paths"])
        or list(derived.val_paths) != list(payload["val_image_paths"])
        or list(derived.holdout_excluded_paths) != list(payload["holdout_excluded_image_paths"])
        or list(derived.boundary_excluded_paths) != list(payload["boundary_excluded_image_paths"])
    ):
        raise ValueError(
            "split artifact does not reproduce the shared geographic membership "
            f"derivation: {split_artifact_path}"
        )
    return derived
