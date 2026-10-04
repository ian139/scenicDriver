"""Shared, explicit validation-membership derivation for baseline regression.

Training and evaluation must agree exactly on which exported rows form the
model's validation set. This module owns that agreement:

- ``training_population_mask`` / ``training_arrays``: the filtered training
  population (fixed benchmark validation/test identities and embedded held-out
  export splits removed), row-aligned over every exported array.
- ``derive_validation_membership``: the single seeded train/validation
  derivation over that filtered population (via
  ``src.scenic_scorer.regression.validation_split_indices``), used by both
  training and evaluation.
- the persisted split artifact written by training and consumed by evaluation
  (``write_split_artifact`` / ``load_validation_membership``). Evaluation never
  guesses membership: it reproduces the artifact through the same derivation
  from the recorded training inputs, and any drift is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.modeling.compare_regression_on_benchmark import (  # noqa: E402
    load_benchmark_split,
    load_exported_image_paths,
)
from src.scenic_scorer.regression import validation_split_indices  # noqa: E402

SPLIT_ARTIFACT_SCHEMA = "scenic-regression-split/1"


def training_population_mask(dataset_path: Path, split_path: Path) -> np.ndarray:
    """Boolean keep-mask over exported rows defining the training population.

    Rows held out by the fixed benchmark split (validation/test) and rows whose
    embedded export split is not "train" are excluded.
    """
    split = load_benchmark_split(split_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        paths = load_exported_image_paths(data)
        keep = np.array(
            [path not in split.index or split.at[path, "split"] == "train" for path in paths]
        )
        if "splits" in data.files:
            assignments = np.asarray(data["splits"])
            if assignments.ndim != 1 or len(assignments) != len(paths):
                raise ValueError("dataset splits must align with image_paths")
            normalized = []
            for path, assignment in zip(paths, assignments, strict=True):
                assignment = (
                    assignment.decode("utf-8") if isinstance(assignment, bytes) else str(assignment)
                )
                assignment = assignment.strip().lower().replace("validation", "val")
                if assignment not in {"train", "val", "test"}:
                    raise ValueError("dataset has invalid split assignments")
                if path in split.index and assignment != split.at[path, "split"]:
                    raise ValueError(f"benchmark split conflicts with dataset split for {path}")
                normalized.append(assignment)
            keep &= np.array([assignment == "train" for assignment in normalized])
    return keep


def training_arrays(dataset_path: Path, split_path: Path) -> dict[str, np.ndarray]:
    """Remove held-out exact identities from every row-aligned export array."""
    keep = training_population_mask(dataset_path, split_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        paths = load_exported_image_paths(data)
        arrays = {}
        for key in data.files:
            values = data[key]
            if values.ndim < 1 or len(values) != len(paths):
                raise ValueError(f"dataset array {key} must align with image_paths")
            arrays[key] = values[keep]
    return arrays


@dataclass(frozen=True)
class ValidationMembership:
    """Explicit train/validation membership over one exported dataset.

    ``train_indices``/``val_indices`` are positions into the filtered training
    population (the order the trainer sees); ``*_dataset_indices`` map them back
    to rows of the full exported dataset.
    """

    val_split: float
    seed: int
    population_indices: np.ndarray  # full-dataset rows kept by the filter
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_paths: tuple[str, ...]
    val_paths: tuple[str, ...]

    @property
    def train_dataset_indices(self) -> np.ndarray:
        return np.asarray(self.population_indices)[list(self.train_indices)]

    @property
    def val_dataset_indices(self) -> np.ndarray:
        return np.asarray(self.population_indices)[list(self.val_indices)]


def derive_validation_membership(
    dataset_path: Path,
    benchmark_split_csv: Path,
    *,
    val_split: float,
    seed: int,
) -> ValidationMembership:
    """Derive the model's exact train/validation membership for one dataset.

    This is the one derivation used by both training and evaluation: the
    filtered training population, split by
    ``src.scenic_scorer.regression.validation_split_indices`` with the given
    seed and split fraction.
    """
    keep = training_population_mask(dataset_path, benchmark_split_csv)
    population_indices = np.flatnonzero(keep)
    with np.load(dataset_path, allow_pickle=False) as data:
        paths = load_exported_image_paths(data)
    train_idx, val_idx = validation_split_indices(len(population_indices), val_split, seed)
    population = [int(i) for i in population_indices]
    return ValidationMembership(
        val_split=float(val_split),
        seed=int(seed),
        population_indices=np.asarray(population, dtype=np.int64),
        train_indices=tuple(train_idx),
        val_indices=tuple(val_idx),
        train_paths=tuple(paths[population[i]] for i in train_idx),
        val_paths=tuple(paths[population[i]] for i in val_idx),
    )


def write_split_artifact(
    path: Path,
    membership: ValidationMembership,
    *,
    dataset_path: Path,
    benchmark_split_csv: Path,
) -> None:
    """Persist the exact validation membership produced during training."""
    payload = {
        "schema": SPLIT_ARTIFACT_SCHEMA,
        "dataset": str(dataset_path),
        "benchmark_split_csv": str(benchmark_split_csv),
        "val_split": membership.val_split,
        "seed": membership.seed,
        "population_size": int(len(membership.population_indices)),
        "train_image_paths": list(membership.train_paths),
        "val_image_paths": list(membership.val_paths),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_validation_membership(dataset_path: Path, split_artifact_path: Path) -> ValidationMembership:
    """Reproduce the trained model's exact validation membership from its artifact.

    The artifact records the training-time derivation inputs and resulting
    identities; membership is reproduced with the shared derivation from those
    exact inputs and must match the artifact exactly. Missing artifacts, dataset
    mismatches, and any drift are hard errors — never a guessed split.
    """
    artifact_path = Path(split_artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing split artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SPLIT_ARTIFACT_SCHEMA:
        raise ValueError(f"split artifact has unsupported schema: {artifact_path}")
    required = (
        "dataset",
        "benchmark_split_csv",
        "val_split",
        "seed",
        "train_image_paths",
        "val_image_paths",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"split artifact missing fields {missing}: {artifact_path}")
    if payload["train_image_paths"] == payload["val_image_paths"]:
        raise ValueError(f"split artifact has degenerate membership: {artifact_path}")

    recorded_dataset = Path(payload["dataset"])
    if recorded_dataset.resolve() != Path(dataset_path).resolve():
        raise ValueError(
            f"split artifact dataset {recorded_dataset} does not match evaluation "
            f"dataset {dataset_path}"
        )

    derived = derive_validation_membership(
        recorded_dataset,
        Path(payload["benchmark_split_csv"]),
        val_split=payload["val_split"],
        seed=payload["seed"],
    )
    if (
        list(derived.train_paths) != list(payload["train_image_paths"])
        or list(derived.val_paths) != list(payload["val_image_paths"])
    ):
        raise ValueError(
            f"split artifact does not reproduce the shared validation-membership "
            f"derivation: {artifact_path}"
        )
    return derived
