"""Fail-closed validation and atomic publication of a stage-one handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import atomic_write_json, jsonable, sha256_file
from .scoring import CANDIDATE_POOL_COLUMNS, SCORING_SCHEMA_VERSION

SCHEMA_VERSION = 1
ABSOLUTE_COLUMNS = (
    "image_path",
    "scenic_human",
    "confidence",
    "skip",
    "annotator_id",
    "timestamp",
    "notes",
)
TILE_COLUMNS = (
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
RUN_ARTIFACTS = {
    "region_manifest": "region_manifest.json",
    "tile_manifest": "tile_manifest.csv",
    "inventory_report": "inventory_report.json",
    "acquisition_preflight": "acquisition_preflight.json",
    "annotation_batch": "annotation_batch.csv",
    "batch_manifest": "batch_manifest.json",
    "candidate_pool": "candidate_pool.csv",
    "feature_embeddings": "feature_embeddings.npz",
    "scoring_manifest": "scoring_manifest.json",
    "selection_diagnostics": "selection_diagnostics.json",
    "geographic_splits": "geographic_splits.csv",
    "leakage_audit": "leakage_audit.json",
}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "t",
        "on",
        "completed",
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing artifact: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON artifact must be an object: {path}"
    return value, None


def _read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"missing artifact: {path}"
    try:
        return pd.read_csv(path, low_memory=False), None
    except (OSError, ValueError) as exc:
        return None, f"invalid CSV {path}: {exc}"


def _existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _artifact(path: Path | None, root: Path, required: bool) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "required": required,
            "sha256": None,
            "bytes": None,
        }
    try:
        name = str(path.relative_to(root))
    except ValueError:
        name = str(path)
    if not path.exists() or not path.is_file():
        return {
            "path": name,
            "exists": False,
            "required": required,
            "sha256": None,
            "bytes": None,
        }
    return {
        "path": name,
        "exists": True,
        "required": required,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _resolve(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _previous_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result: dict[str, str] = {}
    for entry in (
        value.get("artifacts", {}) if isinstance(value, dict) else {}
    ).values():
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str)
        ):
            result[entry["path"]] = entry["sha256"]
    return result


def _validate_tile(path: Path, root: Path, blockers: list[str]) -> tuple[bool, int]:
    frame, error = _read_csv(path)
    if error:
        blockers.append(error)
        return False, 0
    assert frame is not None
    missing = sorted(set(TILE_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        if missing:
            blockers.append(f"tile manifest missing columns: {missing}")
        if frame.empty:
            blockers.append("tile manifest is empty")
        return False, len(frame)
    zooms = pd.to_numeric(frame["z"], errors="coerce")
    if zooms.isna().any() or not zooms.eq(14).all():
        blockers.append("tile manifest must contain only zoom-14 rows")
    if len(frame) > 370_000:
        blockers.append("tile manifest exceeds 370000-coordinate hard cap")
    if frame.duplicated(["region", "z", "x", "y"]).any():
        blockers.append("tile manifest contains duplicate tile identities")

    present = pd.Series(True, index=frame.index)
    for style in ("satellite", "terrain"):
        available = frame[f"{style}_present"].map(_bool)
        if f"{style}_s3_present" in frame and f"{style}_s3_uri" in frame:
            available |= frame[f"{style}_s3_present"].map(_bool) & frame[
                f"{style}_s3_uri"
            ].astype(str).str.startswith("s3://")
        present &= available
    if not bool(present.all()):
        blockers.append("tile manifest contains incomplete image pairs")
    for _, row in frame.iterrows():
        for style in ("satellite", "terrain"):
            raw = str(row.get(f"{style}_path", "")).strip()
            local = Path(raw)
            if not local.is_absolute() and not local.exists():
                local = root / local
            remote = _bool(row.get(f"{style}_s3_present")) and str(
                row.get(f"{style}_s3_uri", "")
            ).startswith("s3://")
            if not local.exists() and not remote:
                blockers.append(f"missing imagery referenced by tile manifest: {raw}")
                return False, len(frame)
    return bool(present.all()), len(frame)


def _validate_annotations(path: Path | None, blockers: list[str]) -> tuple[bool, int]:
    if path is None:
        blockers.append("missing absolute annotations CSV")
        return False, 0
    frame, error = _read_csv(path)
    if error:
        blockers.append(error)
        return False, 0
    assert frame is not None
    missing = sorted(set(ABSOLUTE_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(ABSOLUTE_COLUMNS))
    if missing:
        blockers.append(f"absolute annotations missing columns: {missing}")
    if extra:
        blockers.append(
            f"absolute annotations violate seven-column contract; extra columns: {extra}"
        )
    if frame.empty:
        blockers.append("absolute annotations are empty")
        return False, 0
    if missing or extra:
        return False, 0
    scores = pd.to_numeric(frame["scenic_human"], errors="coerce")
    image_paths = frame["image_path"].astype(str).str.strip()
    annotators = frame["annotator_id"].astype(str).str.strip()
    skipped = frame["skip"].map(_bool)
    completed = scores.notna() & image_paths.ne("") & ~skipped
    if (scores.notna() & ~scores.between(0.0, 10.0)).any():
        blockers.append("absolute annotations contain scores outside [0, 10]")
    valid_confidence = {"low", "medium", "high"}
    confidence = frame["confidence"].astype(str).str.strip().str.lower()
    if (~confidence.isin(valid_confidence)).any():
        blockers.append("absolute annotations contain invalid confidence values")
    if image_paths.eq("").any() or annotators.eq("").any():
        blockers.append("absolute annotations contain empty image or annotator identity")
    if frame.assign(
        _image_path=image_paths, _annotator_id=annotators
    ).duplicated(["_image_path", "_annotator_id"]).any():
        blockers.append("absolute annotations contain duplicate annotator/image records")
    if not bool(completed.any()):
        blockers.append("absolute annotations contain no completed labels")
    valid = not any(
        blocker.startswith("absolute annotations") for blocker in blockers
    )
    return valid and bool(completed.any()), int(completed.sum())


def _validate_batch(path: Path, blockers: list[str]) -> tuple[bool, int]:
    frame, error = _read_csv(path)
    if error:
        blockers.append(error)
        return False, 0
    assert frame is not None
    required = {
        "image_path",
        "selection_reason",
        "selection_score",
        "selection_rank",
        "batch_id",
        "run_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        blockers.append(f"annotation batch missing columns: {missing}")
    if frame.empty:
        blockers.append("annotation batch is empty")
    if "image_path" in frame and frame["image_path"].duplicated().any():
        blockers.append("annotation batch contains duplicate image paths")
    return not missing and not frame.empty, len(frame)


def _validate_splits(
    split_path: Path, audit_path: Path, blockers: list[str]
) -> tuple[bool, int, dict[str, Any]]:
    frame, error = _read_csv(split_path)
    if error:
        blockers.append(error)
        return False, 0, {}
    audit, audit_error = _read_json(audit_path)
    if audit_error:
        blockers.append(audit_error)
    assert frame is not None
    missing = sorted({"image_path", "split"} - set(frame.columns))
    if missing:
        blockers.append(f"geographic splits missing columns: {missing}")
    splits = set(frame["split"].astype(str)) if "split" in frame else set()
    if not splits or not splits.issubset({"train", "val", "validation", "test"}):
        blockers.append("geographic splits contain invalid assignments")
    if frame.empty:
        blockers.append("geographic splits are empty")
    for expected in ("train", "val", "test"):
        aliases = {expected} | ({"validation"} if expected == "val" else set())
        if not splits.intersection(aliases):
            blockers.append(f"geographic split has no {expected} rows")
    if not isinstance(audit, dict) or audit.get("valid") is not True:
        blockers.append("geographic leakage audit is invalid")
    return (
        not missing
        and not frame.empty
        and isinstance(audit, dict)
        and audit.get("valid") is True,
        len(frame),
        audit if isinstance(audit, dict) else {},
    )


def _validate_human_table(
    path: Path | None, blockers: list[str], *, benchmark: bool = False
) -> tuple[bool, int]:
    if path is None:
        blockers.append(
            "missing human benchmark CSV" if benchmark else "missing mixed-label CSV"
        )
        return False, 0
    frame, error = _read_csv(path)
    if error:
        blockers.append(error)
        return False, 0
    assert frame is not None
    if benchmark:
        score = (
            "scenic_human"
            if "scenic_human" in frame
            else "scenic_human_mean"
            if "scenic_human_mean" in frame
            else None
        )
        valid = (
            "image_path" in frame
            and score is not None
            and pd.to_numeric(frame[score], errors="coerce").notna().any()
            and not frame.empty
        )
        if not valid:
            blockers.append("benchmark lacks non-empty human image/score rows")
        if (
            "label_source" in frame
            and frame["label_source"].astype(str).str.lower().eq("heuristic").all()
        ):
            blockers.append("benchmark contains weak labels only")
            valid = False
        return bool(valid), len(frame)
    valid = {"image_path", "scenic_score"}.issubset(frame.columns) and not frame.empty
    if not valid:
        blockers.append("mixed-label CSV lacks image_path/scenic_score rows")
    return bool(valid), len(frame)


def _validate_registry(
    path: Path, checkpoint: Path | None, blockers: list[str]
) -> tuple[bool, dict[str, Any]]:
    registry, error = _read_json(path)
    if error:
        blockers.append(error)
        return False, {}
    assert registry is not None
    active = registry.get("active")
    if not isinstance(active, dict) or not active.get("checkpoint"):
        blockers.append("baseline registry has no active checkpoint")
        return False, {}
    checkpoint = checkpoint or _resolve(str(active["checkpoint"]), Path.cwd())
    if checkpoint is None or not checkpoint.exists():
        blockers.append(f"baseline checkpoint missing: {active.get('checkpoint')}")
        return False, {"active": active}
    return True, {
        "registry_path": str(path),
        "registry_sha256": sha256_file(path),
        "active": active,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _validate_acquisition_preflight(path: Path, blockers: list[str]) -> bool:
    payload, error = _read_json(path)
    if error:
        blockers.append(error)
        return False
    assert payload is not None
    if payload.get("budget_valid") is not True:
        blockers.append("acquisition preflight budget_valid is not true")
        return False
    return True


def _scoring_artifact_hash(
    manifest: Mapping[str, Any], name: str
) -> str | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    candidates = (name, name.removesuffix(".csv"), name.removesuffix(".npz"))
    for key in candidates:
        entry = artifacts.get(key)
        if isinstance(entry, Mapping) and isinstance(entry.get("sha256"), str):
            return str(entry["sha256"])
        if isinstance(entry, str):
            return entry
    return None


def _validate_scoring_artifacts(
    candidate_path: Path,
    embedding_path: Path,
    scoring_path: Path,
    root: Path,
    blockers: list[str],
) -> tuple[bool, int, int]:
    """Validate the selector handoff table, dense arrays, and their hashes."""
    blocker_start = len(blockers)

    candidate, error = _read_csv(candidate_path)
    if error:
        blockers.append(error)
        return False, 0, 0
    scoring, scoring_error = _read_json(scoring_path)
    if scoring_error:
        blockers.append(scoring_error)
        return False, 0, 0
    assert candidate is not None and scoring is not None
    if scoring.get("schema_version") != SCORING_SCHEMA_VERSION:
        blockers.append("scoring manifest schema version mismatch")
    required_columns = {
        "image_path",
        "source_identity",
        "satellite_path",
        "terrain_path",
        "score_status",
        "selector_eligible",
        "heuristic_score",
        "scenic_score",
        "scenic_score_heuristic",
        "regression_prediction",
        "normalized_class_entropy",
        "embedding_row_index",
    }
    missing = sorted(required_columns - set(candidate.columns))
    if missing:
        blockers.append(f"candidate pool missing columns: {missing}")
    if candidate.empty:
        blockers.append("candidate pool is empty")
    scored = (
        candidate["score_status"].astype(str).str.lower().eq("scored")
        if "score_status" in candidate
        else pd.Series(False, index=candidate.index)
    )
    if not bool(scored.any()):
        blockers.append("candidate pool contains no successfully scored rows")
    if bool(scored.any()) and not bool(scored.all()):
        blockers.append("candidate pool contains unscored or failed rows")
    if "regression_prediction" in candidate and not bool(
        pd.to_numeric(candidate.loc[scored, "regression_prediction"], errors="coerce")
        .notna()
        .any()
    ):
        blockers.append("candidate pool contains no active regression predictions")
    row_indices: np.ndarray | None = None
    try:
        with np.load(embedding_path, allow_pickle=False) as arrays:
            if "embeddings" not in arrays or "row_indices" not in arrays:
                blockers.append("feature embeddings NPZ lacks embeddings/row_indices arrays")
                embedding_rows = 0
            else:
                embeddings = np.asarray(arrays["embeddings"])
                row_indices = np.asarray(arrays["row_indices"])
                embedding_rows = int(len(embeddings))
                if embeddings.ndim != 2 or embedding_rows <= 0 or embeddings.shape[1] <= 0:
                    blockers.append("feature embeddings NPZ embeddings array is empty or not 2-D")
                if (
                    row_indices.ndim != 1
                    or len(row_indices) != embedding_rows
                    or not np.issubdtype(row_indices.dtype, np.integer)
                    or len(np.unique(row_indices)) != len(row_indices)
                ):
                    blockers.append("feature embeddings NPZ row_indices are invalid")
                if (
                    not np.issubdtype(embeddings.dtype, np.number)
                    or not np.isfinite(embeddings).all()
                ):
                    blockers.append("feature embeddings NPZ embeddings are not finite numeric values")
    except (OSError, ValueError, TypeError) as exc:
        blockers.append(f"invalid feature embeddings NPZ: {exc}")
        embedding_rows = 0
    if "embedding_row_index" in candidate:
        indices = pd.to_numeric(
            candidate.loc[scored, "embedding_row_index"], errors="coerce"
        )
        invalid_indices = (
            indices.isna()
            | (indices < 0)
            | (indices >= max(embedding_rows, 1))
            | (indices % 1 != 0)
        )
        if invalid_indices.any():
            blockers.append("candidate pool embedding row indices are invalid")
        elif row_indices is not None and len(row_indices) == embedding_rows:
            compact = indices.astype(int).to_numpy()
            expected_source_rows = np.flatnonzero(scored.to_numpy())
            if not np.array_equal(row_indices[compact], expected_source_rows):
                blockers.append(
                    "candidate pool embedding indices do not match NPZ source rows"
                )
    for name, path in (
        ("candidate_pool.csv", candidate_path),
        ("feature_embeddings.npz", embedding_path),
    ):
        expected = _scoring_artifact_hash(scoring, name)
        if expected is None:
            blockers.append(f"scoring manifest lacks hash for {name}")
        elif sha256_file(path).lower() != expected.lower():
            blockers.append(f"scoring artifact hash mismatch: {name}")
    state = scoring.get("state")
    if not isinstance(state, Mapping) or state.get("complete") is not True:
        blockers.append("scoring manifest is not complete")
    if not isinstance(state, Mapping) or state.get("ready_for_selection") is not True:
        blockers.append("scoring manifest is not ready for selection")
    counts = scoring.get("counts")
    if not isinstance(counts, Mapping):
        blockers.append("scoring manifest lacks counts")
    else:
        try:
            manifest_rows = int(counts["manifest_rows"])
            scored_rows = int(counts["scored_rows"])
            missing_rows = int(counts["missing_rows"])
            error_rows = int(counts["error_rows"])
        except (KeyError, TypeError, ValueError, OverflowError):
            blockers.append("scoring manifest contains invalid counts")
        else:
            if (
                manifest_rows != len(candidate)
                or scored_rows != int(scored.sum())
                or missing_rows != 0
                or error_rows != 0
            ):
                blockers.append("scoring manifest reports inconsistent or failed rows")
    return (
        not blockers[blocker_start:]
        and not candidate.empty
        and bool(scored.any())
        and embedding_rows > 0
        and isinstance(state, Mapping)
        and state.get("ready_for_selection") is True,
        int(len(candidate)),
        embedding_rows,
    )
def _validate_lineage(
    paths: Mapping[str, Path | None],
    baseline: Mapping[str, Any],
    blockers: list[str],
) -> bool:
    """Reject stale or cross-run artifacts even when their row counts agree."""
    blocker_start = len(blockers)
    required_json = ("scoring_manifest", "batch_manifest", "selection_diagnostics")
    payloads: dict[str, dict[str, Any]] = {}
    for name in required_json:
        path = paths.get(name)
        if path is None:
            blockers.append(f"missing lineage artifact: {name}")
            continue
        payload, error = _read_json(path)
        if error:
            blockers.append(error)
        elif payload is not None:
            payloads[name] = payload

    tile_path = paths.get("tile_manifest")
    candidate_path = paths.get("candidate_pool")
    scoring = payloads.get("scoring_manifest", {})
    scoring_source = scoring.get("source", {})
    tile_source = (
        scoring_source.get("tile_manifest", {})
        if isinstance(scoring_source, Mapping)
        else {}
    )
    expected_tile_hash = (
        tile_source.get("sha256") if isinstance(tile_source, Mapping) else None
    )
    if (
        tile_path is None
        or not isinstance(expected_tile_hash, str)
        or sha256_file(tile_path).lower() != expected_tile_hash.lower()
    ):
        blockers.append("scoring manifest tile source hash mismatch")

    scoring_models = scoring.get("models", {})
    scoring_regression_hash = (
        scoring_models.get("regression_checkpoint_sha256")
        if isinstance(scoring_models, Mapping)
        else None
    )
    baseline_hash = baseline.get("checkpoint_sha256")
    if (
        not isinstance(scoring_regression_hash, str)
        or not isinstance(baseline_hash, str)
        or scoring_regression_hash.lower() != baseline_hash.lower()
    ):
        blockers.append("scoring regression checkpoint does not match baseline")

    for name in ("batch_manifest", "selection_diagnostics"):
        payload = payloads.get(name, {})
        candidate_input = payload.get("candidate_input", {})
        expected = (
            candidate_input.get("sha256")
            if isinstance(candidate_input, Mapping)
            else None
        )
        if (
            candidate_path is None
            or not isinstance(expected, str)
            or sha256_file(candidate_path).lower() != expected.lower()
        ):
            blockers.append(f"{name} candidate source hash mismatch")

    batch_path = paths.get("annotation_batch")
    annotations_path = paths.get("absolute_annotations")
    if batch_path is None or annotations_path is None:
        return False
    batch, batch_error = _read_csv(batch_path)
    annotations, annotations_error = _read_csv(annotations_path)
    if batch_error:
        blockers.append(batch_error)
    if annotations_error:
        blockers.append(annotations_error)
    if batch is None or annotations is None:
        return False
    batch_manifest = payloads.get("batch_manifest", {})
    try:
        declared_rows = int(batch_manifest["row_count"])
    except (KeyError, TypeError, ValueError, OverflowError):
        blockers.append("batch manifest contains invalid row_count")
    else:
        if declared_rows != len(batch):
            blockers.append("batch manifest row count does not match annotation batch")
    declared_batch_id = str(batch_manifest.get("batch_id", "")).strip()
    if (
        not declared_batch_id
        or "batch_id" not in batch
        or not batch["batch_id"].astype(str).eq(declared_batch_id).all()
    ):
        blockers.append("annotation batch identity does not match batch manifest")

    if "image_path" in batch and set(ABSOLUTE_COLUMNS).issubset(annotations.columns):
        completed = annotations[
            pd.to_numeric(annotations["scenic_human"], errors="coerce").notna()
            & ~annotations["skip"].map(_bool)
        ]
        required_images = set(batch["image_path"].astype(str).str.strip())
        completed_images = set(completed["image_path"].astype(str).str.strip())
        missing_images = sorted(required_images - completed_images)
        if missing_images:
            blockers.append(
                "annotation batch has "
                f"{len(missing_images)} images without completed human labels"
            )
    return not blockers[blocker_start:]




def finalize_stage1(
    run_root: str | Path,
    *,
    run_name: str | None = None,
    annotations_csv: str | Path | None = None,
    mixed_labels_csv: str | Path | None = None,
    benchmark_csv: str | Path | None = None,
    registry_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    seeds: Mapping[str, int] | None = None,
    material_config: Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    handoff_path = root / "stage1_handoff.json"
    blockers: list[str] = []
    paths: dict[str, Path | None] = {
        key: root / value for key, value in RUN_ARTIFACTS.items()
    }
    paths["absolute_annotations"] = (
        Path(annotations_csv)
        if annotations_csv
        else _existing(root, ("absolute_annotations.csv", "annotations.csv"))
    )
    if (
        paths["absolute_annotations"] is None
        and Path("data/raw/labels_human.csv").exists()
    ):
        paths["absolute_annotations"] = Path("data/raw/labels_human.csv")
    paths["mixed_labels"] = (
        Path(mixed_labels_csv)
        if mixed_labels_csv
        else _existing(root, ("mixed_labels.csv", "labels_mixed.csv"))
    )
    paths["benchmark"] = (
        Path(benchmark_csv)
        if benchmark_csv
        else _existing(
            root, ("benchmark_split.csv", "benchmark.csv", "challenge_benchmark.csv")
        )
    )
    paths["baseline_registry"] = (
        Path(registry_path)
        if registry_path
        else Path("data/processed/regression/model_registry.json")
    )
    paths["baseline_checkpoint"] = Path(checkpoint_path) if checkpoint_path else None
    required = set(RUN_ARTIFACTS) | {
        "absolute_annotations",
        "mixed_labels",
        "benchmark",
        "baseline_registry",
    }
    records = {
        key: _artifact(path, root, key in required) for key, path in paths.items()
    }
    for key, entry in records.items():
        if entry["required"] and not entry["exists"]:
            blockers.append(f"missing required artifact: {key}")
    previous = _previous_hashes(handoff_path)
    for relative, digest in previous.items():
        path = _resolve(relative, root)
        if (
            path is not None
            and path.exists()
            and path.name != handoff_path.name
            and sha256_file(path) != digest
        ):
            blockers.append(f"hash mismatch against previous handoff: {relative}")
    for key, digest in (expected_hashes or {}).items():
        path = paths.get(key, root / key)
        if (
            path is None
            or not path.exists()
            or sha256_file(path).lower() != str(digest).lower()
        ):
            blockers.append(f"hash mismatch for {key}")
    region, error = _read_json(
        paths["region_manifest"] or root / RUN_ARTIFACTS["region_manifest"]
    )
    if error:
        blockers.append(error)
    elif (
        region is not None
        and region.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION
    ):
        blockers.append("region manifest schema version mismatch")
    inventory, error = _read_json(
        paths["inventory_report"] or root / RUN_ARTIFACTS["inventory_report"]
    )
    if error:
        blockers.append(error)
    acquisition_budget_ok = _validate_acquisition_preflight(
        paths["acquisition_preflight"] or root / RUN_ARTIFACTS["acquisition_preflight"],
        blockers,
    )
    tile_ok, tile_rows = _validate_tile(
        paths["tile_manifest"] or root / RUN_ARTIFACTS["tile_manifest"], root, blockers
    )
    batch_ok, batch_rows = _validate_batch(
        paths["annotation_batch"] or root / RUN_ARTIFACTS["annotation_batch"], blockers
    )
    annotation_ok, annotation_rows = _validate_annotations(
        paths["absolute_annotations"], blockers
    )
    split_ok, split_rows, audit = _validate_splits(
        paths["geographic_splits"] or root / RUN_ARTIFACTS["geographic_splits"],
        paths["leakage_audit"] or root / RUN_ARTIFACTS["leakage_audit"],
        blockers,
    )
    benchmark_ok, benchmark_rows = _validate_human_table(
        paths["benchmark"], blockers, benchmark=True
    )
    mixed_ok, mixed_rows = _validate_human_table(paths["mixed_labels"], blockers)
    registry_ok, baseline = _validate_registry(
        paths["baseline_registry"]
        or Path("data/processed/regression/model_registry.json"),
        paths["baseline_checkpoint"],
        blockers,
    )
    scoring_ok, scoring_rows, embedding_rows = _validate_scoring_artifacts(
        paths["candidate_pool"] or root / RUN_ARTIFACTS["candidate_pool"],
        paths["feature_embeddings"] or root / RUN_ARTIFACTS["feature_embeddings"],
        paths["scoring_manifest"] or root / RUN_ARTIFACTS["scoring_manifest"],
        root,
        blockers,
    )
    lineage_ok = _validate_lineage(paths, baseline, blockers)
    if scoring_rows != tile_rows:
        blockers.append(
            f"candidate pool row count {scoring_rows} does not match tile manifest {tile_rows}"
        )
        scoring_ok = False
    records = {
        key: _artifact(path, root, entry["required"])
        for key, (path, entry) in (
            (key, (paths[key], value)) for key, value in records.items()
        )
    }
    readiness = {
        "data_complete": bool(region is not None and inventory is not None and tile_ok),
        "acquisition_budget_valid": bool(acquisition_budget_ok),
        "scoring_valid": bool(scoring_ok),
        "annotations_valid": bool(batch_ok and annotation_ok),
        "splits_valid": bool(split_ok),
        "benchmark_valid": bool(benchmark_ok),
        "baseline_valid": bool(registry_ok),
        "hashes_valid": bool(
            lineage_ok
            and not any("hash mismatch" in blocker for blocker in blockers)
        ),
    }
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name or root.name,
        "run_root": str(root),
        "artifacts": records,
        "artifact_hashes": {key: entry["sha256"] for key, entry in records.items()},
        "counts": {
            "tile_rows": tile_rows,
            "batch_rows": batch_rows,
            "annotation_rows": annotation_rows,
            "split_rows": split_rows,
            "benchmark_rows": benchmark_rows,
            "mixed_label_rows": mixed_rows,
            "candidate_pool_rows": scoring_rows,
            "embedding_rows": embedding_rows,
        },
        "baseline": baseline,
        "seeds": jsonable(dict(seeds or {})),
        "material_config": jsonable(dict(material_config or {})),
        "leakage_audit": audit,
        "incomplete_work": blockers,
        "blockers": blockers,
        "readiness": readiness,
        **readiness,
        "ready_for_stage2": bool(all(readiness.values()) and mixed_ok and not blockers),
    }
    if write:
        atomic_write_json(handoff_path, jsonable(handoff))
    return jsonable(handoff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize an active-learning stage-one run"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--mixed-labels-csv", type=Path)
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-hash", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected: dict[str, str] = {}
    for item in args.expected_hash:
        if "=" not in item:
            raise SystemExit("--expected-hash uses NAME=SHA256")
        key, digest = item.split("=", 1)
        expected[key] = digest
    result = finalize_stage1(
        args.run_root,
        run_name=args.run_name,
        annotations_csv=args.annotations_csv,
        mixed_labels_csv=args.mixed_labels_csv,
        benchmark_csv=args.benchmark_csv,
        registry_path=args.registry,
        checkpoint_path=args.checkpoint,
        expected_hashes=expected,
    )
    print(
        json.dumps(
            {
                "stage1_handoff": str(Path(args.run_root) / "stage1_handoff.json"),
                "ready_for_stage2": result["ready_for_stage2"],
                "blockers": result["blockers"],
            },
            sort_keys=True,
        )
    )
    if not result["ready_for_stage2"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
