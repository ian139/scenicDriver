"""Fail-closed validation and atomic publication of a stage-one handoff."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import (
    atomic_write_json,
    atomic_write_text,
    jsonable,
    sha256_bytes,
    sha256_file,
)
from .scoring import SCORING_SCHEMA_VERSION
from .selection import audit_geographic_leakage

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
# Explicit unusable-reason tokens annotators may record in notes for skipped
# decisions (matches the annotator web tool's UNUSABLE_REASONS contract).
UNUSABLE_REASONS = (
    "missing_imagery",
    "corrupted_image",
    "cloud_or_obstruction",
    "excessive_water",
    "duplicate",
    "other",
)
_TILE_ZOOM_RE = re.compile(r"^z(\d+)$", re.IGNORECASE)
_TILE_COORDS_RE = re.compile(r"z(\d+)[/_-]*x(\d+)[/_-]*y(\d+)", re.IGNORECASE)
_TILE_STEM_RE = re.compile(r"^(\d+)[_-](\d+)$")
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
    "filtered_index": "filtered_index.csv",
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
        name = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        name = str(path.resolve())
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


def _registry_checkpoint_candidates(
    registry_path: Path, raw_checkpoint: str
) -> list[Path]:
    """Ordered candidate locations for a stored registry checkpoint.

    The value itself (absolute or working-directory-relative), then relative to
    the registry directory (checkpoints/<sha>.pt), then project-root-relative
    (models/...) against the working directory.
    """
    checkpoint_value = Path(raw_checkpoint)
    candidates = [checkpoint_value]
    if not checkpoint_value.is_absolute():
        candidates += [
            registry_path.parent / checkpoint_value,
            Path.cwd() / checkpoint_value,
        ]
    return candidates


def _resolve_registry_checkpoint(
    registry_path: Path, raw_checkpoint: str
) -> Path | None:
    """Resolve a stored registry checkpoint, preferring an existing file.

    Returns the first existing candidate location, or None when none exists.
    """
    for candidate in _registry_checkpoint_candidates(registry_path, raw_checkpoint):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _previous_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict) or value.get("ready_for_stage2") is not True:
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


def _stable_identity(image_path: Any) -> str:
    """Return the stable ``region/z/x/y`` identity of a canonical image path.

    Canonical tile paths follow ``.../z{zoom}/{region}/{x}_{y}.png`` (plus the
    compact ``z{zoom}x{x}y{y}`` / ``z{zoom}/x{x}/y{y}`` forms), so the region
    is the component immediately after the zoom component.  Paths that cannot
    be parsed into tile coordinates fall back to their exact stripped value so
    non-canonical identities keep matching on the literal path.
    """
    text = str(image_path).strip()
    if not text:
        return ""
    parts = PurePosixPath(text).parts
    zoom: int | None = None
    x: int | None = None
    y: int | None = None
    region = ""
    compact = _TILE_COORDS_RE.search(text)
    if compact is not None:
        zoom, x, y = (int(compact.group(index)) for index in (1, 2, 3))
    else:
        for index, part in enumerate(parts):
            zoom_match = _TILE_ZOOM_RE.match(part)
            if zoom_match is not None:
                zoom = int(zoom_match.group(1))
                if index + 1 < len(parts) and parts[index + 1] != parts[-1]:
                    region = parts[index + 1]
                break
        stem = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
        stem_match = _TILE_STEM_RE.match(stem)
        if stem_match is not None:
            x, y = int(stem_match.group(1)), int(stem_match.group(2))
    if zoom is None or x is None or y is None:
        return text
    return f"{region or 'unknown'}/z{zoom}/x{x}/y{y}"


def _supported_notes_reasons(notes: Any) -> list[str]:
    """Return the supported unusable reasons named by *notes* as bare tokens."""
    tokens = set(re.split(r"[^\w]+", str(notes).strip().lower()))
    return [reason for reason in UNUSABLE_REASONS if reason in tokens]


def _snapshot_annotations_frame(
    source: pd.DataFrame, batch: pd.DataFrame, blockers: list[str]
) -> pd.DataFrame | None:
    """Return source rows whose stable tile identity appears in the batch."""
    if "image_path" not in source.columns:
        blockers.append("absolute annotation source missing image_path")
        return None
    if "image_path" not in batch.columns:
        blockers.append("annotation batch missing image_path for snapshot")
        return None
    batch_identities = set(
        batch["image_path"].astype(str).str.strip().map(_stable_identity)
    )
    if not batch_identities:
        blockers.append("annotation batch has no image paths for snapshot")
        return None
    source_identities = (
        source["image_path"].astype(str).str.strip().map(_stable_identity)
    )
    return source.loc[source_identities.isin(batch_identities)]


def _snapshot_annotations(
    source_path: Path | None,
    batch_path: Path | None,
    destination: Path,
    blockers: list[str],
    *,
    write: bool,
) -> Path | None:
    if source_path is None:
        return None
    if batch_path is None:
        blockers.append("missing annotation batch for absolute annotation snapshot")
        return None
    source, source_error = _read_csv(source_path)
    batch, batch_error = _read_csv(batch_path)
    if source_error:
        blockers.append(source_error)
    if batch_error:
        blockers.append(batch_error)
    if source is None or batch is None:
        return None
    snapshot = _snapshot_annotations_frame(source, batch, blockers)
    if snapshot is None:
        return None
    if write:
        atomic_write_text(
            destination,
            snapshot.to_csv(index=False, lineterminator="\n"),
        )
        return destination
    return source_path


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
            if not local.is_absolute():
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
        blockers.append(
            "absolute annotations contain empty image or annotator identity"
        )
    identities = image_paths.map(_stable_identity)
    duplicate_markers = frame.assign(
        _identity=identities, _annotator_id=annotators
    ).duplicated(["_identity", "_annotator_id"])
    if duplicate_markers.any():
        blockers.append(
            "absolute annotations contain duplicate or conflicting annotator "
            "records for the same tile identity: "
            + ", ".join(sorted(set(identities[duplicate_markers])))
        )
    if skipped.any():
        notes_reasons = frame["notes"].astype(str).map(_supported_notes_reasons)
        missing_reason = sorted(
            identities[skipped][notes_reasons[skipped].map(len).eq(0)]
        )
        conflicting_reasons = sorted(
            identities[skipped][notes_reasons[skipped].map(len).gt(1)]
        )
        if missing_reason:
            blockers.append(
                "absolute annotations contain skipped decisions missing a "
                "supported unusable reason in notes: " + ", ".join(missing_reason)
            )
        if conflicting_reasons:
            blockers.append(
                "absolute annotations contain skipped decisions listing "
                "multiple supported unusable reasons in notes: "
                + ", ".join(conflicting_reasons)
            )
    if not bool(completed.any()):
        blockers.append("absolute annotations contain no completed labels")
    valid = not any(blocker.startswith("absolute annotations") for blocker in blockers)
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
    split_path: Path,
    audit_path: Path,
    blockers: list[str],
    *,
    manifest: Mapping[str, Any] | None = None,
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
    outputs = manifest.get("outputs") if isinstance(manifest, Mapping) else None
    split_record = (
        outputs.get("geographic_splits_csv") if isinstance(outputs, Mapping) else None
    )
    audit_record = (
        outputs.get("leakage_audit_json") if isinstance(outputs, Mapping) else None
    )
    expected_split_hash = (
        split_record.get("sha256") if isinstance(split_record, Mapping) else None
    )
    expected_audit_hash = (
        audit_record.get("sha256") if isinstance(audit_record, Mapping) else None
    )
    if not isinstance(expected_split_hash, str):
        blockers.append("batch manifest lacks geographic splits output hash")
    elif sha256_file(split_path).lower() != expected_split_hash.lower():
        blockers.append("geographic splits hash mismatch against batch manifest")
    if not isinstance(expected_audit_hash, str):
        blockers.append("batch manifest lacks leakage audit output hash")
    elif sha256_file(audit_path).lower() != expected_audit_hash.lower():
        blockers.append("leakage audit hash mismatch against batch manifest")
    if isinstance(audit, dict) and audit.get("valid") is True:
        selection_config = (
            manifest.get("selection_config") if isinstance(manifest, Mapping) else None
        )
        try:
            radius = int(
                selection_config.get("adjacency_radius", 1)
                if isinstance(selection_config, Mapping)
                else 1
            )
        except (TypeError, ValueError):
            radius = 1
        recomputed = audit_geographic_leakage(frame, adjacency_radius=radius)
        if recomputed.get("valid") is not True:
            blockers.append(
                "geographic leakage audit does not match admitted split rows"
            )
    return (
        not missing
        and not frame.empty
        and isinstance(audit, dict)
        and audit.get("valid") is True
        and isinstance(expected_split_hash, str)
        and isinstance(expected_audit_hash, str),
        len(frame),
        audit if isinstance(audit, dict) else {},
    )


def _benchmark_target_column(frame: pd.DataFrame) -> str | None:
    """Return the canonical human-target column of a benchmark table.

    ``scenic_human_mean`` is preferred when present, matching the lineage
    validator and the Stage Two evaluator's per-record fallback order.
    """
    if "scenic_human_mean" in frame.columns:
        return "scenic_human_mean"
    if "scenic_human" in frame.columns:
        return "scenic_human"
    return None


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
        score = _benchmark_target_column(frame)
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
    valid = {"image_path", "scenic_score", "label_source"}.issubset(
        frame.columns
    ) and not frame.empty
    if not valid:
        blockers.append(
            "mixed-label CSV lacks image_path/scenic_score/label_source rows"
        )
        return False, len(frame)
    sources = frame["label_source"].astype(str).str.strip().str.lower()
    if not sources.eq("human_override").any():
        blockers.append("mixed-label CSV contains no human_override provenance")
        valid = False
    return bool(valid), len(frame)


def _validate_benchmark_pre_dataset(
    benchmark_path: Path | None,
    blockers: list[str],
    *,
    name: str,
) -> bool:
    """Validate Stage Two pre-dataset invariants on a benchmark table.

    Mirrors the benchmark-contract checks ``scenic_scorer.active_evaluation``
    applies before evaluation: an explicit ``split`` column, at least one
    ``split=test`` row, human targets drawn from ``scenic_human_mean`` or
    ``scenic_human`` only, finite targets bounded to [0, 10], and unique test
    image identities.  Absence of the file is reported by the required-artifact
    and human-table checks; this helper only fails closed on content.
    """
    if benchmark_path is None:
        return False
    frame, error = _read_csv(benchmark_path)
    if error:
        blockers.append(error)
        return False
    assert frame is not None
    if "split" not in frame.columns:
        blockers.append(f"{name} requires an explicit split column")
        return False
    if "image_path" not in frame.columns:
        blockers.append(f"{name} record lacks image_path")
        return False
    test_mask = frame["split"].astype(str).str.strip().str.lower().eq("test")
    test_rows = frame.loc[test_mask]
    raw_paths = test_rows["image_path"]
    missing_paths = raw_paths.isna() | raw_paths.astype(str).str.strip().eq("")
    if missing_paths.any():
        blockers.append(f"{name} record lacks image_path")
        return False
    if test_rows.empty:
        blockers.append(f"{name} contains no split=test rows")
        return False
    target = _benchmark_target_column(frame)
    if target is None:
        blockers.append(
            f"{name} test target must be scenic_human_mean or scenic_human, "
            "never weak/mixed scenic_score"
        )
        return False
    raw_scores = test_rows[target]
    missing_scores = raw_scores.isna() | raw_scores.astype(str).str.strip().eq("")
    if missing_scores.any():
        blockers.append(
            f"{name} test target must be scenic_human_mean or scenic_human, "
            "never weak/mixed scenic_score"
        )
        return False
    scores = pd.to_numeric(raw_scores, errors="coerce")
    if scores.isna().any() or not scores.between(0.0, 10.0).all():
        blockers.append(f"{name} human targets must be finite and in [0, 10]")
        return False
    test_identities = (
        test_rows["image_path"].astype(str).str.strip().map(_stable_identity)
    )
    duplicate_identities = sorted(set(test_identities[test_identities.duplicated()]))
    if duplicate_identities:
        blockers.append(
            f"{name} contains duplicate test image paths: "
            + ", ".join(duplicate_identities)
        )
        return False
    return True


def _benchmark_test_identities(frame: pd.DataFrame) -> set[str]:
    """Stable identities of a benchmark table's split=test rows."""
    if "image_path" not in frame or "split" not in frame:
        return set()
    test = frame["split"].astype(str).str.strip().str.lower().eq("test")
    return set(
        frame.loc[test, "image_path"].astype(str).str.strip().map(_stable_identity)
    )


def _validate_benchmark_test_overlap(
    benchmark_path: Path | None,
    control_path: Path | None,
    blockers: list[str],
) -> bool:
    """Reject expanded/control benchmark tables sharing split=test identities."""
    if benchmark_path is None or control_path is None:
        return True
    benchmark, benchmark_error = _read_csv(benchmark_path)
    control, control_error = _read_csv(control_path)
    if benchmark_error:
        blockers.append(benchmark_error)
    if control_error:
        blockers.append(control_error)
    if benchmark is None or control is None:
        return False
    overlap = sorted(
        _benchmark_test_identities(benchmark) & _benchmark_test_identities(control)
    )
    if overlap:
        blockers.append(
            "Overlap detected between expanded and control benchmark "
            "split=test image identities: " + ", ".join(overlap)
        )
        return False
    return True


def _validate_benchmark_splits(
    benchmark_path: Path | None,
    split_path: Path,
    blockers: list[str],
) -> bool:
    if benchmark_path is None:
        return False
    benchmark, benchmark_error = _read_csv(benchmark_path)
    splits, split_error = _read_csv(split_path)
    if benchmark_error:
        blockers.append(benchmark_error)
    if split_error:
        blockers.append(split_error)
    if benchmark is None or splits is None:
        return False
    required = {"image_path", "split"}
    if not required.issubset(benchmark.columns):
        blockers.append("benchmark lacks fixed geographic split assignments")
        return False
    if not required.issubset(splits.columns):
        blockers.append("geographic splits lack image_path/split assignments")
        return False
    expected = splits[list(required)].copy()
    observed = benchmark[list(required)].copy()
    for frame in (expected, observed):
        frame["image_path"] = frame["image_path"].astype(str).str.strip()
        frame["split"] = (
            frame["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("validation", "val")
        )
    if (
        expected.groupby("image_path")["split"].nunique().gt(1).any()
        or observed.groupby("image_path")["split"].nunique().gt(1).any()
    ):
        blockers.append(
            "benchmark or geographic splits contain conflicting assignments"
        )
        return False
    expected_by_path = expected.drop_duplicates("image_path").set_index("image_path")[
        "split"
    ]
    observed_by_path = observed.drop_duplicates("image_path").set_index("image_path")[
        "split"
    ]
    missing = observed_by_path.index.difference(expected_by_path.index)
    if len(missing):
        blockers.append("benchmark contains images absent from geographic splits")
        return False
    if not observed_by_path.equals(expected_by_path.loc[observed_by_path.index]):
        blockers.append("benchmark split assignments do not match geographic splits")
        return False
    return True


def _validate_benchmark_lineage(
    benchmark_path: Path | None,
    annotations_path: Path | None,
    split_path: Path,
    blockers: list[str],
) -> bool:
    blocker_start = len(blockers)
    if benchmark_path is None or annotations_path is None:
        blockers.append("benchmark lineage inputs are missing")
        return False
    summary_path = benchmark_path.parent / "summary.json"
    summary, summary_error = _read_json(summary_path)
    if summary_error:
        blockers.append(summary_error)
        return False
    assert summary is not None
    hashes = summary.get("source_hashes")
    if not isinstance(hashes, Mapping):
        blockers.append("benchmark summary lacks source hashes")
        return False
    expected = {
        "annotations_csv": annotations_path,
        "geographic_splits_csv": split_path,
        "benchmark_split_csv": benchmark_path,
    }
    for key, path in expected.items():
        digest = hashes.get(key)
        if (
            not isinstance(digest, str)
            or not path.is_file()
            or sha256_file(path).lower() != digest.lower()
        ):
            blockers.append(f"benchmark {key} hash mismatch")

    annotations, annotation_error = _read_csv(annotations_path)
    benchmark, benchmark_error = _read_csv(benchmark_path)
    if annotation_error:
        blockers.append(annotation_error)
    if benchmark_error:
        blockers.append(benchmark_error)
    if annotations is None or benchmark is None:
        return False
    required_annotations = {"image_path", "scenic_human", "skip"}
    if not required_annotations.issubset(annotations.columns):
        blockers.append("benchmark source annotations lack absolute-label columns")
        return False
    target_column = (
        "scenic_human_mean"
        if "scenic_human_mean" in benchmark.columns
        else "scenic_human"
        if "scenic_human" in benchmark.columns
        else None
    )
    if target_column is None or "image_path" not in benchmark.columns:
        blockers.append("benchmark lacks image_path or human target")
        return False
    scores = pd.to_numeric(annotations["scenic_human"], errors="coerce")
    usable = annotations.loc[scores.notna() & ~annotations["skip"].map(_bool)].copy()
    usable["scenic_human"] = scores.loc[usable.index].astype(float)
    means = usable.groupby(usable["image_path"].astype(str).str.strip(), sort=True)[
        "scenic_human"
    ].mean()
    benchmark_paths = benchmark["image_path"].astype(str).str.strip()
    observed = pd.to_numeric(benchmark[target_column], errors="coerce")
    if benchmark_paths.duplicated().any():
        blockers.append("benchmark contains duplicate image paths")
    elif not benchmark_paths.isin(means.index).all():
        blockers.append("benchmark contains targets absent from source annotations")
    else:
        canonical = means.loc[benchmark_paths].to_numpy(dtype=float)
        if observed.isna().any() or not np.allclose(
            observed.to_numpy(dtype=float), canonical, rtol=0.0, atol=1e-9
        ):
            blockers.append(
                "benchmark human targets do not match source annotation aggregates"
            )
    return not blockers[blocker_start:]


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
    if checkpoint is None:
        checkpoint = _resolve_registry_checkpoint(path, str(active["checkpoint"]))
    if checkpoint is None or not checkpoint.exists():
        candidates = _registry_checkpoint_candidates(path, str(active["checkpoint"]))
        blockers.append(
            "baseline checkpoint missing: "
            f"{active.get('checkpoint')} (tried: "
            + ", ".join(str(c) for c in candidates)
            + ")"
        )
        return False, {"active": active}
    checkpoint_sha256 = sha256_file(checkpoint)
    declared_sha256 = active.get("sha256")
    if declared_sha256 is not None:
        if not isinstance(declared_sha256, str) or not declared_sha256.strip():
            blockers.append("baseline registry active.sha256 is not a valid string")
        elif sha256_file(checkpoint).lower() != declared_sha256.strip().lower():
            blockers.append("baseline registry active.sha256 does not match checkpoint")
    registry_valid = not (
        declared_sha256 is not None
        and (
            not isinstance(declared_sha256, str)
            or not declared_sha256.strip()
            or checkpoint_sha256.lower() != declared_sha256.strip().lower()
        )
    )
    return registry_valid, {
        "registry_path": str(path),
        "registry_sha256": sha256_file(path),
        "active": active,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
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


def _scoring_artifact_hash(manifest: Mapping[str, Any], name: str) -> str | None:
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
    schema_version = scoring.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
        or schema_version > SCORING_SCHEMA_VERSION
    ):
        blockers.append(
            "scoring manifest schema version is not a supported positive integer"
        )
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
                blockers.append(
                    "feature embeddings NPZ lacks embeddings/row_indices arrays"
                )
                embedding_rows = 0
            else:
                embeddings = np.asarray(arrays["embeddings"])
                row_indices = np.asarray(arrays["row_indices"])
                assert row_indices is not None
                embedding_rows = int(len(embeddings))
                if (
                    embeddings.ndim != 2
                    or embedding_rows <= 0
                    or embeddings.shape[1] <= 0
                ):
                    blockers.append(
                        "feature embeddings NPZ embeddings array is empty or not 2-D"
                    )
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
                    blockers.append(
                        "feature embeddings NPZ embeddings are not finite numeric values"
                    )
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
        or not tile_path.is_file()
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
            or not candidate_path.is_file()
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
    outputs = batch_manifest.get("outputs")
    outputs_batch_record = (
        outputs.get("annotation_batch_csv") if isinstance(outputs, Mapping) else None
    )
    expected_batch_hash = (
        outputs_batch_record.get("sha256")
        if isinstance(outputs_batch_record, Mapping)
        else None
    )
    if (
        batch_path is None
        or not batch_path.is_file()
        or not isinstance(expected_batch_hash, str)
        or sha256_file(batch_path).lower() != expected_batch_hash.lower()
    ):
        blockers.append("batch_manifest annotation batch output hash mismatch")
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
    selection_contract = batch_manifest.get("selection_contract")
    selection_digest = batch_manifest.get("selection_contract_sha256")
    if isinstance(selection_contract, Mapping):
        actual_digest = sha256_bytes(
            json.dumps(
                jsonable(dict(selection_contract)),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        ordered_images = (
            batch["image_path"].astype(str).tolist() if "image_path" in batch else []
        )
        if (
            selection_digest != actual_digest
            or declared_batch_id != f"batch-{actual_digest[:16]}"
            or selection_contract.get("selected_identities") != ordered_images
        ):
            blockers.append(
                "annotation batch does not match immutable selection contract"
            )
    else:
        blockers.append("batch manifest lacks immutable selection contract")

    if "image_path" in batch and set(ABSOLUTE_COLUMNS).issubset(annotations.columns):
        completed = annotations[
            pd.to_numeric(annotations["scenic_human"], errors="coerce").notna()
            & ~annotations["skip"].map(_bool)
        ]
        required_identities = set(
            batch["image_path"].astype(str).str.strip().map(_stable_identity)
        )
        completed_identities = set(
            completed["image_path"].astype(str).str.strip().map(_stable_identity)
        )
        skipped_identities = set(
            annotations.loc[annotations["skip"].map(_bool), "image_path"]
            .astype(str)
            .str.strip()
            .map(_stable_identity)
        )
        handled_identities = completed_identities | skipped_identities
        missing_identities = sorted(required_identities - handled_identities)
        if missing_identities:
            blockers.append(
                "annotation batch has "
                f"{len(missing_identities)} images without completed review "
                "decisions: " + ", ".join(missing_identities)
            )
        if "is_qa_overlap" in batch:
            qa_identities = set(
                batch.loc[batch["is_qa_overlap"].map(_bool), "image_path"]
                .astype(str)
                .str.strip()
                .map(_stable_identity)
            )
            annotator_counts = (
                completed.assign(
                    _identity=completed["image_path"]
                    .astype(str)
                    .str.strip()
                    .map(_stable_identity),
                    _annotator_id=completed["annotator_id"].astype(str).str.strip(),
                )
                .groupby("_identity")["_annotator_id"]
                .nunique()
            )
            incomplete_qa = sorted(
                identity
                for identity in qa_identities
                if int(annotator_counts.get(identity, 0)) < 2
            )
            if incomplete_qa:
                blockers.append(
                    "annotation batch has "
                    f"{len(incomplete_qa)} blind QA images without two annotators: "
                    + ", ".join(incomplete_qa)
                )
    return not blockers[blocker_start:]


def finalize_stage1(
    run_root: str | Path,
    *,
    run_name: str | None = None,
    annotations_csv: str | Path | None = None,
    mixed_labels_csv: str | Path | None = None,
    benchmark_csv: str | Path | None = None,
    control_benchmark_csv: str | Path | None = None,
    registry_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    seeds: Mapping[str, int] | None = None,
    material_config: Mapping[str, Any] | None = None,
    risks: Sequence[str] | None = None,
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
    paths["annotation_snapshot"] = root / "absolute_annotations.csv"
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
    paths["control_benchmark"] = (
        Path(control_benchmark_csv)
        if control_benchmark_csv
        else _existing(root, ("control_benchmark.csv",))
    )
    paths["baseline_registry"] = (
        Path(registry_path)
        if registry_path
        else Path("data/processed/regression/model_registry.json")
    )
    paths["baseline_checkpoint"] = Path(checkpoint_path) if checkpoint_path else None
    if handoff_path.is_file():
        try:
            prior_handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_handoff = None
        if (
            isinstance(prior_handoff, dict)
            and prior_handoff.get("ready_for_stage2") is True
        ):
            immutable_blockers: list[str] = []
            prior_artifacts = prior_handoff.get("artifacts", {})
            if not isinstance(prior_artifacts, dict):
                immutable_blockers.append(
                    "previous ready handoff artifacts are malformed"
                )
                prior_artifacts = {}
            for key, entry in prior_artifacts.items():
                if not isinstance(entry, dict):
                    continue
                prior_path = _resolve(entry.get("path"), root)
                prior_digest = entry.get("sha256")
                if (
                    prior_path is None
                    or not prior_path.is_file()
                    or not isinstance(prior_digest, str)
                    or sha256_file(prior_path) != prior_digest
                ):
                    immutable_blockers.append(
                        f"hash mismatch against previous handoff: {entry.get('path')}"
                    )
                    lineage_messages = {
                        "tile_manifest": "scoring manifest tile source hash mismatch",
                        "candidate_pool": "batch_manifest candidate source hash mismatch",
                        "baseline_checkpoint": (
                            "scoring regression checkpoint does not match baseline"
                        ),
                    }
                    if key in lineage_messages:
                        immutable_blockers.append(lineage_messages[key])
            requested_run_name = run_name or root.name
            if prior_handoff.get("run_name") != requested_run_name:
                immutable_blockers.append(
                    "run_name differs from previous ready handoff"
                )
            if prior_handoff.get("seeds", {}) != jsonable(dict(seeds or {})):
                immutable_blockers.append("seeds differ from previous ready handoff")
            if prior_handoff.get("material_config", {}) != jsonable(
                dict(material_config or {})
            ):
                immutable_blockers.append(
                    "material_config differs from previous ready handoff"
                )
            if jsonable(list(prior_handoff.get("risks") or [])) != jsonable(
                list(risks or [])
            ):
                immutable_blockers.append("risks differ from previous ready handoff")
            direct_inputs = {
                "mixed_labels": paths["mixed_labels"],
                "benchmark": paths["benchmark"],
                "control_benchmark": paths["control_benchmark"],
                "baseline_registry": paths["baseline_registry"],
                "baseline_checkpoint": paths["baseline_checkpoint"],
            }
            for key, requested_path in direct_inputs.items():
                prior_entry = prior_artifacts.get(key)
                if requested_path is None or not isinstance(prior_entry, dict):
                    continue
                expected = prior_entry.get("sha256")
                if (
                    not requested_path.is_file()
                    or not isinstance(expected, str)
                    or sha256_file(requested_path) != expected
                ):
                    immutable_blockers.append(
                        f"{key} differs from previous ready handoff"
                    )
            annotation_source = paths["absolute_annotations"]
            annotation_batch = paths["annotation_batch"]
            prior_annotation = prior_artifacts.get("absolute_annotations")
            if (
                annotation_source is not None
                and annotation_batch is not None
                and isinstance(prior_annotation, dict)
            ):
                source_frame, source_error = _read_csv(annotation_source)
                batch_frame, batch_error = _read_csv(annotation_batch)
                if source_error or batch_error:
                    immutable_blockers.append(
                        "annotation inputs differ from previous ready handoff"
                    )
                elif (
                    source_frame is not None
                    and batch_frame is not None
                    and "image_path" in source_frame
                    and "image_path" in batch_frame
                ):
                    batch_identities = set(
                        batch_frame["image_path"]
                        .astype(str)
                        .str.strip()
                        .map(_stable_identity)
                    )
                    source_identities = (
                        source_frame["image_path"]
                        .astype(str)
                        .str.strip()
                        .map(_stable_identity)
                    )
                    snapshot = source_frame.loc[
                        source_identities.isin(batch_identities)
                    ]
                    snapshot_digest = sha256_bytes(
                        snapshot.to_csv(index=False, lineterminator="\n").encode(
                            "utf-8"
                        )
                    )
                    if snapshot_digest != prior_annotation.get("sha256"):
                        immutable_blockers.append(
                            "absolute_annotations differs from previous ready handoff"
                        )
                        immutable_blockers.append(
                            "annotation batch has images without completed review decisions"
                        )
                else:
                    immutable_blockers.append(
                        "annotation inputs differ from previous ready handoff"
                    )
            if not immutable_blockers:
                return jsonable(prior_handoff)
            refused = dict(prior_handoff)
            refused["ready_for_stage2"] = False
            refused["blockers"] = list(prior_handoff.get("blockers", [])) + list(
                dict.fromkeys(immutable_blockers)
            )
            refused["incomplete_work"] = refused["blockers"]
            return jsonable(refused)
    annotation_source_path = paths["absolute_annotations"]
    paths["absolute_annotations"] = _snapshot_annotations(
        paths["absolute_annotations"],
        paths["annotation_batch"],
        paths["annotation_snapshot"],
        blockers,
        write=write,
    )
    filtered_path = paths["filtered_index"]
    candidate_path = paths["candidate_pool"]
    annotation_path = paths["absolute_annotations"]
    if (
        filtered_path is not None
        and candidate_path is not None
        and annotation_path is not None
        and candidate_path.is_file()
        and annotation_path.is_file()
    ):
        candidate_frame, candidate_error = _read_csv(candidate_path)
        annotation_frame, annotation_error = _read_csv(annotation_path)
        if candidate_error:
            blockers.append(candidate_error)
        if annotation_error:
            blockers.append(annotation_error)
        if (
            candidate_frame is not None
            and annotation_frame is not None
            and "image_path" in candidate_frame
            and set(ABSOLUTE_COLUMNS).issubset(annotation_frame.columns)
        ):
            annotation_scores = pd.to_numeric(
                annotation_frame["scenic_human"], errors="coerce"
            )
            annotation_identities = (
                annotation_frame["image_path"]
                .astype(str)
                .str.strip()
                .map(_stable_identity)
            )
            completed_identities = set(
                annotation_identities[
                    annotation_scores.notna() & ~annotation_frame["skip"].map(_bool)
                ]
            )
            skipped_identities = (
                set(annotation_identities[annotation_frame["skip"].map(_bool)])
                - completed_identities
            )
            candidate_identities = (
                candidate_frame["image_path"]
                .astype(str)
                .str.strip()
                .map(_stable_identity)
            )
            filtered = candidate_frame.reset_index(names="candidate_row_index").loc[
                ~candidate_identities.isin(skipped_identities),
                ["candidate_row_index", "image_path"],
            ]
            if filtered.empty:
                blockers.append("filtered training index contains no usable rows")
            elif write:
                atomic_write_text(
                    filtered_path,
                    filtered.to_csv(index=False, lineterminator="\n"),
                )
    required = set(RUN_ARTIFACTS) | {
        "absolute_annotations",
        "mixed_labels",
        "benchmark",
        "control_benchmark",
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
    batch_manifest, batch_manifest_error = _read_json(
        paths["batch_manifest"] or root / RUN_ARTIFACTS["batch_manifest"]
    )
    if batch_manifest_error:
        blockers.append(batch_manifest_error)
    split_ok, split_rows, audit = _validate_splits(
        paths["geographic_splits"] or root / RUN_ARTIFACTS["geographic_splits"],
        paths["leakage_audit"] or root / RUN_ARTIFACTS["leakage_audit"],
        blockers,
        manifest=batch_manifest,
    )
    benchmark_ok, benchmark_rows = _validate_human_table(
        paths["benchmark"], blockers, benchmark=True
    )
    control_benchmark_ok, control_benchmark_rows = _validate_human_table(
        paths["control_benchmark"], blockers, benchmark=True
    )
    benchmark_dataset_ok = _validate_benchmark_pre_dataset(
        paths["benchmark"], blockers, name="benchmark"
    )
    control_dataset_ok = _validate_benchmark_pre_dataset(
        paths["control_benchmark"], blockers, name="control benchmark"
    )
    benchmark_overlap_ok = _validate_benchmark_test_overlap(
        paths["benchmark"], paths["control_benchmark"], blockers
    )
    benchmark_lineage_ok = _validate_benchmark_splits(
        paths["benchmark"],
        paths["geographic_splits"] or root / RUN_ARTIFACTS["geographic_splits"],
        blockers,
    )
    benchmark_provenance_ok = _validate_benchmark_lineage(
        paths["benchmark"],
        annotation_source_path,
        paths["geographic_splits"] or root / RUN_ARTIFACTS["geographic_splits"],
        blockers,
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
        "benchmark_valid": bool(
            benchmark_ok
            and benchmark_dataset_ok
            and benchmark_lineage_ok
            and benchmark_provenance_ok
            and benchmark_overlap_ok
        ),
        "control_benchmark_valid": bool(
            control_benchmark_ok and control_dataset_ok and benchmark_overlap_ok
        ),
        "baseline_valid": bool(registry_ok),
        "hashes_valid": bool(
            lineage_ok and not any("hash mismatch" in blocker for blocker in blockers)
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
            "control_benchmark_rows": control_benchmark_rows,
            "mixed_label_rows": mixed_rows,
            "candidate_pool_rows": scoring_rows,
            "embedding_rows": embedding_rows,
        },
        "baseline": baseline,
        "seeds": jsonable(dict(seeds or {})),
        "material_config": jsonable(dict(material_config or {})),
        "risks": jsonable(list(risks or [])),
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
    parser.add_argument("--control-benchmark-csv", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--risk", action="append", default=[])
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
        control_benchmark_csv=args.control_benchmark_csv,
        registry_path=args.registry,
        checkpoint_path=args.checkpoint,
        risks=args.risk,
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
