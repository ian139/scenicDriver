#!/usr/bin/env python3
"""Fail-closed validation for a Stage One handoff before Stage Two work.

The handoff-only path is unchanged. When supplemental annotations and a
supplemental benchmark are supplied, they are validated as supplemental
evaluation evidence: the benchmark must be exactly reconstructible from the
non-skipped finite annotations that hold immutable fixed split membership in
the stage-one geographic splits artifact, with matching human targets and
split assignments and no cross-split leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REQUIRED_READINESS = (
    "data_complete",
    "annotations_valid",
    "splits_valid",
    "benchmark_valid",
    "baseline_valid",
    "hashes_valid",
    "scoring_valid",
    "control_benchmark_valid",
)
COUNT_ARTIFACTS = {
    "batch_rows": "annotation_batch",
    "benchmark_rows": "benchmark",
    "candidate_pool_rows": "candidate_pool",
    "control_benchmark_rows": "control_benchmark",
    "mixed_label_rows": "mixed_labels",
    "split_rows": "geographic_splits",
    "tile_rows": "tile_manifest",
}
ANNOTATION_SCORE_MIN = 0.0
ANNOTATION_SCORE_MAX = 10.0
ADJACENCY_RADIUS = 1
SKIP_TRUE_VALUES = {"1", "true", "yes"}
SKIP_PARSE_VALUES = {"", "0", "false", "no", "1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(handoff_path: Path, record: Mapping[str, Any]) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("artifact path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = handoff_path.parent / path
    return path


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {path}") from exc
        return sum(1 for _ in reader)


def valid_annotation_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scenic_human", "skip"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"annotation CSV lacks required columns: {path}")
        for row in reader:
            skip = (row.get("skip") or "").strip().lower() in {"1", "true", "yes"}
            if not skip and (row.get("scenic_human") or "").strip():
                count += 1
    return count


def benchmark_identities(path: Path) -> tuple[dict[str, int], set[str]]:
    counts: dict[str, int] = {}
    identities: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"benchmark lacks required columns {sorted(required)}: {path}"
            )
        for row in reader:
            identity = (row.get("image_path") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not identity or split not in {"train", "val", "validation", "test"}:
                raise ValueError(
                    f"benchmark contains invalid identity or split: {path}"
                )
            if identity in identities:
                raise ValueError(f"benchmark contains duplicate image_path: {identity}")
            identities.add(identity)
            normalized = "val" if split == "validation" else split
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts, identities


def _normalise_split(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    return "val" if normalized == "validation" else normalized


def _validate_sha256_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a 64-character hex SHA-256 digest")


def _parse_skip(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in SKIP_PARSE_VALUES:
        raise ValueError(f"annotation has malformed skip decision: {value!r}")
    return normalized in SKIP_TRUE_VALUES


def annotation_targets(path: Path) -> tuple[dict[str, list[float]], set[str]]:
    """Return per-identity human scores and the set of skipped identities.

    A truthy skip decision takes precedence over any vestigial score value;
    every other row must carry a finite score on the annotation contract
    scale. Identities must be unique and non-empty.
    """
    targets: dict[str, list[float]] = {}
    skipped: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "scenic_human", "skip"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"annotation CSV lacks required columns: {path}")
        for row in reader:
            identity = (row.get("image_path") or "").strip()
            if not identity:
                raise ValueError(f"annotation contains empty image_path: {path}")
            if identity in targets or identity in skipped:
                raise ValueError(f"annotation contains duplicate identity: {identity}")
            if _parse_skip(row.get("skip")):
                skipped.add(identity)
                continue
            score_text = (row.get("scenic_human") or "").strip()
            if not score_text:
                raise ValueError(f"annotation is neither scored nor skipped: {identity}")
            try:
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(f"annotation has non-numeric score: {identity}") from exc
            if not math.isfinite(score) or not (
                ANNOTATION_SCORE_MIN <= score <= ANNOTATION_SCORE_MAX
            ):
                raise ValueError(
                    f"annotation score out of range [{ANNOTATION_SCORE_MIN}, "
                    f"{ANNOTATION_SCORE_MAX}]: {identity}"
                )
            targets.setdefault(identity, []).append(score)
    return targets, skipped


def supplemental_benchmark_rows(path: Path) -> dict[str, tuple[str, float]]:
    """Return supplemental benchmark identity -> (split, human target)."""
    rows: dict[str, tuple[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "scenic_human_mean", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"benchmark lacks required columns {sorted(required)}: {path}"
            )
        for row in reader:
            identity = (row.get("image_path") or "").strip()
            split = _normalise_split(row.get("split"))
            if not identity or split not in {"train", "val", "test"}:
                raise ValueError(f"benchmark contains invalid identity or split: {path}")
            if identity in rows:
                raise ValueError(f"benchmark contains duplicate image_path: {identity}")
            try:
                target = float((row.get("scenic_human_mean") or "").strip())
            except ValueError as exc:
                raise ValueError(f"benchmark has non-numeric target: {identity}") from exc
            if not math.isfinite(target) or not (
                ANNOTATION_SCORE_MIN <= target <= ANNOTATION_SCORE_MAX
            ):
                raise ValueError(f"benchmark target out of range: {identity}")
            rows[identity] = (split, target)
    return rows


def split_assignments(
    splits_path: Path, identities: set[str]
) -> dict[str, tuple[str, int, int, int]]:
    """Stream the immutable geographic splits artifact for the given identities.

    Returns identity -> (split, z, x, y). Streaming with a membership filter
    keeps this linear in the artifact size rather than quadratic.
    """
    assignments: dict[str, tuple[str, int, int, int]] = {}
    with splits_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split", "z", "x", "y"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"geographic splits lacks required columns {sorted(required)}: "
                f"{splits_path}"
            )
        for row in reader:
            identity = (row.get("image_path") or "").strip()
            if not identity or identity not in identities:
                continue
            split = _normalise_split(row.get("split"))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"geographic splits contains invalid split: {identity}")
            try:
                z, x, y = int(row["z"]), int(row["x"]), int(row["y"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"geographic splits has non-integer tile coordinates: {identity}"
                ) from exc
            if identity in assignments:
                raise ValueError(
                    f"geographic splits assigns identity to multiple rows: {identity}"
                )
            assignments[identity] = (split, z, x, y)
    return assignments


def _load_handoff(handoff_path: Path) -> dict[str, Any]:
    try:
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid handoff JSON: {handoff_path}") from exc
    if not isinstance(handoff, dict) or handoff.get("schema_version") != 1:
        raise ValueError("stage-one handoff schema_version must equal 1")
    if not isinstance(handoff.get("artifacts"), dict) or not isinstance(
        handoff.get("artifact_hashes"), dict
    ):
        raise ValueError("stage-one handoff lacks artifact contracts")
    return handoff


def _resolve_verified_artifact(
    handoff: Mapping[str, Any], handoff_path: Path, name: str
) -> Path:
    record = handoff["artifacts"].get(name)
    if not isinstance(record, dict):
        raise ValueError(f"stage-one handoff lacks artifact record: {name}")
    path = resolve_artifact(handoff_path, record)
    if not path.is_file():
        raise FileNotFoundError(
            f"supplemental validation requires artifact: {name}: {path}"
        )
    expected_hash = record.get("sha256")
    artifact_hashes = handoff["artifact_hashes"]
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or artifact_hashes.get(name) != expected_hash
        or sha256_file(path) != expected_hash
    ):
        raise ValueError(f"artifact SHA-256 mismatch: {name}")
    return path


def _check_benchmark_adjacency(
    assignments: Mapping[str, tuple[str, int, int, int]],
    identities: set[str],
    radius: int = ADJACENCY_RADIUS,
) -> None:
    """Fail when benchmark tiles from different splits are Chebyshev-adjacent."""
    tile_splits: dict[tuple[int, int, int], str] = {}
    for identity in identities:
        split, z, x, y = assignments[identity]
        tile_splits[(z, x, y)] = split
    for (z, x, y), split in sorted(tile_splits.items()):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                other_split = tile_splits.get((z, x + dx, y + dy))
                if other_split is not None and other_split != split:
                    raise ValueError(
                        f"adjacent tiles assigned to different splits: "
                        f"z{z}/{x}/{y} ({split}) vs "
                        f"z{z}/{x + dx}/{y + dy} ({other_split})"
                    )


def validate_handoff(handoff_path: Path) -> dict[str, int]:
    handoff_path = handoff_path.expanduser().resolve()
    try:
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid handoff JSON: {handoff_path}") from exc
    if not isinstance(handoff, dict) or handoff.get("schema_version") != 1:
        raise ValueError("stage-one handoff schema_version must equal 1")
    if handoff.get("ready_for_stage2") is not True:
        raise ValueError("stage-one handoff is not ready_for_stage2")
    if handoff.get("blockers") != [] or handoff.get("incomplete_work") != []:
        raise ValueError("stage-one handoff has blockers or incomplete work")

    readiness = handoff.get("readiness")
    if not isinstance(readiness, dict):
        raise ValueError("stage-one handoff lacks readiness object")
    for key in REQUIRED_READINESS:
        if handoff.get(key) is not True or readiness.get(key) is not True:
            raise ValueError(f"stage-one readiness is false or missing: {key}")

    artifacts = handoff.get("artifacts")
    artifact_hashes = handoff.get("artifact_hashes")
    if not isinstance(artifacts, dict) or not isinstance(artifact_hashes, dict):
        raise ValueError("stage-one handoff lacks artifact contracts")
    resolved: dict[str, Path] = {}
    for name, raw_record in artifacts.items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"invalid artifact record: {name}")
        path = resolve_artifact(handoff_path, raw_record)
        required = raw_record.get("required") is True
        if required and not path.is_file():
            raise FileNotFoundError(f"required artifact is missing: {name}: {path}")
        if not path.is_file():
            continue
        expected_bytes = raw_record.get("bytes")
        expected_hash = raw_record.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"artifact has invalid byte count: {name}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"artifact byte count mismatch: {name}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"artifact has invalid SHA-256: {name}")
        if artifact_hashes.get(name) != expected_hash:
            raise ValueError(f"artifact_hashes mismatch: {name}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")
        resolved[name] = path

    counts = handoff.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("stage-one handoff lacks counts")
    for count_name, artifact_name in COUNT_ARTIFACTS.items():
        expected = counts.get(count_name)
        if not isinstance(expected, int) or expected <= 0:
            raise ValueError(f"invalid declared count: {count_name}")
        actual = csv_row_count(resolved[artifact_name])
        if actual != expected:
            raise ValueError(
                f"row count mismatch for {artifact_name}: {actual} != {expected}"
            )
    annotation_rows = valid_annotation_count(resolved["absolute_annotations"])
    if annotation_rows != counts.get("annotation_rows"):
        raise ValueError("valid human annotation row count mismatch")

    with np.load(resolved["feature_embeddings"], allow_pickle=False) as embeddings:
        if "embeddings" not in embeddings:
            raise ValueError("feature embeddings lack embeddings array")
        embedding_rows = len(embeddings["embeddings"])
    if embedding_rows != counts.get("embedding_rows"):
        raise ValueError("feature embedding row count mismatch")

    leakage = handoff.get("leakage_audit")
    if not isinstance(leakage, dict):
        raise ValueError("stage-one handoff lacks leakage audit")
    if (
        leakage.get("valid") is not True
        or leakage.get("violation_count") != 0
        or leakage.get("violations") != []
        or leakage.get("duplicate_cross_split") is not False
        or leakage.get("adjacent_cross_split") is not False
        or leakage.get("checked_rows") != counts.get("split_rows")
    ):
        raise ValueError("stage-one leakage invariants failed")
    split_counts = leakage.get("split_counts")
    if not isinstance(split_counts, dict) or set(split_counts) != {
        "train",
        "val",
        "test",
    }:
        raise ValueError("stage-one leakage split counts are invalid")
    if any(not isinstance(value, int) or value <= 0 for value in split_counts.values()):
        raise ValueError("stage-one leakage split has no support")
    if sum(split_counts.values()) != counts["split_rows"]:
        raise ValueError("stage-one leakage split counts do not match split rows")
    audit_file = json.loads(resolved["leakage_audit"].read_text(encoding="utf-8"))
    if audit_file != leakage:
        raise ValueError("embedded leakage audit differs from hashed artifact")

    expanded_splits, expanded_ids = benchmark_identities(resolved["benchmark"])
    _, control_ids = benchmark_identities(resolved["control_benchmark"])
    if expanded_splits != {"train": 19, "val": 1, "test": 4}:
        raise ValueError(f"expanded benchmark split changed: {expanded_splits}")
    if expanded_ids & control_ids:
        raise ValueError("expanded and control benchmarks overlap")

    baseline = handoff.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("stage-one handoff lacks baseline identity")
    if artifact_hashes.get("baseline_registry") != baseline.get("registry_sha256"):
        raise ValueError("baseline registry identity mismatch")
    if artifact_hashes.get("baseline_checkpoint") != baseline.get("checkpoint_sha256"):
        raise ValueError("baseline checkpoint identity mismatch")
    registry = json.loads(resolved["baseline_registry"].read_text(encoding="utf-8"))
    if registry.get("active") != baseline.get("active"):
        raise ValueError("active registry record differs from handoff baseline")

    return {
        "artifacts": len(resolved),
        "candidate_rows": counts["candidate_pool_rows"],
        "split_rows": counts["split_rows"],
        "expanded_rows": counts["benchmark_rows"],
        "control_rows": counts["control_benchmark_rows"],
        "leakage_violations": leakage["violation_count"],
    }


def validate_supplemental(
    handoff_path: Path,
    supplemental_annotations: Path,
    supplemental_benchmark: Path,
    annotations_sha256: str,
    benchmark_sha256: str,
    control_benchmark: Path | None = None,
) -> dict[str, int]:
    """Fail-closed validation of supplemental human benchmark evidence.

    The supplemental benchmark must be exactly reconstructible from the
    supplemental annotations plus the immutable stage-one geographic splits:
    every non-skipped, finite annotation with a fixed split assignment must
    appear exactly once with a matching human target and split, and nothing
    else may appear. Split support must cover train/val/test, identities must
    be disjoint and free of adjacent cross-split tiles, and test identities
    must be disjoint from the control benchmark. Raises ValueError on any
    mismatch.
    """
    handoff_path = handoff_path.expanduser().resolve()
    _validate_sha256_digest(annotations_sha256, "supplemental annotations SHA-256")
    _validate_sha256_digest(benchmark_sha256, "supplemental benchmark SHA-256")
    annotations_path = supplemental_annotations.expanduser().resolve()
    benchmark_path = supplemental_benchmark.expanduser().resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(f"supplemental annotations missing: {annotations_path}")
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"supplemental benchmark missing: {benchmark_path}")
    if sha256_file(annotations_path) != annotations_sha256:
        raise ValueError("supplemental annotations SHA-256 mismatch")
    if sha256_file(benchmark_path) != benchmark_sha256:
        raise ValueError("supplemental benchmark SHA-256 mismatch")

    handoff = _load_handoff(handoff_path)
    splits_path = _resolve_verified_artifact(handoff, handoff_path, "geographic_splits")
    if control_benchmark is None:
        control_path = _resolve_verified_artifact(handoff, handoff_path, "control_benchmark")
    else:
        control_path = control_benchmark.expanduser().resolve()
        if not control_path.is_file():
            raise FileNotFoundError(f"control benchmark missing: {control_path}")
        record = handoff["artifacts"]["control_benchmark"]
        if sha256_file(control_path) != record.get("sha256"):
            raise ValueError("control benchmark SHA-256 mismatch with stage-one handoff")

    targets, skipped = annotation_targets(annotations_path)
    benchmark = supplemental_benchmark_rows(benchmark_path)
    identities = set(targets) | skipped | set(benchmark)
    assignments = split_assignments(splits_path, identities)

    expected = {identity for identity in targets if identity in assignments}
    for identity in expected:
        if identity not in benchmark:
            raise ValueError(f"benchmark missing annotated tile: {identity}")
    for identity, (split, target) in benchmark.items():
        if identity not in expected:
            raise ValueError(
                "benchmark tile has no valid non-skipped annotation with fixed "
                f"split membership: {identity}"
            )
        assigned_split, _, _, _ = assignments[identity]
        if assigned_split != split:
            raise ValueError(
                f"benchmark split mismatch for {identity}: "
                f"{split} != assigned {assigned_split}"
            )
        annotation_mean = math.fsum(targets[identity]) / len(targets[identity])
        if not math.isclose(annotation_mean, target, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"benchmark target mismatch for {identity}: "
                f"{target} != annotation mean {annotation_mean}"
            )

    split_counts: dict[str, int] = {}
    for _, (split, _) in benchmark.items():
        split_counts[split] = split_counts.get(split, 0) + 1
    if set(split_counts) != {"train", "val", "test"} or any(
        value <= 0 for value in split_counts.values()
    ):
        raise ValueError(
            f"supplemental benchmark lacks train/val/test support: {split_counts}"
        )

    _check_benchmark_adjacency(assignments, set(benchmark))

    _, control_identities = benchmark_identities(control_path)
    test_identities = {
        identity for identity, (split, _) in benchmark.items() if split == "test"
    }
    overlap = test_identities & control_identities
    if overlap:
        raise ValueError(
            "supplemental test identities overlap control benchmark: "
            + ", ".join(sorted(overlap)[:3])
        )

    return {
        "supplemental_benchmark_valid": 1,
        "supplemental_rows": len(benchmark),
        "supplemental_val_rows": split_counts.get("val", 0),
        "supplemental_test_rows": split_counts.get("test", 0),
        "supplemental_skipped_rows": len(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--supplemental-annotations", type=Path)
    parser.add_argument("--supplemental-annotations-sha256")
    parser.add_argument("--supplemental-benchmark", type=Path)
    parser.add_argument("--supplemental-benchmark-sha256")
    parser.add_argument("--control-benchmark", type=Path)
    args = parser.parse_args()

    supplemental_flags = (
        "--supplemental-annotations",
        "--supplemental-annotations-sha256",
        "--supplemental-benchmark",
        "--supplemental-benchmark-sha256",
    )
    supplemental_args = (
        args.supplemental_annotations,
        args.supplemental_annotations_sha256,
        args.supplemental_benchmark,
        args.supplemental_benchmark_sha256,
    )
    if any(value is not None for value in supplemental_args):
        missing = [
            flag
            for flag, value in zip(supplemental_flags, supplemental_args)
            if value is None
        ]
        if missing:
            parser.error(
                "supplemental validation requires all of: " + ", ".join(missing)
            )

    metrics = validate_handoff(args.handoff)
    print("METRIC handoff_ready=1")
    for name, value in metrics.items():
        print(f"METRIC {name}={value}")
    if any(value is not None for value in supplemental_args):
        supplemental_metrics = validate_supplemental(
            args.handoff,
            args.supplemental_annotations,
            args.supplemental_benchmark,
            args.supplemental_annotations_sha256,
            args.supplemental_benchmark_sha256,
            args.control_benchmark,
        )
        for name, value in supplemental_metrics.items():
            print(f"METRIC {name}={value}")


if __name__ == "__main__":
    main()
