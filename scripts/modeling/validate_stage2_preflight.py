#!/usr/bin/env python3
"""Fail-closed validation for a Stage One handoff before Stage Two work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
            raise ValueError(f"benchmark lacks required columns {sorted(required)}: {path}")
        for row in reader:
            identity = (row.get("image_path") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not identity or split not in {"train", "val", "validation", "test"}:
                raise ValueError(f"benchmark contains invalid identity or split: {path}")
            if identity in identities:
                raise ValueError(f"benchmark contains duplicate image_path: {identity}")
            identities.add(identity)
            normalized = "val" if split == "validation" else split
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts, identities


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
            raise ValueError(f"row count mismatch for {artifact_name}: {actual} != {expected}")
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
    if not isinstance(split_counts, dict) or set(split_counts) != {"train", "val", "test"}:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    metrics = validate_handoff(args.handoff)
    print("METRIC handoff_ready=1")
    for name, value in metrics.items():
        print(f"METRIC {name}={value}")


if __name__ == "__main__":
    main()
