"""Fail-closed validation and atomic publication of a stage-one handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .common import atomic_write_json, jsonable, sha256_file

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
    "annotation_batch": "annotation_batch.csv",
    "batch_manifest": "batch_manifest.json",
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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on", "completed"}


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
        return pd.read_csv(path), None
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
        return {"path": None, "exists": False, "required": required, "sha256": None, "bytes": None}
    try:
        name = str(path.relative_to(root))
    except ValueError:
        name = str(path)
    if not path.exists() or not path.is_file():
        return {"path": name, "exists": False, "required": required, "sha256": None, "bytes": None}
    return {"path": name, "exists": True, "required": required, "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


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
    for entry in (value.get("artifacts", {}) if isinstance(value, dict) else {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str):
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
    present = frame["satellite_present"].map(_bool) & frame["terrain_present"].map(_bool)
    if not bool(present.all()):
        blockers.append("tile manifest contains incomplete image pairs")
    for column in ("satellite_path", "terrain_path"):
        for raw in frame[column].dropna().astype(str):
            if not raw or raw.lower() in {"nan", "none"}:
                blockers.append(f"tile manifest contains empty {column}")
                break
            path_value = Path(raw)
            if not path_value.is_absolute() and not path_value.exists():
                path_value = root / path_value
            if not path_value.exists():
                blockers.append(f"missing imagery referenced by tile manifest: {raw}")
                break
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
        blockers.append(f"absolute annotations violate seven-column contract; extra columns: {extra}")
    if frame.empty:
        blockers.append("absolute annotations are empty")
        return False, 0
    completed = pd.to_numeric(frame["scenic_human"], errors="coerce").notna()
    completed &= frame["image_path"].astype(str).str.strip().ne("")
    completed &= ~frame["skip"].map(_bool)
    if not bool(completed.any()):
        blockers.append("absolute annotations contain no completed labels")
    return not missing and not extra and bool(completed.any()), int(completed.sum())


def _validate_batch(path: Path, blockers: list[str]) -> tuple[bool, int]:
    frame, error = _read_csv(path)
    if error:
        blockers.append(error)
        return False, 0
    assert frame is not None
    required = {"image_path", "selection_reason", "selection_score", "selection_rank", "batch_id", "run_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        blockers.append(f"annotation batch missing columns: {missing}")
    if frame.empty:
        blockers.append("annotation batch is empty")
    if "image_path" in frame and frame["image_path"].duplicated().any():
        blockers.append("annotation batch contains duplicate image paths")
    return not missing and not frame.empty, len(frame)


def _validate_splits(split_path: Path, audit_path: Path, blockers: list[str]) -> tuple[bool, int, dict[str, Any]]:
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
    return not missing and not frame.empty and isinstance(audit, dict) and audit.get("valid") is True, len(frame), audit if isinstance(audit, dict) else {}


def _validate_human_table(path: Path | None, blockers: list[str], *, benchmark: bool = False) -> tuple[bool, int]:
    if path is None:
        blockers.append("missing human benchmark CSV" if benchmark else "missing mixed-label CSV")
        return False, 0
    frame, error = _read_csv(path)
    if error:
        blockers.append(error)
        return False, 0
    assert frame is not None
    if benchmark:
        score = "scenic_human" if "scenic_human" in frame else "scenic_human_mean" if "scenic_human_mean" in frame else None
        valid = "image_path" in frame and score is not None and pd.to_numeric(frame[score], errors="coerce").notna().any() and not frame.empty
        if not valid:
            blockers.append("benchmark lacks non-empty human image/score rows")
        if "label_source" in frame and frame["label_source"].astype(str).str.lower().eq("heuristic").all():
            blockers.append("benchmark contains weak labels only")
            valid = False
        return bool(valid), len(frame)
    valid = {"image_path", "scenic_score"}.issubset(frame.columns) and not frame.empty
    if not valid:
        blockers.append("mixed-label CSV lacks image_path/scenic_score rows")
    return bool(valid), len(frame)


def _validate_registry(path: Path, checkpoint: Path | None, blockers: list[str]) -> tuple[bool, dict[str, Any]]:
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
    paths: dict[str, Path | None] = {key: root / value for key, value in RUN_ARTIFACTS.items()}
    paths["absolute_annotations"] = Path(annotations_csv) if annotations_csv else _existing(root, ("absolute_annotations.csv", "annotations.csv"))
    if paths["absolute_annotations"] is None and Path("data/raw/labels_human.csv").exists():
        paths["absolute_annotations"] = Path("data/raw/labels_human.csv")
    paths["mixed_labels"] = Path(mixed_labels_csv) if mixed_labels_csv else _existing(root, ("mixed_labels.csv", "labels_mixed.csv"))
    paths["benchmark"] = Path(benchmark_csv) if benchmark_csv else _existing(root, ("benchmark_split.csv", "benchmark.csv", "challenge_benchmark.csv"))
    paths["baseline_registry"] = Path(registry_path) if registry_path else Path("data/processed/regression/model_registry.json")
    paths["baseline_checkpoint"] = Path(checkpoint_path) if checkpoint_path else None
    required = set(RUN_ARTIFACTS) | {"absolute_annotations", "mixed_labels", "benchmark", "baseline_registry"}
    records = {key: _artifact(path, root, key in required) for key, path in paths.items()}
    for key, entry in records.items():
        if entry["required"] and not entry["exists"]:
            blockers.append(f"missing required artifact: {key}")
    previous = _previous_hashes(handoff_path)
    for relative, digest in previous.items():
        path = _resolve(relative, root)
        if path is not None and path.exists() and path.name != handoff_path.name and sha256_file(path) != digest:
            blockers.append(f"hash mismatch against previous handoff: {relative}")
    for key, digest in (expected_hashes or {}).items():
        path = paths.get(key, root / key)
        if path is None or not path.exists() or sha256_file(path).lower() != str(digest).lower():
            blockers.append(f"hash mismatch for {key}")
    region, error = _read_json(paths["region_manifest"] or root / RUN_ARTIFACTS["region_manifest"])
    if error:
        blockers.append(error)
    elif region is not None and region.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        blockers.append("region manifest schema version mismatch")
    inventory, error = _read_json(paths["inventory_report"] or root / RUN_ARTIFACTS["inventory_report"])
    if error:
        blockers.append(error)
    tile_ok, tile_rows = _validate_tile(paths["tile_manifest"] or root / RUN_ARTIFACTS["tile_manifest"], root, blockers)
    batch_ok, batch_rows = _validate_batch(paths["annotation_batch"] or root / RUN_ARTIFACTS["annotation_batch"], blockers)
    annotation_ok, annotation_rows = _validate_annotations(paths["absolute_annotations"], blockers)
    split_ok, split_rows, audit = _validate_splits(paths["geographic_splits"] or root / RUN_ARTIFACTS["geographic_splits"], paths["leakage_audit"] or root / RUN_ARTIFACTS["leakage_audit"], blockers)
    benchmark_ok, benchmark_rows = _validate_human_table(paths["benchmark"], blockers, benchmark=True)
    mixed_ok, mixed_rows = _validate_human_table(paths["mixed_labels"], blockers)
    registry_ok, baseline = _validate_registry(paths["baseline_registry"] or Path("data/processed/regression/model_registry.json"), paths["baseline_checkpoint"], blockers)
    records = {key: _artifact(path, root, entry["required"]) for key, (path, entry) in ((key, (paths[key], value)) for key, value in records.items())}
    readiness = {
        "data_complete": bool(region is not None and inventory is not None and tile_ok),
        "annotations_valid": bool(batch_ok and annotation_ok),
        "splits_valid": bool(split_ok),
        "benchmark_valid": bool(benchmark_ok),
        "baseline_valid": bool(registry_ok),
        "hashes_valid": not any("hash mismatch" in blocker for blocker in blockers),
    }
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name or root.name,
        "run_root": str(root),
        "artifacts": records,
        "artifact_hashes": {key: entry["sha256"] for key, entry in records.items()},
        "counts": {"tile_rows": tile_rows, "batch_rows": batch_rows, "annotation_rows": annotation_rows, "split_rows": split_rows, "benchmark_rows": benchmark_rows, "mixed_label_rows": mixed_rows},
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
    parser = argparse.ArgumentParser(description="Finalize an active-learning stage-one run")
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
    result = finalize_stage1(args.run_root, run_name=args.run_name, annotations_csv=args.annotations_csv, mixed_labels_csv=args.mixed_labels_csv, benchmark_csv=args.benchmark_csv, registry_path=args.registry, checkpoint_path=args.checkpoint, expected_hashes=expected)
    print(json.dumps({"stage1_handoff": str(Path(args.run_root) / "stage1_handoff.json"), "ready_for_stage2": result["ready_for_stage2"], "blockers": result["blockers"]}, sort_keys=True))
    if not result["ready_for_stage2"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
