#!/usr/bin/env python3
"""Check the read-only artifacts required by the New England beta API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config/app_regions.json")
REGISTRY_PATH = Path("data/processed/regression/model_registry.json")
DEFAULT_GRAPH_PATH = Path(
    "data/processed/road_graphs/"
    "new_england_north_burlington_bangor_corridor35/road_graph.json"
)
DEFAULT_RUN_NAME = "new_england_north_z14_v6_learned"
DEFAULT_CHECKPOINT_PATH = Path(
    "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt"
)


def _project_path(root: Path, value: str | Path) -> Path:
    """Resolve a configured path while preserving absolute container paths."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def _configured_new_england(root: Path) -> tuple[Path, str, list[str]]:
    """Read the New England graph/run settings, with canonical fallbacks."""

    graph_path = root / DEFAULT_GRAPH_PATH
    run_name = DEFAULT_RUN_NAME
    issues: list[str] = []
    config_path = root / CONFIG_PATH

    if not config_path.is_file():
        return graph_path, run_name, issues

    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(f"invalid: {CONFIG_PATH} ({exc})")
        return graph_path, run_name, issues

    regions = payload.get("regions") if isinstance(payload, dict) else None
    region = next(
        (
            item
            for item in regions or []
            if isinstance(item, dict) and item.get("region") == "new_england_north"
        ),
        None,
    )
    if not isinstance(region, dict):
        issues.append(f"invalid: {CONFIG_PATH} (new_england_north is not configured)")
        return graph_path, run_name, issues

    configured_graph = region.get("graph")
    configured_run = region.get("run_name")
    if isinstance(configured_graph, str) and configured_graph:
        graph_path = _project_path(root, configured_graph)
    else:
        issues.append(f"invalid: {CONFIG_PATH} (new_england_north graph is missing)")
    if isinstance(configured_run, str) and configured_run:
        run_name = configured_run
    else:
        issues.append(f"invalid: {CONFIG_PATH} (new_england_north run_name is missing)")
    return graph_path, run_name, issues


def _active_checkpoint(root: Path) -> tuple[Path, list[str]]:
    """Resolve the checkpoint named by the active model-registry record."""

    registry_path = root / REGISTRY_PATH
    checkpoint_path = root / DEFAULT_CHECKPOINT_PATH
    issues: list[str] = []
    if not registry_path.is_file():
        return checkpoint_path, issues

    try:
        payload: Any = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(f"invalid: {REGISTRY_PATH} ({exc})")
        return checkpoint_path, issues

    active = payload.get("active") if isinstance(payload, dict) else None
    configured_checkpoint = active.get("checkpoint") if isinstance(active, dict) else None
    if isinstance(configured_checkpoint, str) and configured_checkpoint:
        checkpoint_path = _project_path(root, configured_checkpoint)
    else:
        issues.append(f"invalid: {REGISTRY_PATH} (active checkpoint is missing)")
    return checkpoint_path, issues


def check_artifacts(project_root: Path) -> int:
    root = project_root.resolve()
    graph_path, run_name, issues = _configured_new_england(root)
    checkpoint_path, registry_issues = _active_checkpoint(root)
    issues.extend(registry_issues)

    report_dir = root / "data/processed/heuristic_runs" / run_name / "report"
    required_paths = [
        graph_path,
        report_dir / "report.json",
        report_dir / "route.geojson",
        report_dir / "route_metrics.json",
        root / REGISTRY_PATH,
        checkpoint_path,
    ]

    missing: list[Path] = []
    for path in required_paths:
        if not path.is_file() and path not in missing:
            missing.append(path)

    for path in missing:
        try:
            display_path = path.relative_to(root)
        except ValueError:
            display_path = path
        print(f"missing: {display_path}")
    for issue in issues:
        print(issue)
    return 1 if missing or issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing config, processed artifacts, and models",
    )
    args = parser.parse_args()
    return check_artifacts(args.project_root)


if __name__ == "__main__":
    sys.exit(main())
