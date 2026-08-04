"""Emit deterministic repository-maintenance metrics for autoresearch cleanup."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUFF_TARGETS = ("src", "scripts", "tests", "notebooks")
MAINTENANCE_ROOTS = (
    "src",
    "scripts",
    "notebooks",
    "apps/new_england_north",
    "deploy",
    "docs",
    "archive",
    "config",
    "containers",
    ".github",
)
ROOT_FILES = (
    "README.md",
    "pyproject.toml",
    "compose.beta.yml",
    "compose.remote-training.yml",
    "Dockerfile",
    "Dockerfile.beta-api",
    "Dockerfile.remote-training",
    ".dockerignore",
)
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {"__marimo__", "__pycache__"}


def _ruff_violation_count() -> int:
    result = subprocess.run(
        ["ruff", "check", *RUFF_TARGETS, "--output-format", "json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "ruff check failed")
    violations = json.loads(result.stdout)
    if not isinstance(violations, list):
        raise RuntimeError("ruff returned an unexpected result")
    return len(violations)


def _text_files() -> list[Path]:
    files: set[Path] = set()
    for relative_root in MAINTENANCE_ROOTS:
        root = ROOT / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and not EXCLUDED_PARTS.intersection(path.parts)
            ):
                files.add(path)
    for relative_path in ROOT_FILES:
        path = ROOT / relative_path
        if path.is_file():
            files.add(path)
    return sorted(files)


def _line_count(paths: list[Path]) -> int:
    return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in paths)


def main() -> None:
    files = _text_files()
    maintenance_lines = _line_count(files)
    test_files = sorted((ROOT / "tests").glob("test_*"))
    test_lines = _line_count([path for path in test_files if path.is_file()])
    ruff_violations = _ruff_violation_count()
    maintenance_debt = maintenance_lines + 10_000 * ruff_violations

    print(f"METRIC maintenance_debt={maintenance_debt}")
    print(f"METRIC ruff_violations={ruff_violations}")
    print(f"METRIC maintenance_lines={maintenance_lines}")
    print(f"METRIC active_files={len(files)}")
    print(f"METRIC test_lines={test_lines}")


if __name__ == "__main__":
    main()
