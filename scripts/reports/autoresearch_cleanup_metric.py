"""Emit maintenance lines + 10,000 per Ruff violation over tracked source.

The Git index defines file membership; current working-tree contents define line
counts. Untracked outputs do not affect the objective. Tests remain a separate
reported metric, not maintenance lines. Native and shell sources count alongside
Python, app, configuration and documentation text in the existing roots.
"""

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
    "compose.beta.yml",
    "compose.remote-training.yml",
    "Dockerfile",
    "Dockerfile.beta-api",
    "pyproject.toml",
    "Dockerfile.remote-training",
    ".dockerignore",
)
TEXT_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".css",
    ".html",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".zsh",
}
EXCLUDED_PARTS = {"__marimo__", "__pycache__"}


def _ruff_violation_count(tracked: list[Path]) -> int:
    targets = [
        path.relative_to(ROOT).as_posix() for path in tracked
        if path.suffix in {".py", ".pyi", ".ipynb"}
        and any(path.relative_to(ROOT).is_relative_to(root) for root in RUFF_TARGETS)
    ]
    if not targets:
        return 0
    result = subprocess.run(
        ["ruff", "check", "--force-exclude", *targets, "--output-format", "json"],
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


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git tracked-file inventory failed")
    return sorted({
        ROOT / name
        for name in result.stdout.split("\0")
        if name and (ROOT / name).is_file()
    })


def _text_files(tracked: list[Path]) -> list[Path]:
    files = []
    for path in tracked:
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if relative.as_posix() in ROOT_FILES or (
            path.suffix in TEXT_SUFFIXES
            and (
                any(relative.is_relative_to(root) for root in MAINTENANCE_ROOTS)
                or (len(relative.parts) == 1 and path.suffix in {".sh", ".bash", ".zsh"})
            )
        ):
            files.append(path)
    return files


def _line_count(paths: list[Path]) -> int:
    return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in paths)


def main() -> None:
    tracked = _tracked_files()
    files = _text_files(tracked)
    maintenance_lines = _line_count(files)
    test_files = [
        path for path in tracked
        if path.parent == ROOT / "tests" and path.name.startswith("test_")
    ]
    test_lines = _line_count(test_files)
    ruff_violations = _ruff_violation_count(tracked)
    maintenance_debt = maintenance_lines + 10_000 * ruff_violations

    print(f"METRIC maintenance_debt={maintenance_debt}")
    print(f"METRIC ruff_violations={ruff_violations}")
    print(f"METRIC maintenance_lines={maintenance_lines}")
    print(f"METRIC active_files={len(files)}")
    print(f"METRIC test_lines={test_lines}")


if __name__ == "__main__":
    main()
