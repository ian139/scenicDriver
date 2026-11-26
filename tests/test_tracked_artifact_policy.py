from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Generated data, model weights, caches, and build outputs never belong in git.
_FORBIDDEN = (
    "data/*",
    "models/*",
    "cache/*",
    "*/cache/*",
    "*__pycache__/*",
    "*.pyc",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.so",
    "*.egg-info/*",
    "build/*",
    "dist/*",
    ".venv/*",
    "*/.pytest_cache/*",
    ".pytest_cache/*",
    ".ruff_cache/*",
    "notebooks/__marimo__/*",
)
_ALLOWED = frozenset({"data/README.md", "models/.gitkeep"})


def _tracked_files() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("tracked-artifact policy needs a git checkout")
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return [path for path in output.decode().split("\0") if path]


def test_no_generated_artifacts_are_tracked() -> None:
    tracked = _tracked_files()
    assert "pyproject.toml" in tracked
    violations = [
        path
        for path in tracked
        if path not in _ALLOWED and any(fnmatch(path, pattern) for pattern in _FORBIDDEN)
    ]
    assert violations == []
