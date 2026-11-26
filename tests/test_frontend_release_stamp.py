from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/deploy/stamp_frontend_version.py"


def run_stamp(app_dir: Path, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--app-dir", str(app_dir)]
    if check:
        command.append("--check")
    return subprocess.run(command, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("changed_asset", ["viewer.js", "styles.css"])
def test_stamp_rewrites_checks_and_rejects_changed_assets(tmp_path: Path, changed_asset: str) -> None:
    (tmp_path / "viewer.js").write_bytes(b"console.log('release');\n")
    (tmp_path / "styles.css").write_bytes(b"body { color: green; }\n")
    index = tmp_path / "index.html"
    index.write_text(
        '<link rel="stylesheet" href="/styles.css?v=viewer-hardening-v1">\n'
        '<script type="module" src="/viewer.js?v=viewer-hardening-v1"></script>\n',
        encoding="utf-8",
    )
    original = index.read_text(encoding="utf-8")
    assert run_stamp(tmp_path, check=True).returncode == 1
    stamped = run_stamp(tmp_path)
    assert stamped.returncode == 0, stamped.stderr
    version = hashlib.sha256(
        (tmp_path / "viewer.js").read_bytes() + (tmp_path / "styles.css").read_bytes()
    ).hexdigest()
    assert index.read_text(encoding="utf-8") == original.replace("viewer-hardening-v1", version)
    assert run_stamp(tmp_path, check=True).returncode == 0
    current = index.read_bytes()
    assert run_stamp(tmp_path).returncode == 0
    assert index.read_bytes() == current

    asset = tmp_path / changed_asset
    asset.write_bytes(asset.read_bytes() + b"/* changed */\n")
    stale = run_stamp(tmp_path, check=True)
    assert stale.returncode == 1
    assert "does not match" in stale.stderr
    assert index.read_bytes() == current
    assert run_stamp(tmp_path).returncode == 0
    assert index.read_bytes() != current
    assert run_stamp(tmp_path, check=True).returncode == 0


@pytest.mark.parametrize("references", [0, 2])
def test_stamp_refuses_missing_or_duplicate_asset_references(tmp_path: Path, references: int) -> None:
    (tmp_path / "viewer.js").write_bytes(b"viewer")
    (tmp_path / "styles.css").write_bytes(b"styles")
    index = tmp_path / "index.html"
    index.write_text(
        '<link href="/styles.css?v=old">\n' + '<script src="/viewer.js?v=old"></script>\n' * references,
        encoding="utf-8",
    )
    original = index.read_bytes()
    result = run_stamp(tmp_path)
    assert result.returncode == 1
    assert "Expected exactly one /viewer.js reference" in result.stderr
    assert index.read_bytes() == original


def test_tracked_frontend_release_stamp_is_current() -> None:
    result = run_stamp(PROJECT_ROOT / "apps/new_england_north", check=True)
    assert result.returncode == 0, result.stderr
