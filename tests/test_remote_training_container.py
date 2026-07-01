from __future__ import annotations

import json
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_remote_training_container_files_define_training_entrypoints() -> None:
    # --- File existence ---
    dockerfile = REPO_ROOT / "Dockerfile.remote-training"
    dockerignore = REPO_ROOT / ".dockerignore"
    smoke_script = REPO_ROOT / "scripts" / "remote" / "container_smoke.py"

    assert dockerfile.exists(), f"Missing {dockerfile}"
    assert dockerignore.exists(), f"Missing {dockerignore}"
    assert smoke_script.exists(), f"Missing {smoke_script}"

    # --- Dockerfile substrings ---
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    required_dockerfile_substrings = [
        "FROM pytorch/pytorch:",
        "WORKDIR /workspace",
        "COPY pyproject.toml uv.lock ./",
        "uv sync --frozen",
        'CMD ["python", "scripts/remote/container_smoke.py"]',
    ]
    for substr in required_dockerfile_substrings:
        assert substr in dockerfile_text, f"Dockerfile missing: {substr!r}"

    # --- .dockerignore entries ---
    dockerignore_text = dockerignore.read_text(encoding="utf-8")
    required_ignore_entries = [
        "data/raw/",
        "data/processed/",
        "models/**",
        "cache/",
        "scenic_artifacts/",
        ".venv/",
        ".git/",
        "notebooks/__marimo__/",
    ]
    for entry in required_ignore_entries:
        assert entry in dockerignore_text, f".dockerignore missing: {entry!r}"

    # --- Smoke script JSON output ---
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/remote/container_smoke.py",
        "--check-imports",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, f"Smoke script failed (exit {result.returncode}):\n{result.stderr}"

    output = json.loads(result.stdout.strip())
    assert output["ok"] is True, f"Expected ok=true, got: {output}"
    assert output["device"] == "cpu", f"Expected device=cpu, got: {output}"
    assert "checks" in output, f"Missing 'checks' key: {output}"
    checks = output["checks"]
    assert checks.get("torch") is True, f"torch check failed: {checks}"
    assert checks.get("torchvision") is True, f"torchvision check failed: {checks}"
    assert checks.get("timm") is True, f"timm check failed: {checks}"
    assert checks.get("boto3") is True, f"boto3 check failed: {checks}"
    assert checks.get("src.scenic_scorer.regression") is True, f"src.scenic_scorer.regression check failed: {checks}"
    assert checks.get("src.classifier.model") is True, f"src.classifier.model check failed: {checks}"
