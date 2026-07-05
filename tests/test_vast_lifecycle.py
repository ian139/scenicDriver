from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vast_init_dry_run_renders_bootstrap_without_secrets() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/remote/vast_lifecycle.py"),
        "init",
        "--host",
        "203.0.113.10",
        "--repo-url",
        "https://github.com/Ian139/RemoteTraining.git",
        "--branch",
        "Ian139/RemoteTraining",
        "--image",
        "scenicdriver-training:local",
        "--s3-bucket",
        "scenicdriver-data",
        "--s3-only",
        "--dry-run",
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)

    assert payload["mode"] == "dry-run"
    assert payload["ssh_command"][0] == "ssh"
    assert "root@203.0.113.10" in payload["ssh_command"]

    bootstrap = payload["bootstrap_script"]
    assert "git clone" in bootstrap
    assert "Ian139/RemoteTraining" in bootstrap
    assert "docker run --rm --gpus all" in bootstrap
    assert "SCENIC_S3_BUCKET=scenicdriver-data" in bootstrap
    assert "SCENIC_S3_ONLY=1" in bootstrap
    assert "python scripts/modeling/train_regression_baseline.py --help" in bootstrap
    assert "--env-file /root/.scenic/aws.env" in bootstrap
    assert "AWS_SECRET_ACCESS_KEY=" not in bootstrap
    assert "AWS_ACCESS_KEY_ID=" not in bootstrap
    assert str(REPO_ROOT) not in bootstrap


def test_vast_init_defaults_to_prebuilt_dockerhub_image() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/remote/vast_lifecycle.py"),
        "init",
        "--host",
        "203.0.113.10",
        "--repo-url",
        "https://github.com/Ian139/RemoteTraining.git",
        "--branch",
        "Ian139/RemoteTraining",
        "--s3-bucket",
        "scenicdriver-data",
        "--s3-only",
        "--dry-run",
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    bootstrap = payload["bootstrap_script"]
    assert "docker pull ian139/scenicdriver-remote-training:latest" in bootstrap
    assert "docker run --rm --gpus all" in bootstrap
    assert "Provide --image or --containerfile; no tracked container target is available." not in result.stderr
