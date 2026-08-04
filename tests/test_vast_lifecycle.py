from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import src.remote_training.vast_lifecycle as lifecycle
from src.remote_training.vast_lifecycle import VastInitConfig, build_init_plan


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vast_init_plan_carries_retry_configuration() -> None:
    plan = build_init_plan(
        VastInitConfig(
            host="203.0.113.10",
            repo_url="https://github.com/Ian139/RemoteTraining.git",
            s3_bucket="scenicdriver-data",
            retries=7,
            retry_delay_seconds=11,
        )
    )

    assert plan.retries == 7
    assert plan.retry_delay_seconds == 11


def test_run_init_plan_uses_plan_retry_configuration(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_run(command, *, input, text):
        calls.append((command, input, text))
        return type("Completed", (), {"returncode": 9})()

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)
    monkeypatch.setattr(lifecycle.time, "sleep", sleeps.append)
    plan = lifecycle.VastInitPlan(
        ssh_command=["ssh", "host"],
        scp_command=None,
        bootstrap_script="bootstrap",
        retries=2,
        retry_delay_seconds=13,
    )

    assert lifecycle.run_init_plan(plan, dry_run=False) == 9
    assert len(calls) == 2
    assert sleeps == [13]


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
