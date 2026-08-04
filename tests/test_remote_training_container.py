from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

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
        "cuda12.4",
        "ca-certificates",
        "WORKDIR /workspace",
        "COPY pyproject.toml uv.lock ./",
        "uv export --locked --no-dev --no-emit-project",
        "--prune torch",
        "--prune torchvision",
        "--output-file /tmp/remote-training-requirements.txt",
        "--requirement /tmp/remote-training-requirements.txt",
        "--no-deps -e .",
        'CMD ["python", "scripts/remote/container_smoke.py"]',
    ]
    assert "awscli" not in dockerfile_text
    assert "openssh-client" not in dockerfile_text
    for substr in required_dockerfile_substrings:
        assert substr in dockerfile_text, f"Dockerfile missing: {substr!r}"

    # --- .dockerignore entries ---
    dockerignore_text = dockerignore.read_text(encoding="utf-8")
    # --- Active CMUX lifecycle state ---
    required_ignore_entries = [
        "data/raw/",
        "data/processed/",
        "models/**",
        "cache/",
        "scenic_artifacts/",
        ".venv/",
        ".git/",
        ".secrets/",
        ".cmux-vast/",
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


def test_vast_provision_script_is_fail_fast_and_syncs_s3() -> None:
    script = (REPO_ROOT / "scripts" / "remote" / "provision_vast.sh").read_text(encoding="utf-8")

    required_snippets = [
        "set -euo pipefail",
        'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
        'cd "$PROJECT_ROOT"',
        "verify_identity()",
        "verify_bucket()",
        "boto3.client(\"sts\").get_caller_identity()",
        "boto3.client(\"s3\").head_bucket",
        "require_s3_prefix \"$SCENIC_S3_DATA_PREFIX\" \"data\"",
        "require_s3_prefix \"$SCENIC_S3_MODELS_PREFIX\" \"models\"",
        "python -m src.data_pipeline.s3 check-prefix",
        "python -m src.data_pipeline.s3 download-prefix",
        "python scripts/remote/container_smoke.py --device cuda --check-imports --json",
        "python scripts/remote/minimal_inference.py",
        "python -m src.data_pipeline.s3 upload-prefix",
        "scripts/remote/vast-down.sh <task-name> --copy-artifacts --destroy --yes",
    ]
    for snippet in required_snippets:
        assert snippet in script, f"provision script missing: {snippet!r}"

    forbidden_snippets = [
        "aws ",
        "have_aws_cli",
        "command -v aws",
        "boto3 fallback",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in script, f"stale AWS CLI path found: {snippet!r}"

    assert "--optional || log" not in script


def test_minimal_inference_regression_cli_writes_json(tmp_path: Path) -> None:
    import numpy as np
    import torch

    from src.scenic_scorer.regression import ScenicRegressionModel

    checkpoint = tmp_path / "tiny.pt"
    dataset = tmp_path / "tiny.npz"
    output = tmp_path / "inference.json"

    model = ScenicRegressionModel(vit_dim=4, terrain_dim=2, num_classes=3)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vit_dim": 4,
            "terrain_dim": 2,
            "num_classes": 3,
        },
        checkpoint,
    )
    np.savez(
        dataset,
        vit_embeddings=np.zeros((1, 4), dtype=np.float32),
        terrain_features=np.zeros((1, 2), dtype=np.float32),
        class_logits=np.zeros((1, 3), dtype=np.float32),
        scenic_scores=np.array([5.0], dtype=np.float32),
    )

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/remote/minimal_inference.py",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["mode"] == "regression-inference"
    assert payload["device"] == "cpu"
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["dataset"] == str(dataset)
    assert payload["vit_embedding_shape"] == [1, 4]
    assert payload["terrain_features_shape"] == [1, 2]
    assert payload["class_logits_shape"] == [1, 3]
    assert payload["output_shape"] == [1, 1]


def test_vast_destroy_instance_uses_noninteractive_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.remote import cmux_vast_host

    commands: list[list[str]] = []
    monkeypatch.setattr(cmux_vast_host, "require_commands", lambda commands: None)
    monkeypatch.setattr(
        cmux_vast_host,
        "run_command",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(cmux_vast_host, "update_status", lambda state, status, **updates: state.update(status=status, **updates))

    state = {"instance_id": 12345}
    cmux_vast_host.destroy_instance(state)

    assert commands == [["vastai", "destroy", "instance", "12345", "--yes"]]
    assert state["status"] == "destroyed"


def test_vast_failed_allocation_cleanup_uses_noninteractive_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.remote import cmux_vast_host

    commands: list[list[str]] = []

    monkeypatch.setattr(cmux_vast_host, "check_up_preconditions", lambda args: None)
    monkeypatch.setattr(cmux_vast_host, "select_offer_id_at", lambda query, index: 67890)
    monkeypatch.setattr(cmux_vast_host, "create_instance", lambda offer_id, image, disk_gb: 12345)
    monkeypatch.setattr(
        cmux_vast_host,
        "attach_ssh_key",
        lambda instance_id, public_key_path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda state: None)
    monkeypatch.setattr(cmux_vast_host, "update_status", lambda state, status, **updates: state.update(status=status, **updates))
    monkeypatch.setattr(
        cmux_vast_host,
        "run_command",
        lambda command, **kwargs: commands.append(command),
    )

    args = SimpleNamespace(
        allocation_attempts=1,
        branch="main",
        disk_gb=32,
        identity_file="~/.ssh/id_ed25519",
        image="image",
        local_secrets_env_file=".secrets/aws.env",
        offer_id=None,
        offer_query="query",
        workspace_cwd="/tmp/scenic-drive",
        remote_repo_dir="/workspace/scenic-drive",
        ssh_public_key="~/.ssh/id_ed25519.pub",
        task_name="smoke",
        timeout_seconds=1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        cmux_vast_host.do_up(args)

    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
