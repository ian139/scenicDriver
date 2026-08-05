from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.remote import cmux_vast_host, vast_train


def make_config(**overrides: object) -> vast_train.VastTrainConfig:
    values: dict[str, object] = {
        "task_name": "scenic-train-smoke",
        "run_id": "run-123",
        "train_dataset_key": "processed/regression/smoke/tiny_features.npz",
        "s3_bucket": "scenicdriver-data",
        "s3_data_prefix": "processed/regression/smoke/",
        "s3_models_prefix": "models/smoke/",
        "validation_checkpoint_key": "models/smoke/tiny_regression.pt",
        "s3_output_prefix": "outputs/vast/run-123/",
        "image": vast_train.DEFAULT_IMAGE,
        "offer_query": vast_train.DEFAULT_OFFER_QUERY,
        "offer_id": None,
        "disk_gb": 64,
        "allocation_attempts": 1,
        "identity_file": "~/.ssh/id_ed25519",
        "ssh_public_key": "~/.ssh/id_ed25519.pub",
        "local_secrets_env_file": ".secrets/aws.env",
        "remote_env_file": vast_train.DEFAULT_REMOTE_ENV_FILE,
        "timeout_seconds": 1800,
        "poll_seconds": 30,
        "epochs": 40,
        "batch_size": 128,
        "lr": 1e-3,
        "val_split": 0.15,
        "seed": 42,
        "destroy": True,
        "keep_on_failure": False,
    }
    values.update(overrides)
    return vast_train.VastTrainConfig(**values)


def parse_run_args(tmp_path: Path, *extra: str) -> object:
    identity = tmp_path / "id_ed25519"
    public_key = tmp_path / "id_ed25519.pub"
    secrets = tmp_path / "aws.env"
    identity.write_text("private", encoding="utf-8")
    public_key.write_text("public", encoding="utf-8")
    secrets.write_text("AWS_ACCESS_KEY_ID=x\nAWS_SECRET_ACCESS_KEY=y\n", encoding="utf-8")
    return vast_train.build_parser().parse_args(
        [
            "run",
            "scenic-train-smoke",
            "--train-dataset-key",
            "processed/regression/smoke/tiny_features.npz",
            "--s3-bucket",
            "scenicdriver-data",
            "--identity-file",
            str(identity),
            "--ssh-public-key",
            str(public_key),
            "--local-secrets-env-file",
            str(secrets),
            "--allocation-attempts",
            "1",
            "--timeout-seconds",
            "1",
            "--poll-seconds",
            "1",
            *extra,
        ]
    )


def patch_successful_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    poll_status: str,
) -> tuple[list[list[str]], list[dict]]:
    commands: list[list[str]] = []
    writes: list[dict] = []
    state_file = tmp_path / "state" / "scenic-train-smoke.json"

    def write_state(state: dict) -> None:
        writes.append(dict(state))
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{}", encoding="utf-8")

    def update_status(state: dict, status: str, **updates: object) -> None:
        state.update(updates)
        state["status"] = status
        write_state(state)

    monkeypatch.setattr(vast_train, "state_path", lambda task_name: state_file)
    monkeypatch.setattr(vast_train, "write_state", write_state)
    monkeypatch.setattr(vast_train, "update_status", update_status)
    monkeypatch.setattr(vast_train, "require_commands", lambda required: None)
    monkeypatch.setattr(vast_train, "select_offer_id_at", lambda query, index: 67890)
    monkeypatch.setattr(vast_train, "create_instance", lambda offer_id, image, disk_gb: 12345)
    monkeypatch.setattr(vast_train, "attach_ssh_key", lambda instance_id, public_key_path: None)
    monkeypatch.setattr(vast_train, "reboot_instance", lambda instance_id: None)
    monkeypatch.setattr(vast_train, "wait_for_instance_endpoint", lambda instance_id, timeout_seconds: ("ssh5.vast.ai", 2222))
    monkeypatch.setattr(vast_train, "wait_for_ssh", lambda target, timeout_seconds: None)
    monkeypatch.setattr(vast_train, "copy_secrets", lambda target, local_path, remote_path: None)
    monkeypatch.setattr(vast_train, "launch_remote_training", lambda target, state, script: None)
    monkeypatch.setattr(
        vast_train,
        "poll_remote_training",
        lambda target, state, *, poll_seconds, timeout_seconds: poll_status,
    )
    monkeypatch.setattr(vast_train, "copy_training_artifacts", lambda state, *, required: None)

    def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(vast_train, "run_command", run_command)
    return commands, writes


def test_train_run_requires_exact_dataset_key() -> None:
    with pytest.raises(SystemExit):
        vast_train.build_parser().parse_args(["run", "scenic-train-smoke"])


def test_train_run_requires_dataset_key_under_data_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_run_args(
        tmp_path,
        "--s3-data-prefix",
        "processed/regression/smoke/",
        "--train-dataset-key",
        "processed/regression/other/tiny_features.npz",
    )
    monkeypatch.setattr(vast_train, "state_path", lambda task_name: tmp_path / "missing.json")
    monkeypatch.setattr(vast_train, "require_commands", lambda required: None)

    with pytest.raises(SystemExit, match="--train-dataset-key must be under --s3-data-prefix"):
        vast_train._check_run_preconditions(vast_train._config_from_args(args))


def test_train_run_requires_checkpoint_key_under_models_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_run_args(
        tmp_path,
        "--s3-models-prefix",
        "models/smoke/",
        "--validation-checkpoint-key",
        "models/other/tiny_regression.pt",
    )
    monkeypatch.setattr(vast_train, "state_path", lambda task_name: tmp_path / "missing.json")
    monkeypatch.setattr(vast_train, "require_commands", lambda required: None)

    with pytest.raises(SystemExit, match="--validation-checkpoint-key must be under --s3-models-prefix"):
        vast_train._check_run_preconditions(vast_train._config_from_args(args))


def test_remote_training_script_runs_validation_before_training() -> None:
    script = vast_train.build_remote_training_script(make_config())

    validation_index = script.index("python scripts/remote/vast_train.py validate")
    training_index = script.index("python scripts/modeling/train_regression_baseline.py")

    assert validation_index < training_index


def test_remote_training_script_writes_sentinels_and_uploads_outputs() -> None:
    script = vast_train.build_remote_training_script(make_config())

    required_snippets = [
        "started.json",
        "done.json",
        "failed.json",
        "exit_code.txt",
        "python scripts/remote/vast_train.py validate",
        "export SCENIC_DATASET_PATH=/workspace/data/processed/regression/tiny_features.npz",
        "export SCENIC_CHECKPOINT_PATH=/workspace/models/tiny_regression.pt",
    ]
    for snippet in required_snippets:
        assert snippet in script


def test_remote_training_script_escapes_python_newline_literal() -> None:
    script = vast_train.build_remote_training_script(make_config())

    assert r'handle.write("\n")' in script
    assert 'handle.write("\n")' not in script


def test_cli_config_drives_validation_artifact_exports(tmp_path: Path) -> None:
    args = parse_run_args(
        tmp_path,
        "--s3-data-prefix",
        "processed/regression/custom/",
        "--s3-models-prefix",
        "models/custom/",
        "--train-dataset-key",
        "processed/regression/custom/features_v6.npz",
        "--validation-checkpoint-key",
        "models/custom/scenic_regression_v4.pt",
    )

    script = vast_train.build_remote_training_script(vast_train._config_from_args(args))

    assert "export SCENIC_DATASET_PATH=/workspace/data/processed/regression/features_v6.npz" in script
    assert "export SCENIC_CHECKPOINT_PATH=/workspace/models/scenic_regression_v4.pt" in script


def test_training_statuses_are_accepted_by_shared_state_loader() -> None:
    emitted_statuses = {
        "provisioning",
        "training_running",
        "copying_artifacts",
        "destroying",
        "completed_kept",
        "failed_destroyed",
        "failed_kept",
        "destroyed",
    }

    assert emitted_statuses <= cmux_vast_host.VALID_STATUSES


def test_failed_kept_state_is_not_reused_without_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "state" / "scenic-train-smoke.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text('{"status": "failed_kept"}', encoding="utf-8")
    args = parse_run_args(tmp_path)
    monkeypatch.setattr(vast_train, "state_path", lambda task_name: state_file)

    with pytest.raises(SystemExit, match="run cleanup or choose a new task name"):
        vast_train._check_run_preconditions(vast_train._config_from_args(args))


def test_train_run_destroys_instance_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    args = parse_run_args(tmp_path)

    assert vast_train.handle_run(args) == 0

    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
    assert writes[-1]["status"] == "destroyed"


def test_train_run_destroys_instance_after_failure_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="failed")
    args = parse_run_args(tmp_path)

    assert vast_train.handle_run(args) == 1

    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
    assert writes[-1]["status"] == "failed_destroyed"


def test_train_run_no_destroy_still_destroys_bootstrap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    monkeypatch.setattr(
        vast_train,
        "launch_remote_training",
        lambda target, state, script: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    args = parse_run_args(tmp_path, "--no-destroy")

    assert vast_train.handle_run(args) == 1

    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
    assert writes[-1]["status"] == "failed_destroyed"


def test_train_cleanup_uses_recorded_instance_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    state = {
        "task_name": "scenic-train-smoke",
        "instance_id": 12345,
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 2222,
        "ssh_user": "root",
        "identity_file": str(tmp_path / "id_ed25519"),
        "remote_output_root": "/workspace/scenic_artifacts/vast/run-123/",
        "status": "training_running",
    }
    args = vast_train.build_parser().parse_args(
        ["cleanup", "scenic-train-smoke", "--copy-artifacts", "--destroy", "--yes"]
    )

    monkeypatch.setattr(vast_train, "load_state", lambda task_name: state)
    monkeypatch.setattr(
        vast_train,
        "create_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cleanup must not allocate")),
    )
    monkeypatch.setattr(vast_train, "copy_training_artifacts", lambda current, *, required: commands.append(["copy-artifacts"]))
    monkeypatch.setattr(vast_train, "update_status", lambda current, status, **updates: current.update(status=status, **updates))

    def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(vast_train, "run_command", run_command)

    assert vast_train.handle_cleanup(args) == 0

    assert ["copy-artifacts"] in commands
    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
