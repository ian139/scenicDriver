from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.remote import vast_train
from src.scenic_scorer.regression import ScenicRegressionModel


def run_argv(tmp_path: Path, *extra: str) -> list[str]:
    identity = tmp_path / "id_ed25519"
    public_key = tmp_path / "id_ed25519.pub"
    secrets = tmp_path / "aws.env"
    identity.write_text("private", encoding="utf-8")
    public_key.write_text("public", encoding="utf-8")
    secrets.write_text("AWS_ACCESS_KEY_ID=x\nAWS_SECRET_ACCESS_KEY=y\n", encoding="utf-8")
    return [
        "run",
        "scenic-train-smoke",
        "--train-dataset-key",
        "processed/regression/smoke/tiny_features.npz",
        "--benchmark-split-key",
        "processed/benchmark/smoke/benchmark_split.csv",
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


def parse_run_args(tmp_path: Path, *extra: str) -> object:
    return vast_train.build_parser().parse_args(run_argv(tmp_path, *extra))


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


def test_train_run_requires_explicit_benchmark_split_key() -> None:
    with pytest.raises(SystemExit):
        vast_train.build_parser().parse_args(
            ["run", "scenic-train-smoke", "--train-dataset-key", "processed/regression/smoke/tiny_features.npz"]
        )


def test_train_run_rejects_blank_benchmark_split_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_run_args(tmp_path, "--benchmark-split-key", " ")
    monkeypatch.setattr(vast_train, "state_path", lambda task_name: tmp_path / "absent.json")
    monkeypatch.setattr(vast_train, "require_commands", lambda required: None)

    with pytest.raises(SystemExit, match="--benchmark-split-key is required"):
        vast_train._check_run_preconditions(vast_train._config_from_args(args))


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
    """A confirmed-never-launched allocation is disposable even under --no-destroy."""
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    monkeypatch.setattr(
        vast_train,
        "copy_secrets",
        lambda target, local_path, remote_path: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    args = parse_run_args(tmp_path, "--no-destroy")

    assert vast_train.handle_run(args) == 1

    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
    assert writes[-1]["status"] == "failed_destroyed"
    assert writes[-1].get("launch_attempted") is not True


def test_launch_attempted_is_persisted_before_remote_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    observed: dict[str, object] = {}

    def observe_launch(target: object, state: dict, script: str) -> None:
        observed["state_flag"] = state.get("launch_attempted")
        observed["persisted"] = bool(writes) and writes[-1].get("launch_attempted") is True

    monkeypatch.setattr(vast_train, "launch_remote_training", observe_launch)
    args = parse_run_args(tmp_path)

    assert vast_train.handle_run(args) == 0

    assert observed == {"state_flag": True, "persisted": True}
    assert any(write.get("launch_outcome") == "launched" for write in writes)
    assert writes[-1]["status"] == "destroyed"


@pytest.mark.parametrize("flags", [[], ["--keep-on-failure"], ["--no-destroy"]])
def test_raised_after_launch_retains_host_under_every_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    """An unacknowledged launch is ambiguous: training may be running, so the host is retained."""
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    monkeypatch.setattr(
        vast_train,
        "launch_remote_training",
        lambda target, state, script: (_ for _ in ()).throw(RuntimeError("ssh ack lost")),
    )
    args = parse_run_args(tmp_path, *flags)

    assert vast_train.handle_run(args) == 1

    assert ["vastai", "destroy", "instance", "12345", "--yes"] not in commands
    assert writes[-1]["status"] == "failed_kept"
    assert writes[-1]["launch_attempted"] is True
    assert writes[-1]["launch_outcome"] == "unknown"
    assert "Launch outcome unknown" in capsys.readouterr().err


def test_ambiguous_launch_never_rents_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    creations: list[int] = []
    monkeypatch.setattr(
        vast_train,
        "create_instance",
        lambda offer_id, image, disk_gb: creations.append(offer_id) or 12345,
    )
    monkeypatch.setattr(
        vast_train,
        "launch_remote_training",
        lambda target, state, script: (_ for _ in ()).throw(RuntimeError("ssh ack lost")),
    )
    args = parse_run_args(tmp_path, "--allocation-attempts", "2")

    assert vast_train.handle_run(args) == 1

    assert len(creations) == 1
    assert ["vastai", "destroy", "instance", "12345", "--yes"] not in commands
    assert writes[-1]["launch_outcome"] == "unknown"


def test_interrupt_during_launch_retains_host_even_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(vast_train, "launch_remote_training", interrupt)
    args = parse_run_args(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        vast_train.handle_run(args)

    assert ["vastai", "destroy", "instance", "12345", "--yes"] not in commands
    assert writes[-1]["status"] == "failed_kept"
    assert writes[-1]["launch_attempted"] is True
    assert writes[-1]["launch_outcome"] == "unknown"


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


def _write_validation_fixture(tmp_path: Path, *, compatible: bool = True) -> tuple[Path, Path]:
    vit_dim, terrain_dim, num_classes = (3, 2, 4)
    model = ScenicRegressionModel(
        vit_dim=vit_dim,
        terrain_dim=terrain_dim,
        num_classes=num_classes,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vit_dim": vit_dim,
            "terrain_dim": terrain_dim,
            "num_classes": num_classes,
        },
        checkpoint,
    )
    dataset = tmp_path / "features.npz"
    dataset_vit_dim = vit_dim if compatible else vit_dim + 1
    np.savez(
        dataset,
        vit_embeddings=np.ones((1, dataset_vit_dim), dtype=np.float32),
        terrain_features=np.ones((1, terrain_dim), dtype=np.float32),
        class_logits=np.ones((1, num_classes), dtype=np.float32),
        scenic_scores=np.ones((1,), dtype=np.float32),
    )
    return checkpoint, dataset


def test_validate_runs_finite_forward_pass_for_compatible_cpu_fixture(
    tmp_path: Path,
) -> None:
    checkpoint, dataset = _write_validation_fixture(tmp_path)
    output = tmp_path / "validation.json"
    args = vast_train.build_parser().parse_args(
        [
            "validate",
            "--device",
            "cpu",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ]
    )

    assert vast_train.handle_validate(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["device"] == "cpu"


def test_validate_rejects_incompatible_fixture(tmp_path: Path) -> None:
    checkpoint, dataset = _write_validation_fixture(tmp_path, compatible=False)
    args = vast_train.build_parser().parse_args(
        [
            "validate",
            "--device",
            "cpu",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset),
        ]
    )

    with pytest.raises(ValueError, match="dimensions"):
        vast_train.handle_validate(args)


def test_train_cleanup_persists_failed_kept_when_destroy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "scenic-train-smoke",
        "instance_id": 12345,
        "status": "completed_kept",
    }
    writes: list[dict] = []
    args = vast_train.build_parser().parse_args(
        ["cleanup", "scenic-train-smoke", "--destroy", "--yes"]
    )

    def update_status(current: dict, status: str, **updates: object) -> None:
        current.update(updates, status=status)
        writes.append(dict(current))

    monkeypatch.setattr(vast_train, "load_state", lambda task_name: state)
    monkeypatch.setattr(vast_train, "update_status", update_status)
    monkeypatch.setattr(vast_train, "destroy_recorded_instance", lambda current: False)

    assert vast_train.handle_cleanup(args) == 1
    assert state["status"] == "failed_kept"
    assert state["error"] == "destroy failed during cleanup"
    assert writes[-1]["status"] == "failed_kept"
    assert writes[-1]["error"] == "destroy failed during cleanup"


def test_validate_rejects_corrupt_checkpoint(tmp_path: Path) -> None:
    _, dataset = _write_validation_fixture(tmp_path)
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    args = vast_train.build_parser().parse_args(
        [
            "validate",
            "--device",
            "cpu",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset),
        ]
    )

    with pytest.raises(ValueError, match="corrupt or unreadable checkpoint"):
        vast_train.handle_validate(args)


def test_validate_refuses_malicious_pickle_without_executing_it(tmp_path: Path) -> None:
    _, dataset = _write_validation_fixture(tmp_path)
    marker = tmp_path / "pwned"

    class Exploit:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (os.mkdir, (str(marker),))

    checkpoint = tmp_path / "malicious.pt"
    torch.save({"model_state_dict": Exploit(), "vit_dim": 3, "terrain_dim": 2, "num_classes": 4}, checkpoint)
    args = vast_train.build_parser().parse_args(
        ["validate", "--device", "cpu", "--checkpoint", str(checkpoint), "--dataset", str(dataset)]
    )

    with pytest.raises(ValueError, match="corrupt or unreadable checkpoint"):
        vast_train.handle_validate(args)
    assert not marker.exists()


def _fail_remote_copy_with_false(target: object, remote: str, local: Path, *, required: bool) -> bool:
    return False


def _fail_remote_copy_with_oserror(target: object, remote: str, local: Path, *, required: bool) -> bool:
    raise OSError("scp transport broke")


@pytest.mark.parametrize("copy_remote_path", [_fail_remote_copy_with_false, _fail_remote_copy_with_oserror])
def test_completed_training_keeps_host_when_required_artifact_recovery_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    copy_remote_path: object,
) -> None:
    real_copy_training_artifacts = vast_train.copy_training_artifacts
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")
    monkeypatch.setattr(vast_train, "copy_training_artifacts", real_copy_training_artifacts)
    monkeypatch.setattr(vast_train, "copy_remote_path", copy_remote_path)
    args = parse_run_args(tmp_path)

    assert vast_train.handle_run(args) == 1

    assert ["vastai", "destroy", "instance", "12345", "--yes"] not in commands
    assert writes[-1]["status"] == "failed_kept"
    assert "artifact recovery failed" in writes[-1]["error"]


def test_failed_training_still_destroys_when_best_effort_copy_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_copy_training_artifacts = vast_train.copy_training_artifacts
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="failed")
    monkeypatch.setattr(vast_train, "copy_training_artifacts", real_copy_training_artifacts)
    monkeypatch.setattr(vast_train, "copy_remote_path", _fail_remote_copy_with_oserror)
    args = parse_run_args(tmp_path)

    assert vast_train.handle_run(args) == 1

    assert ["vastai", "destroy", "instance", "12345", "--yes"] in commands
    assert writes[-1]["status"] == "failed_destroyed"
    assert "scp transport broke" in writes[-1]["artifact_copy_error"]


@pytest.mark.parametrize("phase", ["provisioning", "polling"])
@pytest.mark.parametrize(
    ("flags", "retained"),
    [([], False), (["--keep-on-failure"], True), (["--no-destroy"], True)],
)
def test_interrupt_applies_retention_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    flags: list[str],
    retained: bool,
) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(vast_train, "wait_for_ssh" if phase == "provisioning" else "poll_remote_training", interrupt)
    args = parse_run_args(tmp_path, *flags)

    with pytest.raises(KeyboardInterrupt):
        vast_train.handle_run(args)

    destroyed = ["vastai", "destroy", "instance", "12345", "--yes"] in commands
    assert destroyed is not retained
    assert writes[-1]["status"] == ("failed_kept" if retained else "failed_destroyed")
    assert "interrupted" in writes[-1]["error"]


def test_cli_returns_130_after_interrupt_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands, writes = patch_successful_allocation(monkeypatch, tmp_path, poll_status="done")

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(vast_train, "poll_remote_training", interrupt)

    assert vast_train.main(run_argv(tmp_path)) == 130
    assert writes[-1]["status"] == "failed_destroyed"


def test_cleanup_does_not_destroy_when_requested_artifact_copy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    state = {
        "task_name": "scenic-train-smoke",
        "run_id": "run-123",
        "instance_id": 12345,
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 2222,
        "identity_file": str(tmp_path / "id_ed25519"),
        "remote_output_root": "/workspace/scenic_artifacts/vast/run-123",
        "status": "completed_kept",
    }
    monkeypatch.setattr(vast_train, "load_state", lambda task_name: state)
    monkeypatch.setattr(vast_train, "copy_remote_path", lambda target, remote, local, *, required: False)
    monkeypatch.setattr(vast_train, "update_status", lambda current, status, **updates: current.update(status=status, **updates))

    def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(vast_train, "run_command", run_command)
    args = vast_train.build_parser().parse_args(["cleanup", "scenic-train-smoke", "--copy-artifacts", "--destroy", "--yes"])

    assert vast_train.handle_cleanup(args) == 1

    assert commands == []
    assert state["status"] == "completed_kept"