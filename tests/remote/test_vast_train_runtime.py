from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import pytest

from scripts.remote import vast_train


def config(tmp_path: Path, *, seconds: int = 5) -> vast_train.VastTrainConfig:
    args = vast_train.build_parser().parse_args([
        "run", "runtime-test", "--run-id", "runtime-test",
        "--train-dataset-key", "processed/regression/features.npz",
        "--benchmark-split-key", "processed/benchmark/split.csv",
        "--labels-key", "processed/regression/labels.csv",
        "--classifier-checkpoint-key", "models/classifier.pt",
        "--s3-bucket", "test", "--timeout-seconds", str(seconds),
    ])
    return vast_train._config_from_args(args)


def remote_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, old_runtime: bool = False) -> tuple[str, Path]:
    """Execute generated Bash with simulated GPU/S3, not paid cloud services."""
    monkeypatch.setattr(vast_train, "DEFAULT_REMOTE_REPO_DIR", str(vast_train.PROJECT_ROOT))
    monkeypatch.setattr(vast_train, "DEFAULT_REMOTE_OUTPUT_BASE", str(tmp_path / "outputs"))
    monkeypatch.setattr(vast_train, "DEFAULT_TRAIN_DATASET", str(tmp_path / "data/train.npz"))
    monkeypatch.setattr(vast_train, "DEFAULT_REMOTE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(vast_train, "DEFAULT_REMOTE_MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(vast_train, "REMOTE_BENCHMARK_SPLIT_CSV", str(tmp_path / "split.csv"))
    monkeypatch.setattr(vast_train, "REMOTE_LABELS_CSV", str(tmp_path / "labels.csv"))
    monkeypatch.setattr(vast_train, "REMOTE_CLASSIFIER_CHECKPOINT", str(tmp_path / "classifier.pt"))
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        f"__version__ = {'2.5.1' if old_runtime else '2.10.0+cu128'!r}\n"
        "class cuda:\n"
        "    is_available = staticmethod(lambda: True)\n"
        "    get_device_capability = staticmethod(lambda index: (12, 0))\n"
        "    get_arch_list = staticmethod(lambda: ['sm_120'])\n"
        "class version:\n"
        "    cuda = '12.8'\n"
    )
    (modules / "torchvision.py").write_text("__version__ = '0.25.0+cu128'\n")
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == '-m' || \"$1\" == 'scripts/remote/vast_train.py' ]]; then exit 0; fi\n"
        "if [[ \"$1\" == 'scripts/modeling/train_regression_baseline.py' ]]; then\n"
        "  touch \"$SCENIC_OUTPUT_ROOT/training-entered\"\n"
        "  if [[ \"${TEST_HANG:-0}\" == 1 ]]; then\n"
        "    sleep 3\n"
        "    touch \"$SCENIC_OUTPUT_ROOT/escaped-deadline\"\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        f"export PYTHONPATH={shlex.quote(str(modules) + ':' + str(vast_train.PROJECT_ROOT))}\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n"
    )
    python.chmod(0o755)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$TEST_UV_LOG\"\nexit 17\n")
    uv.chmod(0o755)
    monkeypatch.setenv("TEST_UV_LOG", str(tmp_path / "uv.log"))
    script = vast_train.build_remote_training_script(config(tmp_path))
    # Absolute stand-ins survive login-shell PATH initialization.
    script = script.replace("python ", shlex.quote(str(python)) + " ")
    script = script.replace("uv pip", shlex.quote(str(uv)) + " pip")
    return script, tmp_path / "outputs/runtime-test"


def test_image_default_and_docker_runtime_are_aligned(tmp_path: Path) -> None:
    current = config(tmp_path)
    assert current.image == "ian139/scenicdriver-remote-training:torch2.10.0-cu128-20261006"
    assert ":latest" not in current.image
    assert "2.10.0-cuda12.8-cudnn9-runtime@sha256:" in (vast_train.PROJECT_ROOT / "Dockerfile.remote-training").read_text()
    assert config(tmp_path).timeout_seconds == 5
    override = vast_train.build_parser().parse_args([
        "run", "runtime-test", "--train-dataset-key", "x", "--benchmark-split-key", "y", "--image", "example/runtime@sha256:" + "a" * 64,
        "--labels-key", "labels.csv", "--classifier-checkpoint-key", "classifier.pt",
    ])
    assert vast_train._config_from_args(override).image == override.image


@pytest.mark.parametrize("image", ["example/runtime", "example/runtime:latest", "registry:5000/runtime", ""])
def test_image_never_implicitly_uses_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    monkeypatch.setattr(vast_train, "state_path", lambda task: tmp_path / "absent.json")
    monkeypatch.setattr(vast_train, "require_commands", lambda commands: None)
    monkeypatch.setattr(vast_train, "_readable_file", lambda path, label: path)
    with pytest.raises(SystemExit, match="version tag or digest"):
        vast_train._check_run_preconditions(replace(config(tmp_path), image=image))


def test_pinned_wheel_failure_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script, output = remote_fixture(tmp_path, monkeypatch, old_runtime=True)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Pinned PyTorch CUDA wheels unavailable" in result.stderr
    calls = (tmp_path / "uv.log").read_text().splitlines()
    assert len(calls) == 1
    assert "torch==2.10.0" in calls[0] and "torchvision==0.25.0" in calls[0]
    assert "--upgrade torch torchvision" not in script
    assert not (output / "training-entered").exists()
    failed = json.loads((output / ".lifecycle/failed.json").read_text())
    assert failed["exit_code"] == 1
    assert failed["runtime_identity"]["torch_version"] == "2.5.1"


def test_run_artifacts_record_actual_runtime_and_source_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script, output = remote_fixture(tmp_path, monkeypatch)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    identity = json.loads((output / "runtime_identity.json").read_text())
    assert identity["image"] == vast_train.DEFAULT_IMAGE
    assert identity["torch_version"] == "2.10.0+cu128"
    assert identity["torchvision_version"] == "0.25.0+cu128"
    assert identity["cuda_version"] == "12.8"
    assert identity["source_digest"].startswith("sha256:")
    assert len(identity["source_digest"]) == 71
    done = json.loads((output / ".lifecycle/done.json").read_text())
    assert done["runtime_identity"] == identity
    assert done["remote_deadline_seconds"] == 5
    state = vast_train.initial_state(config(tmp_path), 1, 2)
    state.update(ssh_host="host", ssh_port=22)
    monkeypatch.setattr(vast_train, "ARTIFACTS_DIR", tmp_path / "recovered")
    monkeypatch.setattr(vast_train, "write_state", lambda current: None)

    def copy(target: object, remote: str, local: Path, *, required: bool) -> bool:
        shutil.copytree(output, local)
        return True

    monkeypatch.setattr(vast_train, "copy_remote_path", copy)
    vast_train.copy_training_artifacts(state, required=True)
    assert state["runtime_identity"] == identity
    assert state["remote_deadline_seconds"] == 5
    assert state["provider_expiry"] is None


def test_detached_deadline_kills_training_and_preserves_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeout_binary = shutil.which("timeout") or shutil.which("gtimeout")
    if timeout_binary is None:
        # macOS has no GNU coreutils. Exercise the same process-group deadline
        # contract with real signals; Linux runs the actual GNU binary.
        timeout_path = tmp_path / "timeout"
        timeout_code = (
            "import os, signal, subprocess, sys\n"
            "assert sys.argv[1] == '--kill-after=30s'\n"
            "seconds = int(sys.argv[2].removesuffix('s'))\n"
            "child = subprocess.Popen(sys.argv[3:], start_new_session=True)\n"
            "try:\n"
            "    code = child.wait(timeout=seconds)\n"
            "except subprocess.TimeoutExpired:\n"
            "    os.killpg(child.pid, signal.SIGTERM)\n"
            "    try:\n"
            "        child.wait(timeout=30)\n"
            "        code = 124\n"
            "    except subprocess.TimeoutExpired:\n"
            "        os.killpg(child.pid, signal.SIGKILL)\n"
            "        child.wait()\n"
            "        code = 137\n"
            "sys.exit(code)\n"
        )
        timeout_path.write_text(
            "#!/bin/bash\n"
            f"exec {shlex.quote(sys.executable)} -c {shlex.quote(timeout_code)} \"$@\"\n"
        )
        timeout_path.chmod(0o755)
        timeout_binary = str(timeout_path)
    script, output = remote_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("TEST_HANG", "1")
    state = vast_train.initial_state(config(tmp_path, seconds=1), 1, 2)
    commands: list[str] = []

    def ssh(target: object, command: str) -> None:
        commands.append(command)
        command = command.replace("command -v timeout", "command -v " + shlex.quote(timeout_binary))
        command = command.replace("timeout --kill-after=", shlex.quote(timeout_binary) + " --kill-after=")
        subprocess.run(["bash", "-c", command], check=True, timeout=5)

    monkeypatch.setattr(vast_train, "ssh", ssh)
    vast_train.launch_remote_training(object(), state, script)
    assert "timeout --kill-after=30s 1s bash -lc" in commands[0]
    assert "nohup" in commands[0]
    deadline = time.monotonic() + 5
    failed_path = output / ".lifecycle/failed.json"
    while not failed_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    failed = json.loads(failed_path.read_text())
    assert failed["exit_code"] == 124
    assert failed["remote_deadline_seconds"] == 1
    assert failed["runtime_identity"]["image"] == vast_train.DEFAULT_IMAGE
    assert (output / "training-entered").exists()
    time.sleep(3)
    assert not (output / "escaped-deadline").exists()
    assert (output / "runtime_identity.json").exists()
    assert (output / ".lifecycle/train.log").exists()
