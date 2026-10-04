"""State-backed Vast.ai training lifecycle command for Scenic regression runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.remote.cmux_vast_host import (  # noqa: E402
    ARTIFACTS_DIR,
    SshTarget,
    attach_ssh_key,
    copy_remote_path,
    copy_secrets,
    create_instance,
    instance_is_running,
    load_state,
    rel_state_path,
    require_commands,
    reboot_instance,
    run_command,
    select_offer_id_at,
    show_instance,
    ssh,
    state_path,
    utc_now,
    validate_task_name,
    wait_for_instance_endpoint,
    wait_for_ssh,
    write_state,
    update_status,
)

# High-performance Vast filter: >250 DLPerf, >200 TFLOPS, >100 DLPerf/$/hr,
# <$8/hr, with one or more GPUs.
DEFAULT_OFFER_QUERY = (
    "num_gpus>=1 dlperf>250 dlperf_per_dphtotal>100 total_flops>200 dph<8 "
    "verified=true direct_port_count>=1 rentable=true gpu_ram>=12 disk_space>=64"
)
DEFAULT_IMAGE = "ian139/scenicdriver-remote-training:latest"
DEFAULT_REMOTE_REPO_DIR = "/workspace"
DEFAULT_REMOTE_ENV_FILE = "/root/.scenic/aws.env"
DEFAULT_REMOTE_OUTPUT_BASE = "/workspace/scenic_artifacts/vast"
DEFAULT_S3_DATA_PREFIX = "processed/regression/"
DEFAULT_S3_MODELS_PREFIX = "models/"
DEFAULT_TRAIN_DATASET = "/workspace/data/processed/regression/train_features.npz"
DEFAULT_REMOTE_DATA_ROOT = "/workspace/data/processed/regression"
DEFAULT_REMOTE_MODELS_ROOT = "/workspace/models"
REMOTE_BENCHMARK_SPLIT_CSV = "/workspace/data/benchmark/benchmark_split.csv"
DEFAULT_VALIDATION_CHECKPOINT_KEY = "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt"

REUSABLE_STATUSES = {"destroyed", "failed_destroyed"}


@dataclass(frozen=True)
class VastTrainConfig:
    task_name: str
    run_id: str
    train_dataset_key: str
    benchmark_split_key: str
    s3_bucket: str
    s3_data_prefix: str
    s3_models_prefix: str
    validation_checkpoint_key: str
    s3_output_prefix: str
    image: str
    offer_query: str
    offer_id: int | None
    disk_gb: int
    allocation_attempts: int
    identity_file: str
    ssh_public_key: str
    local_secrets_env_file: str
    remote_env_file: str
    timeout_seconds: int
    poll_seconds: int
    epochs: int
    batch_size: int
    lr: float
    val_split: float
    seed: int
    destroy: bool
    keep_on_failure: bool


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def _generated_run_id(task_name: str) -> str:
    return f"{task_name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _remote_output_root(run_id: str) -> str:
    return f"{DEFAULT_REMOTE_OUTPUT_BASE}/{run_id}"


def _remote_lifecycle_dir(run_id: str) -> str:
    return f"{_remote_output_root(run_id)}/.lifecycle"


def _remote_train_output(run_id: str) -> str:
    return f"{_remote_output_root(run_id)}/scenic_regression_baseline.pt"

def _remote_prefix_download_path(key: str, prefix: str, remote_root: str) -> str:
    relative = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else Path(key).name
    return str(PurePosixPath(remote_root) / relative)


def _target_from_state(state: dict) -> SshTarget:
    return SshTarget(
        host=str(state["ssh_host"]),
        port=int(state["ssh_port"]),
        user=str(state.get("ssh_user", "root")),
        identity_file=str(state["identity_file"]),
    )


def _readable_file(path_text: str, label: str) -> str:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    if not os.access(path, os.R_OK):
        raise SystemExit(f"{label} is not readable: {path}")
    return str(path)


def build_remote_training_script(config: VastTrainConfig) -> str:
    remote_output_root = _remote_output_root(config.run_id)
    remote_lifecycle_dir = _remote_lifecycle_dir(config.run_id)
    train_output = _remote_train_output(config.run_id)
    validation_dataset = _remote_prefix_download_path(config.train_dataset_key, config.s3_data_prefix, DEFAULT_REMOTE_DATA_ROOT)
    validation_checkpoint = _remote_prefix_download_path(
        config.validation_checkpoint_key,
        config.s3_models_prefix,
        DEFAULT_REMOTE_MODELS_ROOT,
    )
    timeout_minutes = max(1, math.ceil(config.timeout_seconds / 60))
    return f"""set -Eeuo pipefail
cd {DEFAULT_REMOTE_REPO_DIR}
if [[ -f {_quote(config.remote_env_file)} ]]; then
  set -a
  source {_quote(config.remote_env_file)}
  set +a
fi
export SCENIC_S3_BUCKET={_quote(config.s3_bucket)}
export SCENIC_S3_DATA_PREFIX={_quote(config.s3_data_prefix)}
export SCENIC_S3_MODELS_PREFIX={_quote(config.s3_models_prefix)}
export SCENIC_S3_OUTPUT_PREFIX={_quote(config.s3_output_prefix)}
export SCENIC_RUN_ID={_quote(config.run_id)}
export SCENIC_TIMEOUT_MINUTES={_quote(timeout_minutes)}
export SCENIC_STEP_TIMEOUT_SECONDS=600
export SCENIC_OUTPUT_ROOT={_quote(remote_output_root)}
export SCENIC_LIFECYCLE_DIR={_quote(remote_lifecycle_dir)}
export SCENIC_TRAIN_DATASET_KEY={_quote(config.train_dataset_key)}
export SCENIC_TRAIN_DATASET={_quote(DEFAULT_TRAIN_DATASET)}
export SCENIC_BENCHMARK_SPLIT_KEY={_quote(config.benchmark_split_key)}
export SCENIC_BENCHMARK_SPLIT_CSV={_quote(REMOTE_BENCHMARK_SPLIT_CSV)}
export SCENIC_DATASET_PATH={_quote(validation_dataset)}
export SCENIC_VALIDATION_CHECKPOINT_KEY={_quote(config.validation_checkpoint_key)}
export SCENIC_CHECKPOINT_PATH={_quote(validation_checkpoint)}
export SCENIC_TRAIN_OUTPUT={_quote(train_output)}
export SCENIC_TRAIN_EPOCHS={_quote(config.epochs)}
export SCENIC_TRAIN_BATCH_SIZE={_quote(config.batch_size)}
export SCENIC_TRAIN_LR={_quote(config.lr)}
export SCENIC_TRAIN_VAL_SPLIT={_quote(config.val_split)}
export SCENIC_TRAIN_SEED={_quote(config.seed)}
mkdir -p "$SCENIC_OUTPUT_ROOT" "$SCENIC_LIFECYCLE_DIR" "$(dirname "$SCENIC_TRAIN_DATASET")" "$(dirname "$SCENIC_BENCHMARK_SPLIT_CSV")" "$(dirname "$SCENIC_DATASET_PATH")" "$(dirname "$SCENIC_CHECKPOINT_PATH")" "$(dirname "$SCENIC_TRAIN_OUTPUT")"
ensure_torch_supports_gpu() {{
  set +e
  python - <<'PY'
from __future__ import annotations

import sys

import torch

if not torch.cuda.is_available():
    print("torch CUDA support check skipped: cuda unavailable")
    raise SystemExit(0)
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{{major}}{{minor}}"
arches = set(torch.cuda.get_arch_list())
print(f"torch {{torch.__version__}}; gpu capability {{arch}}; supported arches: {{sorted(arches)}}")
if arch not in arches:
    if major >= 12:
        raise SystemExit(42)
    raise SystemExit(f"Current torch build does not support GPU capability {{arch}}")
PY
  local code=$?
  set -e
  if [[ "$code" -eq 42 ]]; then
    echo "GPU requires a newer PyTorch CUDA wheel; upgrading torch/torchvision with uv"
    uv pip install --system --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 --upgrade "torch==2.7.1" "torchvision==0.22.1" "numpy>=1.26,<2" || \
      uv pip install --system --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 --upgrade torch torchvision "numpy>=1.26,<2"
    python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{{major}}{{minor}}"
arches = set(torch.cuda.get_arch_list())
print(f"upgraded torch {{torch.__version__}}; gpu capability {{arch}}; supported arches: {{sorted(arches)}}")
if arch not in arches:
    raise SystemExit(f"Upgraded torch build still does not support GPU capability {{arch}}")
PY
  elif [[ "$code" -ne 0 ]]; then
    return "$code"
  fi
}}

write_lifecycle_json() {{
  local path="$1"
  local status="$2"
  local exit_code="${{3:-}}"
  python - "$path" "$status" "$exit_code" <<'PY'
import json
import os
import sys
import time

path, status, exit_text = sys.argv[1:4]
payload = {{
    "run_id": os.environ.get("SCENIC_RUN_ID"),
    "status": status,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "output_root": os.environ.get("SCENIC_OUTPUT_ROOT"),
}}
if exit_text:
    payload["exit_code"] = int(exit_text)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
}}
write_lifecycle_json "$SCENIC_LIFECYCLE_DIR/started.json" started
set +e
(
  set -euo pipefail
  ensure_torch_supports_gpu
  python -m src.data_pipeline.s3 download-file \
    --bucket "$SCENIC_S3_BUCKET" \
    --key "$SCENIC_TRAIN_DATASET_KEY" \
    --dest "$SCENIC_DATASET_PATH" \
    --required
  python -m src.data_pipeline.s3 download-file \
    --bucket "$SCENIC_S3_BUCKET" \
    --key "$SCENIC_VALIDATION_CHECKPOINT_KEY" \
    --dest "$SCENIC_CHECKPOINT_PATH" \
    --required
  python -m src.data_pipeline.s3 download-file \
    --bucket "$SCENIC_S3_BUCKET" \
    --key "$SCENIC_TRAIN_DATASET_KEY" \
    --dest "$SCENIC_TRAIN_DATASET" \
    --required
  python -m src.data_pipeline.s3 download-file \
    --bucket "$SCENIC_S3_BUCKET" \
    --key "$SCENIC_BENCHMARK_SPLIT_KEY" \
    --dest "$SCENIC_BENCHMARK_SPLIT_CSV" \
    --required
  python scripts/remote/vast_train.py validate \
    --device cuda \
    --checkpoint "$SCENIC_CHECKPOINT_PATH" \
    --dataset "$SCENIC_DATASET_PATH" \
    --output "$SCENIC_OUTPUT_ROOT/validation_result.json"
  python scripts/modeling/train_regression_baseline.py \
    --dataset "$SCENIC_TRAIN_DATASET" \
    --benchmark-split-csv "$SCENIC_BENCHMARK_SPLIT_CSV" \
    --output "$SCENIC_TRAIN_OUTPUT" \
    --epochs "$SCENIC_TRAIN_EPOCHS" \
    --batch-size "$SCENIC_TRAIN_BATCH_SIZE" \
    --lr "$SCENIC_TRAIN_LR" \
    --val-split "$SCENIC_TRAIN_VAL_SPLIT" \
    --seed "$SCENIC_TRAIN_SEED" \
    --device cuda
  python scripts/remote/vast_train.py validate \
    --device cuda \
    --checkpoint "$SCENIC_TRAIN_OUTPUT" \
    --dataset "$SCENIC_TRAIN_DATASET" \
    --output "$SCENIC_OUTPUT_ROOT/trained_inference_result.json"
  python -m src.data_pipeline.s3 upload-prefix \
    --src "$SCENIC_OUTPUT_ROOT" \
    --bucket "$SCENIC_S3_BUCKET" \
    --prefix "$SCENIC_S3_OUTPUT_PREFIX" \
    --required
)
exit_code=$?
set -e
printf '%s\n' "$exit_code" > "$SCENIC_LIFECYCLE_DIR/exit_code.txt"
if [[ "$exit_code" -eq 0 ]]; then
  write_lifecycle_json "$SCENIC_LIFECYCLE_DIR/done.json" done "$exit_code"
else
  write_lifecycle_json "$SCENIC_LIFECYCLE_DIR/failed.json" failed "$exit_code"
fi
exit "$exit_code"
"""


def initial_state(config: VastTrainConfig, offer_id: int, instance_id: int) -> dict:
    now = utc_now()
    remote_output_root = _remote_output_root(config.run_id)
    remote_lifecycle_dir = _remote_lifecycle_dir(config.run_id)
    return {
        "task_name": config.task_name,
        "run_id": config.run_id,
        "instance_id": instance_id,
        "offer_id": offer_id,
        "image": config.image,
        "disk_gb": config.disk_gb,
        "allocation_attempts": config.allocation_attempts,
        "ssh_host": "",
        "ssh_port": 0,
        "ssh_user": "root",
        "identity_file": config.identity_file,
        "ssh_public_key": config.ssh_public_key,
        "remote_env_file": config.remote_env_file,
        "remote_repo_dir": DEFAULT_REMOTE_REPO_DIR,
        "remote_output_root": remote_output_root,
        "remote_lifecycle_dir": remote_lifecycle_dir,
        "remote_pid_file": f"{remote_lifecycle_dir}/pid.txt",
        "s3_bucket": config.s3_bucket,
        "s3_data_prefix": config.s3_data_prefix,
        "s3_models_prefix": config.s3_models_prefix,
        "s3_output_prefix": config.s3_output_prefix,
        "train_dataset_key": config.train_dataset_key,
        "train_dataset": DEFAULT_TRAIN_DATASET,
        "benchmark_split_key": config.benchmark_split_key,
        "benchmark_split_csv": REMOTE_BENCHMARK_SPLIT_CSV,
        "train_output": _remote_train_output(config.run_id),
        "created_at": now,
        "updated_at": now,
        "timings_seconds": {},
        "launch_attempted": False,
        "launch_outcome": "pending",
        "status": "creating",
    }


def launch_remote_training(target: SshTarget, state: dict, script: str) -> None:
    lifecycle_dir = str(state["remote_lifecycle_dir"])
    pid_file = str(state["remote_pid_file"])
    log_path = f"{lifecycle_dir}/train.log"
    remote_command = (
        f"mkdir -p {shlex.quote(lifecycle_dir)} && "
        f"{{ nohup bash -lc {shlex.quote(script)} </dev/null >> {shlex.quote(log_path)} 2>&1 & "
        f"echo $! > {shlex.quote(pid_file)}; }}"
    )
    ssh(target, remote_command)


def read_remote_training_status(target: SshTarget, state: dict) -> str:
    lifecycle_dir = shlex.quote(str(state["remote_lifecycle_dir"]))
    remote_command = f"""python - {lifecycle_dir} <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

lifecycle = Path(sys.argv[1])
done = lifecycle / "done.json"
failed = lifecycle / "failed.json"
exit_code = lifecycle / "exit_code.txt"
pid = lifecycle / "pid.txt"
if done.exists():
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
        print("done" if int(payload["exit_code"]) == 0 else "failed")
    except Exception:
        print("unknown")
elif failed.exists():
    print("failed")
elif exit_code.exists():
    try:
        print("done" if int(exit_code.read_text(encoding="utf-8").strip()) == 0 else "failed")
    except ValueError:
        print("failed")
elif pid.exists():
    print("running")
else:
    print("pending")
PY"""
    result = ssh(target, remote_command, check=False)
    if result.returncode != 0:
        return "unreachable"
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unknown"


def poll_remote_training(target: SshTarget, state: dict, *, poll_seconds: int, timeout_seconds: int) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"
    while time.monotonic() < deadline:
        last_status = read_remote_training_status(target, state)
        if last_status in {"done", "failed"}:
            return last_status
        time.sleep(max(1, poll_seconds))
    raise RuntimeError(f"Timed out waiting for remote training status; last status: {last_status}")


def copy_training_artifacts(state: dict, *, required: bool) -> None:
    """Copy the remote run output locally; raise when a required copy fails."""
    if not state.get("ssh_host") or not state.get("ssh_port"):
        message = "state has no SSH endpoint; cannot copy training artifacts"
        if required:
            raise RuntimeError(message)
        print("Warning: " + message, file=sys.stderr)
        return
    target = _target_from_state(state)
    task_name = str(state["task_name"])
    run_id = str(state["run_id"])
    local_dir = ARTIFACTS_DIR / task_name / "scenic_artifacts" / "vast" / run_id
    remote_path = str(state["remote_output_root"]).rstrip("/") + "/"
    if not copy_remote_path(target, remote_path, local_dir, required=required) and required:
        raise RuntimeError(f"required training artifacts were not recovered from {remote_path}")


def destroy_recorded_instance(state: dict) -> bool:
    instance_id = state.get("instance_id")
    if instance_id is None:
        return False
    result = run_command(["vastai", "destroy", "instance", str(instance_id), "--yes"], check=False)
    return result.returncode == 0


def _config_from_args(args: argparse.Namespace) -> VastTrainConfig:
    run_id = args.run_id or _generated_run_id(args.task_name)
    s3_bucket = args.s3_bucket or os.environ.get("SCENIC_S3_BUCKET", "")
    s3_output_prefix = args.s3_output_prefix or f"outputs/vast/{run_id}/"
    return VastTrainConfig(
        task_name=args.task_name,
        run_id=run_id,
        train_dataset_key=args.train_dataset_key,
        benchmark_split_key=args.benchmark_split_key,
        s3_bucket=s3_bucket,
        s3_data_prefix=args.s3_data_prefix,
        s3_models_prefix=args.s3_models_prefix,
        validation_checkpoint_key=args.validation_checkpoint_key,
        s3_output_prefix=s3_output_prefix,
        image=args.image,
        offer_query=args.offer_query,
        offer_id=args.offer_id,
        disk_gb=args.disk_gb,
        allocation_attempts=max(1, args.allocation_attempts),
        identity_file=str(Path(args.identity_file).expanduser()),
        ssh_public_key=str(Path(args.ssh_public_key).expanduser()),
        local_secrets_env_file=str(Path(args.local_secrets_env_file).expanduser()),
        remote_env_file=args.remote_env_file,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        destroy=args.destroy,
        keep_on_failure=args.keep_on_failure,
    )


def _check_run_preconditions(config: VastTrainConfig) -> None:
    validate_task_name(config.task_name)
    path = state_path(config.task_name)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("status") not in REUSABLE_STATUSES:
            raise SystemExit(f"State exists: {rel_state_path(config.task_name)}; run cleanup or choose a new task name")
    require_commands(["vastai", "ssh", "scp"])
    _readable_file(config.identity_file, "identity file")
    _readable_file(config.ssh_public_key, "ssh public key")
    _readable_file(config.local_secrets_env_file, "local secrets env file")
    if not config.s3_bucket.strip():
        raise SystemExit("--s3-bucket or SCENIC_S3_BUCKET is required")
    if not config.train_dataset_key.strip():
        raise SystemExit("--train-dataset-key is required")
    if not config.benchmark_split_key.strip():
        raise SystemExit("--benchmark-split-key is required")
    if not config.validation_checkpoint_key.strip():
        raise SystemExit("--validation-checkpoint-key is required")
    if not config.validation_checkpoint_key.startswith(config.s3_models_prefix):
        raise SystemExit("--validation-checkpoint-key must be under --s3-models-prefix")
    if not config.train_dataset_key.startswith(config.s3_data_prefix):
        raise SystemExit("--train-dataset-key must be under --s3-data-prefix")


def _cleanup_command(task_name: str) -> str:
    return f"python scripts/remote/vast_train.py cleanup {task_name} --copy-artifacts --destroy --yes"


def _close_failed_run(config: VastTrainConfig, state: dict, error: str) -> None:
    """Recover what artifacts exist, then destroy or retain per --no-destroy/--keep-on-failure."""
    update_status(state, "copying_artifacts", error=error)
    try:
        copy_training_artifacts(state, required=False)
    except Exception as exc:  # noqa: BLE001 - best-effort recovery must not block the retention policy.
        state["artifact_copy_error"] = str(exc)
        write_state(state)
        print(f"Warning: best-effort artifact copy failed: {exc}", file=sys.stderr)
    if config.destroy and not config.keep_on_failure:
        update_status(state, "destroying")
        if destroy_recorded_instance(state):
            update_status(state, "failed_destroyed")
            print(f"Training failed ({error}); Vast instance destroyed: {config.task_name}", file=sys.stderr)
            return
    update_status(state, "failed_kept")
    print(f"Training failed ({error}); instance kept for {config.task_name}", file=sys.stderr)
    print(f"Cleanup: {_cleanup_command(config.task_name)}", file=sys.stderr)


def _handle_training_result(config: VastTrainConfig, state: dict, remote_status: str) -> int:
    update_status(state, "copying_artifacts", remote_training_status=remote_status)
    if remote_status != "done":
        _close_failed_run(config, state, f"remote training {remote_status}")
        return 1
    try:
        copy_training_artifacts(state, required=True)
    except Exception as exc:  # noqa: BLE001 - any recovery failure must retain the sole copy of outputs.
        # Never destroy the only host holding unrecovered outputs.
        update_status(state, "failed_kept", error=f"required training artifact recovery failed: {exc}")
        print(f"Training completed but artifact recovery failed; instance kept for {config.task_name}: {exc}", file=sys.stderr)
        print(f"Cleanup: {_cleanup_command(config.task_name)}", file=sys.stderr)
        return 1
    if not config.destroy:
        update_status(state, "completed_kept")
        print(f"Training completed; instance kept for {config.task_name}")
        print(f"Cleanup: {_cleanup_command(config.task_name)}")
        return 0
    update_status(state, "destroying")
    if destroy_recorded_instance(state):
        update_status(state, "destroyed")
        print(f"Training completed and Vast instance destroyed: {config.task_name}")
        return 0
    update_status(state, "failed_kept", error="destroy failed after successful training")
    print(f"Training completed but destroy failed; cleanup required for {config.task_name}", file=sys.stderr)
    return 1


class LaunchOutcomeUnknown(RuntimeError):
    """The remote training command may have been delivered; the launch outcome is unknown."""


def _mark_launch_outcome_unknown(state: dict, error: str) -> None:
    """Retain the host: training may be running, so never destroy or re-rent here."""
    update_status(
        state,
        "failed_kept",
        error=f"launch outcome unknown: {error}",
        launch_outcome="unknown",
    )
    print(
        f"Launch outcome unknown; training may be running, host kept for {state['task_name']}. "
        f"Cleanup: {_cleanup_command(str(state['task_name']))}",
        file=sys.stderr,
    )


def _abandon_unlaunched_allocation(state: dict | None, instance_id: int | None, error: str, *, retain: bool) -> None:
    """Close an allocation confirmed to have never launched training; only a recorded instance may be retained."""
    if state is not None:
        state["error"] = error
        write_state(state)
        if retain:
            update_status(state, "failed_kept")
            print(f"Provisioning failed ({error}); instance kept for {state['task_name']}", file=sys.stderr)
            print(f"Cleanup: {_cleanup_command(str(state['task_name']))}", file=sys.stderr)
            return
        update_status(state, "failed_destroyed" if destroy_recorded_instance(state) else "failed_kept")
        return
    if instance_id is None:
        return
    result = run_command(["vastai", "destroy", "instance", str(instance_id), "--yes"], check=False)
    if result.returncode != 0:
        print(
            f"Warning: failed to destroy unrecorded Vast instance {instance_id}; run: vastai destroy instance {instance_id} --yes",
            file=sys.stderr,
        )


def _provision_and_launch(config: VastTrainConfig, attempt: int) -> tuple[dict, SshTarget]:
    state: dict | None = None
    instance_id: int | None = None
    try:
        offer_id = int(config.offer_id) if config.offer_id is not None else select_offer_id_at(config.offer_query, attempt)
        instance_id = create_instance(offer_id, config.image, config.disk_gb)
        state = initial_state(config, offer_id, instance_id)
        state["allocation_attempt"] = attempt + 1
        write_state(state)
        attach_ssh_key(instance_id, config.ssh_public_key)
        update_status(state, "ssh_wait")
        host, port = wait_for_instance_endpoint(instance_id, config.timeout_seconds)
        state.update({"ssh_host": host, "ssh_port": port})
        write_state(state)
        reboot_instance(instance_id)
        host, port = wait_for_instance_endpoint(instance_id, config.timeout_seconds)
        state.update({"ssh_host": host, "ssh_port": port})
        write_state(state)
        target = SshTarget(host=host, port=port, user="root", identity_file=config.identity_file)
        wait_for_ssh(target, config.timeout_seconds)
        update_status(state, "provisioning")
        copy_secrets(target, config.local_secrets_env_file, config.remote_env_file)
        script = build_remote_training_script(config)
        # From here the detached training may exist remotely even when the launch
        # acknowledgement is lost: persist the boundary before firing.
        state["launch_attempted"] = True
        write_state(state)
        launch_remote_training(target, state, script)
        state["launch_outcome"] = "launched"
        write_state(state)
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        error = "interrupted during provisioning" if interrupted else str(exc)
        if state is not None and state.get("launch_attempted"):
            # Ambiguous boundary: the detached training may be running on this
            # host. Retain it unconditionally; only explicit operator cleanup
            # may destroy it. Never re-rent a replacement either.
            _mark_launch_outcome_unknown(state, error)
            if interrupted:
                raise
            raise LaunchOutcomeUnknown(error) from exc
        # Confirmed-never-launched: bad hosts are destroyed before re-renting;
        # an operator interrupt honors --no-destroy/--keep-on-failure.
        _abandon_unlaunched_allocation(
            state,
            instance_id,
            error,
            retain=interrupted and (not config.destroy or config.keep_on_failure),
        )
        raise
    return state, target


def handle_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    try:
        _check_run_preconditions(config)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize precondition failures for CLI output.
        print(str(exc), file=sys.stderr)
        return 1

    attempt = 0
    while True:
        try:
            state, target = _provision_and_launch(config, attempt)
            break
        except LaunchOutcomeUnknown as exc:
            # The retained host may be running training; never rent a replacement.
            print(str(exc), file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - bad Vast hosts are common; rerent before training starts.
            attempt += 1
            if attempt >= config.allocation_attempts:
                print(str(exc), file=sys.stderr)
                return 1
            print(f"Vast allocation attempt {attempt} failed; retrying: {exc}", file=sys.stderr)

    try:
        update_status(state, "training_running")
        remote_status = poll_remote_training(
            target,
            state,
            poll_seconds=config.poll_seconds,
            timeout_seconds=config.timeout_seconds,
        )
        return _handle_training_result(config, state, remote_status)
    except KeyboardInterrupt:
        _close_failed_run(config, state, "interrupted")
        raise
    except Exception as exc:  # noqa: BLE001 - copy logs and apply retention policy on orchestrator failures.
        _close_failed_run(config, state, str(exc))
        return 1


def handle_status(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
    state = load_state(args.task_name)
    payload = dict(state)
    remote_status = "not_checked"
    if state.get("ssh_host") and state.get("ssh_port"):
        remote_status = read_remote_training_status(_target_from_state(state), state)
    payload["remote_training_status"] = remote_status
    instance_id = state.get("instance_id")
    if instance_id is not None:
        try:
            instance = show_instance(int(instance_id))
            payload["instance_running"] = instance_is_running(instance)
        except Exception as exc:  # noqa: BLE001 - status still reports local state without Vast CLI reachability.
            payload["instance_status_error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def handle_cleanup(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
    state = load_state(args.task_name)
    if args.copy_artifacts:
        try:
            copy_training_artifacts(state, required=True)
        except RuntimeError as exc:
            print(f"Artifact recovery failed for {args.task_name}; not destroying: {exc}", file=sys.stderr)
            return 1
    if not (args.destroy and args.yes):
        print(f"Destroy command: vastai destroy instance {state['instance_id']} --yes")
        print("Not destroying without --destroy --yes")
        return 0
    previous_status = str(state.get("status", ""))
    update_status(state, "destroying")
    try:
        destroyed = destroy_recorded_instance(state)
    except Exception as exc:  # noqa: BLE001 - persist a retryable terminal state.
        error = f"destroy failed during cleanup: {exc}"
    else:
        error = "destroy failed during cleanup" if not destroyed else ""
    if error:
        update_status(state, "failed_kept", error=error)
        print(f"Destroy failed for Vast training task: {args.task_name}", file=sys.stderr)
        return 1
    final_status = "failed_destroyed" if previous_status.startswith("failed") else "destroyed"
    update_status(state, final_status)
    print(f"Vast training instance destroyed: {args.task_name}")
    return 0


def handle_validate(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from src.scenic_scorer.regression import ScenicRegressionModel, ScenicScoreDataset
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable")
    checkpoint = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    try:
        checkpoint_payload = torch.load(
            checkpoint,
            map_location=args.device,
            weights_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize corrupt artifact failures.
        raise ValueError(f"corrupt or unreadable checkpoint: {checkpoint}") from exc
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint}")
    required_checkpoint_keys = {"model_state_dict", "vit_dim", "terrain_dim", "num_classes"}
    missing_keys = sorted(required_checkpoint_keys - set(checkpoint_payload))
    if missing_keys:
        raise ValueError(f"checkpoint missing required keys: {missing_keys}")

    dimensions: dict[str, int] = {}
    for name in ("vit_dim", "terrain_dim", "num_classes"):
        value = checkpoint_payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"checkpoint {name} must be a positive integer")
        dimensions[name] = int(value)
    hidden_value = checkpoint_payload.get("hidden_dim", 256)
    if isinstance(hidden_value, bool) or not isinstance(hidden_value, (int, np.integer)) or hidden_value < 2:
        raise ValueError("checkpoint hidden_dim must be an integer >= 2")
    dimensions["hidden_dim"] = int(hidden_value)

    try:
        model = ScenicRegressionModel(**dimensions).to(args.device)
        model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("checkpoint model state is incompatible with ScenicRegressionModel") from exc
    model.eval()

    try:
        feature_dataset = ScenicScoreDataset(dataset_path)
    except Exception as exc:  # noqa: BLE001 - normalize corrupt artifact failures.
        raise ValueError(f"corrupt or incompatible dataset: {dataset_path}") from exc
    if len(feature_dataset) < 1:
        raise ValueError(f"dataset is empty: {dataset_path}")
    for values in (
        feature_dataset.vit_embeddings,
        feature_dataset.terrain_features,
        feature_dataset.class_logits,
        feature_dataset.scenic_scores,
    ):
        if not torch.isfinite(values).all().item():
            raise ValueError(f"dataset contains non-finite values: {dataset_path}")
    vit_embedding, terrain_features, class_logits, _, _ = feature_dataset[0]
    if any(
        values.ndim != 1
        for values in (vit_embedding, terrain_features, class_logits)
    ):
        raise ValueError(f"dataset feature arrays must be rank-2: {dataset_path}")
    actual_dimensions = {
        "vit_dim": int(vit_embedding.shape[0]),
        "terrain_dim": int(terrain_features.shape[0]),
        "num_classes": int(class_logits.shape[0]),
    }
    if any(actual_dimensions[name] != dimensions[name] for name in actual_dimensions):
        raise ValueError(
            "dataset dimensions are incompatible with checkpoint: "
            f"dataset={actual_dimensions}, checkpoint={dimensions}"
        )

    inputs = (
        vit_embedding.unsqueeze(0).to(args.device),
        terrain_features.unsqueeze(0).to(args.device),
        class_logits.unsqueeze(0).to(args.device),
    )
    with torch.inference_mode():
        prediction = model(*inputs)
    if prediction.numel() != 1 or not torch.isfinite(prediction).all().item():
        raise ValueError("model forward pass produced a non-finite result")

    payload = {
        "ok": True,
        "device": args.device,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "torch_version": torch.__version__,
        "dimensions": actual_dimensions,
        "prediction": float(prediction.reshape(-1)[0].item()),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and clean up S3-backed Scenic training on a temporary Vast.ai GPU instance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Start a Vast instance, train, copy artifacts, and destroy by default")
    run.add_argument("task_name")
    run.add_argument("--train-dataset-key", required=True)
    run.add_argument(
        "--benchmark-split-key",
        required=True,
        help="Exact S3 key of the benchmark split CSV whose val/test identities are excluded from training",
    )
    run.add_argument("--s3-bucket")
    run.add_argument("--s3-data-prefix", default=DEFAULT_S3_DATA_PREFIX)
    run.add_argument("--s3-models-prefix", default=DEFAULT_S3_MODELS_PREFIX)
    run.add_argument("--validation-checkpoint-key", default=DEFAULT_VALIDATION_CHECKPOINT_KEY)
    run.add_argument("--s3-output-prefix")
    run.add_argument("--run-id")
    run.add_argument("--image", default=DEFAULT_IMAGE)
    offer = run.add_mutually_exclusive_group()
    offer.add_argument("--offer-query", default=DEFAULT_OFFER_QUERY)
    offer.add_argument("--offer-id", type=int)
    run.add_argument("--disk-gb", type=int, default=64)
    run.add_argument("--allocation-attempts", type=int, default=3)
    run.add_argument("--identity-file", default="~/.ssh/id_ed25519")
    run.add_argument("--ssh-public-key", default="~/.ssh/id_ed25519.pub")
    run.add_argument("--local-secrets-env-file", default=".secrets/aws.env")
    run.add_argument("--remote-env-file", default=DEFAULT_REMOTE_ENV_FILE)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--poll-seconds", type=int, default=30)
    run.add_argument("--epochs", type=int, default=40)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--lr", type=float, default=1e-3)
    run.add_argument("--val-split", type=float, default=0.15)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--no-destroy", dest="destroy", action="store_false", default=True)
    run.add_argument("--keep-on-failure", action="store_true")
    run.set_defaults(func=handle_run)

    status = subparsers.add_parser("status", help="Print local state plus remote training sentinel status")
    status.add_argument("task_name")
    status.set_defaults(func=handle_status)

    cleanup = subparsers.add_parser("cleanup", help="Recover artifacts and optionally destroy a recorded Vast training instance")
    cleanup.add_argument("task_name")

    cleanup.add_argument("--copy-artifacts", action="store_true")
    cleanup.add_argument("--destroy", action="store_true")
    cleanup.add_argument("--yes", action="store_true")
    cleanup.set_defaults(func=handle_cleanup)
    validate = subparsers.add_parser("validate", help="Validate a CUDA runtime and checkpoint/dataset pair")
    validate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output")
    validate.set_defaults(func=handle_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
