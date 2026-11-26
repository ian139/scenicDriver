"""State-backed Vast.ai GPU host allocator with CMUX workspace handoff."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".cmux-vast" / "state"
ARTIFACTS_DIR = PROJECT_ROOT / ".cmux-vast" / "artifacts"
DEFAULT_OFFER_QUERY = "gpu_name=RTX_4090 num_gpus>=1 verified=true direct_port_count>=1 rentable=true"
DEFAULT_IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
DEFAULT_BRANCH = "Ian139/RemoteTraining"
DEFAULT_REMOTE_REPO_DIR = "/workspace/scenic-drive"
DEFAULT_REMOTE_ENV_FILE = "/root/.scenic/aws.env"
DEFAULT_VALIDATION_DATASET_KEY = (
    "processed/regression/features_masswhites_z14_mixed5000_v4_h4.npz"
)
DEFAULT_VALIDATION_CHECKPOINT_KEY = (
    "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt"
)
DEFAULT_CMUX_WORKSPACE_CWD = str(PROJECT_ROOT)
# Every subprocess is bounded: by the active lifecycle_deadline() when a
# lifecycle phase is running, otherwise by this per-command default.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 3600
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 3600
TEARDOWN_TIMEOUT_SECONDS = 3600
DESTROY_CONFIRM_TIMEOUT_SECONDS = 600
DESTROY_CONFIRM_POLL_SECONDS = 10
# Phases whose completion makes a remote artifact mandatory before teardown.
# State files without a completed_phases record are treated as having
# completed every phase, so unknown history never relaxes a requirement.
PHASE_BOOTSTRAP = "bootstrap"
PHASE_CONTAINER_VALIDATION = "container_validation"
PHASE_WORKSPACE_HANDOFF = "workspace_handoff"
ALL_PHASES = (PHASE_BOOTSTRAP, PHASE_CONTAINER_VALIDATION, PHASE_WORKSPACE_HANDOFF)
VALID_STATUSES = {
    "creating",
    "ssh_wait",
    "bootstrapping",
    "ready",
    "workspace_pending",
    "provisioning",
    "training_running",
    "copying_artifacts",
    "destroying",
    "completed_kept",
    "done",
    "failed",
    "failed_destroyed",
    "failed_kept",
    "destroyed",
}
EXCLUDED_ROOTS = {
    ".git",
    ".venv",
    ".cmux-vast",
    ".worktrees",
    ".secrets",
    "data/raw",
    "data/processed",
    "data/NWPU-RESISC45",
    "models",
    "cache",
    "scenic_artifacts",
}
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    user: str
    identity_file: str


class LifecycleDeadlineExceeded(RuntimeError):
    """A subprocess or wait outlived its lifecycle phase deadline."""


class TaskLockedError(RuntimeError):
    """Another process holds the interprocess claim on a task's state."""


_LIFECYCLE_DEADLINE: ContextVar[float | None] = ContextVar("cmux_vast_lifecycle_deadline", default=None)


@contextmanager
def lifecycle_deadline(seconds: float) -> Iterator[None]:
    """Bound every subprocess started inside the block by one phase deadline.

    The deadline replaces any outer one so cleanup phases (e.g. destroy after
    an allocation timeout) get their own budget instead of inheriting an
    already-expired deadline.
    """
    token = _LIFECYCLE_DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _LIFECYCLE_DEADLINE.reset(token)


def lifecycle_remaining_seconds() -> float | None:
    deadline = _LIFECYCLE_DEADLINE.get()
    return None if deadline is None else deadline - time.monotonic()


@contextmanager
def task_lock(task_name: str) -> Iterator[None]:
    """Hold an exclusive, non-blocking interprocess claim on one task's state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / f"{task_name}.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TaskLockedError(f"Task {task_name} is claimed by another cmux_vast_host process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_task_name(task_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", task_name or ""):
        raise SystemExit("task-name may contain only letters, numbers, dot, underscore, and hyphen")


def state_path(task_name: str) -> Path:
    return STATE_DIR / f"{task_name}.json"


def rel_state_path(task_name: str) -> Path:
    return state_path(task_name).relative_to(PROJECT_ROOT)


def load_state(task_name: str) -> dict:
    path = state_path(task_name)
    if not path.exists():
        raise SystemExit(f"No state file for task: {task_name}")
    with path.open() as handle:
        state = json.load(handle)
    if state.get("status") not in VALID_STATUSES:
        raise SystemExit(f"Invalid state status in {path}: {state.get('status')}")
    return state


def maybe_load_state(task_name: str) -> dict | None:
    path = state_path(task_name)
    return load_state(task_name) if path.exists() else None


def write_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    path = state_path(state["task_name"])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def update_status(state: dict, status: str, **updates: object) -> None:
    state.update(updates)
    state["status"] = status
    write_state(state)


def record_startup_timing(state: dict, key: str, started_at: float) -> None:
    timings = dict(state.get("startup_timings_seconds") or {})
    timings[key] = round(time.monotonic() - started_at, 3)
    state["startup_timings_seconds"] = timings
    write_state(state)


def require_commands(commands: list[str]) -> None:
    missing: list[str] = []
    for command in commands:
        result = subprocess.run(["/bin/sh", "-c", "command -v \"$1\"", "sh", command], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            missing.append(command)
    if missing:
        raise SystemExit("Missing required local command(s): " + ", ".join(missing))


def run_command(command: list[str], *, check: bool = True, input_text: str | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    remaining = lifecycle_remaining_seconds()
    timeout = DEFAULT_COMMAND_TIMEOUT_SECONDS if remaining is None else remaining
    if timeout <= 0:
        raise LifecycleDeadlineExceeded(f"Lifecycle deadline expired before: {shlex.join(command)}")
    try:
        result = subprocess.run(command, input=input_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd) if cwd else None, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise LifecycleDeadlineExceeded(f"Command exceeded {timeout:.0f}s deadline: {shlex.join(command)}") from exc
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit {result.returncode}"
        raise RuntimeError(f"Command failed: {shlex.join(command)}\n{detail}")
    return result


def parse_json_text(text: str, description: str) -> object:
    payload = text.strip()
    if not payload:
        raise RuntimeError(f"{description} returned empty output")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} returned non-JSON output: {payload[:500]}") from exc


def parse_json_output(result: subprocess.CompletedProcess[str], description: str) -> object:
    return parse_json_text(result.stdout, description)




def select_offer_id_at(query: str, index: int) -> int:
    result = run_command(["vastai", "search", "offers", query, "-o", "dlperf_usd-", "--raw"])
    offers = parse_json_output(result, "vastai search offers")
    if not isinstance(offers, list) or index >= len(offers):
        raise SystemExit("No Vast offers matched query")
    offer_id = offers[index].get("id") if isinstance(offers[index], dict) else None
    if offer_id is None:
        raise RuntimeError("vastai search offers did not return an offer id")
    return int(offer_id)


def create_instance(offer_id: int, image: str, disk_gb: int) -> int:
    result = run_command(["vastai", "create", "instance", str(offer_id), "--image", image, "--disk", str(disk_gb), "--ssh", "--direct", "--raw"])
    payload = parse_json_output(result, "vastai create instance")
    if not isinstance(payload, dict) or payload.get("new_contract") is None:
        raise RuntimeError("vastai create instance did not return new_contract")
    return int(payload["new_contract"])


def attach_ssh_key(instance_id: int, public_key_path: str) -> None:
    public_key = Path(public_key_path).expanduser().read_text().strip()
    run_command(["vastai", "attach", "ssh", str(instance_id), public_key])


def reboot_instance(instance_id: int) -> None:
    run_command(["vastai", "reboot", "instance", str(instance_id)])


def ssh_url_for_instance(instance_id: int) -> tuple[str, int] | None:
    result = run_command(["vastai", "ssh-url", str(instance_id)], check=False)
    text = result.stdout.strip()
    if result.returncode != 0 or not text.startswith("ssh://"):
        return None
    without_scheme = text[len("ssh://") :]
    if "@" in without_scheme:
        without_scheme = without_scheme.split("@", 1)[1]
    if ":" not in without_scheme:
        return None
    host, port_text = without_scheme.rsplit(":", 1)
    try:
        return host, int(port_text)
    except ValueError:
        return None


def show_instance(instance_id: int) -> dict:
    result = run_command(["vastai", "show", "instance", str(instance_id), "--raw"])
    payload = parse_json_output(result, "vastai show instance")
    if isinstance(payload, list):
        if not payload:
            return {}
        payload = payload[0]
    if not isinstance(payload, dict):
        raise RuntimeError("vastai show instance returned unexpected JSON")
    return payload


def host_port_from_show(payload: dict) -> tuple[str, int] | None:
    host = payload.get("public_ipaddr") or payload.get("ssh_host")
    ports = payload.get("ports")
    port: object | None = None
    if isinstance(ports, dict):
        ssh_ports = ports.get("22/tcp")
        if isinstance(ssh_ports, list) and ssh_ports:
            first = ssh_ports[0]
            if isinstance(first, dict):
                port = first.get("HostPort")
    port = port or payload.get("ssh_port")
    if host and port:
        return str(host), int(port)
    return None


def instance_is_running(payload: dict) -> bool:
    values = {str(payload.get("actual_status", "")).lower(), str(payload.get("cur_state", "")).lower()}
    return bool(values & {"running", "loaded", "started"})


def instance_is_terminal(payload: dict) -> bool:
    intended = str(payload.get("intended_status", "")).lower()
    current = {
        str(payload.get("actual_status", "")).lower(),
        str(payload.get("cur_state", "")).lower(),
    }
    return intended in {"stopped", "destroyed"} and not (
        current & {"running", "loaded", "started"}
    )


def wait_for_instance_endpoint(instance_id: int, timeout_seconds: int) -> tuple[str, int]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            payload = show_instance(instance_id)
            endpoint = ssh_url_for_instance(instance_id) or host_port_from_show(payload)
        except LifecycleDeadlineExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve retries around Vast API flakiness.
            last_error = str(exc)
        else:
            status = {
                "actual_status": payload.get("actual_status"),
                "cur_state": payload.get("cur_state"),
                "intended_status": payload.get("intended_status"),
            }
            if instance_is_terminal(payload):
                raise RuntimeError(
                    f"Vast instance stopped before SSH became ready: {json.dumps(status)}"
                )
            if instance_is_running(payload) and endpoint is not None:
                return endpoint
            last_error = json.dumps(status)
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Vast SSH endpoint: {last_error}")


HOST_KEY_FAILURE_MARKERS = ("Host key verification failed", "REMOTE HOST IDENTIFICATION HAS CHANGED")


class HostKeyUntrustedError(RuntimeError):
    """The SSH host key is unknown to, or conflicts with, trusted known_hosts."""


def ssh_security_options(target: SshTarget) -> list[str]:
    """OpenSSH options shared by every ssh/scp invocation.

    Host authentication is strict: only hosts whose keys are already present in
    the operator's trusted OpenSSH known_hosts files are accepted, and unknown
    or changed keys fail closed. Keys are never accepted on first use, so no
    command (including credential transfer) reaches an unauthenticated host.
    Connect and keepalive timeouts stop a dead peer from hanging a phase.
    """
    return [
        "-i",
        target.identity_file,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
    ]


def ssh_base(target: SshTarget) -> list[str]:
    return ["ssh", *ssh_security_options(target), "-p", str(target.port), f"{target.user}@{target.host}"]


def scp_base(target: SshTarget) -> list[str]:
    return ["scp", *ssh_security_options(target), "-P", str(target.port)]


def ssh(target: SshTarget, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command([*ssh_base(target), remote_command], check=check)


def wait_for_ssh(target: SshTarget, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        result = ssh(target, "echo vast-ssh-ok", check=False)
        if result.returncode == 0 and "vast-ssh-ok" in result.stdout:
            return
        last_error = (result.stderr or result.stdout).strip()
        if any(marker in last_error for marker in HOST_KEY_FAILURE_MARKERS):
            raise HostKeyUntrustedError(
                f"SSH host key for [{target.host}]:{target.port} is not trusted; add the verified host key "
                f"to your OpenSSH known_hosts before retrying: {last_error}"
            )
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for SSH: {last_error}")


def target_from_state(state: dict) -> SshTarget:
    return SshTarget(host=str(state["ssh_host"]), port=int(state["ssh_port"]), user=str(state.get("ssh_user", "root")), identity_file=str(state["identity_file"]))


def remote_quote(value: str) -> str:
    return shlex.quote(value)


def copy_secrets(target: SshTarget, local_path: str, remote_path: str) -> None:
    local = Path(local_path).expanduser()
    if not local.exists():
        raise RuntimeError(f"Local secrets env file not found: {local_path}")
    remote_parent = str(Path(remote_path).parent)
    # This strictly host-authenticated command must succeed before any secret
    # bytes are sent; a failed host key check raises here and nothing is copied.
    ssh(target, f"mkdir -p {remote_quote(remote_parent)} /workspace")
    run_command([*scp_base(target), str(local), f"{target.user}@{target.host}:{remote_path}"])


def normalized_tar_parts(name: str) -> list[str]:
    rel = name[2:] if name.startswith("./") else name
    return [part for part in rel.split("/") if part and part != "."]


def should_exclude(tar_info: tarfile.TarInfo) -> bool:
    name = tar_info.name
    if name in {".", ""}:
        return False
    parts = normalized_tar_parts(name)
    if any(part == ".git" or part in EXCLUDED_NAMES for part in parts):
        return True
    rel_candidates = ["/".join(parts)]
    if len(parts) > 1:
        rel_candidates.append("/".join(parts[1:]))
    for rel in rel_candidates:
        for excluded in EXCLUDED_ROOTS:
            if rel == excluded or rel.startswith(excluded + "/"):
                return True
    return False


def exclusion_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if should_exclude(tar_info) else tar_info


def temp_tarball_path(prefix: str) -> Path:
    temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".tar.gz", delete=False)
    temp.close()
    return Path(temp.name)


def overlay_source_paths(source: Path) -> list[str]:
    """List tracked and untracked-but-not-ignored files under ``source``.

    Git ignore rules are the credential boundary: ignored files (local env
    files, keys, caches) are never archived even when present on disk.
    """
    result = run_command(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=source)
    return sorted({path for path in result.stdout.split("\0") if path})


def make_overlay_tarball(source: Path) -> Path:
    paths = overlay_source_paths(source)
    temp_path = temp_tarball_path("scenic-drive-overlay-")
    with tarfile.open(temp_path, "w:gz") as archive:
        for rel in paths:
            path = source / rel
            if not path.exists() and not path.is_symlink():
                continue  # tracked file deleted from the working tree
            archive.add(path, arcname=rel, recursive=False, filter=exclusion_filter)
    return temp_path


def make_repo_tarball(branch: str) -> Path:
    with tempfile.TemporaryDirectory(prefix="scenic-drive-clone-") as temp_dir:
        temp_root = Path(temp_dir)
        clone_dir = temp_root / "scenic-drive"
        run_command(["git", "clone", "--no-hardlinks", "--branch", branch, str(PROJECT_ROOT), str(clone_dir)])
        overlay = make_overlay_tarball(PROJECT_ROOT)
        try:
            with tarfile.open(overlay, "r:gz") as archive:
                archive.extractall(clone_dir)
        finally:
            overlay.unlink(missing_ok=True)
        temp_path = temp_tarball_path("scenic-drive-src-")
        with tarfile.open(temp_path, "w:gz") as archive:
            archive.add(clone_dir, arcname="scenic-drive", filter=exclusion_filter)
        return temp_path


def upload_repo(target: SshTarget, remote_repo_dir: str, branch: str) -> None:
    tarball = make_repo_tarball(branch)
    try:
        run_command([*scp_base(target), str(tarball), f"{target.user}@{target.host}:/workspace/scenic-drive-src.tar.gz"])
    finally:
        tarball.unlink(missing_ok=True)
    remote_parent = str(Path(remote_repo_dir).parent)
    remote_name = Path(remote_repo_dir).name
    ssh(target, f"rm -rf {remote_quote(remote_repo_dir)} && mkdir -p {remote_quote(remote_parent)} && tar -xzf /workspace/scenic-drive-src.tar.gz -C {remote_quote(remote_parent)} && test -d {remote_quote(remote_repo_dir)} || mv {remote_quote(remote_parent + '/scenic-drive')} {remote_quote(remote_repo_dir)}")
    if remote_name != "scenic-drive":
        ssh(target, f"test -d {remote_quote(remote_repo_dir)}")


def bootstrap_remote(target: SshTarget, remote_repo_dir: str, remote_env_file: str) -> None:
    dataset_path = (
        f"{remote_repo_dir.rstrip('/')}/data/processed/regression/"
        f"{Path(DEFAULT_VALIDATION_DATASET_KEY).name}"
    )
    checkpoint_path = (
        f"{remote_repo_dir.rstrip('/')}/models/"
        f"{Path(DEFAULT_VALIDATION_CHECKPOINT_KEY).name}"
    )
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl ca-certificates python3 python3-venv python3-pip build-essential tmux fuse libfuse2
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
cd {remote_quote(remote_repo_dir)}
set -a
. {remote_quote(remote_env_file)}
set +a
: "${{SCENIC_S3_BUCKET:?SCENIC_S3_BUCKET is required in the remote env file}}"
export SCENIC_VALIDATION_DATASET_KEY={remote_quote(DEFAULT_VALIDATION_DATASET_KEY)}
export SCENIC_VALIDATION_CHECKPOINT_KEY={remote_quote(DEFAULT_VALIDATION_CHECKPOINT_KEY)}
export SCENIC_DATASET_PATH={remote_quote(dataset_path)}
export SCENIC_CHECKPOINT_PATH={remote_quote(checkpoint_path)}
mkdir -p "$(dirname "$SCENIC_DATASET_PATH")" "$(dirname "$SCENIC_CHECKPOINT_PATH")"
uv sync --python 3.11
uv run python -m scenicdriver.data_pipeline.s3 download-file \
  --bucket "$SCENIC_S3_BUCKET" \
  --key "$SCENIC_VALIDATION_DATASET_KEY" \
  --dest "$SCENIC_DATASET_PATH" \
  --required
uv run python -m scenicdriver.data_pipeline.s3 download-file \
  --bucket "$SCENIC_S3_BUCKET" \
  --key "$SCENIC_VALIDATION_CHECKPOINT_KEY" \
  --dest "$SCENIC_CHECKPOINT_PATH" \
  --required
uv run python scripts/modeling/train_regression_baseline.py --help >/tmp/scenic_train_help.txt
""".strip()
    ssh(target, "bash -lc " + shlex.quote(script))


def run_remote_container_validation(target: SshTarget, remote_repo_dir: str, remote_env_file: str) -> None:
    dataset_path = (
        f"{remote_repo_dir.rstrip('/')}/data/processed/regression/"
        f"{Path(DEFAULT_VALIDATION_DATASET_KEY).name}"
    )
    checkpoint_path = (
        f"{remote_repo_dir.rstrip('/')}/models/"
        f"{Path(DEFAULT_VALIDATION_CHECKPOINT_KEY).name}"
    )
    script = f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd {remote_quote(remote_repo_dir)}
set -a
. {remote_quote(remote_env_file)}
set +a
export SCENIC_DATASET_PATH={remote_quote(dataset_path)}
export SCENIC_CHECKPOINT_PATH={remote_quote(checkpoint_path)}
uv run python scripts/remote/vast_train.py validate --device cuda \
  --checkpoint "$SCENIC_CHECKPOINT_PATH" \
  --dataset "$SCENIC_DATASET_PATH" \
  --output /tmp/scenic_container_validation.json
""".strip()
    ssh(target, "bash -lc " + shlex.quote(script))


def verify_remote_validation(target: SshTarget) -> None:
    result = ssh(target, "cat /tmp/scenic_container_validation.json")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("device") != "cuda":
        raise RuntimeError("Remote CUDA validation failed: " + result.stdout.strip())





def build_initial_state(args: argparse.Namespace, offer_id: int, instance_id: int) -> dict:
    now = utc_now()
    return {
        "task_name": args.task_name,
        "instance_id": instance_id,
        "offer_id": offer_id,
        "image": args.image,
        "disk_gb": args.disk_gb,
        "ssh_host": "",
        "ssh_port": 0,
        "ssh_user": "root",
        "identity_file": str(Path(args.identity_file).expanduser()),
        "ssh_public_key": str(Path(args.ssh_public_key).expanduser()),
        "remote_repo_dir": args.remote_repo_dir,
        "remote_env_file": DEFAULT_REMOTE_ENV_FILE,
        "local_secrets_env_file": args.local_secrets_env_file,
        "repo_source": "git-checkout-upload",
        "branch": args.branch,
        "cmux_workspace_id": None,
        "cmux_workspace_ref": None,
        "cmux_workspace_registered": False,
        "cmux_workspace_observed": False,
        "cmux_workspace_path": None,
        "cmux_workspace_name": None,
        "cmux_agent": None,
        "workspace_cwd": str(getattr(args, "workspace_cwd", DEFAULT_CMUX_WORKSPACE_CWD)),
        "completed_phases": [],
        "created_at": now,
        "updated_at": now,
        "status": "creating",
    }


def mark_phase_completed(state: dict, phase: str) -> None:
    phases = state.get("completed_phases")
    if not isinstance(phases, list):
        return  # unknown history already requires every phase's artifacts
    if phase not in phases:
        state["completed_phases"] = [*phases, phase]
        write_state(state)


def completed_phases(state: dict) -> frozenset[str]:
    phases = state.get("completed_phases")
    if not isinstance(phases, list):
        return frozenset(ALL_PHASES)
    return frozenset(phase for phase in phases if isinstance(phase, str))


def lifecycle_seconds_left() -> float:
    remaining = lifecycle_remaining_seconds()
    if remaining is None:
        raise RuntimeError("lifecycle_seconds_left() requires an active lifecycle_deadline()")
    return max(0.0, remaining)


def check_up_preconditions(args: argparse.Namespace) -> None:
    validate_task_name(args.task_name)
    existing = maybe_load_state(args.task_name)
    if existing is not None and existing.get("status") != "destroyed":
        raise SystemExit(f"State exists: {rel_state_path(args.task_name)}; run `python scripts/remote/cmux_vast_host.py down` or choose a new task name")
    identity_file = str(Path(args.identity_file).expanduser())
    public_key = str(Path(args.ssh_public_key).expanduser())
    require_commands(["vastai", "ssh", "scp", "tar", "git"])
    if not Path(identity_file).exists():
        raise SystemExit(f"identity file not found: {identity_file}")
    if not Path(public_key).exists():
        raise SystemExit(f"ssh public key not found: {public_key}")


def do_up(args: argparse.Namespace) -> dict:
    check_up_preconditions(args)
    identity_file = str(Path(args.identity_file).expanduser())
    public_key = str(Path(args.ssh_public_key).expanduser())
    state: dict | None = None
    target: SshTarget | None = None
    max_attempts = max(1, int(args.allocation_attempts))
    for attempt in range(max_attempts):
        allocation_started = time.monotonic()
        instance_id: int | None = None
        state = None
        target = None
        try:
            # One deadline bounds the whole allocation attempt, including every
            # Vast/SSH subprocess, rather than granting each wait a fresh budget.
            with lifecycle_deadline(args.timeout_seconds):
                offer_id = int(args.offer_id) if args.offer_id is not None else select_offer_id_at(args.offer_query, attempt)
                instance_id = create_instance(offer_id, args.image, args.disk_gb)
                state = build_initial_state(args, offer_id, instance_id)
                state["allocation_attempt"] = attempt + 1
                write_state(state)
                attach_ssh_key(instance_id, public_key)
                update_status(state, "ssh_wait")
                host, port = wait_for_instance_endpoint(instance_id, lifecycle_seconds_left())
                state.update({"ssh_host": host, "ssh_port": port})
                write_state(state)
                reboot_instance(instance_id)
                host, port = wait_for_instance_endpoint(instance_id, lifecycle_seconds_left())
                state.update({"ssh_host": host, "ssh_port": port})
                write_state(state)
                target = SshTarget(host=host, port=port, user="root", identity_file=identity_file)
                wait_for_ssh(target, lifecycle_seconds_left())
            record_startup_timing(state, "vast_allocation", allocation_started)
            break
        except Exception as exc:  # noqa: BLE001 - bad Vast hosts are common; rerent before bootstrapping.
            if state is not None:
                state["error"] = str(exc)
                write_state(state)
            if instance_id is not None:
                try:
                    destroy_instance_confirmed(instance_id)
                except Exception as destroy_exc:  # noqa: BLE001 - never rent a replacement for an unconfirmed destroy.
                    error = f"{exc}; destroy of instance {instance_id} unconfirmed: {destroy_exc}"
                    if state is not None:
                        update_status(state, "failed_kept", error=error)
                    raise RuntimeError(f"Not allocating a replacement host: {error}") from exc
                if state is not None:
                    update_status(state, "destroyed")
            elif isinstance(exc, LifecycleDeadlineExceeded):
                # A timed-out create may still have rented an unknown instance.
                raise
            if isinstance(exc, HostKeyUntrustedError) or attempt + 1 >= max_attempts:
                raise
            if instance_id is None:
                print(f"Vast allocation attempt {attempt + 1} failed before instance creation; retrying", file=sys.stderr)
            else:
                print(f"Vast allocation attempt {attempt + 1} failed; confirmed destroy of instance {instance_id}; retrying", file=sys.stderr)
    if state is None or target is None:
        raise RuntimeError("Vast allocation did not produce an SSH-ready host")
    try:

        if getattr(args, "bootstrap", "full") == "none":
            update_status(state, "ready", ssh_host=target.host, ssh_port=target.port)
            return state

        update_status(state, "bootstrapping")
        with lifecycle_deadline(args.bootstrap_timeout_seconds):
            copy_secrets(target, args.local_secrets_env_file, DEFAULT_REMOTE_ENV_FILE)

            phase_started = time.monotonic()
            try:
                upload_repo(target, args.remote_repo_dir, args.branch)
            finally:
                record_startup_timing(state, "repo_upload", phase_started)

            phase_started = time.monotonic()
            try:
                bootstrap_remote(target, args.remote_repo_dir, DEFAULT_REMOTE_ENV_FILE)
            finally:
                record_startup_timing(state, "uv_sync_bootstrap", phase_started)
            mark_phase_completed(state, PHASE_BOOTSTRAP)

            phase_started = time.monotonic()
            try:
                run_remote_container_validation(target, args.remote_repo_dir, DEFAULT_REMOTE_ENV_FILE)
                verify_remote_validation(target)
            finally:
                record_startup_timing(state, "container_validation", phase_started)
            mark_phase_completed(state, PHASE_CONTAINER_VALIDATION)

        update_status(state, "ready", ssh_host=target.host, ssh_port=target.port)
        return state
    except Exception as exc:  # noqa: BLE001 - preserve instance for explicit teardown.
        update_status(state, "failed", error=str(exc))
        raise


def handle_up(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
    try:
        with task_lock(args.task_name):
            state = do_up(args)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Vast CMUX host ready: {args.task_name}")
    print(f"State: {rel_state_path(args.task_name)}")
    print(f"Remote repo: {state['remote_repo_dir']}")
    return 0




def add_up_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offer-query", default=DEFAULT_OFFER_QUERY)
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--allocation-attempts", type=int, default=3)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk-gb", type=int, default=64)
    parser.add_argument("--identity-file", default="~/.ssh/id_ed25519")
    parser.add_argument("--ssh-public-key", default="~/.ssh/id_ed25519.pub")
    parser.add_argument("--local-secrets-env-file", default=".secrets/aws.env")
    parser.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--workspace-cwd", default=DEFAULT_CMUX_WORKSPACE_CWD)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--bootstrap", default="full", choices=("full", "none"), help="full: install dependencies and smoke test; none: stop after SSH ready.")
    parser.add_argument("--bootstrap-timeout-seconds", type=int, default=DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS, help="Deadline for secrets copy, repo upload, dependency bootstrap, and container validation.")


def build_cmux_workspace_command(name: str, cwd: str) -> list[str]:
    """Build the documented CMUX opening command without executing it."""
    return [
        "cmux",
        "new-workspace",
        "--name",
        name,
        "--cwd",
        cwd,
        "--focus",
        "false",
    ]


def _cmux_scalar_identity(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    identity = str(value).strip()
    return identity or None


def build_cmux_workspace_registration_command(name: str, cwd: str) -> list[str]:
    """Build the CMUX JSON-RPC workspace creation command."""
    params = {"cwd": cwd, "focus": False, "name": name}
    return [
        "cmux",
        "rpc",
        "workspace.create",
        json.dumps(params, separators=(",", ":"), sort_keys=True),
    ]


def parse_cmux_workspace_creation(payload: object) -> tuple[str, str]:
    """Extract the CMUX ref and ID without using cwd or other metadata."""
    if not isinstance(payload, dict):
        raise RuntimeError("cmux workspace create returned a non-object JSON value")

    # `cmux rpc` currently returns the workspace object directly.  Accept the
    # envelope and nested workspace forms used by older/newer socket clients,
    # while checking them before a generic `id` key (which may be a request ID).
    for key in ("result", "workspace"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            try:
                return parse_cmux_workspace_creation(nested)
            except RuntimeError:
                pass

    ref = next(
        (
            identity
            for key in ("workspace_ref", "ref")
            if (identity := _cmux_scalar_identity(payload.get(key))) is not None
        ),
        None,
    )
    workspace_id = next(
        (
            identity
            for key in ("workspace_id", "workspaceId", "id")
            if (identity := _cmux_scalar_identity(payload.get(key))) is not None
        ),
        None,
    )
    if ref is None and workspace_id is None:
        raise RuntimeError("cmux workspace create JSON did not include a ref or ID")
    # A CLI version may return only one identifier.  Keep both state keys
    # populated with the exact returned token in that case.
    return ref or workspace_id or "", workspace_id or ref or ""


def register_cmux_workspace(name: str, cwd: str) -> tuple[str, str]:
    command = build_cmux_workspace_registration_command(name, cwd)
    result = run_command(command)
    payload = parse_json_output(result, "cmux workspace create")
    return parse_cmux_workspace_creation(payload)


def handle_start_task(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
    try:
        with task_lock(args.task_name):
            return start_task_locked(args)
    except TaskLockedError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def start_task_locked(args: argparse.Namespace) -> int:
    state = maybe_load_state(args.task_name)
    if state is None or state.get("status") == "destroyed":
        try:
            state = do_up(args)
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 1
    if state.get("status") == "ready" and getattr(args, "bootstrap", "full") == "none":
        print(f"Vast host ready (bootstrap none): {args.task_name}")
        print(f"SSH: {' '.join(ssh_base(target_from_state(state)))}")
        print(f"State: {rel_state_path(args.task_name)}")
        return 0
    if state.get("status") == "workspace_pending":
        name = str(state.get("cmux_workspace_name") or getattr(args, "workspace_name", None) or args.task_name)
        cwd = str(state.get("workspace_cwd") or getattr(args, "workspace_cwd", None) or DEFAULT_CMUX_WORKSPACE_CWD)
    elif state.get("status") == "ready":
        name = str(getattr(args, "workspace_name", None) or args.task_name)
        cwd = str(state.get("workspace_cwd") or getattr(args, "workspace_cwd", None) or DEFAULT_CMUX_WORKSPACE_CWD)
        state["cmux_workspace_name"] = name
        state["cmux_workspace_path"] = cwd
        state["cmux_agent"] = getattr(args, "agent", "none")
        state["workspace_cwd"] = cwd
        state["cmux_workspace_ref"] = None
        state["cmux_workspace_id"] = None
        state["cmux_workspace_registered"] = False
        state["cmux_workspace_observed"] = False
        state.pop("cmux_workspace_observed_ref", None)
        update_status(state, "workspace_pending")
    else:
        raise SystemExit(f"State {rel_state_path(args.task_name)} has status {state.get('status')}")

    registered = (
        state.get("cmux_workspace_registered") is True
        and (
            _cmux_scalar_identity(state.get("cmux_workspace_ref")) is not None
            or _cmux_scalar_identity(state.get("cmux_workspace_id")) is not None
        )
    )
    if registered:
        print(
            "CMUX workspace already registered: "
            f"ref={state.get('cmux_workspace_ref')} id={state.get('cmux_workspace_id')}"
        )
        print("Watch state with: python scripts/remote/cmux_vast_host.py watch --once")
        return 0

    if getattr(args, "manual", False):
        update_status(
            state,
            "workspace_pending",
            cmux_workspace_ref=None,
            cmux_workspace_id=None,
            cmux_workspace_registered=False,
            cmux_workspace_observed=False,
        )
        state.pop("cmux_workspace_observed_ref", None)
        command = build_cmux_workspace_command(name, cwd)
        print(f"CMUX workspace command (manual, unregistered): {shlex.join(command)}")
        print("CMUX workspace creation was left to the operator; no identity was recorded.")
        print("Watch state with: python scripts/remote/cmux_vast_host.py watch --once")
        return 0

    try:
        # The handoff phase makes scenic_artifacts/ mandatory at teardown, so
        # create it before registering and mark the phase only on success.
        prepare_workspace_artifacts(state)
        workspace_ref, workspace_id = register_cmux_workspace(name, cwd)
    except Exception as exc:  # noqa: BLE001 - registration must remain retryable.
        update_status(
            state,
            "workspace_pending",
            cmux_workspace_ref=None,
            cmux_workspace_id=None,
            cmux_workspace_registered=False,
            cmux_workspace_observed=False,
            error=str(exc),
        )
        state.pop("cmux_workspace_observed_ref", None)
        print(f"CMUX workspace registration failed; task remains pending: {exc}", file=sys.stderr)
        return 1

    mark_phase_completed(state, PHASE_WORKSPACE_HANDOFF)
    update_status(
        state,
        "workspace_pending",
        cmux_workspace_ref=workspace_ref,
        cmux_workspace_id=workspace_id,
        cmux_workspace_registered=True,
        cmux_workspace_observed=False,
    )
    state.pop("cmux_workspace_observed_ref", None)
    print(f"CMUX workspace registered: ref={workspace_ref} id={workspace_id}")
    print("Watch state with: python scripts/remote/cmux_vast_host.py watch --once")
    return 0


def prepare_workspace_artifacts(state: dict) -> None:
    remote_root = str(state["remote_repo_dir"]).strip().rstrip("/")
    ssh(target_from_state(state), f"mkdir -p {remote_quote(remote_root + '/scenic_artifacts')}")


def remote_file_exists(target: SshTarget, remote_path: str) -> bool:
    result = ssh(target, f"test -e {shlex.quote(remote_path)}", check=False)
    return result.returncode == 0


def copy_remote_path(target: SshTarget, remote_path: str, local_path: Path, *, required: bool) -> bool:
    if not remote_file_exists(target, remote_path):
        message = f"remote path missing: {remote_path}"
        print(("Warning: " if not required else "Warning (required): ") + message, file=sys.stderr)
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    destination = str(local_path)
    if remote_path.endswith("/"):
        local_path.mkdir(parents=True, exist_ok=True)
        destination = str(local_path)
    result = run_command([*scp_base(target), "-r", f"{target.user}@{target.host}:{remote_path}", destination], check=False)
    if result.returncode != 0:
        message = f"failed to copy remote path: {remote_path}"
        print(("Warning: " if not required else "Warning (required): ") + message, file=sys.stderr)
        return False
    return True


def copy_artifacts(state: dict) -> bool:
    """Copy remote artifacts; return False if any phase-required artifact is missing.

    An artifact is required only once the phase that produces it has completed,
    so hosts that failed or stopped early can still be torn down.
    """
    phases = completed_phases(state)
    bootstrap_done = PHASE_BOOTSTRAP in phases
    validation_done = PHASE_CONTAINER_VALIDATION in phases
    handoff_done = PHASE_WORKSPACE_HANDOFF in phases
    anything_required = bootstrap_done or validation_done or handoff_done
    remote_root = state.get("remote_repo_dir")
    if (
        not isinstance(remote_root, str)
        or not remote_root.strip()
        or not remote_root.strip().startswith("/")
        or not remote_root.strip().strip("/")
    ):
        print("Warning: state has no valid remote_repo_dir; cannot copy artifacts", file=sys.stderr)
        return not anything_required
    if not state.get("ssh_host") or not state.get("ssh_port"):
        print("Warning: state has no SSH endpoint; cannot copy artifacts", file=sys.stderr)
        return not anything_required
    try:
        target = target_from_state(state)
        task_name = str(state["task_name"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Warning: invalid state for artifact copy: {exc}", file=sys.stderr)
        return not anything_required

    remote_root = remote_root.strip().rstrip("/")
    artifact_dir = ARTIFACTS_DIR / task_name
    required_ok = True
    copy_requests = (
        (f"{remote_root}/data/processed/regression/", artifact_dir / "data" / "processed" / "regression", bootstrap_done),
        (f"{remote_root}/models/", artifact_dir / "models", False),
        (f"{remote_root}/scenic_artifacts/", artifact_dir / "scenic_artifacts", handoff_done),
        ("/tmp/scenic_container_validation.json", artifact_dir / "scenic_container_validation.json", validation_done),
    )
    for remote_path, local_path, required in copy_requests:
        try:
            copied = copy_remote_path(target, remote_path, local_path, required=required)
        except Exception as exc:  # noqa: BLE001 - an artifact failure must block teardown.
            print(f"Warning: failed to copy {remote_path}: {exc}", file=sys.stderr)
            copied = False
        if required and not copied:
            required_ok = False
    return required_ok


def workspace_closed(state: dict) -> bool:
    if state.get("cmux_workspace_registered") is False:
        return False
    recorded_values = {
        identity
        for identity in (
            _cmux_scalar_identity(state.get("cmux_workspace_ref")),
            _cmux_scalar_identity(state.get("cmux_workspace_id")),
        )
        if identity is not None
    }
    if not recorded_values:
        return False

    try:
        result = run_command(["cmux", "workspace", "list", "--json"], check=False)
    except Exception:
        return False
    if result.returncode != 0:
        return False
    try:
        payload = parse_json_output(result, "cmux workspace list")
    except Exception:
        return False

    entries: object = payload
    if isinstance(payload, dict):
        entries = payload.get("workspaces", payload.get("result"))
    if not isinstance(entries, list):
        return False
    if not entries:
        # An observed workspace may be the last workspace in the window.  An
        # empty, successful JSON list is therefore a valid absence signal,
        # while an unobserved empty list remains safely pending.
        return (
            state.get("cmux_workspace_observed") is True
            and _cmux_scalar_identity(state.get("cmux_workspace_observed_ref")) in recorded_values
        )
    if any(not isinstance(entry, dict) for entry in entries):
        return False

    identity_present = False
    observed_identity: str | None = None
    for entry in entries:
        entry_ref = _cmux_scalar_identity(entry.get("ref"))
        if "ref" in entry and entry.get("ref") is not None and entry_ref is None:
            return False
        entry_identities = [
            identity
            for identity in (
                entry_ref,
                *(
                    _cmux_scalar_identity(entry.get(key))
                    for key in ("id", "workspace_id", "workspaceId")
                ),
            )
            if identity is not None
        ]
        if not entry_identities:
            return False
        matched_identity = next(
            (identity for identity in entry_identities if identity in recorded_values),
            None,
        )
        if matched_identity is not None:
            identity_present = True
            observed_identity = matched_identity

    if identity_present:
        state["cmux_workspace_registered"] = True
        state["cmux_workspace_observed"] = True
        state["cmux_workspace_observed_ref"] = observed_identity
        try:
            if state.get("task_name"):
                write_state(state)
        except Exception:
            state.pop("cmux_workspace_observed_ref", None)
            state["cmux_workspace_observed"] = False
            return False
        return False
    return (
        state.get("cmux_workspace_observed") is True
        and _cmux_scalar_identity(state.get("cmux_workspace_observed_ref"))
        in recorded_values
    )

def listed_instance_ids() -> set[int]:
    result = run_command(["vastai", "show", "instances", "--raw"])
    payload = parse_json_output(result, "vastai show instances")
    if not isinstance(payload, list):
        raise RuntimeError("vastai show instances returned unexpected JSON")
    ids: set[int] = set()
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("id") is None:
            raise RuntimeError("vastai show instances returned an entry without an id")
        ids.add(int(entry["id"]))
    return ids


def destroy_instance_confirmed(instance_id: int) -> None:
    """Destroy an instance and return only after the account listing omits it.

    A successful destroy command is not proof of release; callers may rent a
    replacement or record ``destroyed`` only after this returns.
    """
    with lifecycle_deadline(DESTROY_CONFIRM_TIMEOUT_SECONDS):
        run_command(["vastai", "destroy", "instance", str(instance_id), "--yes"])
        while True:
            try:
                if instance_id not in listed_instance_ids():
                    return
                last_error = "instance still listed"
            except LifecycleDeadlineExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed listing is not confirmation.
                last_error = str(exc)
            remaining = lifecycle_seconds_left()
            if remaining <= DESTROY_CONFIRM_POLL_SECONDS:
                raise RuntimeError(f"Destroy of Vast instance {instance_id} not confirmed: {last_error}")
            time.sleep(DESTROY_CONFIRM_POLL_SECONDS)


def destroy_instance(state: dict) -> None:
    try:
        destroy_instance_confirmed(int(state["instance_id"]))
    except Exception as exc:  # noqa: BLE001 - preserve a terminal cleanup state.
        error = f"destroy failed: {exc}"
        update_status(state, "failed_kept", error=error)
        raise RuntimeError(error) from exc
    update_status(state, "destroyed")


def process_watch_state(state: dict, *, destroy: bool, yes: bool) -> None:
    status = state.get("status")
    task_name = state["task_name"]
    if status in {"destroyed", "done"}:
        return
    if status != "workspace_pending":
        print(f"No CMUX workspace to watch for {task_name}: {status}")
        return
    if not workspace_closed(state):
        print(f"CMUX workspace still open or unconfirmed: {task_name}")
        return
    try:
        artifacts_ok = copy_artifacts(state)
    except Exception as exc:  # noqa: BLE001 - failed copies must leave the host retryable.
        print(f"Warning: artifact copy failed for {task_name}: {exc}", file=sys.stderr)
        artifacts_ok = False
    if not artifacts_ok:
        print(f"Required artifact copy incomplete; keeping Vast task: {task_name}", file=sys.stderr)
        return
    if destroy and yes:
        destroy_instance(state)
        print(f"Vast task destroyed: {task_name}")
    else:
        update_status(state, "done")
        print(f"Task is closed; destroy with python scripts/remote/cmux_vast_host.py down {task_name} --destroy --yes")


def task_names_to_watch(task_name: str | None) -> list[str]:
    if task_name:
        validate_task_name(task_name)
        return [task_name]
    if not STATE_DIR.exists():
        return []
    return [path.stem for path in sorted(STATE_DIR.glob("*.json"))]


def load_watchable_state(task_name: str, *, explicit: bool) -> dict | None:
    if explicit:
        return load_state(task_name)
    path = state_path(task_name)
    if not path.exists():
        return None
    with path.open() as handle:
        state = json.load(handle)
    return state if state.get("status") in VALID_STATUSES else None


def handle_watch(args: argparse.Namespace) -> int:
    while True:
        for task_name in task_names_to_watch(args.task_name):
            try:
                # State is (re)loaded under the claim so a concurrent
                # start-task/down cannot race this teardown decision.
                with task_lock(task_name):
                    state = load_watchable_state(task_name, explicit=bool(args.task_name))
                    if state is None:
                        continue
                    with lifecycle_deadline(TEARDOWN_TIMEOUT_SECONDS):
                        process_watch_state(state, destroy=args.destroy, yes=args.yes)
            except TaskLockedError as exc:
                print(f"Skipping watch: {exc}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"Watch failed for {task_name}: {exc}", file=sys.stderr)
                if args.once:
                    return 1
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


def handle_down(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
    try:
        with task_lock(args.task_name), lifecycle_deadline(TEARDOWN_TIMEOUT_SECONDS):
            return down_locked(args)
    except TaskLockedError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def down_locked(args: argparse.Namespace) -> int:
    state = load_state(args.task_name)
    destroying = args.destroy and args.yes
    artifacts_ok = True
    # Every destroy copies first so phase-required artifacts gate teardown
    # regardless of status or --copy-artifacts.
    if (
        (destroying and state["status"] != "destroyed")
        or args.copy_artifacts
        or state["status"] in {"workspace_pending", "done", "failed"}
    ):
        try:
            artifacts_ok = copy_artifacts(state)
        except Exception as exc:  # noqa: BLE001 - never destroy after a failed copy.
            print(f"Warning: artifact copy failed for {args.task_name}: {exc}", file=sys.stderr)
            artifacts_ok = False
    if not destroying:
        print(f"Destroy command: vastai destroy instance {state['instance_id']}")
        print("Not destroying without --destroy --yes")
        return 0
    if not artifacts_ok:
        print(f"Required artifact copy incomplete; refusing to destroy {args.task_name}", file=sys.stderr)
        return 1
    try:
        destroy_instance(state)
    except Exception as exc:  # noqa: BLE001 - destroy_instance persists failed_kept.
        print(f"Destroy failed for Vast task {args.task_name}: {exc}", file=sys.stderr)
        return 1
    print(f"Vast task destroyed: {args.task_name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create, watch, and tear down Scenic Drive Vast.ai hosts with CMUX handoff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    up = subparsers.add_parser("up", help="Create a Vast.ai GPU host")
    up.add_argument("task_name")
    add_up_options(up)
    up.set_defaults(func=handle_up)

    start_task = subparsers.add_parser("start-task", help="Create and register a CMUX workspace for a Vast host")
    start_task.add_argument("task_name")
    start_task.add_argument("--agent", default="none")
    start_task.add_argument("--workspace-name")
    start_task.add_argument(
        "--manual",
        dest="manual",
        action="store_true",
        help="Print a manual CMUX command and leave the workspace explicitly unregistered.",
    )
    add_up_options(start_task)
    start_task.set_defaults(func=handle_start_task)

    watch = subparsers.add_parser("watch", help="Watch CMUX workspaces and stop Vast hosts when closed")
    watch.add_argument("--task-name")
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--interval-seconds", type=int, default=60)
    watch.add_argument("--destroy", action="store_true")
    watch.add_argument("--yes", action="store_true")
    watch.set_defaults(func=handle_watch)

    down = subparsers.add_parser("down", help="Copy artifacts and optionally destroy a Vast.ai instance")
    down.add_argument("task_name")
    down.add_argument("--destroy", action="store_true")
    down.add_argument("--copy-artifacts", action="store_true")
    down.add_argument("--yes", action="store_true")
    down.set_defaults(func=handle_down)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
