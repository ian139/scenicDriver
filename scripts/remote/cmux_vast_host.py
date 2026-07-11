"""State-backed Vast.ai GPU host allocator with CMUX workspace handoff."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
LEGACY_STATE_DIR = PROJECT_ROOT / ".orca-vast" / "state"
ARTIFACTS_DIR = PROJECT_ROOT / ".cmux-vast" / "artifacts"
DEFAULT_OFFER_QUERY = "gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1 rentable=true"
DEFAULT_IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
DEFAULT_BRANCH = "Ian139/RemoteTraining"
DEFAULT_REMOTE_REPO_DIR = "/workspace/scenic-drive"
DEFAULT_REMOTE_ENV_FILE = "/root/.scenic/aws.env"
DEFAULT_CMUX_WORKSPACE_CWD = str(PROJECT_ROOT)
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
    # Keep legacy state out of remote overlays during migration.
    ".orca-vast",
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


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_task_name(task_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", task_name or ""):
        raise SystemExit("task-name may contain only letters, numbers, dot, underscore, and hyphen")


def state_path(task_name: str) -> Path:
    return STATE_DIR / f"{task_name}.json"


def legacy_state_path(task_name: str) -> Path:
    return LEGACY_STATE_DIR / f"{task_name}.json"


def rel_state_path(task_name: str) -> Path:
    return state_path(task_name).relative_to(PROJECT_ROOT)


def _normalize_state(state: dict) -> dict:
    """Normalize legacy Orca keys in memory; writes always use CMUX keys."""
    legacy_to_cmux = {
        "orca_worktree_id": "cmux_workspace_id",
        "orca_worktree_path": "cmux_workspace_path",
        "orca_worktree_name": "cmux_workspace_name",
        "orca_agent": "cmux_agent",
    }
    for legacy_key, cmux_key in legacy_to_cmux.items():
        if cmux_key not in state and state.get(legacy_key) is not None:
            state[cmux_key] = state[legacy_key]
    if state.get("status") in {"orca_serving", "worktree_running"}:
        state["status"] = "workspace_pending"
    for key in tuple(state):
        if key.startswith("orca_"):
            state.pop(key, None)
    return state


def _state_source_path(task_name: str) -> Path | None:
    current = state_path(task_name)
    if current.exists():
        return current
    legacy = legacy_state_path(task_name)
    return legacy if legacy.exists() else None


def load_state(task_name: str) -> dict:
    path = _state_source_path(task_name)
    if path is None:
        raise SystemExit(f"No state file for task: {task_name}")
    with path.open() as handle:
        state = _normalize_state(json.load(handle))
    if state.get("status") not in VALID_STATUSES:
        raise SystemExit(f"Invalid state status in {path}: {state.get('status')}")
    return state


def maybe_load_state(task_name: str) -> dict | None:
    return load_state(task_name) if _state_source_path(task_name) is not None else None


def write_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = _normalize_state(state)
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
    result = subprocess.run(command, input=input_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd) if cwd else None)
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




def select_offer_id(query: str) -> int:
    result = run_command(["vastai", "search", "offers", query, "-o", "dlperf_usd-", "--raw"])
    offers = parse_json_output(result, "vastai search offers")
    if not isinstance(offers, list) or not offers:
        raise SystemExit("No Vast offers matched query")
    offer_id = offers[0].get("id") if isinstance(offers[0], dict) else None
    if offer_id is None:
        raise RuntimeError("vastai search offers did not return an offer id")
    return int(offer_id)


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


def wait_for_instance_endpoint(instance_id: int, timeout_seconds: int) -> tuple[str, int]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            payload = show_instance(instance_id)
            endpoint = ssh_url_for_instance(instance_id) or host_port_from_show(payload)
            if instance_is_running(payload) and endpoint is not None:
                return endpoint
            last_error = json.dumps({"actual_status": payload.get("actual_status"), "cur_state": payload.get("cur_state")})
        except Exception as exc:  # noqa: BLE001 - preserve retry behavior around Vast flakiness.
            last_error = str(exc)
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Vast SSH endpoint: {last_error}")


def ssh_base(target: SshTarget) -> list[str]:
    return [
        "ssh",
        "-i",
        target.identity_file,
        "-p",
        str(target.port),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{target.user}@{target.host}",
    ]


def scp_base(target: SshTarget) -> list[str]:
    return [
        "scp",
        "-i",
        target.identity_file,
        "-P",
        str(target.port),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


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


def make_overlay_tarball(source: Path) -> Path:
    temp = tempfile.NamedTemporaryFile(prefix="scenic-drive-overlay-", suffix=".tar.gz", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    with tarfile.open(temp_path, "w:gz") as archive:
        def tar_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            if should_exclude(tar_info):
                return None
            return tar_info

        archive.add(source, arcname=".", filter=tar_filter)
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
        temp = tempfile.NamedTemporaryFile(prefix="scenic-drive-src-", suffix=".tar.gz", delete=False)
        temp_path = Path(temp.name)
        temp.close()
        with tarfile.open(temp_path, "w:gz") as archive:
            def tar_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
                if should_exclude(tar_info):
                    return None
                return tar_info

            archive.add(clone_dir, arcname="scenic-drive", filter=tar_filter)
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
    script = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl ca-certificates python3 python3-venv python3-pip build-essential tmux fuse libfuse2
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
cd {remote_quote(remote_repo_dir)}
set -a; . {remote_quote(remote_env_file)}; set +a
uv sync --python 3.11
uv run python scripts/modeling/train_regression_baseline.py --help >/tmp/scenic_train_help.txt
""".strip()
    ssh(target, "bash -lc " + shlex.quote(script))


def run_remote_container_smoke(target: SshTarget, remote_repo_dir: str, remote_env_file: str) -> None:
    script = f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd {remote_quote(remote_repo_dir)}
set -a; . {remote_quote(remote_env_file)}; set +a
uv run python scripts/remote/container_smoke.py --device cuda --json >/tmp/scenic_container_smoke.json
""".strip()
    ssh(target, "bash -lc " + shlex.quote(script))


def verify_remote_smoke(target: SshTarget) -> None:
    result = ssh(target, "cat /tmp/scenic_container_smoke.json")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("device") != "cuda":
        raise RuntimeError("Remote CUDA smoke failed: " + result.stdout.strip())



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
        "cmux_workspace_path": None,
        "cmux_workspace_name": None,
        "cmux_agent": None,
        "workspace_cwd": str(getattr(args, "workspace_cwd", DEFAULT_CMUX_WORKSPACE_CWD)),
        "created_at": now,
        "updated_at": now,
        "status": "creating",
    }


def check_up_preconditions(args: argparse.Namespace) -> None:
    validate_task_name(args.task_name)
    existing = maybe_load_state(args.task_name)
    if existing is not None and existing.get("status") != "destroyed":
        raise SystemExit(f"State exists: {rel_state_path(args.task_name)}; run vast-down or choose a new task name")
    require_commands(["vastai", "ssh", "scp", "tar", "git", "uv"])
    identity_file = str(Path(args.identity_file).expanduser())
    public_key = str(Path(args.ssh_public_key).expanduser())
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
            offer_id = int(args.offer_id) if args.offer_id is not None else select_offer_id_at(args.offer_query, attempt)
            instance_id = create_instance(offer_id, args.image, args.disk_gb)
            state = build_initial_state(args, offer_id, instance_id)
            state["allocation_attempt"] = attempt + 1
            write_state(state)
            attach_ssh_key(instance_id, public_key)
            update_status(state, "ssh_wait")
            host, port = wait_for_instance_endpoint(instance_id, args.timeout_seconds)
            state.update({"ssh_host": host, "ssh_port": port})
            write_state(state)
            reboot_instance(instance_id)
            host, port = wait_for_instance_endpoint(instance_id, args.timeout_seconds)
            state.update({"ssh_host": host, "ssh_port": port})
            write_state(state)
            target = SshTarget(host=host, port=port, user="root", identity_file=identity_file)
            wait_for_ssh(target, args.timeout_seconds)
            record_startup_timing(state, "vast_allocation", allocation_started)
            break
        except Exception as exc:  # noqa: BLE001 - bad Vast hosts are common; rerent before bootstrapping.
            if state is not None:
                state["error"] = str(exc)
                write_state(state)
            if instance_id is not None:
                run_command(["vastai", "destroy", "instance", str(instance_id), "--yes"], check=False)
                if state is not None:
                    update_status(state, "destroyed")
            if attempt + 1 >= max_attempts:
                raise
            if instance_id is None:
                print(f"Vast allocation attempt {attempt + 1} failed before instance creation; retrying", file=sys.stderr)
            else:
                print(f"Vast allocation attempt {attempt + 1} failed; destroyed instance {instance_id}; retrying", file=sys.stderr)
    if state is None or target is None:
        raise RuntimeError("Vast allocation did not produce an SSH-ready host")
    try:

        if getattr(args, "bootstrap", "full") == "none":
            update_status(state, "ready", ssh_host=target.host, ssh_port=target.port)
            return state

        update_status(state, "bootstrapping")
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

        phase_started = time.monotonic()
        try:
            run_remote_container_smoke(target, args.remote_repo_dir, DEFAULT_REMOTE_ENV_FILE)
            verify_remote_smoke(target)
        finally:
            record_startup_timing(state, "container_smoke", phase_started)

        update_status(state, "ready", ssh_host=target.host, ssh_port=target.port)
        return state
    except Exception as exc:  # noqa: BLE001 - preserve instance for explicit teardown.
        update_status(state, "failed", error=str(exc))
        raise


def handle_up(args: argparse.Namespace) -> int:
    try:
        state = do_up(args)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Vast CMUX host ready: {args.task_name}")
    print(f"State: {rel_state_path(args.task_name)}")
    print(f"Remote repo: {state['remote_repo_dir']}")
    print(f"Next: scripts/remote/vast-start-task.sh {args.task_name} '<prompt>'")
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


def handle_start_task(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
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
        name = str(state.get("cmux_workspace_name") or args.task_name)
        cwd = str(state.get("workspace_cwd") or DEFAULT_CMUX_WORKSPACE_CWD)
    elif state.get("status") == "ready":
        name = str(getattr(args, "workspace_name", None) or args.task_name)
        cwd = str(getattr(args, "workspace_cwd", None) or state.get("workspace_cwd") or DEFAULT_CMUX_WORKSPACE_CWD)
        state["cmux_workspace_name"] = name
        state["cmux_workspace_path"] = cwd
        state["cmux_agent"] = getattr(args, "agent", "none")
        state["workspace_cwd"] = cwd
        update_status(state, "workspace_pending")
    else:
        raise SystemExit(f"State {rel_state_path(args.task_name)} has status {state.get('status')}")
    command = build_cmux_workspace_command(name, cwd)
    print(f"CMUX workspace command: {shlex.join(command)}")
    print("CMUX workspace creation is left to the operator; no CMUX command was executed.")
    print("Watch: scripts/remote/vast-watch.sh --once")
    return 0


def remote_file_exists(target: SshTarget, remote_path: str) -> bool:
    result = ssh(target, f"test -e {shlex.quote(remote_path)}", check=False)
    return result.returncode == 0


def copy_remote_path(target: SshTarget, remote_path: str, local_path: Path, *, required: bool) -> None:
    if not remote_file_exists(target, remote_path):
        print(f"Warning: remote path missing: {remote_path}", file=sys.stderr)
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    destination = str(local_path)
    if remote_path.endswith("/"):
        local_path.mkdir(parents=True, exist_ok=True)
        destination = str(local_path)
    result = run_command([*scp_base(target), "-r", f"{target.user}@{target.host}:{remote_path}", destination], check=False)
    if result.returncode != 0:
        message = f"failed to copy remote path: {remote_path}"
        if required:
            raise RuntimeError(message)
        print("Warning: " + message, file=sys.stderr)


def copy_artifacts(state: dict) -> None:
    if not state.get("ssh_host") or not state.get("ssh_port"):
        print("Warning: state has no SSH endpoint; cannot copy artifacts", file=sys.stderr)
        return
    target = target_from_state(state)
    task_name = state["task_name"]
    artifact_dir = ARTIFACTS_DIR / task_name
    remote_root = state.get("remote_repo_dir")
    copy_remote_path(target, f"{remote_root}/data/processed/regression/", artifact_dir / "data" / "processed" / "regression", required=False)
    copy_remote_path(target, f"{remote_root}/models/", artifact_dir / "models", required=False)
    copy_remote_path(target, f"{remote_root}/scenic_artifacts/", artifact_dir / "scenic_artifacts", required=False)
    copy_remote_path(target, "/tmp/scenic_container_smoke.json", artifact_dir / "scenic_container_smoke.json", required=False)


def workspace_closed(state: dict) -> bool:
    workspace_id = state.get("cmux_workspace_id")
    workspace_path = state.get("cmux_workspace_path") or state.get("workspace_cwd")
    result = run_command(["cmux", "workspace", "list", "--json"], check=False)
    if result.returncode != 0:
        return True
    try:
        payload = parse_json_output(result, "cmux workspace list")
    except RuntimeError:
        return True
    entries: object = payload
    if isinstance(payload, dict):
        entries = payload.get("workspaces", payload.get("result", payload))
    if not isinstance(entries, list):
        return True
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id") or entry.get("workspace_id") or entry.get("workspaceId")
        entry_path = entry.get("current_directory") or entry.get("currentDirectory") or entry.get("cwd")
        if (workspace_id and str(entry_id) == str(workspace_id)) or (workspace_path and str(entry_path) == str(workspace_path)):
            return False
    return True


def destroy_instance(state: dict) -> None:
    require_commands(["vastai"])
    run_command(["vastai", "destroy", "instance", str(state["instance_id"]), "--yes"])
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
        print(f"CMUX workspace still open: {task_name}")
        return
    copy_artifacts(state)
    if destroy and yes:
        destroy_instance(state)
        print(f"Vast task destroyed: {task_name}")
    else:
        update_status(state, "done")
        print(f"Task is closed; destroy with scripts/remote/vast-down.sh {task_name} --destroy --yes")


def states_to_watch(task_name: str | None) -> list[dict]:
    if task_name:
        validate_task_name(task_name)
        return [load_state(task_name)]
    states: list[dict] = []
    seen: set[str] = set()
    for directory in (STATE_DIR, LEGACY_STATE_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            with path.open() as handle:
                state = _normalize_state(json.load(handle))
            if state.get("status") in VALID_STATUSES:
                states.append(state)
    return states


def handle_watch(args: argparse.Namespace) -> int:
    while True:
        for state in states_to_watch(args.task_name):
            try:
                process_watch_state(state, destroy=args.destroy, yes=args.yes)
            except Exception as exc:  # noqa: BLE001
                print(f"Watch failed for {state.get('task_name')}: {exc}", file=sys.stderr)
                if args.once:
                    return 1
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


def handle_down(args: argparse.Namespace) -> int:
    validate_task_name(args.task_name)
    state = load_state(args.task_name)
    if args.copy_artifacts or state["status"] in {"workspace_pending", "done", "failed"}:
        copy_artifacts(state)
    if not (args.destroy and args.yes):
        print(f"Destroy command: vastai destroy instance {state['instance_id']}")
        print("Not destroying without --destroy --yes")
        return 0
    destroy_instance(state)
    print(f"Vast task destroyed: {args.task_name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create, watch, and tear down Scenic Drive Vast.ai hosts with CMUX handoff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    up = subparsers.add_parser("up", help="Create a Vast.ai GPU host")
    up.add_argument("task_name")
    add_up_options(up)
    up.set_defaults(func=handle_up)

    start_task = subparsers.add_parser("start-task", help="Prepare a CMUX workspace opening command for a Vast host")
    start_task.add_argument("task_name")
    start_task.add_argument("prompt", help="Operator prompt retained for wrapper compatibility; CMUX opening is manual.")
    start_task.add_argument("--agent", default="none")
    start_task.add_argument("--workspace-name")
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
