"""Build and push the remote-training Docker image on a bare Vast.ai host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.remote.cmux_vast_host import (  # noqa: E402
    SshTarget,
    attach_ssh_key,
    create_instance,
    instance_is_running,
    reboot_instance,
    run_command,
    select_offer_id,
    show_instance,
    ssh,
    ssh_base,
    utc_now,
    validate_task_name,
    wait_for_instance_endpoint,
)


STATE_DIR = PROJECT_ROOT / ".cmux-vast" / "state"
DEFAULT_OFFER_QUERY = "num_gpus=1 dph<0.20 cpu_cores>=2 disk_space>=30 direct_port_count>=1 verified=true rentable=true"
DEFAULT_IMAGE = "ubuntu:22.04"
DEFAULT_REPO_URL = "https://github.com/ian139/scenicdriver.git"
DEFAULT_BRANCH = "Ian139/RemoteTraining"
DEFAULT_REPO_DIR = "/workspace/scenic-drive"
DEFAULT_TARGET_IMAGE = "ian139/scenicdriver-remote-training:latest"
DEFAULT_DOCKERFILE = "Dockerfile.remote-training"
MISSING_TOKEN = "missing DOCKER_HUB_TOKEN; export a Docker Hub access token with push permission for ian139/scenicdriver-remote-training"


def state_path(task_name: str) -> Path:
    return STATE_DIR / f"{task_name}.json"


def rel_state_path(task_name: str) -> Path:
    return state_path(task_name).relative_to(PROJECT_ROOT)


def write_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    path = state_path(state["task_name"])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def update_state(state: dict, status: str, **updates: object) -> None:
    state.update(updates)
    state["status"] = status
    write_state(state)


def record_timing(state: dict, key: str, started_at: float) -> None:
    timings = dict(state.get("timings_seconds") or {})
    timings[key] = round(time.monotonic() - started_at, 3)
    state["timings_seconds"] = timings
    write_state(state)


def check_preconditions(args: argparse.Namespace) -> tuple[str, str, str]:
    validate_task_name(args.task_name)
    path = state_path(args.task_name)
    if path.exists():
        with path.open() as handle:
            existing = json.load(handle)
        status = existing.get("status")
        if status != "destroyed":
            raise SystemExit(f"State {rel_state_path(args.task_name)} has status {status}; destroy or remove it before retrying")

    token = os.environ.get(args.dockerhub_token_env)
    if not token:
        raise SystemExit(MISSING_TOKEN)

    identity_file = str(Path(args.identity_file).expanduser())
    public_key = str(Path(args.ssh_public_key).expanduser())
    if not Path(identity_file).exists():
        raise SystemExit(f"identity file not found: {identity_file}")
    if not Path(public_key).exists():
        raise SystemExit(f"ssh public key not found: {public_key}")

    for command in ("vastai", "ssh"):
        result = run_command(["/bin/sh", "-c", "command -v \"$1\"", "sh", command], check=False)
        if result.returncode != 0:
            raise SystemExit(f"missing required local command: {command}")
    return token, identity_file, public_key


def initial_state(args: argparse.Namespace, offer_id: int, instance_id: int, identity_file: str, public_key: str) -> dict:
    now = utc_now()
    return {
        "task_name": args.task_name,
        "status": "creating",
        "offer_id": offer_id,
        "instance_id": instance_id,
        "image": args.image,
        "disk_gb": args.disk_gb,
        "ssh_host": None,
        "ssh_port": None,
        "ssh_user": "root",
        "identity_file": identity_file,
        "ssh_public_key": public_key,
        "repo_url": args.repo_url,
        "branch": args.branch,
        "repo_dir": args.repo_dir,
        "target_image": args.target_image,
        "dockerfile": args.dockerfile,
        "created_at": now,
        "updated_at": now,
        "timings_seconds": {},
    }


def remote_bash(script: str) -> str:
    return "bash -lc " + shlex.quote(script)


def run_ssh_phase(target: SshTarget, script: str) -> None:
    result = ssh(target, remote_bash(script), check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"remote command failed with exit {result.returncode}")


def wait_for_ssh_true(target: SshTarget, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        result = ssh(target, "true", check=False)
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout).strip()
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for SSH: {last_error}")


def docker_install_script() -> str:
    return r"""
set -euo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git docker.io
service docker start || true
if ! docker info >/dev/null 2>&1; then
  nohup dockerd >/tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do
    if docker info >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
docker version
if ! docker buildx version; then
  mkdir -p /usr/libexec/docker/cli-plugins /usr/lib/docker/cli-plugins
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64) BUILDX_ARCH=amd64 ;;
    aarch64|arm64) BUILDX_ARCH=arm64 ;;
    *) echo "unsupported buildx arch: $ARCH" >&2; exit 1 ;;
  esac
  curl -fsSL "https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-$BUILDX_ARCH" -o /usr/libexec/docker/cli-plugins/docker-buildx
  chmod +x /usr/libexec/docker/cli-plugins/docker-buildx
  ln -sf /usr/libexec/docker/cli-plugins/docker-buildx /usr/lib/docker/cli-plugins/docker-buildx
  docker buildx version
fi
""".strip()


def clone_script(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"rm -rf {shlex.quote(args.repo_dir)}",
            f"git clone --branch {shlex.quote(args.branch)} {shlex.quote(args.repo_url)} {shlex.quote(args.repo_dir)}",
            f"test -f {shlex.quote(args.repo_dir + '/' + args.dockerfile)}",
        ]
    )


def build_push_script(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(args.repo_dir)}",
            f"docker buildx build --platform linux/amd64 -f {shlex.quote(args.dockerfile)} -t {shlex.quote(args.target_image)} --push .",
        ]
    )


def verify_script(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"rm -rf {shlex.quote(args.repo_dir)}",
            f"docker run --rm {shlex.quote(args.target_image)} python -c \"import torch; print('CUDA:', torch.cuda.is_available(), 'Torch:', torch.__version__)\"",
        ]
    )


def docker_login(target: SshTarget, username: str, token: str) -> None:
    result = run_command([*ssh_base(target), f"docker login -u {shlex.quote(username)} --password-stdin"], input_text=f"{token}\n", check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError("docker login failed")


def destroy_instance(instance_id: int) -> bool:
    run_command(["vastai", "destroy", "instance", str(instance_id)], check=False)
    result = run_command(["vastai", "show", "instance", str(instance_id), "--raw"], check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return False
    if isinstance(payload, dict) and payload.get("instances") is None:
        return True
    try:
        shown = show_instance(instance_id)
    except Exception:
        return True
    return not instance_is_running(shown)


def build(args: argparse.Namespace) -> int:
    token, identity_file, public_key = check_preconditions(args)
    state: dict | None = None
    instance_id: int | None = None
    destroy_verified = False
    try:
        phase_started = time.monotonic()
        offer_id = int(args.offer_id) if args.offer_id is not None else select_offer_id(args.offer_query)
        instance_id = create_instance(offer_id, args.image, args.disk_gb)
        state = initial_state(args, offer_id, instance_id, identity_file, public_key)
        write_state(state)
        attach_ssh_key(instance_id, public_key)
        reboot_instance(instance_id)
        update_state(state, "ssh_wait")
        host, port = wait_for_instance_endpoint(instance_id, args.timeout_seconds)
        state.update({"ssh_host": host, "ssh_port": port})
        write_state(state)
        target = SshTarget(host=host, port=port, user="root", identity_file=identity_file)
        wait_for_ssh_true(target, args.timeout_seconds)
        record_timing(state, "vast_create_and_ssh", phase_started)

        update_state(state, "docker_install")
        phase_started = time.monotonic()
        run_ssh_phase(target, docker_install_script())
        record_timing(state, "docker_install", phase_started)

        update_state(state, "dockerhub_login")
        phase_started = time.monotonic()
        docker_login(target, args.dockerhub_username, token)
        record_timing(state, "dockerhub_login", phase_started)

        update_state(state, "repo_clone")
        phase_started = time.monotonic()
        run_ssh_phase(target, clone_script(args))
        record_timing(state, "repo_clone", phase_started)

        update_state(state, "docker_build_push")
        phase_started = time.monotonic()
        run_ssh_phase(target, build_push_script(args))
        record_timing(state, "docker_build_push", phase_started)

        update_state(state, "image_verify")
        phase_started = time.monotonic()
        run_ssh_phase(target, verify_script(args))
        record_timing(state, "image_verify", phase_started)
        return 0
    except Exception as exc:  # noqa: BLE001 - state and destroy must reflect any failure.
        if state is not None:
            state["error"] = str(exc)
            write_state(state)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if instance_id is not None:
            destroy_started = time.monotonic()
            try:
                destroy_verified = destroy_instance(instance_id)
                if state is not None:
                    record_timing(state, "vast_destroy", destroy_started)
                    if destroy_verified:
                        final_status = "destroyed" if "error" not in state else "failed_destroyed"
                        update_state(state, final_status)
                    else:
                        state["error"] = (state.get("error") or "") + "; failed to verify Vast destroy"
                        update_state(state, "failed")
            except Exception as exc:  # noqa: BLE001
                if state is not None:
                    state["error"] = (state.get("error") or "") + f"; destroy failed: {exc}"
                    write_state(state)
                print(f"destroy failed: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push the Scenic Drive remote-training image on a bare Vast.ai host")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Rent a bare Vast builder, build, push, verify, and destroy it")
    build_parser.add_argument("task_name")
    build_parser.add_argument("--offer-query", default=DEFAULT_OFFER_QUERY)
    build_parser.add_argument("--offer-id", type=int)
    build_parser.add_argument("--image", default=DEFAULT_IMAGE)
    build_parser.add_argument("--disk-gb", type=int, default=40)
    build_parser.add_argument("--identity-file", default="~/.ssh/id_ed25519")
    build_parser.add_argument("--ssh-public-key", default="~/.ssh/id_ed25519.pub")
    build_parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    build_parser.add_argument("--branch", default=DEFAULT_BRANCH)
    build_parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    build_parser.add_argument("--dockerhub-username", default="ian139")
    build_parser.add_argument("--dockerhub-token-env", default="DOCKER_HUB_TOKEN")
    build_parser.add_argument("--target-image", default=DEFAULT_TARGET_IMAGE)
    build_parser.add_argument("--dockerfile", default=DEFAULT_DOCKERFILE)
    build_parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command != "build":
        raise SystemExit(f"unknown command: {args.command}")
    raise SystemExit(build(args))


if __name__ == "__main__":
    main()
