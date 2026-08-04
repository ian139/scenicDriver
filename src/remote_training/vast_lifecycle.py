"""Plan and execute Vast.ai instance initialization for remote training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time


DEFAULT_REMOTE_TRAINING_IMAGE = "ian139/scenicdriver-remote-training:latest"


@dataclass(frozen=True)
class VastInitConfig:
    host: str
    user: str = "root"
    port: int = 22
    identity_file: Path | None = None
    repo_url: str = ""
    branch: str = "Ian139/RemoteTraining"
    repo_dir: str = "/workspace/scenic-drive"
    image: str | None = None
    containerfile: str | None = None
    container_name: str = "scenic-remote-training"
    s3_bucket: str = ""
    s3_only: bool = True
    remote_env_file: str = "/root/.scenic/aws.env"
    local_secrets_env_file: Path | None = None
    data_root: str = "/workspace/scenic-data"
    models_root: str = "/workspace/scenic-models"
    smoke_command: str = "python scripts/modeling/train_regression_baseline.py --help"
    force_reset: bool = False
    require_artifacts: bool = False
    pull_image: bool = True
    retries: int = 3
    retry_delay_seconds: int = 5


@dataclass(frozen=True)
class VastInitPlan:
    ssh_command: list[str]
    scp_command: list[str] | None
    bootstrap_script: str
    retries: int = 3
    retry_delay_seconds: int = 5


def validate_config(config: VastInitConfig) -> None:
    if not config.host:
        raise ValueError("--host is required")
    if not config.repo_url:
        raise ValueError("--repo-url is required")
    if not config.s3_bucket:
        raise ValueError("--s3-bucket or SCENIC_S3_BUCKET is required")
    if config.retries < 1:
        raise ValueError("--retries must be at least 1")
    if config.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds must be non-negative")


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _ssh_command(config: VastInitConfig) -> list[str]:
    command = ["ssh"]
    if config.identity_file is not None:
        command.extend(["-i", str(config.identity_file)])
    command.extend(["-p", str(config.port), f"{config.user}@{config.host}", "bash -s"])
    return command


def _scp_command(config: VastInitConfig) -> list[str] | None:
    if config.local_secrets_env_file is None:
        return None
    command = ["scp"]
    if config.identity_file is not None:
        command.extend(["-i", str(config.identity_file)])
    command.extend(["-P", str(config.port), str(config.local_secrets_env_file), f"{config.user}@{config.host}:{config.remote_env_file}"])
    return command


def _docker_run(config: VastInitConfig, image: str, command: str) -> str:
    return " \\\n  ".join(
        [
            "docker run --rm --gpus all",
            f"--name {_q(config.container_name)}",
            f"--env-file {_q(config.remote_env_file)}",
            "-e SCENIC_S3_BUCKET",
            "-e SCENIC_S3_ONLY",
            f"-v {_q(config.repo_dir)}:{_q(config.repo_dir)}",
            f"-v {_q(config.data_root)}/raw:{_q(config.repo_dir)}/data/raw",
            f"-v {_q(config.data_root)}/processed:{_q(config.repo_dir)}/data/processed",
            f"-v {_q(config.models_root)}:{_q(config.repo_dir)}/models",
            f"-w {_q(config.repo_dir)}",
            f"{_q(image)} {command}",
        ]
    )


def _artifact_flag(config: VastInitConfig) -> str:
    return "--required" if config.require_artifacts else "--optional"


def _bootstrap_script(config: VastInitConfig) -> str:
    image = config.image or DEFAULT_REMOTE_TRAINING_IMAGE
    s3_only = "1" if config.s3_only else "0"
    artifact_flag = _artifact_flag(config)
    s3_smoke = (
        "python -c "
        + _q(
            "import os, torch, boto3; "
            "bucket=os.environ['SCENIC_S3_BUCKET']; "
            "assert os.environ.get('SCENIC_S3_ONLY') in {'1','true','yes'}; "
            "assert torch.cuda.is_available(); "
            "boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix='raw/', MaxKeys=1); "
            "print('vast smoke ok:', torch.cuda.get_device_name(0))"
        )
    )
    lines = [
        "set -euo pipefail",
        "",
        "echo '== prereqs =='",
        "command -v git >/dev/null",
        "command -v docker >/dev/null",
        "command -v nvidia-smi >/dev/null",
        "docker info >/dev/null",
        "nvidia-smi",
        "",
        "echo '== repo =='",
        f"REPO_DIR={_q(config.repo_dir)}",
        f"BRANCH={_q(config.branch)}",
        f"REPO_URL={_q(config.repo_url)}",
        'if [ ! -d "$REPO_DIR/.git" ]; then',
        '  mkdir -p "$(dirname "$REPO_DIR")"',
        '  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"',
        "else",
        '  if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then',
    ]
    if config.force_reset:
        lines.extend(
            [
                "    echo 'remote repo has local changes; discarding because --force-reset was supplied'",
                "  fi",
                '  git -C "$REPO_DIR" fetch origin "$BRANCH"',
                '  git -C "$REPO_DIR" checkout "$BRANCH"',
                '  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"',
                "fi",
            ]
        )
    else:
        lines.extend(
            [
                "    echo 'remote repo has local changes; rerun with --force-reset to discard them' >&2",
                "    exit 1",
                "  fi",
                '  git -C "$REPO_DIR" fetch origin "$BRANCH"',
                '  git -C "$REPO_DIR" checkout "$BRANCH"',
                '  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"',
                "fi",
            ]
        )
    lines.extend(
        [
            "",
            "echo '== environment =='",
            'mkdir -p "$REPO_DIR/data/raw" "$REPO_DIR/data/processed" "$REPO_DIR/models"',
            f"mkdir -p {_q(config.data_root)}/raw {_q(config.data_root)}/processed {_q(config.models_root)}",
            f"export SCENIC_S3_BUCKET={_q(config.s3_bucket)}",
            f"export SCENIC_S3_ONLY={s3_only}",
            'echo "using remote env file: ' + config.remote_env_file.replace('"', '\\"') + '"',
            "",
            "echo '== container =='",
        ]
    )
    if config.containerfile:
        lines.append(f"docker build --pull -f {_q(config.containerfile)} -t {_q(image)} {_q(config.repo_dir)}")
    elif config.pull_image:
        lines.append(f"docker pull {_q(image)}")
    lines.extend(
        [
            "",
            "echo '== S3 =='",
            _docker_run(
                config,
                image,
                "python -m src.data_pipeline.s3 check-prefix --bucket \"$SCENIC_S3_BUCKET\" --prefix raw/ --required",
            ),
            _docker_run(
                config,
                image,
                f"python -m src.data_pipeline.s3 download-prefix --bucket \"$SCENIC_S3_BUCKET\" --prefix processed/regression/ --dest data/processed/regression {artifact_flag}",
            ),
            _docker_run(
                config,
                image,
                f"python -m src.data_pipeline.s3 download-prefix --bucket \"$SCENIC_S3_BUCKET\" --prefix models/ --dest models {artifact_flag}",
            ),
            _docker_run(
                config,
                image,
                f"python -m src.data_pipeline.s3 download-file --bucket \"$SCENIC_S3_BUCKET\" --key raw/labels_human.csv --dest data/raw/labels_human.csv {artifact_flag}",
            ),
            "",
            "echo '== smoke =='",
            _docker_run(config, image, s3_smoke),
            _docker_run(config, image, config.smoke_command),
        ]
    )
    return "\n".join(lines) + "\n"


def build_init_plan(config: VastInitConfig) -> VastInitPlan:
    validate_config(config)
    return VastInitPlan(
        ssh_command=_ssh_command(config),
        scp_command=_scp_command(config),
        bootstrap_script=_bootstrap_script(config),
        retries=config.retries,
        retry_delay_seconds=config.retry_delay_seconds,
    )


def _redacted_command(command: list[str]) -> str:
    if command and command[0] == "scp" and len(command) >= 2:
        return " ".join(shlex.quote(part) for part in [*command[:-2], "<redacted>", command[-1]])
    return " ".join(shlex.quote(part) for part in command)


def _run_with_retries(command: list[str], *, stdin: str | None, retries: int, retry_delay_seconds: int) -> int:
    last_returncode = 1
    for attempt in range(1, retries + 1):
        completed = subprocess.run(command, input=stdin, text=True)
        if completed.returncode == 0:
            return 0
        last_returncode = completed.returncode
        if attempt < retries:
            time.sleep(retry_delay_seconds)
    print(f"command failed after {retries} attempt(s): {_redacted_command(command)}", file=sys.stderr)
    return last_returncode

def _remote_env_parent_script(scp_command: list[str]) -> str:
    destination = scp_command[-1]
    _, _, remote_path = destination.partition(":")
    parent = PurePosixPath(remote_path).parent
    return f"mkdir -p {_q(str(parent))}\n"


def run_init_plan(plan: VastInitPlan, *, dry_run: bool) -> int:
    if dry_run:
        return 0
    if plan.scp_command is not None:
        mkdir_status = _run_with_retries(
            plan.ssh_command,
            stdin=_remote_env_parent_script(plan.scp_command),
            retries=plan.retries,
            retry_delay_seconds=plan.retry_delay_seconds,
        )
        if mkdir_status != 0:
            return mkdir_status
        scp_status = _run_with_retries(
            plan.scp_command,
            stdin=None,
            retries=plan.retries,
            retry_delay_seconds=plan.retry_delay_seconds,
        )
        if scp_status != 0:
            return scp_status
    return _run_with_retries(
        plan.ssh_command,
        stdin=plan.bootstrap_script,
        retries=plan.retries,
        retry_delay_seconds=plan.retry_delay_seconds,
    )
