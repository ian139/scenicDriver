"""Initialize a reachable Vast.ai host for containerized Scenic Drive training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.s3 import normalize_s3_only  # noqa: E402
from src.remote_training.vast_lifecycle import VastInitConfig, build_init_plan, run_init_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a Vast.ai remote training instance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init", help="Render or execute the Vast.ai initialization flow")
    init.add_argument("--host", required=True)
    init.add_argument("--user", default="root")
    init.add_argument("--port", type=int, default=22)
    init.add_argument("--identity-file", type=Path)
    init.add_argument("--repo-url", required=True)
    init.add_argument("--branch", default="Ian139/RemoteTraining")
    init.add_argument("--repo-dir", default="/workspace/scenic-drive")
    init.add_argument("--image")
    init.add_argument("--containerfile")
    init.add_argument("--container-name", default="scenic-remote-training")
    init.add_argument("--s3-bucket", default=os.environ.get("SCENIC_S3_BUCKET", ""))
    init.add_argument("--s3-only", dest="s3_only", action="store_true", default=None)
    init.add_argument("--no-s3-only", dest="s3_only", action="store_false")
    init.add_argument("--remote-env-file", default="/root/.scenic/aws.env")
    init.add_argument("--local-secrets-env-file", type=Path, metavar="<untracked-env-file>")
    init.add_argument("--data-root", default="/workspace/scenic-data")
    init.add_argument("--models-root", default="/workspace/scenic-models")
    init.add_argument("--smoke-command", default="uv run python scripts/modeling/train_regression_baseline.py --help")
    init.add_argument("--force-reset", action="store_true")
    init.add_argument("--require-artifacts", action="store_true")
    init.add_argument("--no-pull-image", dest="pull_image", action="store_false", default=True)
    init.add_argument("--retries", type=int, default=3)
    init.add_argument("--retry-delay-seconds", type=int, default=5)
    init.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _effective_s3_only(value: bool | None) -> bool:
    if value is not None:
        return value
    return normalize_s3_only(os.environ.get("SCENIC_S3_ONLY"), default=True)


def _config_from_args(args: argparse.Namespace) -> VastInitConfig:
    return VastInitConfig(
        host=args.host,
        user=args.user,
        port=args.port,
        identity_file=args.identity_file,
        repo_url=args.repo_url,
        branch=args.branch,
        repo_dir=args.repo_dir,
        image=args.image,
        containerfile=args.containerfile,
        container_name=args.container_name,
        s3_bucket=args.s3_bucket,
        s3_only=_effective_s3_only(args.s3_only),
        remote_env_file=args.remote_env_file,
        local_secrets_env_file=args.local_secrets_env_file,
        data_root=args.data_root,
        models_root=args.models_root,
        smoke_command=args.smoke_command,
        force_reset=args.force_reset,
        require_artifacts=args.require_artifacts,
        pull_image=args.pull_image,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )


def _redacted_scp(command: list[str] | None) -> list[str] | None:
    if command is None:
        return None
    if len(command) < 2:
        return command
    return [*command[:-2], "<redacted>", command[-1]]


def main() -> None:
    args = parse_args()
    if args.command != "init":
        raise SystemExit(f"unknown command: {args.command}")
    try:
        config = _config_from_args(args)
        plan = build_init_plan(config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "ssh_command": plan.ssh_command,
                    "scp_command": _redacted_scp(plan.scp_command),
                    "bootstrap_script": plan.bootstrap_script,
                },
                indent=2,
            )
        )
        return
    raise SystemExit(run_init_plan(plan, dry_run=False))


if __name__ == "__main__":
    main()
