"""State-backed Vast.ai CPU runner for the full-bbox routing benchmark.

The orchestrator deliberately keeps credentials on the local machine: the generated
remote shell script only references the remote environment file copied by SSH.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.remote.cmux_vast_host import (  # noqa: E402
    ARTIFACTS_DIR,
    SshTarget,
    VALID_STATUSES,
    attach_ssh_key,
    copy_remote_path,
    copy_secrets,
    create_instance,
    make_overlay_tarball,
    require_commands,
    run_command,
    scp_base,
    select_offer_id_at,
    ssh,
    ssh_base,
    utc_now,
    validate_task_name,
    wait_for_instance_endpoint,
    wait_for_ssh,
)

STATE_DIR = PROJECT_ROOT / ".cmux-vast" / "state"
DEFAULT_OFFER_QUERY = (
    "cpu_arch=amd64 cpu_cores>=4 cpu_ram>=32 verified=true "
    "direct_port_count>=1 rentable=true disk_space>=64"
)
DEFAULT_IMAGE = "ubuntu:22.04"
DEFAULT_REMOTE_REPO_DIR = "/workspace/scenic-drive"
DEFAULT_REMOTE_ENV_FILE = "/root/.scenic/aws.env"
DEFAULT_REMOTE_RUN_ROOT = "/workspace/scenic_artifacts/vast-route"
DEFAULT_S3_CHECKPOINT_PREFIX = "checkpoints"
DEFAULT_ARTIFACT_S3_BUCKET = "scenicdriver-data"
DEFAULT_ARTIFACT_S3_PREFIX = "releases/routeOptimizer/prompt-two-exp02-20260810/"
DEFAULT_PER_WORKER_RAM_MB = 24_576
DEFAULT_RESERVED_RAM_MB = 8_192
DEFAULT_MAX_WORKERS = 32
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 60


@dataclass(frozen=True)
class VastRouteConfig:
    task_name: str
    run_id: str = ""
    s3_bucket: str = ""
    s3_prefix: str = ""
    artifact_s3_bucket: str = DEFAULT_ARTIFACT_S3_BUCKET
    artifact_s3_prefix: str = DEFAULT_ARTIFACT_S3_PREFIX
    image: str = DEFAULT_IMAGE
    offer_query: str = DEFAULT_OFFER_QUERY
    offer_id: int | None = None
    disk_gb: int = 64
    allocation_attempts: int = 3
    identity_file: str = "~/.ssh/id_ed25519"
    ssh_public_key: str = "~/.ssh/id_ed25519.pub"
    local_secrets_env_file: str = ".secrets/aws.env"
    remote_env_file: str = DEFAULT_REMOTE_ENV_FILE
    remote_repo_dir: str = DEFAULT_REMOTE_REPO_DIR
    corpus: str = "scripts/routing/production_benchmark_pairs.json"
    graph: str = (
        "data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3"
    )
    report: str = "data/processed/heuristic_runs/prompt_two_candidate_exp02_fresh_test20_20260810/report/report.json"
    output: str = "data/processed/routing_benchmarks/production_artifact_benchmark.json"
    workers: int | None = None
    group_size: int | None = None
    per_worker_ram_mb: int = DEFAULT_PER_WORKER_RAM_MB
    reserved_ram_mb: int = DEFAULT_RESERVED_RAM_MB
    max_workers: int = DEFAULT_MAX_WORKERS
    checkpoint_interval_seconds: int = DEFAULT_CHECKPOINT_INTERVAL_SECONDS
    case_timeout_seconds: float = 10.0
    strict_service_full: bool = False
    timeout_seconds: int = 1800
    destroy: bool = False


def quote(value: object) -> str:
    """Shell quote one value; used for every path and user-controlled argument."""
    return shlex.quote(str(value))


_quote = quote


def generated_run_id(task_name: str) -> str:
    return f"{task_name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def state_path(task_name: str) -> Path:
    validate_task_name(task_name)
    return STATE_DIR / f"{task_name}.json"


def load_state(task_name: str) -> dict[str, Any]:
    path = state_path(task_name)
    if not path.is_file():
        raise SystemExit(f"No state file for task: {task_name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid state file: {path}: {exc}") from exc
    if payload.get("status") not in VALID_STATUSES:
        raise SystemExit(f"Invalid state status in {path}: {payload.get('status')}")
    return payload


def write_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = utc_now()
    path = state_path(str(state["task_name"]))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def update_state(state: dict[str, Any], **updates: Any) -> None:
    state.update(updates)
    write_state(state)


def validate_offer_config(offer_query: str, offer_id: int | None = None) -> None:
    if offer_id is not None and int(offer_id) <= 0:
        raise ValueError("offer-id must be a positive integer")
    if offer_id is None and not str(offer_query or "").strip():
        raise ValueError("offer-query must not be empty when offer-id is omitted")


def validate_worker_overrides(
    workers: int | None,
    group_size: int | None,
    *,
    cpu_count: int | None = None,
    ram_mb: int | None = None,
    per_worker_ram_mb: int = DEFAULT_PER_WORKER_RAM_MB,
    reserved_ram_mb: int = DEFAULT_RESERVED_RAM_MB,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    if group_size is not None and group_size < 1:
        raise ValueError("group-size must be at least 1")
    if workers is not None and group_size is not None and group_size < workers:
        raise ValueError("group-size must be at least workers")
    if per_worker_ram_mb < 1 or reserved_ram_mb < 0:
        raise ValueError("worker memory budget is invalid")
    if max_workers < 1:
        raise ValueError("max-workers must be at least 1")
    if cpu_count is not None and cpu_count < 1:
        raise ValueError("remote CPU count must be positive")
    if ram_mb is not None and ram_mb < 1:
        raise ValueError("remote RAM must be positive")
    if workers is not None and cpu_count is not None and workers > cpu_count:
        raise ValueError(f"workers={workers} exceeds remote CPU count={cpu_count}")
    if workers is not None and ram_mb is not None:
        capacity = max(0, (ram_mb - reserved_ram_mb) // per_worker_ram_mb)
        if workers > capacity:
            raise ValueError(
                f"workers={workers} exceeds conservative RAM capacity={capacity}"
            )
    if workers is not None and workers > max_workers:
        raise ValueError(f"workers={workers} exceeds max-workers={max_workers}")


def derive_worker_count(
    ram_mb: int,
    cpu_count: int,
    *,
    per_worker_ram_mb: int = DEFAULT_PER_WORKER_RAM_MB,
    reserved_ram_mb: int = DEFAULT_RESERVED_RAM_MB,
    max_workers: int = DEFAULT_MAX_WORKERS,
    explicit_workers: int | None = None,
) -> int:
    """Return a bounded worker count using CPU and reserved-memory capacity."""
    validate_worker_overrides(
        explicit_workers,
        None,
        cpu_count=cpu_count,
        ram_mb=ram_mb,
        per_worker_ram_mb=per_worker_ram_mb,
        reserved_ram_mb=reserved_ram_mb,
        max_workers=max_workers,
    )
    capacity = min(
        cpu_count, max(0, (ram_mb - reserved_ram_mb) // per_worker_ram_mb), max_workers
    )
    if capacity < 1:
        raise ValueError(
            "remote host has insufficient CPU/RAM for one benchmark worker"
        )
    if explicit_workers is not None:
        return explicit_workers
    return capacity


def remote_paths(config: VastRouteConfig) -> dict[str, str]:
    root = PurePosixPath(config.remote_repo_dir)
    output = root / config.output
    checkpoint = output.with_suffix(".jsonl")
    run_root = PurePosixPath(DEFAULT_REMOTE_RUN_ROOT) / config.run_id
    s3_prefix = config.s3_prefix.strip("/")
    checkpoint_key = f"{s3_prefix}/{DEFAULT_S3_CHECKPOINT_PREFIX}/{config.task_name}/{config.task_name}.jsonl".strip(
        "/"
    )
    final_key = f"{s3_prefix}/{config.task_name}.json".strip("/")
    return {
        "repo": str(root),
        "output": str(output),
        "checkpoint": str(checkpoint),
        "log": str(run_root / "benchmark.log"),
        "script": str(run_root / "run.sh"),
        "checkpoint_s3": f"s3://{config.s3_bucket}/{checkpoint_key}",
        "final_s3": f"s3://{config.s3_bucket}/{final_key}",
    }


def build_preflight_script(config: VastRouteConfig) -> str:
    repo = quote(config.remote_repo_dir)
    env = quote(config.remote_env_file)
    return f"""set -Eeuo pipefail
# Raw host probe must work even if the project checkout is not usable.
printf 'nproc='; nproc
printf 'MemTotal_kB='; awk '/^MemTotal:/ {{print $2}}' /proc/meminfo
printf 'disk_available_kB='; df -Pk / | awk 'NR==2 {{print $4}}'
cd {repo}
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON=3.11
command -v python >/dev/null || command -v python3 >/dev/null
command -v uv >/dev/null
if [[ ! -r {env} ]]; then echo 'remote AWS environment file is missing' >&2; exit 20; fi
set -a; source {env}; set +a
aws sts get-caller-identity >/dev/null
uv run python scripts/routing/check_beta_artifacts.py --project-root {repo}
""".strip()


def parse_resource_probe(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in {"nproc", "MemTotal_kB", "MemTotal"}:
            try:
                values[key] = int(value.strip())
            except ValueError:
                continue
    cpu = values.get("nproc")
    ram_kb = values.get("MemTotal_kB", values.get("MemTotal"))
    if cpu is None or ram_kb is None:
        raise RuntimeError("remote preflight did not report nproc and MemTotal")
    return cpu, ram_kb // 1024


def remote_preflight(target: SshTarget, config: VastRouteConfig) -> tuple[int, int]:
    result = ssh(target, "bash -lc " + quote(build_preflight_script(config)))
    return parse_resource_probe(result.stdout)


def build_remote_script(
    config: VastRouteConfig, *, workers: int, group_size: int
) -> str:
    """Build a detached, resumable benchmark script without embedding secret values."""
    paths = remote_paths(config)
    repo = quote(paths["repo"])
    env = quote(config.remote_env_file)
    corpus = quote(str(PurePosixPath(config.remote_repo_dir) / config.corpus))
    graph = (
        quote(str(PurePosixPath(config.remote_repo_dir) / config.graph))
        if config.graph
        else ""
    )
    report = quote(str(PurePosixPath(config.remote_repo_dir) / config.report))
    output = quote(paths["output"])
    checkpoint = quote(paths["checkpoint"])
    checkpoint_s3 = quote(paths["checkpoint_s3"])
    final_s3 = quote(paths["final_s3"])
    graph_arg = f"--graph {graph}" if graph else ""
    strict_service_arg = "--strict-service-full" if config.strict_service_full else ""
    spatial_index_guard = (
        f"""if [[ ! -f {graph}.edge_projection_index ]]; then
  echo "required manifest edge projection index is missing" >&2
  exit 21
fi"""
        if graph
        else ""
    )
    return (
        f"""#!/usr/bin/env bash
set -Eeuo pipefail
# Probe raw host capacity before depending on the project, env, or uv.
printf 'nproc='; nproc
printf 'MemTotal_kB='; awk '/^MemTotal:/ {{print $2}}' /proc/meminfo
printf 'disk_available_kB='; df -Pk / | awk 'NR==2 {{print $4}}'
cd {repo}
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON=3.11
set -a
source {env}
set +a
mkdir -p {quote(str(PurePosixPath(paths["log"]).parent))} "$(dirname {output})"
{spatial_index_guard}
CHECKPOINT={checkpoint}
OUTPUT={output}
CHECKPOINT_S3={checkpoint_s3}
FINAL_S3={final_s3}
validate_checkpoint() {{
  python - "${{1:-$CHECKPOINT}}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
seen: set[str] = set()
fingerprint = None
for line in path.open(encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("case_id"):
            raise SystemExit(1)
        row_fingerprint = row.get("checkpoint_fingerprint")
        if not isinstance(row_fingerprint, str) or not row_fingerprint:
            raise SystemExit(1)
        if fingerprint is None:
            fingerprint = row_fingerprint
        elif row_fingerprint != fingerprint:
            raise SystemExit(1)
        seen.add(str(row["case_id"]))
if fingerprint is None:
    raise SystemExit(1)
if len(seen) != sum(1 for _ in path.open(encoding="utf-8") if _.strip()):
    raise SystemExit(1)
PY
}}
copy_checkpoint() {{
  if [[ ! -s "$CHECKPOINT" ]]; then return 0; fi
  tmp="${{CHECKPOINT}}.snapshot.$$"
  # Snapshot first; never validate or upload a file still being appended.
  cp "$CHECKPOINT" "$tmp"
  if validate_checkpoint "$tmp"; then
    aws s3 cp "$tmp" "$CHECKPOINT_S3" --only-show-errors
  fi
  rm -f "$tmp"
}}
if aws s3 cp "$CHECKPOINT_S3" "$CHECKPOINT" --only-show-errors; then
  validate_checkpoint "$CHECKPOINT" || rm -f "$CHECKPOINT"
fi
if [[ -f "$CHECKPOINT" ]]; then
  RESUME=(--resume)
else
  RESUME=()
fi
BENCHMARK=(uv run python scripts/routing/production_benchmark.py
  --corpus {corpus} {graph_arg} --report {report} --output {output}
  --case-timeout-seconds {quote(config.case_timeout_seconds)} --workers {quote(workers)}
  --group-size {quote(group_size)} {strict_service_arg})
"${{BENCHMARK[@]}}" "${{RESUME[@]}}" &
BENCH_PID=$!
while kill -0 "$BENCH_PID" 2>/dev/null; do
  copy_checkpoint || true
  sleep {quote(max(1, config.checkpoint_interval_seconds))}
done
wait "$BENCH_PID"
copy_checkpoint
python - "$OUTPUT" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
matrix = payload.get("matrix")
rows = payload.get("results")
if not isinstance(matrix, dict) or not isinstance(rows, list):
    raise SystemExit("refusing final upload: malformed benchmark payload")
expected = matrix.get("planned_cases")
if not isinstance(expected, int) or len(rows) != expected:
    raise SystemExit("refusing final upload: unexpected planned case count")
if any(
    not isinstance(row, dict) or not row.get("case_id")
    for row in rows
):
    raise SystemExit("refusing final upload: duplicate or missing case IDs")
ids = [str(row["case_id"]) for row in rows]
if len(ids) != len(set(ids)) or len(ids) != expected:
    raise SystemExit("refusing final upload: duplicate or missing case IDs")
if any(
    not isinstance(row.get("checkpoint_fingerprint"), str)
    or not row["checkpoint_fingerprint"]
    for row in rows
):
    raise SystemExit("refusing final upload: checkpoint fingerprints are inconsistent")
fingerprints = {{str(row["checkpoint_fingerprint"]) for row in rows}}
if len(fingerprints) != 1 or fingerprints != {{str(payload.get("checkpoint_fingerprint"))}}:
    raise SystemExit("refusing final upload: checkpoint fingerprints are inconsistent")
if matrix.get("all_cases_persisted") is not True:
    raise SystemExit("refusing final upload: matrix.all_cases_persisted is not true")
PY
aws s3 cp "$OUTPUT" "$FINAL_S3" --only-show-errors
""".strip()
        + "\n"
    )


def build_script(config: VastRouteConfig, workers: int, group_size: int) -> str:
    return build_remote_script(config, workers=workers, group_size=group_size)


def build_initial_state(
    config: VastRouteConfig,
    offer_id: int,
    instance_id: int,
    *,
    workers: int | None = None,
    group_size: int | None = None,
    cpu_count: int | None = None,
    ram_mb: int | None = None,
    ssh_target: SshTarget | None = None,
) -> dict[str, Any]:
    paths = remote_paths(config)
    state = {
        "task_name": config.task_name,
        "run_id": config.run_id,
        "status": "creating",
        "created_at": utc_now(),
        "instance_id": int(instance_id),
        "offer_id": int(offer_id),
        "ssh_host": ssh_target.host if ssh_target else None,
        "ssh_port": ssh_target.port if ssh_target else None,
        "ssh_user": ssh_target.user if ssh_target else "root",
        "identity_file": config.identity_file,
        "remote_repo_dir": config.remote_repo_dir,
        "remote_script": paths["script"],
        "remote_log": paths["log"],
        "output_path": paths["output"],
        "checkpoint_path": paths["checkpoint"],
        "checkpoint_s3": paths["checkpoint_s3"],
        "final_s3": paths["final_s3"],
        "workers": workers,
        "worker_count": workers,
        "group_size": group_size,
        "cpu_count": cpu_count,
        "ram_mb": ram_mb,
    }
    # Flat fields remain convenient for scripts; grouped fields make the
    # instance/SSH/run/checkpoint/worker/group contract explicit.
    state.update(
        instance={"id": int(instance_id), "offer_id": int(offer_id)},
        ssh={
            "host": state["ssh_host"],
            "port": state["ssh_port"],
            "user": state["ssh_user"],
        },
        run={"id": config.run_id, "status": state["status"], "pid": None},
        checkpoint={"path": paths["checkpoint"], "s3": paths["checkpoint_s3"]},
        worker={"count": workers, "cpu_count": cpu_count, "ram_mb": ram_mb},
        group={"size": group_size},
    )
    return state


def upload_source_overlay(target: SshTarget, remote_repo_dir: str) -> None:
    overlay = make_overlay_tarball(PROJECT_ROOT)
    try:
        destination = f"{target.user}@{target.host}:/tmp/scenic-drive-overlay.tar.gz"
        run_command([*scp_base(target), str(overlay), destination])
    finally:
        overlay.unlink(missing_ok=True)
    ssh(
        target,
        "mkdir -p {repo} && tar -xzf /tmp/scenic-drive-overlay.tar.gz -C {repo} && rm -f /tmp/scenic-drive-overlay.tar.gz".format(
            repo=quote(remote_repo_dir)
        ),
    )


def bootstrap_remote_project(target: SshTarget, config: VastRouteConfig) -> None:
    repo = quote(config.remote_repo_dir)
    manifest = quote(
        str(PurePosixPath(config.remote_repo_dir) / "deploy/beta_artifacts.json")
    )
    env = quote(config.remote_env_file)
    script = f"""set -Eeuo pipefail
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends build-essential ca-certificates curl awscli tar python3 python3-venv python-is-python3
fi
command -v curl >/dev/null
command -v aws >/dev/null
command -v tar >/dev/null
command -v python >/dev/null || command -v python3 >/dev/null
cd {repo}
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
uv python install 3.11
printf '3.11\n' > .python-version
export UV_PYTHON=3.11
test -f {manifest}
if [[ ! -r {env} ]]; then echo 'remote AWS environment file is missing' >&2; exit 20; fi
set -a; source {env}; set +a
uv run python scripts/deploy/bootstrap_beta_artifacts.py --manifest {manifest} --project-root {repo} \
  --s3-bucket {quote(config.artifact_s3_bucket)} --s3-prefix {quote(config.artifact_s3_prefix)}
""".strip()
    ssh(target, "bash -lc " + quote(script))


def upload_remote_script(
    target: SshTarget, config: VastRouteConfig, script_text: str
) -> None:
    paths = remote_paths(config)
    run_command(
        [
            *ssh_base(target),
            f"mkdir -p {quote(str(PurePosixPath(paths['script']).parent))} && cat > {quote(paths['script'])}",
        ],
        input_text=script_text,
    )
    ssh(target, f"chmod 700 {quote(paths['script'])}")


def launch_remote_script(target: SshTarget, config: VastRouteConfig) -> int:
    paths = remote_paths(config)
    command = f"nohup bash {quote(paths['script'])} > {quote(paths['log'])} 2>&1 < /dev/null & echo $!"
    result = ssh(target, command)
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("remote detached benchmark did not return a PID") from exc


def recover_outputs(state: dict[str, Any], *, required: bool = False) -> list[Path]:
    if not state.get("ssh_host") or not state.get("ssh_port"):
        if required:
            raise RuntimeError("state has no SSH endpoint")
        return []
    target = SshTarget(
        host=str(state["ssh_host"]),
        port=int(state["ssh_port"]),
        user=str(state.get("ssh_user", "root")),
        identity_file=str(state["identity_file"]),
    )
    local_dir = ARTIFACTS_DIR / str(state["task_name"])
    local_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for field, name in (
        ("checkpoint_path", "checkpoint.jsonl"),
        ("remote_log", "benchmark.log"),
        ("output_path", "final.json"),
    ):
        remote = state.get(field)
        if not remote:
            continue
        destination = local_dir / name
        if copy_remote_path(target, str(remote), destination, required=required):
            copied.append(destination)
    return copied


def destroy_instance(state: dict[str, Any]) -> bool:
    instance_id = state.get("instance_id")
    if instance_id is None:
        return False
    result = run_command(
        ["vastai", "destroy", "instance", str(instance_id), "--yes"], check=False
    )
    if result.returncode == 0:
        update_state(state, status="destroyed", destroyed_at=utc_now())
        return True
    update_state(
        state,
        status="failed_kept",
        destroy_error=(result.stderr or result.stdout).strip(),
    )
    return False


def config_from_args(args: argparse.Namespace) -> VastRouteConfig:
    run_id = args.run_id or generated_run_id(args.task_name)
    prefix = args.s3_prefix or f"outputs/vast/{run_id}"
    return VastRouteConfig(
        task_name=args.task_name,
        run_id=run_id,
        s3_bucket=args.s3_bucket or os.environ.get("SCENIC_S3_BUCKET", ""),
        s3_prefix=prefix,
        artifact_s3_bucket=args.artifact_s3_bucket
        or os.environ.get("SCENIC_ARTIFACT_S3_BUCKET", DEFAULT_ARTIFACT_S3_BUCKET),
        artifact_s3_prefix=args.artifact_s3_prefix
        or os.environ.get("SCENIC_ARTIFACT_S3_PREFIX", DEFAULT_ARTIFACT_S3_PREFIX),
        image=args.image,
        offer_query=args.offer_query,
        offer_id=args.offer_id,
        disk_gb=args.disk_gb,
        allocation_attempts=max(1, args.allocation_attempts),
        identity_file=str(Path(args.identity_file).expanduser()),
        ssh_public_key=str(Path(args.ssh_public_key).expanduser()),
        local_secrets_env_file=str(Path(args.local_secrets_env_file).expanduser()),
        remote_env_file=args.remote_env_file,
        remote_repo_dir=args.remote_repo_dir,
        corpus=args.corpus,
        graph=args.graph,
        report=args.report,
        output=args.output,
        workers=args.workers,
        group_size=args.group_size,
        per_worker_ram_mb=args.per_worker_ram_mb,
        reserved_ram_mb=args.reserved_ram_mb,
        max_workers=args.max_workers,
        checkpoint_interval_seconds=args.checkpoint_interval_seconds,
        case_timeout_seconds=args.case_timeout_seconds,
        strict_service_full=args.strict_service_full,
        timeout_seconds=args.timeout_seconds,
        destroy=args.destroy,
    )


def validate_config(config: VastRouteConfig, *, check_files: bool = True) -> None:
    validate_task_name(config.task_name)
    validate_offer_config(config.offer_query, config.offer_id)
    validate_worker_overrides(
        config.workers,
        config.group_size,
        per_worker_ram_mb=config.per_worker_ram_mb,
        reserved_ram_mb=config.reserved_ram_mb,
        max_workers=config.max_workers,
    )
    if not config.s3_bucket.strip():
        raise ValueError("--s3-bucket or SCENIC_S3_BUCKET is required")
    if config.disk_gb < 1:
        raise ValueError("disk-gb must be positive")
    if config.allocation_attempts < 1:
        raise ValueError("allocation-attempts must be at least 1")
    if check_files:
        for path, label in (
            (config.identity_file, "identity file"),
            (config.ssh_public_key, "SSH public key"),
            (config.local_secrets_env_file, "local secrets env file"),
        ):
            if not Path(path).is_file():
                raise ValueError(f"{label} not found: {path}")


def allocate_instance(config: VastRouteConfig) -> tuple[int, int]:
    last_error: BaseException | None = None
    for attempt in range(config.allocation_attempts):
        try:
            offer_id = (
                int(config.offer_id)
                if config.offer_id is not None
                else select_offer_id_at(config.offer_query, attempt)
            )
            return offer_id, create_instance(offer_id, config.image, config.disk_gb)
        except (RuntimeError, SystemExit) as exc:
            last_error = exc
    assert last_error is not None
    raise RuntimeError(
        f"failed to allocate a Vast instance after {config.allocation_attempts} attempts: {last_error}"
    ) from last_error


def handle_run(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    try:
        validate_config(config, check_files=not args.dry_run)
    except (ValueError, SystemExit) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.dry_run:
        workers = config.workers or 1
        group = config.group_size or workers
        print(build_remote_script(config, workers=workers, group_size=group))
        return 0
    existing = state_path(config.task_name)
    if existing.exists():
        raise SystemExit(
            f"State exists: {existing}; use recover/cleanup or choose another task"
        )
    require_commands(["vastai", "ssh", "scp", "tar"])
    state: dict[str, Any] | None = None
    try:
        offer_id, instance_id = allocate_instance(config)
        state = build_initial_state(config, offer_id, instance_id)
        write_state(state)
        attach_ssh_key(instance_id, config.ssh_public_key)
        host, port = wait_for_instance_endpoint(instance_id, config.timeout_seconds)
        target = SshTarget(
            host=host, port=port, user="root", identity_file=config.identity_file
        )
        update_state(state, status="ssh_wait", ssh_host=host, ssh_port=port)
        wait_for_ssh(target, config.timeout_seconds)
        update_state(state, status="provisioning")
        # Copying the overlay and bootstrap installs uv before preflight checks.
        copy_secrets(target, config.local_secrets_env_file, config.remote_env_file)
        ssh(target, f"chmod 600 {quote(config.remote_env_file)}")
        upload_source_overlay(target, config.remote_repo_dir)
        bootstrap_remote_project(target, config)
        update_state(state, status="provisioning")
        cpu_count, ram_mb = remote_preflight(target, config)
        workers = derive_worker_count(
            ram_mb,
            cpu_count,
            per_worker_ram_mb=config.per_worker_ram_mb,
            reserved_ram_mb=config.reserved_ram_mb,
            max_workers=config.max_workers,
            explicit_workers=config.workers,
        )
        group_size = config.group_size or workers
        validate_worker_overrides(workers, group_size)
        state.update(
            cpu_count=cpu_count,
            ram_mb=ram_mb,
            workers=workers,
            worker_count=workers,
            group_size=group_size,
            worker={"count": workers, "cpu_count": cpu_count, "ram_mb": ram_mb},
            group={"size": group_size},
        )
        write_state(state)
        script = build_remote_script(config, workers=workers, group_size=group_size)
        upload_remote_script(target, config, script)
        pid = launch_remote_script(target, config)
        state["run"] = {"id": config.run_id, "status": "training_running", "pid": pid}
        update_state(
            state, status="training_running", remote_pid=pid, launched_at=utc_now()
        )
        print(f"Vast CPU benchmark launched: {config.task_name}")
        print(f"State: {state_path(config.task_name)}")
        return 0
    except Exception as exc:  # preserve state for recover/cleanup
        if state is not None:
            update_state(state, status="failed_kept", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1


def handle_status(args: argparse.Namespace) -> int:
    state = load_state(args.task_name)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def handle_recover(args: argparse.Namespace) -> int:
    state = load_state(args.task_name)
    copied = recover_outputs(state, required=False)
    print(
        json.dumps(
            {"task_name": args.task_name, "recovered": [str(path) for path in copied]},
            sort_keys=True,
        )
    )
    return 0


def handle_cleanup(args: argparse.Namespace) -> int:
    state = load_state(args.task_name)
    # Recovery is intentionally first and never implies destruction.
    try:
        recover_outputs(state, required=False)
    except Exception as exc:
        print(f"Recovery warning: {exc}", file=sys.stderr)
    command = f"vastai destroy instance {state.get('instance_id')} --yes"
    if not (args.destroy and args.yes):
        print(f"Destroy command: {command}")
        print("Not destroying without --destroy --yes")
        return 0
    return 0 if destroy_instance(state) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable full-bbox routing benchmark on a Vast.ai CPU host"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="allocate a CPU host and launch a detached benchmark"
    )
    run.add_argument("task_name")
    run.add_argument("--artifact-s3-bucket", default=None)
    run.add_argument("--artifact-s3-prefix", default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--run-id")
    run.add_argument("--s3-bucket")
    run.add_argument("--s3-prefix")
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
    run.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR)
    run.add_argument(
        "--corpus", default="scripts/routing/production_benchmark_pairs.json"
    )
    run.add_argument(
        "--graph",
        default="data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3",
    )
    run.add_argument(
        "--report",
        default="data/processed/heuristic_runs/prompt_two_candidate_exp02_fresh_test20_20260810/report/report.json",
    )
    run.add_argument(
        "--output",
        default="data/processed/routing_benchmarks/production_artifact_benchmark.json",
    )
    run.add_argument("--workers", type=int)
    run.add_argument("--group-size", type=int)
    run.add_argument("--per-worker-ram-mb", type=int, default=DEFAULT_PER_WORKER_RAM_MB)
    run.add_argument("--reserved-ram-mb", type=int, default=DEFAULT_RESERVED_RAM_MB)
    run.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    run.add_argument(
        "--checkpoint-interval-seconds",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    )
    run.add_argument("--case-timeout-seconds", type=float, default=10.0)
    run.add_argument("--strict-service-full", action="store_true")
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--destroy", action="store_true")
    run.set_defaults(func=handle_run)
    for name, handler, help_text in (
        ("status", handle_status, "print persisted state"),
        ("recover", handle_recover, "copy remote checkpoint/log/final outputs"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("task_name")
        command.set_defaults(func=handler)
    cleanup = sub.add_parser(
        "cleanup", help="recover and optionally destroy the CPU host"
    )
    cleanup.add_argument("task_name")
    cleanup.add_argument("--destroy", action="store_true")
    cleanup.add_argument("--yes", action="store_true")
    cleanup.set_defaults(func=handle_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
