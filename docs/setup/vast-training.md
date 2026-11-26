# Vast.ai Remote Training

This runbook covers the operator-controlled GPU lifecycle for Scenic regression
training. It deliberately separates the reusable container image from data and
model artifacts: the image contains the runtime and repository code, while S3
holds input data, checkpoints, and outputs.

The canonical GPU image definition is [`Dockerfile.remote-training`](../../Dockerfile.remote-training).
Use `-f Dockerfile.remote-training` in scripted Docker/Vast commands. The
release selected by `vast_train.py`'s `DEFAULT_IMAGE` is
`ian139/scenicdriver-remote-training:torch2.10.0-cu128-20261006`.
This is a **publish-once** release tag, not a claim that it is already published.
Build and publish it below before renting; never overwrite a released tag.
For registry-enforced immutability, pass the pushed `repo@sha256:...` digest via
`--image`. Untagged images and `:latest` are rejected.

> **Safety gate:** The provisioning path performs S3, GPU, CUDA-import, and
> minimal-inference checks before it reaches a manual stop point. Do not start
> a long training job until every validation gate passes. Keep credentials out
> of image layers, shell history, and Git.

## Prerequisites

- A Docker host with one NVIDIA GPU, the Docker GPU runtime (`--gpus all`), and
  drivers compatible with the CUDA 12.8 base image.
- At least 64 GB of disk for the image, downloaded S3 prefixes, and output
  artifacts.
- Vast.ai CLI access and an SSH key (the examples use `~/.ssh/id_ed25519`).
- AWS credentials supplied at runtime. The container accepts an env file such
  as `/root/.scenic/aws.env` containing `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and `SCENIC_S3_BUCKET`.
  Follow [`aws-s3.md`](aws-s3.md) for bucket and credential setup. Container
  provisioning checks identity and bucket access with `boto3`, then uses the
  repository's `python -m scenicdriver.data_pipeline.s3` helper for every S3 prefix
  check and transfer.
- `uv` and the checked-out repository for the state-backed wrapper path.

The container does not need an AWS CLI layer: `boto3` and the repository S3
helper are the only S3 interfaces used by its provisioning path. Never bake
credentials into the image.

## Build, smoke-test, and publish the image

Build the canonical image for an amd64 Vast host:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.remote-training \
  -t scenicdriver/remote-training:vast-smoke .
```

Run canonical validation locally. Use CPU on a non-CUDA host and CUDA on a host
exposing an NVIDIA GPU. The image ships no checkpoint or dataset, so download
both inputs first and mount them into the container:

```bash
uv run python -m scenicdriver.data_pipeline.s3 download-file \
  --bucket "$SCENIC_S3_BUCKET" \
  --key <checkpoint-key> \
  --dest models/<checkpoint>.pt \
  --required
uv run python -m scenicdriver.data_pipeline.s3 download-file \
  --bucket "$SCENIC_S3_BUCKET" \
  --key <features-key> \
  --dest data/processed/regression/<features>.npz \
  --required

docker run --rm --platform linux/amd64 \
  -v "$PWD/models:/workspace/models:ro" \
  -v "$PWD/data/processed/regression:/workspace/data/processed/regression:ro" \
  scenicdriver/remote-training:vast-smoke \
  python scripts/remote/vast_train.py validate --device cpu \
  --checkpoint models/<checkpoint>.pt --dataset data/processed/regression/<features>.npz

docker run --rm --gpus all \
  -v "$PWD/models:/workspace/models:ro" \
  -v "$PWD/data/processed/regression:/workspace/data/processed/regression:ro" \
  scenicdriver/remote-training:vast-smoke \
  python scripts/remote/vast_train.py validate --device cuda \
  --checkpoint models/<checkpoint>.pt --dataset data/processed/regression/<features>.npz
```

Publishing is separate from GPU rental. The amd64 base is digest-pinned to
PyTorch 2.10.0 / torchvision 0.25.0 / CUDA 12.8, matching the local lock's
torch/torchvision versions. Dependency installation resolves against these
constraints rather than silently keeping the older 2.5.1 runtime.
Tag, push, and prove that the registry serves the image before allocating:

```bash
export SCENIC_RUNTIME_IMAGE="$(uv run python -c 'from scripts.remote.vast_train import DEFAULT_IMAGE; print(DEFAULT_IMAGE)')"
docker tag scenicdriver/remote-training:vast-smoke "$SCENIC_RUNTIME_IMAGE"
docker push "$SCENIC_RUNTIME_IMAGE"
docker pull --platform linux/amd64 "$SCENIC_RUNTIME_IMAGE"
docker image inspect "$SCENIC_RUNTIME_IMAGE" --format '{{index .RepoDigests 0}}'
```

Use the printed digest as `vast_train.py run ... --image <repo@sha256:digest>`
when a registry cannot enforce publish-once tags. A changed source tree or
dependency set requires a new release tag and an updated `DEFAULT_IMAGE`.
GPU compatibility repair installs only `torch==2.10.0` and
`torchvision==0.25.0` from the CUDA 12.8 wheel index. Unavailable wheels or an
unsupported GPU fail the run; no unversioned upgrade is attempted.

## Short validation run

Use tiny S3 smoke prefixes for validation rather than broad production
prefixes. Set a unique run id and upload a small feature dataset and checkpoint
before starting the host:

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_RUN_ID=vast-smoke-$(date -u +%Y%m%dT%H%M%SZ)
export SCENIC_S3_DATA_PREFIX=processed/regression/vast-smoke/$SCENIC_RUN_ID/
export SCENIC_S3_MODELS_PREFIX=models/vast-smoke/$SCENIC_RUN_ID/
export SCENIC_S3_OUTPUT_PREFIX=outputs/vast/$SCENIC_RUN_ID/

uv run python -m scenicdriver.data_pipeline.s3 upload-prefix \
  --src /path/to/tiny_features.npz \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_DATA_PREFIX" \
  --required
uv run python -m scenicdriver.data_pipeline.s3 upload-prefix \
  --src /path/to/tiny_regression.pt \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_MODELS_PREFIX" \
  --required
```

Search for a suitable offer, start a temporary instance, and verify SSH and GPU
visibility before pulling the image:

```bash
vastai search offers 'num_gpus>=1 dlperf>250 dlperf_per_dphtotal>100 total_flops>200 dph<8 verified=true direct_port_count>=1 rentable=true gpu_ram>=12 disk_space>=64' \
  -o 'dlperf_usd-' --raw
vastai create instance <offer-id> \
  --image nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  --disk 64 --ssh --direct --raw
vastai attach ssh <instance-id> "$(ssh-keygen -y -f ~/.ssh/id_ed25519)"
vastai ssh-url <instance-id>
ssh -i ~/.ssh/id_ed25519 -p <ssh-port> root@<ssh-host> nvidia-smi
```

Pull the published image and run the repository provisioning script. Copy the
AWS env file to the host out of band; do not put it in the image or command
arguments:

```bash
ssh -i ~/.ssh/id_ed25519 -p <ssh-port> root@<ssh-host>
# In the remote shell, select the release just published (or its registry digest).
export SCENIC_RUNTIME_IMAGE=ian139/scenicdriver-remote-training:torch2.10.0-cu128-20261006
docker pull "$SCENIC_RUNTIME_IMAGE"
mkdir -p /workspace/scenic-data /workspace/scenic-models /workspace/scenic-artifacts /root/.scenic
# Copy /root/.scenic/aws.env out-of-band; do not bake credentials into the image.

# Exact S3 keys chosen for this run (the same values passed to
# `vast_train.py run --train-dataset-key/--validation-checkpoint-key`).
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_TRAIN_DATASET_KEY='<features-key>'
export SCENIC_VALIDATION_CHECKPOINT_KEY='<checkpoint-key>'

docker run --rm \
  --env-file /root/.scenic/aws.env \
  -v /workspace/scenic-data:/workspace/data/processed/regression \
  -v /workspace/scenic-models:/workspace/models \
  "$SCENIC_RUNTIME_IMAGE" \
  python -m scenicdriver.data_pipeline.s3 download-file \
  --bucket "$SCENIC_S3_BUCKET" \
  --key "$SCENIC_VALIDATION_CHECKPOINT_KEY" \
  --dest models/validation_checkpoint.pt \
  --required
docker run --rm \
  --env-file /root/.scenic/aws.env \
  -v /workspace/scenic-data:/workspace/data/processed/regression \
  -v /workspace/scenic-models:/workspace/models \
  "$SCENIC_RUNTIME_IMAGE" \
  python -m scenicdriver.data_pipeline.s3 download-file \
  --bucket "$SCENIC_S3_BUCKET" \
  --key "$SCENIC_TRAIN_DATASET_KEY" \
  --dest data/processed/regression/validation_features.npz \
  --required

docker run --rm --gpus all --name scenic-vast-validate \
  -v /workspace/scenic-data:/workspace/data/processed/regression \
  -v /workspace/scenic-models:/workspace/models \
  -v /workspace/scenic-artifacts:/workspace/scenic_artifacts \
  "$SCENIC_RUNTIME_IMAGE" \
  python scripts/remote/vast_train.py validate --device cuda \
  --checkpoint models/validation_checkpoint.pt \
  --dataset data/processed/regression/validation_features.npz \
  --output scenic_artifacts/validation_result.json
```

### Provisioning gates

The canonical `vast_train.py validate` command performs CUDA and input checks;
S3 transfers use the repository helper for every prefix check and transfer.

```bash
python - <<'PY'
import boto3
boto3.client("sts").get_caller_identity()
PY
python - <<'PY'
import os
import boto3
boto3.client("s3").head_bucket(Bucket=os.environ["SCENIC_S3_BUCKET"])
PY
python -m scenicdriver.data_pipeline.s3 check-prefix \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_DATA_PREFIX" \
  --required
python -m scenicdriver.data_pipeline.s3 check-prefix \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_MODELS_PREFIX" \
  --required
python -m scenicdriver.data_pipeline.s3 download-prefix \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_DATA_PREFIX" \
  --dest data/processed/regression \
  --required
python -m scenicdriver.data_pipeline.s3 download-prefix \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_MODELS_PREFIX" \
  --dest models \
  --required
nvidia-smi
python scripts/remote/vast_train.py validate --device cuda \
  --checkpoint models/<checkpoint>.pt \
  --dataset data/processed/regression/<features>.npz \
  --output scenic_artifacts/vast/<run-id>/inference_result.json
python -m scenicdriver.data_pipeline.s3 upload-prefix \
  --src scenic_artifacts/vast/<run-id>/ \
  --bucket "$SCENIC_S3_BUCKET" \
  --prefix "$SCENIC_S3_OUTPUT_PREFIX" \
  --required
```

The script fails closed when required data/model prefixes or local inference
inputs are missing. `SCENIC_ALLOW_MISSING_ARTIFACTS=1` is an explicit
smoke-only fallback and must not be used to authorize a real training run. On
success it prints a manual stop point and does **not** start long training.

Validation checklist:

- [ ] The image builds with `Dockerfile.remote-training`.
- [ ] The image pulls from Docker Hub on the Vast host.
- [ ] `vast_train.py validate --device cuda` succeeds.
- [ ] `python -m scenicdriver.data_pipeline.s3 upload-prefix` uploads the output
      directory to the output prefix.
- [ ] No long training command starts before every item above passes.

## Large CPU route and graph runs

OSM graph conversion and the production routing benchmark are CPU/memory
workloads; they do not need a GPU. Use a verified Vast CPU offer with at least
32 GB RAM (64 GB preferred), four vCPUs, 64 GB disk, and direct SSH access.
The local preload measured about 5.8 GB peak RSS, but live isolated workers on
the 64 GB Vast host reached roughly 15–20 GB each. The wrapper therefore
reserves 24,576 MiB per worker plus 8,192 MiB for the OS and parent process.

The state-backed runner derives workers as:

```text
min(nproc, floor((MemTotal_MiB - 8192) / 24576), 32)
```

Set `--workers` explicitly when a smaller, predictable footprint is preferred.
`--group-size` bounds the number of cases submitted in each checkpoint window.
The graph, learned report, corpus, and output defaults are the canonical
full-bbox paths; override them only for a deliberate artifact variant.

First verify the generated remote script without allocating a host:

```bash
uv run python scripts/remote/vast_route_benchmark.py run full-bbox-v1 \
  --dry-run \
  --s3-bucket "$SCENIC_S3_BUCKET" \
  --s3-prefix "$SCENIC_S3_PREFIX" |
  bash -n
```

Create `.secrets/aws.env` outside Git with the remote S3 credentials, then
launch the resumable run:

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_S3_PREFIX=outputs/vast/new-england-north-full-bbox-v1
chmod 600 .secrets/aws.env

uv run python scripts/remote/vast_route_benchmark.py run full-bbox-v1 \
  --s3-bucket "$SCENIC_S3_BUCKET" \
  --s3-prefix "$SCENIC_S3_PREFIX" \
  --local-secrets-env-file .secrets/aws.env \
  --workers 2 \
  --group-size 64
```

The runner records the Vast instance, SSH endpoint, remote PID, resource
probe, checkpoint path, and S3 keys in
`.cmux-vast/state/full-bbox-v1.json`. It snapshots the JSONL checkpoint before
each S3 upload and resumes only when the stored checkpoint fingerprint matches
the graph, report, corpus, timeout, worker, and grouping configuration. The
final JSON is uploaded only when every planned case has a unique persisted row
and a consistent fingerprint.

Monitor or recover without destroying the host:

```bash
uv run python scripts/remote/vast_route_benchmark.py status full-bbox-v1
uv run python scripts/remote/vast_route_benchmark.py recover full-bbox-v1
```

After validating the recovered checkpoint and final JSON, recover once more and
destroy the host:

```bash
uv run python scripts/remote/vast_route_benchmark.py cleanup full-bbox-v1 \
  --destroy --yes
```

On interruption, run `recover` before `cleanup`; cleanup always attempts
recovery first. Never reuse a task name with an incompatible graph, report,
corpus, worker, or group configuration.

## State-backed train-and-close lifecycle

For a normal run, use the repository wrapper. It allocates a host, copies the
runtime secret env file, validates the remote setup before training, copies
artifacts back, and destroys the instance by default. Set `SCENIC_S3_BUCKET` in
the environment or pass `--s3-bucket`; `--train-dataset-key` is the S3 object key
for the training dataset and `--benchmark-split-key` is the exact S3 object key
of the benchmark split CSV whose val/test identities are excluded from training.
Both are required and downloaded with `--required`, so a missing key fails the
run before training starts.

```bash
export SCENIC_S3_BUCKET=scenicdriver-data

uv run python scripts/remote/vast_train.py run <task-name> \
  --train-dataset-key <key> \
  --benchmark-split-key <split-csv-key> \
  --epochs 1 \
  --batch-size 64

uv run python scripts/remote/vast_train.py status <task-name>

uv run python scripts/remote/vast_train.py cleanup <task-name> --copy-artifacts --destroy --yes
```

`--epochs 1 --batch-size 64` is a cost-controlled smoke train. Omit those
overrides for the default training path. The wrapper defaults to the
published image, 64 GB disk, three allocation attempts, and a 30-minute
lifecycle timeout. The local secrets file defaults to `.secrets/aws.env`; keep
that file untracked. State is recorded at
`.cmux-vast/state/<task-name>.json` (including the instance id and remote
sentinel paths), so if the local orchestrator dies, rerun the cleanup command.
Use `--no-destroy` only when deliberately retaining the host for investigation;
otherwise failed runs are destroyed by default unless `--keep-on-failure` is
requested. One exception is never destroyed automatically: when the remote
training launch itself is unacknowledged (the call raising after the detached
command may already be running), the run records `launch_outcome: "unknown"`
in the state file and keeps the host regardless of `--no-destroy` /
`--keep-on-failure`, because training may be running on it. Recover or destroy
that host only via the explicit `cleanup ... --copy-artifacts --destroy --yes`
command.

### Runtime identity and deadline

Each output bundle includes `runtime_identity.json`: `image` (the exact
reference passed to Vast), actual `torch_version`, `torchvision_version`,
`cuda_version`, and `source_digest`. The digest is SHA-256 of sorted,
length-delimited paths and bytes from `pyproject.toml`, `uv.lock`, `src/`,
`scripts/`, `notebooks/`, and `config/` inside the running image; Python caches
and host native binaries are excluded. Lifecycle sentinels and validation JSON
also carry this identity; the manifest identifies the accompanying checkpoint
and logs. Artifact recovery copies it into local run state as `runtime_identity`.
`image` is the submitted release tag or digest, **not** a fabricated resolved
registry digest; choose `--image repo@sha256:...` to record a registry digest.

`--timeout-seconds` (default 1800) also caps the **entire detached remote
training/upload script** with GNU `timeout --kill-after=30s`. State records
`remote_deadline_seconds`, `remote_kill_after_seconds`, and
`remote_deadline_at` (UTC launch-boundary deadline). Exported remote knobs are
`SCENIC_REMOTE_DEADLINE_SECONDS`, `SCENIC_TIMEOUT_MINUTES`, and
`SCENIC_STEP_TIMEOUT_SECONDS`; the outer deadline is authoritative. Timeout
terminates the whole process group independently of the local orchestrator.
A surviving supervisor records failure (exit 124, or 137 after forced kill),
leaving outputs available for cleanup/recovery.

The installed Vast CLI and [create-instance reference](https://docs.vast.ai/cli/reference/create-instance)
provide no instance-expiry option, so state records `provider_expiry: null`.
**The training deadline does not destroy the rental or end billing.** Retained
hosts (including ambiguous launches and failed artifact recovery) still require
explicit `cleanup ... --copy-artifacts --destroy --yes`; no watchdog discards
their only copy of artifacts. If the local orchestrator dies, training stops
after the deadline plus at most 30 seconds, but rental/storage charges continue
until you recover and destroy the instance.

## Monitoring and emergency cleanup

Watch the container, GPU, and Vast instance while a validation or training task
is active:

```bash
docker logs -f scenic-vast-validate
watch -n 30 nvidia-smi
vastai show instance <instance-id> --raw
```

If a command stalls or a smoke step fails, capture the exact command and the
last log lines, then stop the work and destroy the instance:

```bash
docker logs scenic-vast-validate --tail 200
docker rm -f scenic-vast-validate || true
vastai destroy instance <instance-id> --yes
```

For CMUX-managed hosts/tasks, use the repository wrappers. The start wrapper
allocates and bootstrap-checks the host, then creates and registers the CMUX
workspace. The watch/down wrappers use the recorded workspace identity before
collecting artifacts or destroying the host:

```bash
uv run python scripts/remote/cmux_vast_host.py start-task scenic-vast-smoke \
  --agent none \
  --allocation-attempts 3 \
  --disk-gb 64 \
  --image "$SCENIC_RUNTIME_IMAGE" \
  --timeout-seconds 1800

uv run python scripts/remote/cmux_vast_host.py watch --interval-seconds 60 --destroy --yes
uv run python scripts/remote/cmux_vast_host.py down scenic-vast-smoke --copy-artifacts --destroy --yes
```

The CMUX state file keeps the workspace reference and id tied to the unique
task identity. If workspace creation or registration fails, the task remains
`workspace_pending` with no recorded identity; rerun the same start command to
retry safely. Do not manually destroy a host before recovering the state when
the wrappers are managing it.

## Optional head orchestration

For repository workspace and orchestration conventions, see [`../internal/cmux-workflow.md`](../internal/cmux-workflow.md).

## Related documentation

- [`aws-s3.md`](aws-s3.md) — S3 bucket, credentials, and local synchronization
- [`../../scripts/remote/vast_train.py`](../../scripts/remote/vast_train.py) — state-backed lifecycle implementation
- [`../../scripts/remote/vast_route_benchmark.py`](../../scripts/remote/vast_route_benchmark.py) — resumable full-bbox CPU benchmark lifecycle
