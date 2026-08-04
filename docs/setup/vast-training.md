# Vast.ai Remote Training

This runbook covers the operator-controlled GPU lifecycle for Scenic regression
training. It deliberately separates the reusable container image from data and
model artifacts: the image contains the runtime and repository code, while S3
holds input data, checkpoints, and outputs.

The canonical GPU image definition is [`Dockerfile.remote-training`](../../Dockerfile.remote-training).
Use `-f Dockerfile.remote-training` in scripted Docker/Vast commands. The
published image used by the lifecycle scripts is
`ian139/scenicdriver-remote-training:latest`.

> **Safety gate:** The provisioning path performs S3, GPU, CUDA-import, and
> minimal-inference checks before it reaches a manual stop point. Do not start
> a long training job until every validation gate passes. Keep credentials out
> of image layers, shell history, and Git.

## Prerequisites

- A Docker host with one NVIDIA GPU, the Docker GPU runtime (`--gpus all`), and
  drivers compatible with the CUDA 12.4 base image.
- At least 64 GB of disk for the image, downloaded S3 prefixes, and output
  artifacts.
- Vast.ai CLI access and an SSH key (the examples use `~/.ssh/id_ed25519`).
- AWS credentials supplied at runtime. The container accepts an env file such
  as `/root/.scenic/aws.env` containing `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and `SCENIC_S3_BUCKET`.
  Follow [`aws-s3.md`](aws-s3.md) for bucket and credential setup. The
  provisioning script uses the AWS CLI when available and falls back to the
  repository's `boto3` S3 helper.
- `uv` and the checked-out repository for the state-backed wrapper path.

The image does not need an extra AWS CLI layer when the `boto3` fallback is
available. Never bake credentials into the image.

## Build, smoke-test, and publish the image

Build the canonical image for an amd64 Vast host:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.remote-training \
  -t scenicdriver/remote-training:vast-smoke .
```

Run the import/device smoke locally. Use the CPU check on a non-CUDA host and
the CUDA check on a host exposing an NVIDIA GPU:

```bash
docker run --rm --platform linux/amd64 scenicdriver/remote-training:vast-smoke \
  python scripts/remote/container_smoke.py --check-imports --device cpu

# On a CUDA host:
docker run --rm --gpus all scenicdriver/remote-training:vast-smoke \
  python scripts/remote/container_smoke.py --check-imports --device cuda
```

Publishing is separate from GPU rental. Tag, push, and prove that the registry
can serve the image before allocating a Vast instance:

```bash
docker tag scenicdriver/remote-training:vast-smoke \
  ian139/scenicdriver-remote-training:latest
docker push ian139/scenicdriver-remote-training:latest
docker pull --platform linux/amd64 ian139/scenicdriver-remote-training:latest
```

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

aws s3 cp /path/to/tiny_features.npz "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_DATA_PREFIX"
aws s3 cp /path/to/tiny_regression.pt "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_MODELS_PREFIX"
```

Search for a suitable offer, start a temporary instance, and verify SSH and GPU
visibility before pulling the image:

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1 rentable=true' \
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
docker pull ian139/scenicdriver-remote-training:latest
mkdir -p /workspace/scenic-data /workspace/scenic-models /workspace/scenic-artifacts /root/.scenic
# Copy /root/.scenic/aws.env out-of-band; do not bake credentials into the image.

docker run --gpus all --name scenic-vast-validate \
  --env-file /root/.scenic/aws.env \
  -e SCENIC_S3_BUCKET=scenicdriver-data \
  -e SCENIC_S3_DATA_PREFIX="$SCENIC_S3_DATA_PREFIX" \
  -e SCENIC_S3_MODELS_PREFIX="$SCENIC_S3_MODELS_PREFIX" \
  -e SCENIC_S3_OUTPUT_PREFIX="$SCENIC_S3_OUTPUT_PREFIX" \
  -e SCENIC_TIMEOUT_MINUTES=30 \
  -v /workspace/scenic-data:/workspace/data/processed/regression \
  -v /workspace/scenic-models:/workspace/models \
  -v /workspace/scenic-artifacts:/workspace/scenic_artifacts \
  ian139/scenicdriver-remote-training:latest \
  bash scripts/remote/provision_vast.sh
```

### Provisioning gates

`scripts/remote/provision_vast.sh` runs these checks in order. The commands
below show the AWS CLI form; inside the container the script uses its `boto3`
fallback when the AWS CLI is absent:

```bash
aws sts get-caller-identity
aws s3api head-bucket --bucket "$SCENIC_S3_BUCKET"
aws s3 sync "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_DATA_PREFIX" data/processed/regression
aws s3 sync "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_MODELS_PREFIX" models
nvidia-smi
python scripts/remote/container_smoke.py --device cuda --check-imports --json
python scripts/remote/minimal_inference.py \
  --device cuda \
  --checkpoint models/<checkpoint>.pt \
  --dataset data/processed/regression/<features>.npz \
  --output scenic_artifacts/vast/<run-id>/inference_result.json
aws s3 sync scenic_artifacts/vast/<run-id>/ \
  "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_OUTPUT_PREFIX"
```

The script fails closed when required data/model prefixes or local inference
inputs are missing. `SCENIC_ALLOW_MISSING_ARTIFACTS=1` is an explicit
smoke-only fallback and must not be used to authorize a real training run. On
success it prints a manual stop point and does **not** start long training.

Validation checklist:

- [ ] The image builds with `Dockerfile.remote-training`.
- [ ] The image pulls from Docker Hub on the Vast host.
- [ ] `scripts/remote/provision_vast.sh` pulls required data and model prefixes
      from S3; missing prefixes fail unless `SCENIC_ALLOW_MISSING_ARTIFACTS=1`.
- [ ] `nvidia-smi` succeeds on the host and through the container path.
- [ ] `container_smoke.py --device cuda --check-imports` succeeds.
- [ ] `minimal_inference.py` writes `inference_result.json` quickly.
- [ ] `aws s3 sync` uploads the output directory to the output prefix.
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
for the training dataset.

```bash
export SCENIC_S3_BUCKET=scenicdriver-data

scripts/remote/vast-train.sh run <task-name> \
  --train-dataset-key <key> \
  --epochs 1 \
  --batch-size 64

scripts/remote/vast-train.sh status <task-name>

scripts/remote/vast-train.sh cleanup <task-name> --copy-artifacts --destroy --yes
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
requested.

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
scripts/remote/vast-start-task.sh scenic-vast-smoke \
  --agent none \
  --allocation-attempts 3 \
  --disk-gb 64 \
  --image ian139/scenicdriver-remote-training:latest \
  --timeout-seconds 1800

scripts/remote/vast-watch.sh --interval-seconds 60 --destroy --yes
scripts/remote/vast-down.sh scenic-vast-smoke --copy-artifacts --destroy --yes
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
- [`../../compose.remote-training.yml`](../../compose.remote-training.yml) — local GPU/container smoke service
- [`../../scripts/remote/provision_vast.sh`](../../scripts/remote/provision_vast.sh) — fail-fast validation script
- [`../../scripts/remote/vast_train.py`](../../scripts/remote/vast_train.py) — state-backed lifecycle implementation
- [`../../scripts/remote/vast_route_benchmark.py`](../../scripts/remote/vast_route_benchmark.py) — resumable full-bbox CPU benchmark lifecycle
