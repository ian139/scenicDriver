# Scenic Route Planner

## Overview

An ML-driven system that scores scenic beauty from satellite imagery and terrain data, then uses those scores for route planning.

This project combines:
- **Satellite imagery** (Mapbox) for visual features (vegetation, water, land use)
- **Terrain-RGB tiles** (Mapbox) for elevation, slope, and relief analysis
- **RESISC-45 classifier** for land-use classification
- **Heuristic scoring** that combines all features into a scenic score (0-10)

Current direction:
- Keep classifier signal, but move away from fixed manual class weights.
- Train a learned scenic regressor using satellite embeddings + terrain features.
- Retrain/expand classifier data to improve domain fit (especially Northeast).
- Add a manual scenic annotation set and use it as the primary quality benchmark.
- Prioritize shipping a usable MVP trip-planning app first; continue model improvements in parallel.

## Goals

- Score scenic beauty from satellite/terrain tiles (0-10).
- Build regional heatmaps + reports for QA.
- Prepare routing inputs (road graph + scenic edge cost).

## Current focus

`v6` is the active learned checkpoint for Northeast runs:

- `models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt`
- Source of truth: `data/processed/regression/model_registry.json`

Current focus is MVP app build-out around the New England North web UI, with routing/system hardening underneath.

The web MVP includes:
- New England North scenic heatmap
- route comparison and scenic controls
- active remote-training evaluation

## Tech stack

- ML: PyTorch, timm, numpy, pandas
- Geo: rasterio (optional), shapely (optional), osmnx (optional for routing)
- Tooling: uv, marimo
- Viewer/Web: static MapLibre app under `apps/new_england_north/`
- API: FastAPI under `src/app_api/`

## Workflow

1. Download tiles (Mapbox) for a bbox at fixed zoom.
2. Generate heuristic labels + report.
3. Add manual scenic labels for a stratified subset (benchmark + calibration).
4. Export regression features (satellite embeddings + terrain + logits).
5. Train/evaluate regressor on human-only and mixed-label datasets.
6. Build a road graph and score edges with scenic tiles.

Use grouped workflow CLIs under `scripts/annotation/`, `scripts/ingest/`, `scripts/modeling/`, `scripts/reports/`, and `scripts/routing/`. Use marimo notebooks under `notebooks/` for training, scoring, and annotation workflows.

## Quick start

```bash
uv sync
export MAPBOX_ACCESS_TOKEN=<your-token>
uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload
cd apps/new_england_north && python3 -m http.server 3000
uv run marimo edit notebooks/train.mo.py
```

Open `http://localhost:3000` for the web MVP after the API is running on `:8080`.

### Beta deployment

The beta runs as a single public Nginx origin backed by an internal FastAPI
service. Processed route/heatmap artifacts and model weights are deployment
prerequisites; they remain outside Git and Docker image layers and are mounted
read-only from `data/processed/` and `models/`.

```bash
cp .env.beta.example .env.beta
# Set MAPBOX_ACCESS_TOKEN in the untracked .env.beta, and optionally change SCENIC_WEB_PORT.
docker compose --env-file .env.beta -f compose.beta.yml up --build
```

Open `http://localhost:${SCENIC_WEB_PORT:-80}`. Stop the beta with:

```bash
docker compose --env-file .env.beta -f compose.beta.yml down
```

Before startup, sync the canonical ignored New England graph, learned run,
registry, and active registry checkpoint into their documented
`data/processed/` and `models/` paths.

## Common workflows

### Ingest Mapbox tiles

```bash
uv run python scripts/ingest/download_bbox_tiles.py \
  --min-lat 40.28 --min-lon -105.70 \
  --max-lat 40.35 --max-lon -105.58 \
  --zoom 16 --style mapbox.satellite \
  --output data/raw/images/satellite

uv run python scripts/ingest/download_bbox_tiles.py \
  --min-lat 40.28 --min-lon -105.70 \
  --max-lat 40.35 --max-lon -105.58 \
  --zoom 16 --style mapbox.terrain-rgb \
  --output data/raw/images/terrain
```

### Generate reports

```bash
uv run python scripts/reports/heuristic_report.py \
  --run-name masswhites_z14_learned_h4_v2 \
  --scoring learned \
  --regression-ckpt models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt \
  --satellite-dir data/raw/images/satellite/z14/masswhites \
  --terrain-dir data/raw/images/terrain/z14/masswhites \
  --max-tiles 5000 \
  --s3-only
```

### Annotate scenic labels

Notebook option:

```bash
uv run marimo edit notebooks/annotate_scenic.mo.py
```

Browser option:

```bash
uv run python scripts/annotation/annotate_scenic_web.py \
  --labels-csv data/processed/heuristic_runs/masswhites_z14_flat_5k_seamfix/labels.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --annotations-csv data/raw/labels_human.csv \
  --sample-size 500 \
  --stratify-by-class
```

Benchmark and overlap batches:

```bash
uv run python scripts/annotation/build_human_benchmark.py \
  --annotations-csv data/raw/labels_human.csv \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --output-dir data/processed/regression \
  --run-name masswhites_human_benchmark_v2 \
  --val-frac 0.2 --test-frac 0.2 --seed 42

uv run python scripts/annotation/build_overlap_batch.py \
  --annotations-csv data/raw/labels_human.csv \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --source-annotator ian \
  --target-annotator paperspace \
  --sample-size 200 \
  --seed 42 \
  --output-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv
```

### Train and evaluate learned scoring

```bash
uv run python scripts/modeling/export_regression_dataset.py \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --output data/processed/regression/features_masswhites_z14_mixed5000.npz \
  --label-source-column label_source \
  --human-weight 4.0 \
  --heuristic-weight 1.0

uv run python scripts/modeling/train_regression_baseline.py \
  --dataset data/processed/regression/features_masswhites_z14_mixed5000.npz \
  --output models/scenic_regression_baseline_masswhites_z14_mixed5000.pt \
  --use-sample-weights

uv run python scripts/modeling/evaluate_regression_baseline.py \
  --dataset data/processed/regression/features_masswhites_z14_mixed5000.npz \
  --checkpoint models/scenic_regression_baseline_masswhites_z14_mixed5000.pt \
  --metrics-json data/processed/regression/baseline_metrics_masswhites_z14_mixed5000.json
```

### Build and compare routes

Build a road graph from an OSM bbox:

```bash
uv run --with osmnx python scripts/routing/build_graph_from_osm.py \
  --min-lat 42.35 --min-lon -72.57 \
  --max-lat 42.39 --max-lon -72.52 \
  --run-name amherst_core
```

This writes:
- `data/processed/road_graphs/amherst_core/road_graph.json`
- `data/processed/road_graphs/amherst_core/run.json`

Generate scenic vs baseline route overlay:

```bash
uv run python scripts/routing/route_compare_service.py \
  --start 42.40 -72.70 \
  --end 42.48 -72.62 \
  --scenic-weight 0.8 \
  --run-name masswhites_z14_learned_h4_v2 \
  --graph-geojson data/processed/sample_road_graph.geojson
```

The report viewer auto-loads:
- `route.geojson` (single file with scenic + baseline features), or
- fallback pair `route_scenic.geojson` and `route_fast.geojson`.

Route comparison metrics (distance/time/scenic deltas) are rendered in-map when route overlay data is present.


### Vast.ai cloud GPU workflow

The canonical Dockerfile for GPU runs is `Dockerfile.remote-training` (the root
`Dockerfile` is a convenience symlink). Use `-f Dockerfile.remote-training` in
scripted Docker/Vast commands. The image is only the reusable environment; S3
holds data and model weights. Image publishing remains separate from GPU rental:
training pulls `ian139/scenicdriver-remote-training:latest`. The provisioning
script prepares a temporary Vast.ai instance for one short validation run, then
stops at a manual gate before any long job. `scripts/remote/vast-train.sh`
reuses that gate, runs the regression trainer, syncs artifacts, and destroys the
instance by default.

Head orchestrator and subagent model:

```bash
# Head orchestrator terminal
omp --advisor --model "openai-codex/gpt-5.5" --thinking xhigh

# Planning/execution subagents
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

Start the cost-controlled training lifecycle from the head orchestrator:

```bash
/goal vast-auto-training-lifecycle
# Objective: start Vast, validate S3/GPU smoke, train regression model, sync outputs, destroy instance
```

Local image build and smoke:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.remote-training \
  -t scenicdriver/remote-training:vast-smoke .

docker run --rm --platform linux/amd64 scenicdriver/remote-training:vast-smoke \
  python scripts/remote/container_smoke.py --check-imports --device cpu

# On a CUDA host:
docker run --rm --gpus all scenicdriver/remote-training:vast-smoke \
  python scripts/remote/container_smoke.py --check-imports --device cuda
```

Tag, push, and prove the registry pull:

```bash
docker tag scenicdriver/remote-training:vast-smoke \
  ian139/scenicdriver-remote-training:latest
docker push ian139/scenicdriver-remote-training:latest
docker pull --platform linux/amd64 ian139/scenicdriver-remote-training:latest
```

Vast.ai launch requirements:

- One NVIDIA GPU with Docker GPU runtime (`--gpus all`) and driver compatible
  with CUDA 12.4.
- At least 64 GB disk for the image, downloaded data/model prefixes, and output
  artifacts.
- AWS credentials supplied at runtime only, usually via `/root/.scenic/aws.env`
  containing `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`,
  and `SCENIC_S3_BUCKET`. The container uses `boto3` for S3 if the AWS CLI is
  absent, so the Docker image does not need to carry the extra AWS CLI layer.
- No long job starts until the smoke and minimal inference steps below pass.

For the fast validation run, use tiny S3 smoke prefixes instead of broad
production prefixes. Example:

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_RUN_ID=vast-smoke-$(date -u +%Y%m%dT%H%M%SZ)
export SCENIC_S3_DATA_PREFIX=processed/regression/vast-smoke/$SCENIC_RUN_ID/
export SCENIC_S3_MODELS_PREFIX=models/vast-smoke/$SCENIC_RUN_ID/
export SCENIC_S3_OUTPUT_PREFIX=outputs/vast/$SCENIC_RUN_ID/
aws s3 cp /path/to/tiny_features.npz "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_DATA_PREFIX"
aws s3 cp /path/to/tiny_regression.pt "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_MODELS_PREFIX"
```

Example Vast search/start and SSH preflight:

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

Pull and run the prebuilt image on the Vast host:

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
The provisioning script runs these required S3/GPU/inference gates in order
(shown with AWS CLI syntax; the script falls back to the repo's boto3 S3 helper
inside the container):

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
aws s3 sync scenic_artifacts/vast/<run-id>/ "s3://$SCENIC_S3_BUCKET/$SCENIC_S3_OUTPUT_PREFIX"
```

Validation checklist:

- [ ] Docker image builds with `Dockerfile.remote-training`.
- [ ] Image pulls from Docker Hub on the Vast host.
- [ ] `scripts/remote/provision_vast.sh` pulls required data and model prefixes
      from S3; missing prefixes fail unless `SCENIC_ALLOW_MISSING_ARTIFACTS=1`.
- [ ] `nvidia-smi` succeeds on the host and inside the container path.
- [ ] `container_smoke.py --device cuda --check-imports` succeeds.
- [ ] `minimal_inference.py` writes `inference_result.json` quickly.
- [ ] `aws s3 sync` uploads the output directory back to S3.
- [ ] No long training command starts before every item above passes.

Automatic train-and-close path:

```bash
scripts/remote/vast-train.sh run <task-name> \
  --train-dataset-key <key> \
  --epochs 1 \
  --batch-size 64

scripts/remote/vast-train.sh status <task-name>

scripts/remote/vast-train.sh cleanup <task-name> --copy-artifacts --destroy --yes
```

Use `--epochs 1 --batch-size 64` for a cost-controlled smoke train. Omit those
overrides for the real default training path. If the local orchestrator dies,
rerun the cleanup command; it uses the recorded `.cmux-vast/state/<task-name>.json`
instance id and remote sentinel paths.

Monitoring and cost controls:

```bash
docker logs -f scenic-vast-validate
watch -n 30 nvidia-smi
vastai show instance <instance-id> --raw
```

If a command stalls or a smoke step fails, capture the exact command and last
log lines, then stop the instance:

```bash
docker logs scenic-vast-validate --tail 200
docker rm -f scenic-vast-validate || true
vastai destroy instance <instance-id> --yes
```

For CMUX-managed Vast hosts/tasks, use the repo wrappers:

```bash
scripts/remote/vast-start-task.sh scenic-vast-smoke 'Validate prebuilt image and S3-backed smoke run.' \
  --agent none \
  --allocation-attempts 3 \
  --disk-gb 64 \
  --image ian139/scenicdriver-remote-training:latest \
  --timeout-seconds 1800

scripts/remote/vast-watch.sh --interval-seconds 60 --destroy --yes
scripts/remote/vast-down.sh scenic-vast-smoke --copy-artifacts --destroy --yes
```

The `vast-start-task.sh` wrapper allocates/bootstrap-checks the Vast host and
prints the documented `cmux new-workspace --name ... --cwd ... --focus false`
command. It does not execute CMUX workspace creation; run that command only
after the checkout path is ready. `vast-watch.sh` observes `cmux workspace list
--json` before collecting artifacts or destroying the host.

## Repository layout

```text
apps/
  new_england_north/           # canonical web UI
docs/
  setup/aws-s3.md
  architecture/infrastructure.md
  research/ml-research-log.md
  roadmap.md
  internal/cmux-workflow.md
scripts/
  annotation/
  ingest/
  modeling/
  reports/
  routing/
archive/
  archive.md
  archive_scan_summary.json
  scripts/
  notebooks/
  notes/
```

## Documentation map

- [`data/README.md`](data/README.md)
- [`docs/setup/aws-s3.md`](docs/setup/aws-s3.md)
- [`docs/research/ml-research-log.md`](docs/research/ml-research-log.md)
- [`docs/architecture/infrastructure.md`](docs/architecture/infrastructure.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`archive/archive.md`](archive/archive.md)
- [`docs/internal/cmux-workflow.md`](docs/internal/cmux-workflow.md)

## Current boundaries

- No native mobile app implementation (responsive web first).
- No full production-scale platform hardening/SRE rollout yet.
- No full-US NAIP processing pipeline.

## Requirements

- Python 3.11+
- `uv` package manager
- Mapbox access token (free tier works)
- GPU recommended for training

## License

Private repository - not for distribution.
