# Scenic Route Planner

## Overview

Scenic Route Planner scores scenic beauty from satellite imagery and terrain data, then uses those scores to compare and plan routes. The project combines Mapbox satellite and Terrain-RGB tiles, a RESISC-45 land-use classifier, learned scenic regression, and route-planning services.

### Current focus

The active product work is the New England North web MVP, with routing and system hardening underneath it. The viewer currently provides a scenic heatmap, route comparison controls, and a remote-training results panel.

The active learned checkpoint is recorded in [`data/processed/regression/model_registry.json`](data/processed/regression/model_registry.json):

`models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt`

Model weights, generated route/report data, datasets, and credentials are intentionally outside a normal source checkout. See the release boundary below and [`NEXT_STEPS.md`](NEXT_STEPS.md) for the artifact requirements.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- A Mapbox access token for map tiles and location search
- Node.js 20+ only when running the viewer test
- A CUDA-capable GPU is recommended for training; Docker is required for the container and beta workflows
- AWS credentials only for S3-backed data, artifact, or remote-training workflows

## Quick start (local source preview)

Install the locked development environment, then start the API and static viewer in separate terminals:

```bash
uv sync --frozen --extra dev

export MAPBOX_ACCESS_TOKEN=<your-token>
uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload
```

```bash
cd apps/new_england_north
python3 -m http.server 3000
```

Open <http://localhost:3000>. The viewer uses the local API at `http://localhost:8080` when served on port 3000. Check the API directly with:

```bash
curl http://localhost:8080/v1/healthz
```

For notebook-based training or annotation, run for example:

```bash
uv run marimo edit notebooks/train.mo.py
```

### App/API boundary

- `apps/new_england_north/` is a static MapLibre viewer; it has no separate frontend build step.
- `src/app_api/main.py` is the FastAPI service. Local development runs it on `:8080`; the hosted beta reverse-proxies it under `/api`.
- The API exposes health, region, heatmap, route-comparison, training-result, and search endpoints under `/v1/`. Mapbox-backed search needs `MAPBOX_ACCESS_TOKEN`.
- A fresh checkout can start the API and health/shell endpoints without ignored runtime artifacts. Learned heatmaps and route comparison require the generated artifacts and model weights described in [`NEXT_STEPS.md`](NEXT_STEPS.md).

### Hosted beta

The beta is a separate deployment path: Nginx serves the viewer and proxies an internal FastAPI service. It mounts processed artifacts and model weights read-only; those files are not copied into the image.

Before startup, bootstrap the canonical ignored artifacts and verify their checksums:

```bash
SCENIC_S3_BUCKET=scenicdriver-data SCENIC_S3_PREFIX=releases/routeOptimizer/75ee0431/ \
  uv run python scripts/deploy/bootstrap_beta_artifacts.py
SCENIC_S3_BUCKET=scenicdriver-data SCENIC_S3_PREFIX=releases/routeOptimizer/75ee0431/ \
  uv run python scripts/deploy/bootstrap_beta_artifacts.py --check-only
```

Keep S3 credentials and `.env.beta` outside Git. Then start the beta:

```bash
cp .env.beta.example .env.beta
# Set MAPBOX_ACCESS_TOKEN in the untracked .env.beta; optionally change SCENIC_WEB_PORT.
docker compose --env-file .env.beta -f compose.beta.yml up --build
```

Open `http://localhost:${SCENIC_WEB_PORT:-80}`. Stop it with:

```bash
docker compose --env-file .env.beta -f compose.beta.yml down
```

## Common workflows

The grouped command-line workflows live under `scripts/`; notebooks under `notebooks/` are useful for interactive training, scoring, and annotation.

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

See [`docs/setup/aws-s3.md`](docs/setup/aws-s3.md) for S3 credentials, bucket setup, and synchronization.

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

Notebook:

```bash
uv run marimo edit notebooks/annotate_scenic.mo.py
```

Browser-based annotation and benchmark preparation:

```bash
uv run python scripts/annotation/annotate_scenic_web.py \
  --labels-csv data/processed/heuristic_runs/masswhites_z14_flat_5k_seamfix/labels.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --annotations-csv data/raw/labels_human.csv \
  --sample-size 500 \
  --stratify-by-class

uv run python scripts/annotation/build_human_benchmark.py \
  --annotations-csv data/raw/labels_human.csv \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --output-dir data/processed/regression \
  --run-name masswhites_human_benchmark_v2 \
  --val-frac 0.2 --test-frac 0.2 --seed 42
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

Build a road graph from an OSM bounding box:

```bash
uv run --with osmnx python scripts/routing/build_graph_from_osm.py \
  --min-lat 42.35 --min-lon -72.57 \
  --max-lat 42.39 --max-lon -72.52 \
  --run-name amherst_core
```

This writes `data/processed/road_graphs/amherst_core/road_graph.json` and `run.json`. Generate a scenic-versus-baseline overlay:

```bash
uv run python scripts/routing/route_compare_service.py \
  --start 42.40 -72.70 \
  --end 42.48 -72.62 \
  --scenic-weight 0.8 \
  --run-name masswhites_z14_learned_h4_v2 \
  --graph-geojson data/processed/sample_road_graph.geojson
```

The viewer accepts `route.geojson`, or the fallback pair `route_scenic.geojson` and `route_fast.geojson`, and renders distance/time/scenic deltas when overlay data is available.

#### Canonical New England North full-bbox graph

The active New England North route artifact is the ignored SQLite graph
`data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3`.
It covers the configured bbox (`42.488301979602255..47.50235895196859`,
`-73.5205078125..-66.796875`). Source PBFs and OSMnx intermediates remain in
`cache/osm-pbf/new_england_north_full_bbox_v1/` and
`cache/osmnx/new_england_north_full_bbox_v1/`; generated graph and run metadata
remain under `data/processed/` and are not committed.

Build the canonical artifact from the dated, checksum-verified extracts:

```bash
uv run --with 'osmnx==2.1.0' python scripts/routing/build_graph_from_osm.py \
  --min-lat 42.488301979602255 --min-lon -73.5205078125 \
  --max-lat 47.50235895196859 --max-lon -66.796875 \
  --network drive --run-name new_england_north_full_bbox_v1 \
  --graph-format sqlite3 \
  --cache-folder cache/osmnx/new_england_north_full_bbox_v1 \
  --require-source-checksums \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/new-york-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/vermont-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/new-hampshire-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/maine-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/massachusetts-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/quebec-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/new-brunswick-260716.osm.pbf \
  --coverage-probe rutland_usps 43.60784414 -72.98226538 \
  --coverage-probe lisbon_police 44.02516775 -70.10003245 \
  --coverage-probe burlington 44.475884 -73.214003 \
  --coverage-probe bangor 44.801616 -68.771305
```

Validate before activation and after bootstrap:

```bash
uv run python scripts/routing/check_beta_artifacts.py --project-root .
```

For rollback, keep the prior corridor graph and manifest objects, restore the
previous graph/config paths, rerun the checker, and only then restart the beta
API. Do not delete the old artifact until the new deployment passes its smoke
probes.

### Remote GPU workflow

Use [`docs/setup/vast-training.md`](docs/setup/vast-training.md) for the complete Vast.ai runbook: canonical GPU image build/publish, S3 and GPU smoke gates, minimal inference, state-backed training, monitoring, artifact recovery, and destroy/cleanup commands. The reusable image is `Dockerfile.remote-training`; S3 remains the source for data and model weights. Do not start a long job until the validation gates pass.

## Repository layout

```text
apps/new_england_north/       # canonical static web UI
src/app_api/                  # FastAPI service
src/route_planner/            # route planning and scenic-cost logic
scripts/
  annotation/ ingest/ modeling/ reports/ routing/
  remote/                     # container and Vast.ai workflows
notebooks/                    # marimo workflows
config/                       # region and lifecycle configuration
data/                         # local/generated data (mostly ignored)
models/                       # local model weights (ignored except .gitkeep)
docs/                          # setup, architecture, research, and roadmap
archive/                       # curated historical material
```

## Documentation map

- [`data/README.md`](data/README.md) — data layout and provenance notes
- [`docs/setup/aws-s3.md`](docs/setup/aws-s3.md) — S3 setup and synchronization
- [`docs/setup/vast-training.md`](docs/setup/vast-training.md) — remote GPU lifecycle and safety gates
- [`docs/research/ml-research-log.md`](docs/research/ml-research-log.md) — model experiments
- [`docs/architecture/infrastructure.md`](docs/architecture/infrastructure.md) — system and artifact architecture
- [`docs/roadmap.md`](docs/roadmap.md) — planned work
- [`archive/archive.md`](archive/archive.md) — curated archive index
- [`docs/internal/cmux-workflow.md`](docs/internal/cmux-workflow.md) — internal CMUX workflow reference

## Current boundaries

- This is a private source preview, not an open-source distribution; the [`LICENSE`](LICENSE) reserves all rights.
- Source checkouts do not contain model weights, generated route/report data, datasets, or credentials. A hosted beta is not a claim that the source preview is self-contained.
- The product is responsive web first; there is no native mobile app implementation.
- Full production-scale platform hardening/SRE rollout is not complete.
- There is no full-US NAIP processing pipeline.
- Remote GPU training is an operator-run workflow, not a hosted training service; S3 credentials must be supplied at runtime and instances must be destroyed or explicitly retained by the operator.

## License

Private repository — not for distribution. All rights reserved; see [`LICENSE`](LICENSE). No source, model, data, artifact, or credential redistribution is authorized by this notice.
