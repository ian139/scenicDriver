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

`v5` is the active learned checkpoint for Northeast runs:

- `models/scenic_regression_baseline_masswhites_z14_mixed5000_v5_weighted_h4.pt`
- Source of truth: `data/processed/regression/model_registry.json`

Current focus is MVP app build-out (hosted route compare + web/mobile UX), with routing/system hardening underneath.

Web MVP now includes:
- live geocoding + autocomplete
- route compare cards (scenic vs baseline)
- saved trips + share links
- mobile tab bar + settings bottom sheet
- MapTiler basemap support with OSM fallback

## Tech stack

- ML: PyTorch, timm, numpy, pandas
- Geo: rasterio (optional), shapely (optional), osmnx (optional for routing)
- Tooling: uv, marimo
- Viewer: MapLibre under `apps/web/` with OSM tiles + optional Mapbox satellite
- API: FastAPI under `src/app_api/`
- Web: static MapLibre app under `apps/web/`
- Mobile: Expo shell under `apps/mobile/`

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
cd apps/web && python3 -m http.server 3000
uv run marimo edit notebooks/train.mo.py
```

Open `http://localhost:3000` for the web MVP after the API is running on `:8080`.

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
  --regression-ckpt models/scenic_regression_baseline_masswhites_z14_mixed5000_v5_weighted_h4.pt \
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

## Repository layout

```text
apps/
  web/                         # moved from web/
  mobile/                      # moved from mobile/
docs/
  setup/aws-s3.md
  architecture/infrastructure.md
  research/ml-research-log.md
  roadmap.md
  internal/orca-workflow.md
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
- [`docs/internal/orca-workflow.md`](docs/internal/orca-workflow.md)

## Current boundaries

- No full native mobile app implementation (responsive web first).
- No full production-scale platform hardening/SRE rollout yet.
- No full-US NAIP processing pipeline.

## Requirements

- Python 3.11+
- `uv` package manager
- Mapbox access token (free tier works)
- GPU recommended for training

## License

Private repository - not for distribution.
