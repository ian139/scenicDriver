# Scenic Route Planner

An ML-driven system that scores scenic beauty from satellite imagery and terrain data, then uses those scores for route planning.

## Overview

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

## Next Step (Current)

`v5` is the active learned checkpoint for Northeast runs:

- `models/scenic_regression_baseline_masswhites_z14_mixed5000_v5_weighted_h4.pt`
- Source of truth: `data/processed/regression/model_registry.json`

Current focus is MVP app build-out (hosted route compare + web/mobile UX), with routing/system hardening underneath:

```bash
# 1) Deterministic graph cache build (writes road_graph.json + run.json)
uv run python scripts/build_graph_from_osm.py \
  --min-lat 42.35 --min-lon -72.57 \
  --max-lat 42.39 --max-lon -72.52 \
  --run-name amherst_core

# 2) Scenic vs baseline route overlay (single call)
uv run python scripts/route_compare_service.py \
  --start 42.40 -72.70 \
  --end 42.48 -72.62 \
  --scenic-weight 0.8 \
  --run-name masswhites_z14_learned_h4_v2 \
  --graph-geojson data/processed/sample_road_graph.geojson
```

## Quick Start

```bash
# Install dependencies
uv sync

# Set Mapbox token
export MAPBOX_ACCESS_TOKEN=<your-token>

# Download tiles for a region
uv run python scripts/download_bbox_tiles.py \
  --min-lat 40.28 --min-lon -105.70 \
  --max-lat 40.35 --max-lon -105.58 \
  --zoom 16 --style mapbox.satellite \
  --output data/raw/images/satellite

uv run python scripts/download_bbox_tiles.py \
  --min-lat 40.28 --min-lon -105.70 \
  --max-lat 40.35 --max-lon -105.58 \
  --zoom 16 --style mapbox.terrain-rgb \
  --output data/raw/images/terrain

# Run heuristic labeling
uv run python scripts/heuristic_report.py --max-tiles 2000 --write-raw-labels

# Train the model
uv run marimo run notebooks/regression.mo.py
```

## Learned Scoring Scaffold (Step 3)

Notebook-first option:

```bash
uv run marimo edit notebooks/learned_scoring.mo.py
```

Annotation workflows:
- `notebooks/annotate_scenic.mo.py` (marimo)
- `scripts/annotate_scenic_web.py` (browser UI)

```bash
uv run marimo edit notebooks/annotate_scenic.mo.py
```

Web UI alternative (procedural annotation in browser):

```bash
uv run python scripts/annotate_scenic_web.py \
  --labels-csv data/processed/heuristic_runs/masswhites_z14_flat_5k_seamfix/labels.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --annotations-csv data/raw/labels_human.csv \
  --sample-size 500 \
  --stratify-by-class
```

Then open `http://localhost:8765`.

Overlap pass (same tiles labeled by two annotators):

```bash
# Build overlap batch for second annotator
uv run python scripts/build_overlap_batch.py \
  --annotations-csv data/raw/labels_human.csv \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --source-annotator ian \
  --target-annotator paperspace \
  --sample-size 200 \
  --seed 42 \
  --output-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv

# Annotate exactly that overlap batch
uv run python scripts/annotate_scenic_web.py \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --batch-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --annotations-csv data/raw/labels_human.csv \
  --annotator-id paperspace \
  --sample-size 200 \
  --stratify-by-class
```

```bash
# 0) Build human benchmark split + agreement report
uv run python scripts/build_human_benchmark.py \
  --annotations-csv data/raw/labels_human.csv \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --output-dir data/processed/regression \
  --run-name masswhites_human_benchmark_v1 \
  --val-frac 0.2 \
  --test-frac 0.2 \
  --seed 42

# 1) Export feature dataset for learned scenic regression
uv run python scripts/export_regression_dataset.py \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --output data/processed/regression/features_masswhites_z14_mixed5000.npz \
  --label-source-column label_source \
  --human-weight 4.0 \
  --heuristic-weight 1.0

# 2) Train baseline regressor
uv run python scripts/train_regression_baseline.py \
  --dataset data/processed/regression/features_masswhites_z14_mixed5000.npz \
  --output models/scenic_regression_baseline_masswhites_z14_mixed5000.pt \
  --use-sample-weights

# 3) Evaluate baseline on validation split
uv run python scripts/evaluate_regression_baseline.py \
  --dataset data/processed/regression/features_masswhites_z14_mixed5000.npz \
  --checkpoint models/scenic_regression_baseline_masswhites_z14_mixed5000.pt \
  --metrics-json data/processed/regression/baseline_metrics_masswhites_z14_mixed5000.json
```

Current recommendation from weight sweep (`data/processed/regression/weight_sweep_masswhites_z14.json`):
- Use `human_weight=4.0`, `heuristic_weight=1.0` for mixed-supervision training.
- Current benchmark split/agreement output: `data/processed/regression/masswhites_human_benchmark_v1/summary.json`.

Promotion gate:

```bash
uv run python scripts/promote_regression_model.py \
  --candidate-metrics data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v4_weighted_h4.json \
  --baseline-metrics data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v2_weighted_h4.json \
  --candidate-checkpoint models/scenic_regression_baseline_masswhites_z14_mixed5000_v4_weighted_h4.pt \
  --run-name masswhites_z14_mixed5000_v4_weighted_h4
```

Held-out benchmark compare (v4 vs v2):

```bash
uv run python scripts/compare_regression_on_benchmark.py \
  --dataset data/processed/regression/features_masswhites_z14_mixed5000_v4_h4.npz \
  --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000_v4.csv \
  --benchmark-split-csv data/processed/regression/masswhites_human_benchmark_v4/benchmark_split.csv \
  --baseline-checkpoint models/scenic_regression_baseline_masswhites_z14_mixed5000_v2_weighted_h4.pt \
  --candidate-checkpoint models/scenic_regression_baseline_masswhites_z14_mixed5000_v4_weighted_h4.pt \
  --output-json data/processed/regression/benchmark_compare_masswhites_v4_vs_v2.json
```

## Current Working Sets

| Region | Type | Tiles |
|--------|------|-------|
| masswhites (z14) | Northeast regional coverage | 25,544 per layer |
| amherst_ma (z16) | Local validation area | 5,544 per layer |
| Human annotations | Scenic manual labels | 500 rows (`labels_human.csv`) |

See [`data/README.md`](data/README.md) for detailed tile regions, bboxes, and data structure.

## Project Structure

```
├── notebooks/
│   ├── classifier.mo.py    # Stage 1: RESISC-45 classifier training
│   ├── regression.mo.py    # Stage 2/3: Heuristic labels + regression
│   └── train.mo.py         # Entry point
├── scripts/
│   ├── download_bbox_tiles.py   # Mapbox tile downloader
│   ├── heuristic_report.py      # Generate labels + reports
│   ├── annotate_scenic_web.py   # Browser annotation UI
│   └── train/eval/export scripts for learned scoring
├── src/
│   ├── classifier/         # Land-use classification model
│   ├── heuristics/         # Scenic scoring logic
│   ├── data_pipeline/      # Mapbox API, data loading
│   └── terrain/            # Elevation processing
├── data/
│   ├── raw/images/         # Satellite + terrain tiles
│   ├── raw/labels.csv      # Heuristic-generated labels
│   └── processed/          # Run logs, reports
└── models/                 # Checkpoints (not committed)
```

## Documentation

- [`AGENTS.md`](AGENTS.md) - Development workflow and principles
- [`data/README.md`](data/README.md) - Data structure and tile regions
- [`archive/archive.md`](archive/archive.md) - Archived scripts/models and restore commands

## S3 Storage (Optional)

Set a bucket and sync local data:

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
bash scripts/s3_sync.sh
```

Apply lifecycle rules (raw/processed → colder storage):

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket "$SCENIC_S3_BUCKET" \
  --lifecycle-configuration file://scripts/s3_lifecycle.json
```

S3-only report generation (no large local tile copy):

`scripts/heuristic_report.py` now defaults to:
- `SCENIC_S3_BUCKET=scenicdriver-data`
- `SCENIC_S3_ONLY=1`

Override these env vars explicitly if you want a different bucket or local-first mode.

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_S3_ONLY=1
uv run python scripts/heuristic_report.py \
  --run-name masswhites_z14 \
  --satellite-dir data/raw/images/satellite/z14/masswhites \
  --terrain-dir data/raw/images/terrain/z14/masswhites \
  --max-tiles 5000
```

Learned-scoring report generation (uses trained regression checkpoint):

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
uv run python scripts/heuristic_report.py \
  --run-name masswhites_z14_learned_h4_v2 \
  --scoring learned \
  --regression-ckpt models/scenic_regression_baseline_masswhites_z14_mixed5000_v4_weighted_h4.pt \
  --satellite-dir data/raw/images/satellite/z14/masswhites \
  --terrain-dir data/raw/images/terrain/z14/masswhites \
  --max-tiles 5000 \
  --s3-only
```

## Routing MVP

Build a road graph from OSM bbox (requires geo extras):

```bash
uv sync --extra geo
uv run python scripts/build_graph_from_osm.py \
  --min-lat 42.35 --min-lon -72.57 \
  --max-lat 42.39 --max-lon -72.52 \

## MVP API (New)

Hosted route compare + contributor endpoints:

```bash
uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload
```

Quick checks:

```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/regions
```

Web MVP shell (separate terminal):

```bash
cd web
python3 -m http.server 3000
```

Open `http://localhost:3000` (ensure API is running on `:8080`).

Address support:
- In the web UI, switch `Input Mode` to `Street Addresses`.
- The app calls `GET /v1/geocode?q=...` and uses the top result.
- Address fields now support live autocomplete suggestions from geocoding results.
- Coordinates mode remains available for direct lat/lon input.
- Auto-matching now normalizes points to canonical syntax:
  - `latlon`: `40.123456,-75.123456`
  - `wkt`: `POINT(-75.123456 40.123456)`
  --run-name amherst_core
```

This writes:
- `data/processed/road_graphs/amherst_core/road_graph.json`
- `data/processed/road_graphs/amherst_core/run.json`

Plan a scenic route from a LineString GeoJSON graph and write route overlay:

```bash
uv run python scripts/route_compare_service.py \
  --start 42.40 -72.70 \
  --end 42.48 -72.62 \
  --scenic-weight 0.8 \
  --run-name masswhites_z14_learned_h4_v2 \
  --graph-geojson data/processed/sample_road_graph.geojson
```

The report viewer auto-loads:
- `route.geojson` (single file with scenic + baseline features), or
- fallback pair `route_scenic.geojson` and `route_fast.geojson`.
- Route comparison metrics (distance/time/scenic deltas) are rendered in-map when route overlay data is present.

Routing now uses per-edge travel times from OSM `maxspeed` (with road-type fallbacks) and honors one-way directionality.

## Requirements

- Python 3.11+
- `uv` package manager
- Mapbox access token (free tier works)
- GPU recommended for training

## License

Private repository - not for distribution.
