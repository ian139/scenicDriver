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

## Current Dataset

| Region | Type | Tiles |
|--------|------|-------|
| Olympic Peninsula, WA | Dense forest | 575 |
| Big Sur, CA | Coastal cliffs | 361 |
| Rocky Mountains, CO | Alpine/mountains | 391 |
| Philadelphia suburbs, PA | Flat suburban | 112 |
| **Total** | | **1,439** |

See [`data/README.md`](data/README.md) for detailed tile regions, bboxes, and data structure.

## Project Structure

```
├── notebooks/
│   ├── classifier.mo.py    # Stage 1: RESISC-45 classifier training
│   ├── regression.mo.py    # Stage 2/3: Heuristic labels + regression
│   └── train.mo.py         # Entry point
├── scripts/
│   ├── download_bbox_tiles.py   # Mapbox tile downloader
│   └── heuristic_report.py      # Generate labels + reports
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

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_S3_ONLY=1
uv run python scripts/heuristic_report.py \
  --run-name masswhites_z14 \
  --satellite-dir data/raw/images/satellite/z14 \
  --terrain-dir data/raw/images/terrain/z14 \
  --max-tiles 5000
```

## Requirements

- Python 3.11+
- `uv` package manager
- Mapbox access token (free tier works)
- GPU recommended for training

## License

Private repository - not for distribution.
