# Data Directory

This directory contains all imagery, terrain, and label data for the Scenic Route Planner.

## Directory Structure

```
data/
├── raw/
│   ├── images/
│   │   ├── satellite/z16/   # Mapbox satellite tiles (256x256 PNG)
│   │   │   ├── olympic_peninsula/
│   │   │   ├── big_sur/
│   │   │   ├── rocky_mountains/
│   │   │   └── philadelphia/
│   │   └── terrain/z16/     # Mapbox terrain-rgb tiles (elevation encoded)
│   │       ├── olympic_peninsula/
│   │       ├── big_sur/
│   │       ├── rocky_mountains/
│   │       └── philadelphia/
│   └── labels.csv           # Heuristic-generated labels (optional)
├── processed/               # Run logs, caches, heuristic reports
└── NWPU-RESISC45/           # Classifier pre-training dataset
```

## Tile Regions

All tiles are zoom level 16 from Mapbox. Tile naming: `{x}_{y}.png`

| Region | Type | Bbox (lat/lon) | Tiles | X-range |
|--------|------|----------------|-------|---------|
| **Olympic Peninsula, WA** | Dense forest | 47.82, -123.88 to 47.90, -123.75 | 575 | 10216-10240 |
| **Big Sur, CA** | Coastal cliffs | 36.20, -121.88 to 36.28, -121.78 | 361 | 10580-10598 |
| **Rocky Mountains, CO** | Alpine/mountains | 40.28, -105.70 to 40.35, -105.58 | 391 | 13525-13547 |
| **Philadelphia suburbs, PA** | Flat suburban | 40.018, -75.2284 to 40.0734, -75.185 | 112 | 19073-19080 |

**Total: 1,439 paired satellite/terrain tiles**

## Download Commands

Requires `MAPBOX_ACCESS_TOKEN` environment variable.

```bash
# Example: Download a new region
uv run python scripts/download_bbox_tiles.py \
  --min-lat 40.28 --min-lon -105.70 \
  --max-lat 40.35 --max-lon -105.58 \
  --zoom 16 \
  --style mapbox.satellite \
  --output data/raw/images/satellite

uv run python scripts/download_bbox_tiles.py \
  --min-lat 40.28 --min-lon -105.70 \
  --max-lat 40.35 --max-lon -105.58 \
  --zoom 16 \
  --style mapbox.terrain-rgb \
  --output data/raw/images/terrain
```

## Heuristic Reports

Generate a report for a single region (example uses the Rocky Mountains tiles):

```bash
uv run python scripts/heuristic_report_region.py --region rocky_mountains
```

Generate a report for all tiles (all regions combined):

```bash
uv run python scripts/heuristic_report.py --run-name heuristic_all
```

Reports are written to `data/processed/heuristic_runs/{run_name}/report/`.

If you want to also update the training labels at `data/raw/labels.csv`, add `--write-raw-labels`.

## Terrain-RGB Decoding

Mapbox terrain-rgb tiles encode elevation in RGB channels:

```python
elevation = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1  # meters
```

## Labels Format

`labels.csv` columns:
- `image_path`: Relative path to satellite tile (from `data/raw/`)
- `scenic_score`: Heuristic score 0-10 (higher = more scenic)
- `lat`, `lon`: Tile center coordinates (placeholder 0,0 for now)
- `class_id`: RESISC-45 class prediction (0-44)

## Heuristic Scoring Formula

```
score = 2.5 * class_score      # From RESISC-45 classifier
      + 2.0 * tanh(relief/500)  # Elevation range
      + 1.5 * tanh(roughness/200)  # Elevation std dev
      + 1.5 * tanh(slope_mean/15)  # Average slope
      + 1.5 * water_proxy       # Low+flat regions
      + 1.0 * veg_proxy         # Green channel ratio
      - 1.2 * water_fraction    # Ocean/large water penalty
```

## Troubleshooting

- `uv run` fails with `ModuleNotFoundError`: run `uv sync` first.
- Classifier weights missing: ensure `models/classifier/best_model.pt` exists.
- Map tiles missing: set `MAPBOX_ACCESS_TOKEN` before downloads.
- Viewer map blank: allow `https://unpkg.com/` (MapLibre) and set `data/processed/report_config.json` with `{"mapbox_token": "..."}` for satellite layer.
- `timm`/`pandas` import errors: re-run `uv sync` to refresh the environment.

## Adding New Regions

1. Choose a bbox with interesting terrain (mountains, coast, forest)
2. Run download script for both `mapbox.satellite` and `mapbox.terrain-rgb`
3. Move tiles to appropriate subfolder in `z16/`
4. Update this README with the new region details
5. Re-run heuristic labeling: `uv run python scripts/heuristic_report_region.py --region <name> --write-raw-labels`

## Notes

- The heuristic labeler scans recursively and supports region subfolders under `z16/`.
- Tile naming: `{x}_{y}.png` inside `.../z16/<region>/` is supported for heatmap rendering.
- Keep large tile datasets out of git (already in `.gitignore`).

## S3 Sync (Optional)

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
./scripts/s3_sync.sh
```

Lifecycle policy template: `scripts/s3_lifecycle.json`
