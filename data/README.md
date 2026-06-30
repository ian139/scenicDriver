# Data Layout

## Local Paths

- `data/raw/images/`: local tile cache (optional when using S3-first mode)
- `data/raw/labels.csv`: weak labels from heuristic scoring
- `data/raw/labels_human.csv`: manual annotations (`scenic_human`, confidence, notes)
- `data/processed/`: reports, feature exports, run metadata, regression artifacts

## S3 Canonical Paths

Keep tile keys under:

- `raw/images/satellite/z{zoom}/{region}/{x}_{y}.png`
- `raw/images/terrain/z{zoom}/{region}/{x}_{y}.png`

No extra nested zoom folder under region.

Report tooling defaults (unless overridden in env):
- `SCENIC_S3_BUCKET=scenicdriver-data`
- `SCENIC_S3_ONLY=1`

## Annotation Workflow Notes

- For local tile loading in annotator: `raw_dir = data/raw`
- For S3 tile loading in annotator: `raw_dir = s3://<bucket>/raw`
- Manual labels are appended/upserted to `data/raw/labels_human.csv`

## Current Active Working Sets

- `masswhites` at `z14` (primary region)
- `amherst_ma` at `z16` (secondary/local validation)

## Regression Artifacts (Step 3)

- Feature exports may include:
  - `sample_weights` (for mixed supervision, e.g. human=3.0 / heuristic=1.0)
  - `image_paths` (row alignment/audit)
- Recommended mixed labels source:
  - `data/processed/regression/labels_masswhites_z14_mixed5000.csv`

## Human Benchmark Artifacts

- Build benchmark split + agreement stats:
  - `uv run python scripts/annotation/build_human_benchmark.py --annotations-csv data/raw/labels_human.csv --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv --output-dir data/processed/regression --run-name masswhites_human_benchmark_v1 --val-frac 0.2 --test-frac 0.2 --seed 42`
- Build overlap batch for cross-annotator agreement collection:
  - `uv run python scripts/annotation/build_overlap_batch.py --annotations-csv data/raw/labels_human.csv --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv --source-annotator ian --target-annotator paperspace --sample-size 200 --seed 42 --output-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv`
- Output folder layout:
  - `data/processed/regression/<run_name>/benchmark_tiles.csv`
  - `data/processed/regression/<run_name>/benchmark_split.csv`
  - `data/processed/regression/<run_name>/agreement_by_annotator.csv`
  - `data/processed/regression/<run_name>/agreement_by_pair.csv`
  - `data/processed/regression/<run_name>/summary.json`

## Report Overlay Artifacts

Report viewer route overlay supports either layout in `data/processed/heuristic_runs/<run_name>/report/`:

- `route.geojson` (recommended; one file containing scenic + baseline features), or
- split files: `route_scenic.geojson` and `route_fast.geojson`.

When route overlay is present, the viewer also displays route comparison metrics:
- scenic vs baseline distance
- scenic vs baseline travel time
- scenic score delta

Generate combined overlay directly:

- `uv run python scripts/routing/route_compare_service.py --start 42.40 -72.70 --end 42.48 -72.62 --scenic-weight 0.8 --run-name <run_name> --graph-geojson <graph>`

## Git Policy

Large datasets and generated artifacts are ignored via `.gitignore`.
