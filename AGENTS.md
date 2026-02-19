# Scenic Route Planner: What We're Building

We are building an ML‑driven system that scores scenic beauty from imagery + terrain, then uses those scores for route planning. The current focus is shipping a web/mobile-friendly MVP trip planner while continuing model improvements in parallel.

This repo is intentionally notebook‑first (marimo). Keep the workflow tight, reproducible, and data‑centric.

## Public Repo Contract
- No secrets, tokens, or internal paths in tracked files.
- Keep large datasets and model weights out of git.
- One clear path per task (no duplicate stacks for the same use case).

## Execution Environment
- Use `uv` as the Python package manager (not `pip`).
- Use marimo notebooks (not Jupyter) for training/research workflows.
- Manual tile annotation may run in either marimo (`notebooks/annotate_scenic.mo.py`) or the web UI (`scripts/annotate_scenic_web.py`).
- Set `MAPBOX_ACCESS_TOKEN` in the shell before Mapbox downloads.

## Data
See [`data/README.md`](data/README.md) for tile regions, download commands, and labeling details.

## Project Structure (Current)
- `notebooks/classifier.mo.py`: Stage 1 RESISC45 classifier training.
- `notebooks/regression.mo.py`: Stage 2/3 heuristic labels + multitask regression/classification.
- `notebooks/train.mo.py`: lightweight hub/entry point.
- `data/raw/images/`: imagery tiles.
- `data/raw/labels.csv`: `image_path, scenic_score, lat, lon, class_id`.
- `data/raw/labels_human.csv`: manual scenic labels (`scenic_human`, confidence, notes).
- `data/processed/`: run logs, sample grids, caches.
- `models/`: checkpoints (do not commit).
- `scripts/download_bbox_tiles.py`: bbox downloader (supports `mapbox.satellite` + `mapbox.terrain-rgb`).
- `scripts/annotate_scenic_web.py`: browser-based procedural annotation UI.

## Commands (Primary)
- `marimo edit notebooks/classifier.mo.py`
- `marimo run notebooks/classifier.mo.py`
- `marimo edit notebooks/regression.mo.py`
- `marimo run notebooks/regression.mo.py`
- `marimo edit notebooks/learned_scoring.mo.py`
- `marimo run notebooks/learned_scoring.mo.py`
- `marimo edit notebooks/annotate_scenic.mo.py`
- `marimo run notebooks/annotate_scenic.mo.py`
- `uv run python scripts/annotate_scenic_web.py`
- `uv run python scripts/annotate_scenic_web.py --labels-csv data/processed/heuristic_runs/masswhites_z14_flat_5k_seamfix/labels.csv --raw-dir s3://$SCENIC_S3_BUCKET/raw --annotations-csv data/raw/labels_human.csv --sample-size 500 --stratify-by-class`
- `uv run python scripts/build_overlap_batch.py --annotations-csv data/raw/labels_human.csv --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv --source-annotator ian --target-annotator paperspace --sample-size 200 --seed 42 --output-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv`
- `uv run python scripts/annotate_scenic_web.py --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv --batch-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv --raw-dir s3://$SCENIC_S3_BUCKET/raw --annotations-csv data/raw/labels_human.csv --annotator-id paperspace --sample-size 200 --stratify-by-class`
- `uv run python scripts/build_human_benchmark.py --annotations-csv data/raw/labels_human.csv --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv --output-dir data/processed/regression --run-name masswhites_human_benchmark_v2 --val-frac 0.2 --test-frac 0.2 --seed 42`
- `uv run python scripts/heuristic_report.py --run-name masswhites_z14_learned_h4_v2 --scoring learned --regression-ckpt models/scenic_regression_baseline_masswhites_z14_mixed5000_v2_weighted_h4.pt --satellite-dir data/raw/images/satellite/z14/masswhites --terrain-dir data/raw/images/terrain/z14/masswhites --max-tiles 5000 --s3-only`
- `uv run python scripts/build_graph_from_osm.py --min-lat 42.35 --min-lon -72.57 --max-lat 42.39 --max-lon -72.52 --run-name amherst_core`
- `uv run python scripts/route_demo_geojson.py --geojson data/processed/sample_road_graph.geojson --start 42.40 -72.70 --end 42.48 -72.62 --scenic-weight 0.6 --output-geojson data/processed/sample_route.geojson`
- `uv run python scripts/route_demo_geojson.py --geojson data/processed/sample_road_graph.geojson --start 42.40 -72.70 --end 42.48 -72.62 --scenic-weight 0.8 --output-geojson data/processed/sample_route.geojson --report-dir data/processed/heuristic_runs/masswhites_z14_learned_h4_v2/report`
- `uv sync`
- `uv run python scripts/download_bbox_tiles.py --min-lat 40.018 --min-lon -75.2284 --max-lat 40.0734 --max-lon -75.185 --zoom 16 --style mapbox.satellite --output data/raw/images/satellite`
- `uv run python scripts/download_bbox_tiles.py --min-lat 40.018 --min-lon -75.2284 --max-lat 40.0734 --max-lon -75.185 --zoom 16 --style mapbox.terrain-rgb --output data/raw/images/terrain`

## Principles
- One notebook, one workflow: keep `notebooks/train.mo.py` authoritative.
- Fail fast: validate `labels.csv` and image paths before training.
- Minimal, explicit steps: no hidden preprocessing or side effects.
- Reproducibility first: seed all RNGs and log runs to `data/processed/`.
- Clean surfaces: small helpers, no duplicate pipelines.

## Current Technology Plan
- Phase 0 (priority): ship MVP planner UX + hosted route-compare API using current promoted model.
- Phase 1 (parallel): continue model-quality improvements without blocking MVP delivery.
- Retrain the RESISC45 classifier on larger/higher-quality data to reduce regional mislabeling.
- Keep classifier signal, but shift from fixed manual class weights to model features/auxiliary loss.
- Replace heuristic class-weight scoring with a learned scenic regressor using satellite embeddings + terrain features (optional class probabilities).
- During mixed supervision runs, weight human labels higher than heuristic labels (via `label_source` -> `sample_weights`; current default human weight: `4.0`).
- Add a manual scenic annotation workflow and treat human labels as primary supervision for model calibration/evaluation.
- Keep a deterministic human benchmark split + agreement report under `data/processed/regression/<run_name>/` (`benchmark_split.csv`, `agreement_by_annotator.csv`, `agreement_by_pair.csv`, `summary.json`).
- Run an explicit overlap pass (same tiles annotated by two annotators) before relying on pairwise agreement metrics.
- Keep heuristic scoring as fallback/baseline for sanity checks and quick previews.
- Active default checkpoint: `models/scenic_regression_baseline_masswhites_z14_mixed5000_v5_weighted_h4.pt` (see `data/processed/regression/model_registry.json`).
- Current execution focus: MVP app surface (route compare API + web/mobile UX), with routing hardening and model iteration continuing behind it.

## Git Workflow
- Default to working on `main` for quick fixes.
- If a change is more than a quick fix (new feature, refactor, multi-file edits), create a short-lived branch first.
- Name branches by intent, e.g. `feat/terrain-features`, `fix/report-viewer`, `chore/docs-cleanup`.

## Archive Policy
- Archived/non-primary files are tracked in `archive/archive.md`.
- Before introducing a second workflow path, check the archive and restore only what is needed.
- If a file is restored from archive, document why in the related PR/commit message.

## Craftsmanship Rubric (Any Change)
- Intent: does this improve correctness, reproducibility, or throughput?
- Uniqueness: are we creating a second way to do the same task?
- Surface: did we add a new config knob unnecessarily?
- Data integrity: are shapes, dtypes, and bounds validated?
- Repro: is the run fully reconstructable later?

## Research Workflow
1. Download tiles (Mapbox) for a bbox at a fixed zoom.
2. Generate heuristic labels (`labels.csv`) for broad weak supervision.
3. Collect manual scenic labels on a stratified tile subset (human benchmark set).
4. Run `notebooks/train.mo.py` / `notebooks/learned_scoring.mo.py` with mixed supervision.
5. Save best checkpoint + run summary JSON + benchmark metrics.
6. Iterate on model + scoring features.

## S3 Workflow Notes
- When using S3-first mode, keep tile keys under `raw/images/...` so report tooling can resolve prefixes.
- Canonical tile key layout: `raw/images/{satellite|terrain}/z{zoom}/{region}/{x}_{y}.png` (no extra nested zoom folder).
- Report scripts default to `SCENIC_S3_BUCKET=scenicdriver-data` and `SCENIC_S3_ONLY=1` unless already set in env.
- Override defaults when needed via explicit exports (for other buckets/modes):
  - `export SCENIC_S3_BUCKET=<your-bucket>`
  - `export SCENIC_S3_ONLY=0|1`
- Prefer `bash scripts/s3_sync.sh` (script may not be executable on all machines).
- Upload Step 3 artifacts after runs (use run-specific names, not only `features_v1`):
  - `aws s3 cp data/processed/regression/<features>.npz s3://$SCENIC_S3_BUCKET/processed/regression/<features>.npz`
  - `aws s3 cp models/<checkpoint>.pt s3://$SCENIC_S3_BUCKET/models/<checkpoint>.pt`
  - `aws s3 cp data/processed/regression/<metrics>.json s3://$SCENIC_S3_BUCKET/processed/regression/<metrics>.json`
- Viewer route overlays:
  - Preferred: write a combined `route.geojson` (scenic + baseline features).
  - Supported fallback: `route_scenic.geojson` + `route_fast.geojson`.
  - Viewer now renders route comparison metrics (distance/time/scenic deltas) when route overlay is present.

## Training Contract
- Multi‑task output: regression (scenic score 0–10) + classification (45 classes).
- Default split: 70/15/15 (train/val/test) with early stopping (patience=5).
- Save best checkpoint and `data/processed/{run_name}_train_run.json`.

## Non‑Goals (For Now)
- No full native mobile app implementation (responsive web first).
- No full production-scale platform hardening/SRE rollout yet.
- No full‑US NAIP processing pipeline.
