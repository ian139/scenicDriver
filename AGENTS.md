# Scenic Route Planner: What We're Building

We are building an ML‑driven system that scores scenic beauty from imagery + terrain, then uses those scores for route planning. The current focus is regression training and NAIP/Mapbox data ingestion.

This repo is intentionally notebook‑first (marimo). Keep the workflow tight, reproducible, and data‑centric.

## Public Repo Contract
- No secrets, tokens, or internal paths in tracked files.
- Keep large datasets and model weights out of git.
- One clear path per task (no duplicate stacks for the same use case).

## Execution Environment
- Use `uv` as the Python package manager (not `pip`).
- Use marimo notebooks (not Jupyter) for all workflows.
- Set `MAPBOX_ACCESS_TOKEN` in the shell before Mapbox downloads.

## Data
See [`data/README.md`](data/README.md) for tile regions, download commands, and labeling details.

## Project Structure (Current)
- `notebooks/classifier.mo.py`: Stage 1 RESISC45 classifier training.
- `notebooks/regression.mo.py`: Stage 2/3 heuristic labels + multitask regression/classification.
- `notebooks/train.mo.py`: lightweight hub/entry point.
- `data/raw/images/`: imagery tiles.
- `data/raw/labels.csv`: `image_path, scenic_score, lat, lon, class_id`.
- `data/processed/`: run logs, sample grids, caches.
- `models/`: checkpoints (do not commit).
- `scripts/download_bbox_tiles.py`: bbox downloader (supports `mapbox.satellite` + `mapbox.terrain-rgb`).

## Commands (Primary)
- `marimo edit notebooks/classifier.mo.py`
- `marimo run notebooks/classifier.mo.py`
- `marimo edit notebooks/regression.mo.py`
- `marimo run notebooks/regression.mo.py`
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
- Phase 1 (priority): improve model quality before routing.
- Retrain the RESISC45 classifier on larger/higher-quality data to reduce regional mislabeling.
- Keep classifier signal, but shift from fixed manual class weights to model features/auxiliary loss.
- Replace heuristic class-weight scoring with a learned scenic regressor using satellite embeddings + terrain features (optional class probabilities).
- Keep heuristic scoring as fallback/baseline for sanity checks and quick previews.

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
2. Prepare `labels.csv` with `image_path` + targets.
3. Run `notebooks/train.mo.py` for multi‑task training.
4. Save best checkpoint + run summary JSON.
5. Iterate on model + scoring features.

## S3 Workflow Notes
- When using S3-first mode, keep tile keys under `raw/images/...` so report tooling can resolve prefixes.
- For S3-only report generation, set `SCENIC_S3_BUCKET` and `SCENIC_S3_ONLY=1`.
- Prefer `bash scripts/s3_sync.sh` (script may not be executable on all machines).

## Training Contract
- Multi‑task output: regression (scenic score 0–10) + classification (45 classes).
- Default split: 70/15/15 (train/val/test) with early stopping (patience=5).
- Save best checkpoint and `data/processed/{run_name}_train_run.json`.

## Non‑Goals (For Now)
- No mobile app implementation.
- No production API deployment.
- No full‑US NAIP processing pipeline.
