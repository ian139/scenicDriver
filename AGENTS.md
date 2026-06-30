# Scenic Drive Agent Guide

## Project
Scenic Drive scores scenic beauty from satellite imagery and terrain data, then uses those scores for route planning. Current work prioritizes the MVP route-planning API and web/mobile app while model improvements continue in parallel.

## Active Surfaces
- `apps/web/`: static MapLibre web MVP.
- `apps/mobile/`: Expo React Native shell.
- `src/app_api/`: FastAPI route-compare and contributor endpoints.
- `src/route_planner/`: graph, cost, planner, and route service logic.
- `src/classifier/`, `src/scenic_scorer/`, `src/terrain/`, `src/heuristics/`, `src/data_pipeline/`: ML/data pipeline code.
- `notebooks/`: marimo training, scoring, and annotation workflows.
- `scripts/`: workflow CLIs grouped by annotation, ingest, modeling, reports, and routing.

## Commands
- `uv sync`
- `uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload`
- `cd apps/web && python3 -m http.server 3000`
- `uv run marimo edit notebooks/train.mo.py`
- `uv run marimo edit notebooks/regression.mo.py`
- `uv run marimo edit notebooks/learned_scoring.mo.py`
- `uv run marimo edit notebooks/annotate_scenic.mo.py`
- `uv run python scripts/ingest/download_bbox_tiles.py ...`
- `uv run python scripts/reports/heuristic_report.py ...`
- `uv run python scripts/routing/route_compare_service.py ...`

## Data And Artifact Policy
- Keep large data, generated reports, caches, and model weights out of git.
- Local generated paths are `data/raw/`, `data/processed/`, `data/NWPU-RESISC45/`, `models/`, `cache/`, and `scenic_artifacts/`.
- Preserve `data/README.md` as the canonical data layout contract.
- Preserve `archive/archive.md` as the archive manifest.

## Development Rules
- Use `uv`, not `pip`, for Python dependency management.
- Use marimo notebooks, not Jupyter, for training/research workflows.
- Prefer one canonical workflow path; archive superseded alternatives instead of leaving duplicate active paths.
- Run focused tests or smoke checks for changed API, script, notebook, or app paths before yielding.
- Agent/Orca workflow details live in `docs/internal/orca-workflow.md`.
- Keep ML workflows in marimo notebooks (`notebooks/`).
- Use grouped `scripts/` subdirectories for workflow CLIs only.
- Keep large datasets and model weights out of git.
- Store run artifacts under `data/processed/` (ignored).
