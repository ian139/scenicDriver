# Scenic Drive Agent Guide

## Project
Scenic Drive scores scenic beauty from satellite imagery and terrain data, then uses those scores for route planning. Current work prioritizes the MVP route-planning API and New England North web app while model improvements continue in parallel.

## Active Surfaces
- `apps/new_england_north/`: canonical static MapLibre web MVP.
- `src/app_api/`: FastAPI route-compare and contributor endpoints.
- `src/route_planner/`: graph, cost, planner, and route service logic.
- `src/classifier/`, `src/scenic_scorer/`, `src/terrain/`, `src/heuristics/`, `src/data_pipeline/`: ML/data pipeline code.
- `notebooks/`: marimo training, scoring, and annotation workflows.
- `scripts/`: workflow CLIs grouped by annotation, ingest, modeling, reports, and routing.

## Commands
- `uv sync`
- `uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload`
- `cd apps/new_england_north && python3 -m http.server 3000`
- `uv run marimo edit notebooks/train.mo.py`
- `uv run marimo edit notebooks/regression.mo.py`
- `uv run marimo edit notebooks/learned_scoring.mo.py`
- `uv run marimo edit notebooks/annotate_scenic.mo.py`
- `uv run python scripts/ingest/download_bbox_tiles.py ...`
- `uv run python scripts/reports/heuristic_report.py ...`
- `uv run python scripts/routing/route_compare_service.py ...`
- `cp .env.beta.example .env.beta` and populate `MAPBOX_ACCESS_TOKEN`
- `docker compose --env-file .env.beta -f compose.beta.yml up --build`
- Open `http://localhost:${SCENIC_WEB_PORT:-80}`
- `docker compose --env-file .env.beta -f compose.beta.yml down`

## Data And Artifact Policy
- Keep large data, generated reports, caches, and model weights out of git.
- Local generated paths are `data/raw/`, `data/processed/`, `data/NWPU-RESISC45/`, `models/`, `cache/`, and `scenic_artifacts/`.
- Preserve `data/README.md` as the canonical data layout contract.
- Preserve `archive/archive.md` as the archive manifest.
- Beta deployment requires the canonical processed graph, learned run, registry, and active model checkpoint mounted from ignored `data/processed/` and `models/`; they remain outside Git and Docker image layers.

## Development Rules
- Use `uv`, not `pip`, for Python dependency management.
- Use marimo notebooks, not Jupyter, for training/research workflows.
- Prefer one canonical workflow path; archive superseded alternatives instead of leaving duplicate active paths.
- Run focused tests or smoke checks for changed API, script, notebook, or app paths before yielding.
- OMP/CMUX workflow details live in `docs/internal/cmux-workflow.md`.
- Keep ML workflows in marimo notebooks (`notebooks/`).
- Use grouped `scripts/` subdirectories for workflow CLIs only.
- Keep large datasets and model weights out of git.
- Store run artifacts under `data/processed/` (ignored).

## Subagent Model Roles

These roles resolve through the OMP `modelRoles` map. Each inherits the base provider config from that map; only the model and any role-specific flags are listed here.

- `designer`: `openrouter/glm/5.2` — UI/UX design and visual refinement.
- `commit`: `ollama-cloud/deepseek-v4-flash` — commit-message generation.
- `task`: `ollama-cloud/deepseek-v4-pro --thinking high` — general subagent implementation work.
