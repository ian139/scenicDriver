# CODEX

## Project
Scenic Route Planner is an ML-driven system that scores scenic beauty from satellite imagery + terrain data, then uses those scores for route planning. Current focus is shipping an MVP planner app (web/mobile-friendly) backed by hosted route-compare services, while improving models in parallel.

## Goals
- Score scenic beauty from satellite/terrain tiles (0-10).
- Build regional heatmaps + reports for QA.
- Prepare routing inputs (road graph + scenic edge cost).

## Current Status
- RESISC45 classifier: available and integrated.
- Heuristic labeling + report viewer: working.
- Terrain feature extraction: working.
- Web annotation UI: working (`scripts/annotate_scenic_web.py`).
- Manual scenic annotations: 500 rows collected (`data/raw/labels_human.csv`).
- Weighted mixed-supervision baseline: implemented and trained (`heuristic_weight=1.0`, configurable human weight).
- Human-weight sweep complete (`h2/h3/h4`); current recommendation is `human_weight=4.0`.
- Routing MVP restored: road graph loader + scenic route planner + GeoJSON route CLI.
- In progress: MVP app/API build-out, classifier quality improvements, and edge-level scenic data integration.

## Tech Stack (Current)
- ML: PyTorch, timm, numpy, pandas
- Geo: rasterio (optional), shapely (optional), osmnx (optional for routing)
- Tooling: uv, marimo
- Viewer: MapLibre (OSM tiles) + optional Mapbox satellite

## High-Level Pipeline
1. Download tiles (Mapbox) for a bbox at fixed zoom.
2. Generate heuristic labels + report.
3. Add manual scenic labels for a stratified subset (benchmark + calibration).
4. Export regression features (satellite embeddings + terrain + logits).
5. Train/evaluate regressor on human-only and mixed-label datasets.
6. Build a road graph and score edges with scenic tiles.

## Repo Conventions
- Keep ML workflows in marimo notebooks (`notebooks/`).
- Use `scripts/` for data prep, downloads, and inference utilities only.
- Keep large datasets and model weights out of git.
- Store run artifacts under `data/processed/` (ignored).

## Non-Goals (For Now)
- No full native mobile app implementation (responsive web first).
- No full production-scale platform hardening/SRE rollout yet.
- No full-US NAIP processing pipeline.
