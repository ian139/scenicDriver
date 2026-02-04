# CODEX

## Project
Scenic Route Planner (working name: "Progno") is an AI-powered route planning system that prioritizes scenic drives across the United States using landscape classification, terrain analysis, and pathfinding.

## Goals
- Score scenic beauty from satellite/aerial imagery and terrain data.
- Build a continental scenic score heatmap (US).
- Provide routing that balances time with scenery.
- Expose an API and mobile app for planning and navigation.

## Current Status
- Landscape classification: completed (ViT on RESISC45).
- Terrain analysis pipeline: completed.
- Scenic scoring formula: completed.
- Mapbox tile processing: completed.
- In progress: regression model training, NAIP processing, S3 batch pipeline.

## Tech Stack
- ML: PyTorch, Transformers, scikit-learn, OpenCV
- Geo: GDAL, PostGIS, Shapely, Rasterio
- Backend: Python, FastAPI, PostgreSQL + PostGIS
- Mobile: React Native or Swift (iOS)
- Infra: AWS S3, Docker, CI/CD

## High-Level Pipeline
1. Landscape classification (ViT on 224x224 imagery).
2. Terrain analysis (DEM slope, elevation variation, water proximity, vegetation indices).
3. Scenic score regression (0-10 scale).
4. Continental dataset generation (NAIP tiles + batch inference).
5. Scenic route planning (A*/Dijkstra with custom cost).
6. API + mobile client.

## Basics / Local Setup
This repo is currently a shell. When code lands, add:
- `requirements.txt` and/or `pyproject.toml`
- `src/` with classifier, scorer, and route planner
- `data/` and `models/` (or document external storage)
- `docker/` or `Dockerfile` if needed

Example target usage (once implemented):
```python
from src.classifier import SceneryClassifier
from src.scenic_scorer import TerrainScorer
from src.route_planner import ScenicRoutePlanner
```

## Repo Conventions
- Keep ML training workflows in marimo notebooks (`notebooks/`).
- Use `scripts/` for data prep, downloads, and inference utilities only.
- Keep model weights out of git; use S3 or a model registry.
- Document any external data sources and access credentials in `docs/`.

## Roadmap (Next)
- Implement regression model training.
- Build NAIP S3 ingestion + batch inference.
- Create spatial index for route scoring.
- Implement scenic pathfinding cost function.
- Stand up FastAPI endpoints.

## Contact
- GitHub: @Ian139
- Issues: github.com/ian139/scenicDriver/issues
