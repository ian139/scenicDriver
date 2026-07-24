# Scenic Drive

Scenic Drive finds routes that trade a small amount of travel time for better scenery. It combines satellite imagery, terrain signals, learned scenic scores, and a directed road graph behind a FastAPI service and a lightweight MapLibre web app.

The current product surface is the **New England North** web app. It compares the fastest route with a scenic alternative, renders regional scenic scores, and exposes the active model result.

## What is here

- **Route comparison** — fastest and scenic routes with distance, duration, and scenic-score deltas.
- **Learned scenic scoring** — satellite embeddings, terrain features, classifier signals, and mixed human/heuristic labels.
- **Regional web app** — a static MapLibre client with no frontend build step.
- **Reproducible operations** — locked Python dependencies, artifact manifests, container definitions, and focused CI checks.

```mermaid
flowchart LR
    A[Satellite + Terrain Tiles] --> B[Feature and Label Pipelines]
    B --> C[Scenic Model]
    C --> D[Scored Regional Artifacts]
    E[OSM Road Data] --> F[Directed Road Graph]
    D --> G[FastAPI Route Service]
    F --> G
    G --> H[MapLibre Web App]
```

## Quick start

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- A Mapbox access token for map tiles and address search
- Node.js 20+ only for the viewer test

Install the locked development environment:

```bash
uv sync --frozen --extra dev
```

Start the API:

```bash
export MAPBOX_ACCESS_TOKEN=<your-token>
uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload
```

In another terminal, serve the static app:

```bash
cd apps/new_england_north
python3 -m http.server 3000
```

Open <http://localhost:3000>. Useful API endpoints:

- Health: <http://localhost:8080/v1/healthz>
- Interactive API reference: <http://localhost:8080/scalar>

A clean checkout can run the API and app shell. Heatmaps and route comparison require the ignored graph, report, registry, and model artifacts described in the [deployment guide](docs/setup/deployment.md).

## Hosted beta

The beta uses Nginx for the static app and reverse proxy, plus FastAPI for the API. Runtime artifacts and credentials are mounted or supplied externally; they are never copied into image layers.

```bash
cp .env.beta.example .env.beta
# Set MAPBOX_ACCESS_TOKEN in .env.beta

docker compose --env-file .env.beta -f compose.beta.yml up --build
```

Open `http://localhost:${SCENIC_WEB_PORT:-80}`. Before deployment, bootstrap and validate the canonical artifacts using the [deployment runbook](docs/setup/deployment.md).

## Repository layout

```text
apps/new_england_north/  Static MapLibre web app
src/app_api/             FastAPI application and contributor endpoints
src/route_planner/       Graph, cost, planning, cancellation, and service logic
src/classifier/          Landscape classifier training and inference
src/scenic_scorer/       Learned scenic regression
src/terrain/             Terrain-RGB feature extraction
src/heuristics/          Labeling and report generation
src/data_pipeline/       Tile and S3 data access
scripts/                 Grouped workflow CLIs
notebooks/               Interactive marimo workflows
config/                  Region and storage configuration
deploy/                  Versioned beta artifact manifests
docs/                    Architecture, setup, research, and roadmap
archive/                 Curated historical material
```

## Common workflows

| Goal | Entry point |
|---|---|
| Download regional tiles | `scripts/ingest/download_bbox_tiles.py` |
| Annotate scenic labels | `notebooks/annotate_scenic.mo.py` or `scripts/annotation/annotate_scenic_web.py` |
| Train and evaluate scoring | `notebooks/regression.mo.py` and `scripts/modeling/` |
| Generate regional reports | `scripts/reports/heuristic_report.py` |
| Build an OSM road graph | `scripts/routing/build_graph_from_osm.py` |
| Compare routes | `scripts/routing/route_compare_service.py` |
| Run remote training | `scripts/remote/` and the [Vast.ai runbook](docs/setup/vast-training.md) |

Run a marimo workflow with, for example:

```bash
uv run marimo edit notebooks/regression.mo.py
```

## Documentation

- [Architecture](docs/architecture/infrastructure.md) — current offline and online system boundaries
- [Data layout](data/README.md) — canonical local and S3 artifact paths
- [Deployment](docs/setup/deployment.md) — required beta artifacts, bootstrap, validation, and startup
- [AWS/S3 setup](docs/setup/aws-s3.md) — credentials, bucket layout, sync, and lifecycle policy
- [Remote training](docs/setup/vast-training.md) — container and Vast.ai lifecycle
- [Roadmap](docs/roadmap.md) — current product, model, data, and routing priorities
- [ML research log](docs/research/ml-research-log.md) — learned-score model history and promotion evidence
- [Routing performance log](docs/research/routing-performance-autoresearch.md) — benchmark results and retained experiments
- [Archive manifest](archive/archive.md) — historical, non-primary material

## Tests

```bash
uv run pytest -q
node --test tests/test_new_england_north_viewer.mjs
```

These are the same two gates run by GitHub Actions.

## Project boundaries

- Model weights, datasets, generated reports, road graphs, caches, and credentials stay outside Git.
- The responsive web app is canonical; the archived native app is not active.
- Remote training is operator-run, not a hosted training service.
- The repository is a private source preview, not a self-contained data or model release.

## License

Private repository — not for distribution. All rights reserved; see [LICENSE](LICENSE).
