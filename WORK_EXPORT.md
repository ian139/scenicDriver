# ScenicDrive Work Export

_Last updated: April 20, 2026_

## Summary

ScenicDrive has moved from a mostly research/reporting workflow into a working MVP route-planning web app. The current app can geocode addresses, compare scenic vs baseline routes, show route metrics, render routes on a map, and overlay scenic heatmap data from heuristic/model report runs.

The main branch has been merged and pushed. Current branch state at export time: `main` is aligned with `origin/main`.

## Current Product Shape

The MVP is a local web navigation app backed by FastAPI.

- Web UI lives in `web/`.
- API lives in `src/app_api/`.
- Routing logic lives in `src/route_planner/`.
- Scenic score reports live under `data/processed/heuristic_runs/`.
- Road graph regions live under `data/processed/road_graphs/`.

The main user-facing flow is:

1. Start the API.
2. Serve the web UI.
3. Pick a region.
4. Enter `From` and `To` addresses.
5. Plan a route.
6. Compare scenic route vs baseline route.
7. Optionally toggle scenic heatmap overlay.

## How To Run

Start the API:

```bash
cd "/Users/ian/Projects/Scenic Drive"
export MAPBOX_ACCESS_TOKEN="<your-mapbox-token>"
uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload
```

Serve the web app:

```bash
cd "/Users/ian/Projects/Scenic Drive/web"
python3 -m http.server 3000
```

Open:

```text
http://localhost:3000
```

Useful API checks:

```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/regions
curl "http://localhost:8080/v1/geocode?q=target%20south%20philly&region=philadelphia"
```

## Web App Work

The web UI was built as a lightweight static app using MapLibre.

Key features:

- Map-first layout with floating route controls.
- Address-first `From` / `To` route planner.
- Settings menu for advanced controls.
- Coordinate input mode available from settings.
- Region selector.
- Scenic weight slider.
- Max detour factor slider.
- Route planning loading overlay.
- Route metrics cards.
- Scenic heatmap toggle.
- Heatmap rendering that transitions by zoom level:
  - Smooth/blurred score overlay at wider zooms.
  - Tile fill polygons at higher zooms.
- Heatmap colors now use an absolute `0-10` scenic-score scale instead of relative percentile coloring.
- Tile borders are disabled for a cleaner high-zoom heatmap.
- Z14 heatmap tile fill transitions earlier than Z16.

Important files:

- `web/index.html`
- `web/styles.css`
- `web/app.js`

## API Work

A FastAPI app was added under `src/app_api/`.

Current endpoints:

- `GET /`
- `GET /v1/healthz`
- `GET /v1/regions`
- `GET /v1/geocode`
- `POST /v1/route/compare`
- `GET /v1/heatmap`
- `POST /v1/contrib/session/start`
- `GET /v1/contrib/tasks/next`
- `POST /v1/contrib/labels`
- `GET /v1/contrib/profile`
- `GET /v1/contrib/leaderboard`
- `POST /v1/admin/contrib/review/run`

Important behavior:

- `/v1/geocode` uses Mapbox Geocoding v5.
- Coordinate input is still parsed directly.
- Region bbox is used as a Mapbox geocode bias.
- `/v1/regions` discovers both `road_graph.geojson` and `road_graph.json`.
- `/v1/route/compare` supports both GeoJSON road graphs and serialized RoadGraph JSONs.
- `pittsfield` resolves to Masswhites-style report names.
- Derived graph names like `masswhites_z14_wide` can resolve back to the Masswhites run report.
- `/v1/heatmap` overlays scenic scores from the matching heuristic run.
- Heatmap response includes both point GeoJSON and tile polygon GeoJSON.

Important files:

- `src/app_api/main.py`
- `src/app_api/schemas.py`
- `src/app_api/contrib_repo.py`
- `src/route_planner/service.py`

## Routing Work

Route comparison was wrapped into API and script-friendly interfaces.

Key behavior:

- Scenic route and baseline route are returned together.
- Metrics include distance, estimated duration, average scenic score, and deltas.
- No-route cases return structured `422` responses instead of uncaught server errors.
- Route planning can apply tile score reports to road graph edges.
- Road graph loading now supports:
  - `road_graph.geojson`
  - `road_graph.json`

Relevant files:

- `src/route_planner/service.py`
- `src/route_planner/planner.py`
- `src/route_planner/graph.py`
- `scripts/route_compare_service.py`
- `scripts/route_demo_geojson.py`

## Road Graph Regions

Existing discovered regions include:

- `philadelphia`
- `pittsfield`
- `masswhites_z14_wide`

The wide Masswhites graph was built to cover the same broad area as the Masswhites Z14 heuristic/model run. The app originally only found `road_graph.geojson`; this was fixed so `road_graph.json` is also accepted.

The Masswhites wide graph was created locally with `osmnx`. If it needs to be rebuilt:

```bash
uv run --with osmnx python scripts/build_graph_from_osm.py \
  --min-lat 41.19518982948958 \
  --min-lon -73.5205078125 \
  --max-lat 44.512176171071054 \
  --max-lon -72.97119140625 \
  --run-name masswhites_z14_wide \
  --timeout 600
```

For more road classes:

```bash
uv run --with osmnx python scripts/build_graph_from_osm.py \
  --min-lat 41.19518982948958 \
  --min-lon -73.5205078125 \
  --max-lat 44.512176171071054 \
  --max-lon -72.97119140625 \
  --run-name masswhites_z14_wide_all \
  --network all_public \
  --timeout 900
```

## Heatmap Work

The heatmap went through several iterations.

Current behavior:

- Uses absolute scenic score normalization: `0 -> 10`.
- High scenic scores can still be red.
- Low scores are blue.
- Middle scores transition through green/yellow.
- Border artifacts are reduced by not using border tiles for robust normalization work earlier, then final color scale was changed to absolute `0-10`.
- High zoom switches to tile polygons instead of dots.
- Tile fill starts earlier for Z14 runs.
- Tile fill has no visible borders.

Important endpoint:

```text
GET /v1/heatmap?region=<region>&run_name=<optional>&max_points=3000
```

Response includes:

- `tile_zoom`
- `normalization`
- `geojson`
- `geojson_tiles`

## Geocoding Work

The geocoding path was intentionally simplified after experimentation.

What was tried:

- Nominatim fallback.
- Mapbox Search Box suggest/retrieve.
- Query expansion and intent ranking.
- Strict POI token filtering.

Final decision:

- Use a simple Mapbox Geocoding v5 path.
- Bias with region bbox.
- Avoid overengineering query matching.

This made behavior more predictable and easier to debug. Searches work correctly when the selected region covers the searched place.

Example:

```bash
curl "http://localhost:8080/v1/geocode?q=target%20south%20philly&region=philadelphia"
```

## Model And Reporting Work

The project now treats MVP app delivery as a near-term priority while model improvement continues.

Major ML/reporting changes:

- V4 pipeline scripts were added.
- Model promotion gate was added.
- Route compare service wrapper was added.
- Benchmark comparison script was added.
- V5 model was promoted as the active practical checkpoint.
- Heuristic report defaults were updated toward the current active checkpoint.
- Report generation supports region-oriented runs.
- Report/viewer work includes cluster view and route overlay support.

Important files:

- `scripts/run_regression_v4_pipeline.py`
- `scripts/promote_regression_model.py`
- `scripts/compare_regression_on_benchmark.py`
- `scripts/route_compare_service.py`
- `scripts/heuristic_report.py`
- `scripts/heuristic_report_region.py`
- `src/heuristics/labeler.py`
- `src/heuristics/report.py`
- `src/scenic_scorer/regression.py`
- `data/processed/regression/model_registry.json`

## Contributor / Annotation Workflow

A contributor label flow was added to support future credit-based annotation workflows.

Included pieces:

- contributor profile creation
- task assignment
- label submission
- leaderboard
- admin review/promotion endpoint

This is still MVP-level and local-first.

Important files:

- `src/app_api/contrib_repo.py`
- `src/app_api/schemas.py`
- `tests/test_app_api.py`

## Tests And Verification

Useful checks:

```bash
uv run python -m py_compile src/app_api/main.py web/app.js
uv run --with pytest,httpx pytest tests/test_app_api.py -q
uv run --with pytest pytest tests/test_promotion_and_route_service.py -q
```

Recent checks during development:

- `node --check web/app.js`
- `uv run python -m py_compile src/app_api/main.py`
- `uv run python -m py_compile src/route_planner/service.py`

## Git State At Export

Recent notable commits:

```text
0d337321 Polish MVP planner UX and heatmap readability to make the app feel production-ready across devices. Harden routing/geocoding behavior and docs to align with current checkpoint and app progress.
c242e4f0 Merge branch 'local'
61f71fc0 V1 Really done
1da27c63 v1 routes
18261336 Add web MVP UI with address autocomplete and robust geocoding/route errors
072c49e3 Add MVP FastAPI backend for route compare and contributor workflows
3a1ab4e2 Prioritize MVP app delivery in project docs and TODOs
```

At the time of this export:

```text
main...origin/main
```

## Known Limitations

- This is still a local MVP, not production deployment.
- Mapbox token must be supplied by environment variable.
- Region search quality depends on selecting the correct region bbox.
- Routing only works where road graph coverage exists.
- Scenic scores only apply where report tile coverage overlaps road graph edges.
- The web UI is plain HTML/CSS/JS, which is good for speed but may need a framework later if the app grows.
- Contributor/credits workflow is functional scaffolding, not a full trust/payment/rewards system.

## Good Next Steps

1. Finish UI polish on the navigation panel.
2. Add route alternatives list and route selection.
3. Add explicit heatmap run selector.
4. Cache geocode results locally in the browser.
5. Add a route details drawer with distance, time, scenic score, and detour explanation.
6. Add production config separation for local/dev/prod API base URLs.
7. Improve mobile layout after desktop UI stabilizes.
8. Later, return to manual labels and model improvement once MVP flow is solid.

