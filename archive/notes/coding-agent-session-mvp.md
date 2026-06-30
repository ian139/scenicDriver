# Coding Agent Session: ScenicDrive MVP

## Why This Session Stands Out

This session started with a research-heavy machine learning project and turned it into a working product loop. ScenicDrive already had promising pieces: scenic score modeling, heuristic report generation, road graph experiments, and tile data. What it did not yet have was a cohesive user-facing experience where someone could type two places, plan a route, compare scenic vs baseline paths, and visually understand where scenic regions were.

The session I am proud of was the one where we moved from research artifacts into an MVP navigation app. We paired through backend API design, frontend UI iteration, route graph integration, geocoding, heatmap rendering, and model-report interoperability. The result was not just code: it was a clearer product direction.

## What We Built

We built a local web navigation MVP for ScenicDrive.

The app lets a user:

1. Pick a region.
2. Search for a start and destination.
3. Generate a scenic route and a baseline route.
4. Compare route metrics.
5. Toggle a scenic heatmap overlay.
6. Use wider road graph regions when available.

The implementation added a FastAPI backend, a lightweight MapLibre frontend, routing comparison endpoints, geocoding support, heatmap data generation, and contributor annotation scaffolding.

Important areas touched:

- `src/app_api/`
- `src/route_planner/`
- `web/`
- `scripts/`
- `tests/`
- project docs and TODOs

## The Starting Point

At the start, ScenicDrive was mostly a data and model pipeline. It had heuristic reports, learned scenic scoring, road graph scripts, and model checkpoints. The user had been training and evaluating models, running local and S3-backed reports, and generating regional tile data.

The next question was: what should this become?

The answer became: first make a usable MVP. Model quality still mattered, but the highest-impact next step was turning scenic scores into something interactive: a road trip planning app.

That decision shaped the rest of the session.

## Backend API

We added a FastAPI app that became the bridge between model artifacts and a product interface.

Key endpoints:

- `GET /`
- `GET /v1/healthz`
- `GET /v1/regions`
- `GET /v1/geocode`
- `POST /v1/route/compare`
- `GET /v1/heatmap`
- contributor annotation endpoints

The route comparison endpoint wraps existing route planner logic so the web app can ask for one response containing both scenic and baseline routes. It also returns metrics and route GeoJSON in one payload.

One important improvement was error handling. Early versions surfaced route failures as raw 500 errors. We changed no-route cases into structured `422` responses with useful hints. That made frontend behavior much easier to reason about.

## Geocoding

Geocoding went through several iterations.

We tried:

- simple Mapbox Geocoding
- Nominatim fallback
- query expansion
- local slang handling
- POI intent filtering
- Mapbox Search Box suggest/retrieve

At one point, queries like `target south philly` returned completely wrong places because the geocoder matched fuzzy text outside the intended region. We debugged with real curl responses and adjusted.

Eventually, we intentionally simplified. The user correctly called out that we were overengineering it. The final approach uses Mapbox Geocoding v5 with region bbox bias and keeps behavior predictable. That was a good engineering moment: the better solution was not more cleverness, but less.

## Route Graphs And Region Coverage

A major issue appeared when the scenic run covered a broad Masswhites area but the road graph only covered Pittsfield. The heatmap could show a wide scored region, but routes only worked where the graph existed.

We diagnosed that routing was limited by graph coverage, not model coverage.

The user generated a wider Masswhites road graph using the same bbox as the Masswhites Z14 scenic run. Then we patched the API to discover both:

- `road_graph.geojson`
- `road_graph.json`

We also updated route loading so serialized RoadGraph JSONs can be used directly.

That change made the app much more flexible. A region no longer had to be a hand-authored GeoJSON graph; it could use generated graph JSON output too.

## Heatmap Rendering

The heatmap work was one of the most iterative parts of the session.

The first version used a MapLibre heatmap layer. It looked smooth, but the colors were density-relative. That meant colors changed when zooming and high-density areas could turn red even if they were not actually high scenic scores.

We fixed that by changing the meaning of color:

- Red means high absolute scenic score.
- Blue means low absolute scenic score.
- The scale is fixed to `0..10`.

Then another issue appeared: when zoomed in, heatmap points looked like dots. We solved that by returning tile polygons from the backend and switching rendering by zoom level:

- wider zoom: smooth/blurred score overlay
- high zoom: colored tile polygons

For Z14 runs, the tile fill appears earlier. For Z16, the transition stays later. This made the heatmap feel more natural at different tile resolutions.

We also removed tile borders so the high-zoom layer looked cleaner.

## Frontend UI

The frontend started as a straightforward route form and evolved into a more navigation-like map app.

Major UI changes:

- Map-first layout.
- Floating route controls.
- `From` and `To` address fields.
- Settings menu for advanced controls.
- Coordinate mode moved into settings.
- Loading animation for route planning.
- Route metrics cards.
- Heatmap toggle.

There were several rounds of design critique. Some versions looked too much like a generic AI-generated dashboard: too large, too flashy, too much spacing, and too developmental. We simplified the structure, removed the top header, moved settings into a menu, made the route panel smaller, and replaced the log-like status area with a compact status pill.

The UI still has room to improve, but the session moved it from “debug interface” to “early product surface.”

## Contributor Annotation Workflow

We also added scaffolding for a future annotation workflow.

The API can now support:

- contributor sessions
- label tasks
- submitted labels
- contributor profiles
- leaderboard
- admin review flow

This connects to a larger product idea: users could help annotate scenic imagery and earn credits or reputation. It is still MVP scaffolding, but the backend shape is in place.

## Model And Pipeline Work

The session also touched the ML pipeline and model promotion direction.

Work included:

- v4 pipeline scripts
- promotion gate script
- benchmark compare script
- route compare service wrapper
- MPS-aware training paths
- current active model/registry documentation
- report generation fixes

We also made a practical call: use the current v5 checkpoint as the working model for the MVP, rather than continuing to train indefinitely. That kept momentum focused on product integration.

## Testing And Verification

We repeatedly used small verification loops:

- `node --check web/app.js`
- `uv run python -m py_compile ...`
- API curl checks
- route endpoint checks
- region discovery checks
- test suite checks for API and route service behavior

This was important because we were iterating quickly across frontend, backend, and data artifacts. Small compile and endpoint checks helped keep the work from drifting into a broken state.

## Technical Decisions I Am Proud Of

The best decisions were not always the most complex ones.

The biggest one was deciding to prioritize the MVP. Instead of spending more time only improving model quality, we built the route planning loop. That made the project easier to understand and evaluate.

Another good decision was simplifying geocoding after overcomplicating it. We tried clever ranking and provider orchestration, but the simpler Mapbox path was easier to maintain and reason about.

The heatmap evolution was also important. We moved from a visually misleading density heatmap to an absolute score-based overlay. That made the visualization more honest.

Finally, supporting both graph formats made the region system more robust. Generated graph JSONs can now be used directly by the app.

## What Changed In The Repo

Large additions:

- FastAPI backend under `src/app_api/`
- Static web app under `web/`
- Route comparison API integration
- Heatmap endpoint and rendering
- Contributor annotation API scaffolding
- v4/v5 model pipeline support scripts
- promotion and benchmark comparison tooling
- new tests for API and promotion/route service behavior

Important files:

- `src/app_api/main.py`
- `src/app_api/schemas.py`
- `src/app_api/contrib_repo.py`
- `src/route_planner/service.py`
- `web/index.html`
- `web/styles.css`
- `web/app.js`
- `scripts/run_regression_v4_pipeline.py`
- `scripts/promote_regression_model.py`
- `scripts/compare_regression_on_benchmark.py`
- `tests/test_app_api.py`
- `tests/test_promotion_and_route_service.py`

## Final State

By the end of the session, ScenicDrive had:

- a working local API
- a working local web app
- address search
- route comparison
- scenic vs baseline route metrics
- route rendering
- heatmap overlay
- wide road graph support
- contributor annotation scaffolding
- updated docs and TODO priorities

The project went from “promising ML research pipeline” to “usable scenic navigation MVP.”

## Why I Am Proud Of This Session

This was a strong example of coding-agent collaboration because it required more than writing isolated code. We had to keep product direction, ML artifacts, geospatial data, routing, frontend UX, and developer workflow aligned at the same time.

There were several moments where the right move was to listen to the user’s feedback and simplify. The user pushed back on overengineered geocoding and AI-looking UI. Those critiques made the final system better.

The session shows the kind of work I value most: turning scattered but promising technical pieces into a coherent product loop, while staying flexible enough to change direction when the user’s intuition is right.

