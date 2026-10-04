# Archive Manifest

`archive/` is a **frozen, superseded snapshot**. Nothing under it is an active
workflow, an active instruction, or an active dependency of the current Scenic
Drive training, reporting, app, or routing paths. It exists to preserve project
history and reduce operational risk, not to be restored casually.

This manifest describes exactly what `archive/` contains. It does not track the
active tree and must not be used to infer which active scripts exist: current
paths and commands live in [`README.md`](../README.md), [`docs/`](../docs/), and
[`data/README.md`](../data/README.md). No file under `src/`, `scripts/`, `apps/`,
or `tests/` imports or executes anything here, and `pytest` collects only
`tests/`.

## Policy

- Non-primary scripts, notebooks, and notes are moved here instead of deleted.
- Archived files are records or historical utilities. Restore one only after
  confirming it does not duplicate an active workflow, and drop the archived copy
  once it is active again.
- Nothing under `archive/` is an active instruction. Do not run an archived entry
  point as part of a current pipeline.

## Contents

### `archive/apps/`

- `archive/apps/web/` — superseded browser planner: `index.html`, `app.js`,
  `styles.css`, and `data/masswhites_heatmap_cells.js`. The New England North app
  under `apps/new_england_north/` is canonical.
- `archive/apps/mobile/` — superseded Expo shell: `app.json`, `App.tsx`,
  `designTokens.ts`, `package.json`, `tsconfig.json`. The responsive web app is
  canonical.

### `archive/scripts/`

- `build_graph_from_osm.py` — historical copy; the active replacement is
  `scripts/routing/build_graph_from_osm.py`.
- `classify_demo.py`
- `download_naip_bbox.py`
- `download_naip_state.py`
- `download_resisc45.py`
- `download_sample_tiles.py`
- `extract_terrain_features.py`
- `heuristic_report_server.py`
- `route_demo_geojson.py`
- `route_demo_graph_json.py`

These are superseded acquisition, feature, report, and routing demos. The active
open-data acquisition path is `scripts/ingest/plan_active_learning_region.py` with
`scripts/ingest/naip_3dep_workflow.py`; none of the archived download scripts is
part of it.

### `archive/notes/`

- `cmux-workflow-legacy.md` — historical, non-active workflow. The current
  replacement is [`docs/internal/cmux-workflow.md`](../docs/internal/cmux-workflow.md).

### `archive/archive.md`

This manifest.
