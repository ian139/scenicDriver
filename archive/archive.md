# Archive Manifest

Purpose: preserve non-primary files while keeping the active workspace focused on the current Scenic Drive training, reporting, app, and routing paths.

## Archive Policy
- Active paths live in:
  - `notebooks/train.mo.py`
  - `notebooks/classifier.mo.py`
  - `notebooks/regression.mo.py`
  - `notebooks/learned_scoring.mo.py`
  - `notebooks/annotate_scenic.mo.py`
  - `scripts/ingest/download_bbox_tiles.py`
  - `scripts/reports/heuristic_report.py`
  - `scripts/reports/rebuild_report_from_labels.py`
  - `scripts/annotation/annotate_scenic_web.py`
  - `scripts/modeling/*` regression pipeline scripts
  - `scripts/routing/build_graph_from_osm.py`
  - `scripts/routing/route_compare_service.py`
  - `apps/web/`
  - `apps/mobile/`
- Non-primary scripts, notebooks, and notes are moved to `archive/` instead of deleted.
- Archived files are records or historical utilities; restore only after confirming they do not create a duplicate active workflow.

## Archived Scripts
Moved to `archive/scripts/`:
- `archive/scripts/build_graph_from_osm.py` — historical copy; active replacement is `scripts/routing/build_graph_from_osm.py`.
- `archive/scripts/classify_demo.py`
- `archive/scripts/download_naip_bbox.py`
- `archive/scripts/download_naip_state.py`
- `archive/scripts/download_resisc45.py`
- `archive/scripts/download_sample_tiles.py`
- `archive/scripts/extract_terrain_features.py`
- `archive/scripts/heuristic_report_server.py`
- `archive/scripts/route_demo_geojson.py`
- `archive/scripts/route_demo_graph_json.py`

## Archived Notebooks
Moved to `archive/notebooks/`:
- `archive/notebooks/heuristic_ui.mo.py`

## Archived Notes
Moved to `archive/notes/`:
- `archive/notes/work-export-2026-04-20.md`
- `archive/notes/coding-agent-session-mvp.md`
- `archive/notes/archive-review-rebuild-guide.md`
- `archive/notes/misplaced-job-application-AGENTS.md`

## Notes
- This archive is intended to reduce confusion and operational risk, not to remove project history.
- Active docs must use the current grouped paths under `apps/`, `docs/`, and `scripts/`.
- Historical docs under `archive/notes/` may retain old command paths because they are records, not active instructions.
