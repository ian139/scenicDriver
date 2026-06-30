# Archive Review And Rebuild Guide

This document reviews the archived filesystem snapshot and provides a clean rebuild plan.

## Snapshot Scope

- Archive root: `archive/`
- New full snapshot: `archive/root_snapshot_20260215_203601/`
- Existing historical archive content retained in:
- `archive/scripts/`
- `archive/models/`
- `archive/archive.md`
- Root files intentionally left outside archive:
- `AGENTS.md`
- `AWS_BUCKET_SETUP.md`
- `CODEX.md`
- `INFRASTRUCTURE.md`
- `README.md`
- `TODO.md`

## What Was Moved

Moved into `archive/root_snapshot_20260215_203601/`:

- `.claude`
- `.env`
- `.env.example`
- `.gitignore`
- `.venv`
- `data`
- `models`
- `notebooks`
- `progno.egg-info`
- `pyproject.toml`
- `scripts`
- `src`
- `tests`
- `uv.lock`

Move manifest: `archive/root_snapshot_20260215_203601/MOVE_MANIFEST.txt`

## Archive-Wide File Scan

All files under `archive/` were scanned:

- Total files: `103,905`
- Text files read: `30,632`
- Binary files seen: `73,273`
- Read errors: `0`

Scan summary artifact: `archive/archive_scan_summary.json`

Largest footprint drivers:

- Python environment binaries under `archive/root_snapshot_20260215_203601/.venv/`
- Model checkpoints under:
- `archive/models/`
- `archive/root_snapshot_20260215_203601/models/`
- Tile assets under `archive/root_snapshot_20260215_203601/data/raw/images/`

## How The Archived System Is Built

### 1) Data Ingestion Layer

Primary implementation:

- `archive/root_snapshot_20260215_203601/src/data_pipeline/mapbox.py`
- `archive/root_snapshot_20260215_203601/scripts/download_bbox_tiles.py`

Pattern:

- BBox -> XYZ tile conversion
- Download via Mapbox API with retry/rate limiting
- Local caching by zoom/tile id
- Optional S3 upload and S3-only mode

Historical/secondary ingestion:

- NAIP download scripts in `archive/scripts/` (`download_naip_bbox.py`, `download_naip_state.py`)

### 2) Heuristic Labeling + Reporting

Primary implementation:

- `archive/root_snapshot_20260215_203601/src/heuristics/labeler.py`
- `archive/root_snapshot_20260215_203601/src/heuristics/report.py`
- `archive/root_snapshot_20260215_203601/scripts/heuristic_report.py`
- `archive/root_snapshot_20260215_203601/scripts/heuristic_report_region.py`

Pattern:

- Pair satellite + terrain tiles
- Decode terrain-rgb -> elevation-derived features
- Compute heuristic scenic score
- Optional classifier prior
- Emit `labels.csv` and interactive HTML report (`report.json`, map overlays, thumbs)

### 3) Learned Scoring (Step 3 Scaffold)

Primary implementation:

- `archive/root_snapshot_20260215_203601/scripts/export_regression_dataset.py`
- `archive/root_snapshot_20260215_203601/scripts/train_regression_baseline.py`
- `archive/root_snapshot_20260215_203601/scripts/evaluate_regression_baseline.py`
- `archive/root_snapshot_20260215_203601/src/scenic_scorer/regression.py`
- `archive/root_snapshot_20260215_203601/notebooks/learned_scoring.mo.py`

Pattern:

- Export fused training tensors (embeddings + terrain + logits + targets)
- Train baseline regressor
- Evaluate MAE/RMSE/correlation

### 4) Manual Annotation Workflow

Primary implementation:

- `archive/root_snapshot_20260215_203601/notebooks/annotate_scenic.mo.py`

Pattern:

- Batch selection (random/stratified)
- Human scenic score + confidence + notes
- Append/upsert to `labels_human.csv`
- Summary view in notebook

### 5) Routing Prototype

Primary implementation:

- `archive/root_snapshot_20260215_203601/src/route_planner/graph.py`
- `archive/root_snapshot_20260215_203601/src/route_planner/cost.py`
- `archive/root_snapshot_20260215_203601/src/route_planner/planner.py`

Historical routing utilities:

- `archive/scripts/build_graph_from_osm.py`
- `archive/scripts/route_demo_geojson.py`
- `archive/scripts/route_demo_graph_json.py`

Pattern:

- Graph abstraction with node/edge model
- Scenic-aware edge cost
- A* search over road graph
- OSM ingest prototype with version-compat fallbacks

### 6) API Skeleton

- `archive/root_snapshot_20260215_203601/src/api/`

Pattern:

- Early FastAPI structure present, not primary execution path

### 7) Test Surface

- `archive/root_snapshot_20260215_203601/tests/test_heuristics_labeler.py`
- `archive/root_snapshot_20260215_203601/tests/test_route_planner.py`
- `archive/root_snapshot_20260215_203601/tests/test_scenic_scorer.py`

Pattern:

- Unit-level coverage for core formula/graph behavior
- Limited end-to-end, limited performance/regression guarantees

## Specific Implementation Gaps And Improvements

### A) Reproducibility/Versioning

Observed:

- Dataset/feature/model versioning is not uniformly enforced across all scripts.
- S3/local path contracts are partly implicit.

Improve:

- Introduce explicit manifest contract:
- `dataset_manifest.json` (bbox, zoom, source, tile count, hash)
- `features_manifest.json` (encoder hash, preprocessing version, dtype, dims)
- `model_manifest.json` (train config hash, dataset hash, commit hash)

### B) Code Surface Duplication

Observed:

- Overlap between active scripts and archived script variants.
- Multiple paths for similar tasks (routing demos, NAIP ingestion variants).

Improve:

- Collapse to one canonical CLI per capability:
- ingest
- label/report
- feature export
- train/eval
- route build/query
- Keep historical scripts only as references in archive, not executable defaults.

### C) Data/Compute Separation

Observed:

- Snapshot includes runtime environment (`.venv`) and generated binaries mixed with source.

Improve:

- Enforce clean artifact boundaries:
- source only in repo
- reproducible env lock files only
- data/models/cache externalized (S3 + manifests)

### D) Routing Readiness

Observed:

- `RoadGraph.find_nearest_node` currently brute-force.
- Spatial index TODO remains unimplemented.
- OSM compatibility handling is fragile across osmnx versions.

Improve:

- Add deterministic OSM adapter with pinned versions and validated API wrapper.
- Add spatial index (R-tree or H3/S2 index) for nearest-node and segment lookup.
- Precompute `road_segment_id -> tile_ids` index as a first-class artifact.

### E) Model Quality Controls

Observed:

- Heuristic labels dominate target generation.
- Human labels exist as workflow but not yet fully integrated as primary benchmark/training anchor.

Improve:

- Promote human-labeled split to authoritative eval set.
- Train mixed-supervision head:
- high weight on human labels
- low weight on heuristic pseudo-labels
- Add per-region calibration metrics and drift checks.

### F) Notebook Reliability

Observed:

- marimo workflows are productive but can break from cell wiring/redefinition issues.

Improve:

- Keep notebook logic thin; move logic to importable modules.
- Add notebook smoke tests for critical notebooks:
- `notebooks/regression.mo.py`
- `notebooks/learned_scoring.mo.py`
- `notebooks/annotate_scenic.mo.py`

### G) Testing/CI

Observed:

- Current tests cover units but do not protect end-to-end flows.

Improve:

- Add pipeline integration tests:
- tile pairing
- feature export schema validation
- training/eval run sanity checks
- report generation integrity (`report.json` contract)

## Clean Rebuild Plan (Recommended)

### Phase 0: Repository Reset Contract

- Recreate minimal source tree only (`src/`, `scripts/`, `notebooks/`, `tests/`).
- Keep archival snapshot immutable under `archive/`.

### Phase 1: Data Contract First

- Define canonical manifests for tiles/features/models.
- Enforce manifest validation before any training/inference.

### Phase 2: Step 3 Core Pipeline

- Keep one production path:
- export features
- train baseline
- evaluate baseline
- register artifacts (local + S3 + manifest)

### Phase 3: Human Label Integration

- Finalize annotation tool contract.
- Build benchmark split and agreement reports.
- Retrain with mixed supervision.

### Phase 4: Routing Integration

- Build stable road graph ingest with pinned OSM stack.
- Precompute road-tile index.
- Add fast online route scoring over cached embeddings/features.

## Regeneration Checklist

- Recreate dependencies from lockfile, not from archived `.venv`.
- Rebuild tiles/features from manifests, not from ad-hoc folder assumptions.
- Rebuild model checkpoints from tracked config + seed + dataset version.
- Re-enable routing only after spatial index and road-tile index are stable.

## Final Notes

The archive now contains both:

- historical non-primary components (`archive/scripts`, `archive/models`)
- a full project snapshot (`archive/root_snapshot_20260215_203601`)

This is enough to reconstruct the project cleanly with a stricter architecture:

- single workflow per task
- explicit data contracts
- versioned artifacts
- human-grounded evaluation
- routing over cached ML features instead of raw imagery at query time
