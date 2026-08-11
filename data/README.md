# Data Layout

## Local Paths

- `data/raw/images/`: local tile cache (optional when using S3-first mode)
- `data/raw/labels.csv`: weak labels from heuristic scoring
- `data/raw/labels_human.csv`: manual annotations (`scenic_human`, confidence, notes)
- `data/processed/`: reports, feature exports, run metadata, regression artifacts

## S3 Canonical Paths

Keep tile keys under:

- `raw/images/satellite/z{zoom}/{region}/{x}_{y}.png`
- `raw/images/terrain/z{zoom}/{region}/{x}_{y}.png`

No extra nested zoom folder under region.

Lifecycle rules target the parent `raw/images/` prefix so both satellite and terrain tiles share the same raw image transition policy.

## Open-Data Acquisition Layout

The active ML acquisition contract is `config/data_sources/naip_3dep_v1.json`.
It writes only beneath the source-versioned root:

- `data/raw/sources/naip_3dep_v1/images/satellite/z14/<region>/<x>_<y>.png`
- `data/raw/sources/naip_3dep_v1/images/terrain/z14/<region>/<x>_<y>.png`
- `data/raw/sources/naip_3dep_v1/cache/` for content-addressed source objects
- `data/raw/sources/naip_3dep_v1/catalog/` for hash-pinned local catalog snapshots

The region contract is `config/data_sources/regions_v1.json`; its ignored
Census jurisdiction and GSHHG land artifacts live under
`data/raw/boundaries/` and must match the SHA-256 values in that contract.
Planning is offline by default. Missing catalogs produce a
`discovery_authorization_request.json`; no requester-pays access occurs until
positive caps and the explicit requester-pays acknowledgement are supplied.
Legacy Mapbox images remain historical inputs only and must never share cache,
feature, prediction, or report identities with NAIP/3DEP artifacts.

Report tooling defaults (unless overridden in env):
- `SCENIC_S3_BUCKET=scenicdriver-data`
- `SCENIC_S3_ONLY=1`

## Annotation Workflow Notes

- For local tile loading in annotator: `raw_dir = data/raw`
- For S3 tile loading in annotator: `raw_dir = s3://<bucket>/raw`
- Manual labels are appended/upserted to `data/raw/labels_human.csv`

## Active-Learning Run Artifacts

Every Stage-One run owns one ignored directory:
`data/processed/active_learning/<run_name>/`. Its canonical artifacts are:

- `region_manifest.json`, `tile_manifest.csv`, `inventory_report.json`, and
  `acquisition_preflight.json` for bounded acquisition;
- `candidate_pool.csv`, `feature_embeddings.npz`, and
  `scoring_manifest.json` for resumable weak-label/model inference;
- `annotation_batch.csv`, `batch_manifest.json`, `selection_diagnostics.json`,
  `geographic_splits.csv`, and `leakage_audit.json` for deterministic
  selection and fixed train/validation/test membership;
- `mixed_labels.csv` for per-run mixed supervision (aggregated human overrides overlaying weak heuristic labels);
- `benchmark_split.csv`, `benchmark_tiles.csv`, `agreement_by_annotator.csv`,
  `agreement_by_pair.csv`, and `summary.json` for per-run fixed-geographic benchmark splits and agreement reports;
- `stage1_handoff.json` for fail-closed Stage-Two admission.

### Cumulative vs. Per-Run Snapshot Semantics

- `data/raw/labels_human.csv` is cumulative across runs and retains all historical manual annotations across runs.
- Per-run builders (`build_mixed_labels.py` and `build_human_benchmark.py`) take cumulative annotations as input and filter them to the tile identities present in the run's fixed `geographic_splits.csv`.
- The generated per-run artifacts (`mixed_labels.csv`, `benchmark_split.csv`, `benchmark_tiles.csv`, etc.) are per-run snapshots strictly aligned with that run's fixed geographic assignments.

`candidate_pool.csv` contains scalar scores and identities only. Dense
embeddings, class outputs, and terrain vectors stay in
`feature_embeddings.npz` with explicit row indices. Missing imagery remains a
status row and is never selector-eligible. Do not overwrite a prior run name
or move generated artifacts into Git.


## Training and Validation Working Sets

These datasets support model development and are not, by themselves, a list of
deployed application regions:

- `masswhites` at `z14`: primary learned-model training set.
- `amherst_ma` at `z16`: secondary local validation set.

## Configured and Deployed Regions

The application region catalog is `config/app_regions.json`. It currently
defines `new_england_north`, `masswhites`, `philadelphia`, and `pittsfield`;
their graph paths and bounding boxes in that file are the source of truth for
configured regions. Configuration alone does not mean that the corresponding
artifacts are present or deployed.

The hosted beta deployment is specifically the New England North release
described in [`docs/setup/deployment.md`](../docs/setup/deployment.md). Its
required graph, report, and model artifacts are listed in that runbook. The
training/validation working sets above should not be interpreted as deployed
regions; `amherst_ma`, in particular, is not an entry in the application
region catalog.

## Regression Artifacts (Step 3)

- Feature exports may include:
  - `sample_weights` (for mixed supervision, e.g. human=3.0 / heuristic=1.0)
  - `image_paths` (row alignment/audit)
- Recommended mixed labels source:
  - `data/processed/regression/labels_masswhites_z14_mixed5000.csv`

## Human Benchmark Artifacts

- Build per-run benchmark split + agreement stats:
  - `uv run python scripts/annotation/build_human_benchmark.py --annotations-csv data/raw/labels_human.csv --geographic-splits-csv data/processed/active_learning/<run_name>/geographic_splits.csv --output-dir data/processed/active_learning --run-name <run_name>`
- Build overlap batch for cross-annotator agreement collection:
  - `uv run python scripts/annotation/build_overlap_batch.py --annotations-csv data/raw/labels_human.csv --labels-csv data/processed/regression/labels_masswhites_z14_mixed5000.csv --source-annotator ian --target-annotator paperspace --sample-size 200 --seed 42 --output-csv data/processed/regression/overlap_batch_ian_to_paperspace_200.csv`
- Output folder layout (`data/processed/active_learning/<run_name>/` or `data/processed/regression/<run_name>/`):
  - `benchmark_tiles.csv`
  - `benchmark_split.csv`
  - `agreement_by_annotator.csv`
  - `agreement_by_pair.csv`
  - `summary.json`
## Report Overlay Artifacts

Report viewer route overlay supports either layout in `data/processed/heuristic_runs/<run_name>/report/`:

- `route.geojson` (recommended; one file containing scenic + baseline features), or
- split files: `route_scenic.geojson` and `route_fast.geojson`.

When route overlay is present, the viewer also displays route comparison metrics:
- scenic vs baseline distance
- scenic vs baseline travel time
- scenic score delta

Generate combined overlay directly:

- `uv run python scripts/routing/route_compare_service.py --start 42.40 -72.70 --end 42.48 -72.62 --scenic-weight 0.8 --run-name <run_name> --graph-geojson <graph>`

## Canonical Full-Bbox Road Graph

The deployed New England North graph is the ignored SQLite artifact
`data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3`.
Its source extracts are cached under
`cache/osm-pbf/new_england_north_full_bbox_v1/`, conversion intermediates under
`cache/osmnx/new_england_north_full_bbox_v1/`, and build metadata beside the
graph. These paths stay outside Git; beta bootstrap downloads the compressed
artifact from the deployment manifest.

Build it with the canonical full-bbox command in the [deployment runbook](../docs/setup/deployment.md), then run:

```bash
uv run python scripts/routing/check_beta_artifacts.py --project-root .
```

The checker validates the SQLite format/schema, configured bbox, row counts,
coverage probes, and integrity before startup. Keep the previous corridor
artifact and manifest for rollback: restore its graph/config/manifest paths,
rerun the checker, and restart only after the rollback artifact passes.

## Git Policy

Large datasets and generated artifacts are ignored via `.gitignore`.
