# TODO

## Done
- [x] Extract heuristic labeling into shared module (`src/heuristics/labeler.py`).
- [x] Add report generator with histogram + heatmap + side panel (`src/heuristics/report.py`).
- [x] Add CLI to run heuristic labeling + report (`scripts/heuristic_report.py`).
- [x] Add local report server (`scripts/heuristic_report_server.py`, now archived at `archive/scripts/heuristic_report_server.py`).
- [x] Update regression notebook to use shared labeler (`notebooks/regression.mo.py`).
- [x] Add tests for labeler (pairing, determinism, parsing) (`tests/test_heuristics_labeler.py`).
- [x] Add region helper CLI (`scripts/heuristic_report_region.py`).
- [x] Update data docs (`data/README.md`).
- [x] Add troubleshooting section to `data/README.md` (timm/pandas, uv run).
- [x] Add browser-based annotation UI (`scripts/annotate_scenic_web.py`) with procedural navigation + CSV persistence.
- [x] Fix terrain seam artifact handling (zero-elevation border) to avoid inflated relief in `4863_*` tiles.
- [x] Verify classifier loads with `uv run` and checkpoint `models/classifier/best_model.pt`.
- [x] Train/evaluate baseline on human-only (`~500`) and mixed (`5000`, with human overrides) datasets.
- [x] Run human-weight sweep for mixed supervision (`h2/h3/h4`) and record recommendation (`h4`) in `data/processed/regression/weight_sweep_masswhites_z14.json`.
- [x] Wire learned-scoring report mode (`--scoring learned`) into `scripts/heuristic_report.py` and `scripts/heuristic_report_region.py`.

## Next Steps

### Phase 1: Model Quality (Priority)
- [ ] Retrain classifier on a larger/higher-quality dataset (or expanded labels) to improve domain fit and reduce class mislabeling in Northeast regions.
- [ ] Keep classifier signal but shift it to model features/auxiliary loss instead of fixed manual class weights.
- [ ] Replace heuristic class-weight scoring with a learned scenic regressor that uses satellite embeddings + terrain features (and optional class probabilities) as inputs.
- [x] Learned-scoring scaffold: feature export script (`scripts/export_regression_dataset.py`) + baseline trainer (`scripts/train_regression_baseline.py`) + evaluator (`scripts/evaluate_regression_baseline.py`).
- [x] Build manual scenic annotation MVP (`notebooks/annotate_scenic.mo.py`) with 0-10 scoring, skip, confidence, and CSV output.
- [x] Add stratified tile sampler for annotation batches (cross-region and class-balanced).
- [x] Create human-labeled benchmark split and report agreement stats (tile overlap, annotator variance) via `scripts/build_human_benchmark.py` (latest run: `data/processed/regression/masswhites_human_benchmark_v1/summary.json`).
- [x] Train with weighted mixed supervision: human scenic labels (primary, higher weight) + heuristic labels (weak, lower weight).
- [x] Add explicit overlap annotation tooling (`scripts/build_overlap_batch.py`) and annotator `--batch-csv` mode for same-tile cross-annotator labeling.
- [x] Complete overlap annotation pass (label generated overlap batch) so pairwise agreement metrics are non-empty (`data/processed/regression/masswhites_human_benchmark_v2/agreement_by_pair.csv`).
- [ ] Expand manual labels from 500 -> 1000-1500 and rerun mixed training/eval.

### Phase 2: Data + Reporting
- [ ] Add new region pipeline (download → label → terrain features → train).
- [ ] Run per-region reports for: `rocky_mountains`, `olympic_peninsula`, `philadelphia` (Big Sur done).
- [ ] Optionally add "cluster view" for multi-region heatmap (group by region).

### Phase 3: Routing
- [x] Routing: stabilize OSM ingest (osmnx version compatibility, bbox validation, smaller demo).
- [ ] Routing: add cached graph builder output under `data/processed/` with a deterministic run name.
- [x] Routing: implement scenic edge scoring (tile score → edge score aggregation).
- [x] Routing: enforce one-way/direction-aware traversal + parsed speed-limit travel times from OSM metadata.
- [x] Routing: add CLI for best-route (start/end, scenic weight, output GeoJSON).
- [x] Routing: wire route overlay into heuristic viewer (auto-load `route.geojson` if present).
