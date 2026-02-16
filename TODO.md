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

## Next Steps

### Phase 1: Model Quality (Priority)
- [ ] Retrain classifier on a larger/higher-quality dataset (or expanded labels) to improve domain fit and reduce class mislabeling in Northeast regions.
- [ ] Keep classifier signal but shift it to model features/auxiliary loss instead of fixed manual class weights.
- [ ] Replace heuristic class-weight scoring with a learned scenic regressor that uses satellite embeddings + terrain features (and optional class probabilities) as inputs.
- [ ] Verify classifier loads with `uv run` and checkpoint `models/classifier/best_model.pt`.
- [x] Learned-scoring scaffold: feature export script (`scripts/export_regression_dataset.py`) + baseline trainer (`scripts/train_regression_baseline.py`) + evaluator (`scripts/evaluate_regression_baseline.py`).
- [ ] Build manual scenic annotation MVP (`notebooks/annotate_scenic.mo.py`) with 0-10 scoring, skip, confidence, and CSV output.
- [ ] Add stratified tile sampler for annotation batches (cross-region and class-balanced).
- [ ] Create human-labeled benchmark split and report agreement stats (tile overlap, annotator variance).
- [ ] Train with mixed supervision: human scenic labels (primary, higher weight) + heuristic labels (weak, lower weight).

### Phase 2: Data + Reporting
- [ ] Add new region pipeline (download → label → terrain features → train).
- [ ] Run per-region reports for: `rocky_mountains`, `olympic_peninsula`, `philadelphia` (Big Sur done).
- [ ] Optionally add "cluster view" for multi-region heatmap (group by region).

### Phase 3: Routing
- [ ] Routing: stabilize OSM ingest (osmnx version compatibility, bbox validation, smaller demo).
- [ ] Routing: add cached graph builder output under `data/processed/` with a deterministic run name.
- [ ] Routing: implement scenic edge scoring (tile score → edge score aggregation).
- [ ] Routing: add CLI for best-route (start/end, scenic weight, output GeoJSON).
- [ ] Routing: wire route overlay into heuristic viewer (auto-load `route.geojson` if present).
