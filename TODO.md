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

### Phase 0: MVP App (Priority)
- [ ] Ship hosted route-compare API wrapper for app integration (stable request/response contract).
- [ ] Build responsive web MVP for road-trip planning (start/end, scenic weight, route compare, map overlay).
- [ ] Add optional signed-in contributor annotation mode with credits + shadow QA queue.
- [ ] Add region selector/data registry in app and validate at least: `pittsfield`, `philadelphia`.

### Phase 1: Model Quality (Priority)
- [ ] Retrain classifier on a larger/higher-quality dataset (or expanded labels) to improve domain fit and reduce class mislabeling in Northeast regions.
- [ ] Keep classifier signal but shift it to model features/auxiliary loss instead of fixed manual class weights.
- [ ] Replace heuristic class-weight scoring with a learned scenic regressor that uses satellite embeddings + terrain features (and optional class probabilities) as inputs.
- [x] Learned-scoring scaffold: feature export script (`scripts/export_regression_dataset.py`) + baseline trainer (`scripts/train_regression_baseline.py`) + evaluator (`scripts/evaluate_regression_baseline.py`).
- [x] Build manual scenic annotation MVP (`notebooks/annotate_scenic.mo.py`) with 0-10 scoring, skip, confidence, and CSV output.
- [x] Add stratified tile sampler for annotation batches (cross-region and class-balanced).
- [x] Create human-labeled benchmark split and report agreement stats (tile overlap, annotator variance) via `scripts/build_human_benchmark.py` (latest run: `data/processed/regression/masswhites_human_benchmark_v2/summary.json`).
- [x] Train with weighted mixed supervision: human scenic labels (primary, higher weight) + heuristic labels (weak, lower weight).
- [x] Add explicit overlap annotation tooling (`scripts/build_overlap_batch.py`) and annotator `--batch-csv` mode for same-tile cross-annotator labeling.
- [x] Complete overlap annotation pass (label generated overlap batch) so pairwise agreement metrics are non-empty (`data/processed/regression/masswhites_human_benchmark_v2/agreement_by_pair.csv`).
- [x] Keep `v2` regression checkpoint as active default after v1/v3 parity check (documented in `MLResearch.md`).
- [x] Promote `v4` as active checkpoint after passing regression + held-out benchmark gates (`data/processed/regression/model_registry.json`, `data/processed/regression/benchmark_compare_masswhites_v4_vs_v2.json`).
- [ ] Expand manual labels from 500 -> 1000-1500 and rerun mixed training/eval.

### Phase 2: Data + Reporting
- [ ] Add new region pipeline (download → label → terrain features → train).
- [ ] Run per-region reports for: `rocky_mountains`, `olympic_peninsula` (`philadelphia` done, Big Sur done).
- [ ] Optionally add "cluster view" for multi-region heatmap (group by region).

### Phase 3: Routing
- [x] Routing: stabilize OSM ingest (osmnx version compatibility, bbox validation, smaller demo).
- [x] Routing: add cached graph builder output under `data/processed/` with a deterministic run name (`scripts/build_graph_from_osm.py` auto run folder + `run.json`).
- [x] Routing: implement scenic edge scoring (tile score → edge score aggregation).
- [x] Routing: enforce one-way/direction-aware traversal + parsed speed-limit travel times from OSM metadata.
- [x] Routing: add CLI for best-route (start/end, scenic weight, output GeoJSON).
- [x] Routing: wire route overlay into heuristic viewer (auto-load `route.geojson` if present).
- [x] Routing: viewer route overlay fallback for split files (`route_scenic.geojson` + `route_fast.geojson`) when `route.geojson` is absent.
- [x] Routing: show scenic vs baseline travel-time/distance/scenic deltas directly in viewer route panel.
- [x] Attempted `v3` rerun and documented results in `MLResearch.md` (`v3` reused v1-equivalent labels and should not be promoted).
- [x] Build `v4` mixed labels from overlap-aware source (`data/processed/heuristic_runs/masswhites_z14_learned_h4_v2/labels.csv`) plus latest human annotations.
- [x] Export `features_masswhites_z14_mixed5000_v4_h4.npz` and verify it is a new dataset artifact.
- [x] Train/evaluate `v4` and compare against v2; promoted via gate + registry update.
- [ ] Expand annotations to 1000+ and rerun benchmark + mixed training for `v5`.
