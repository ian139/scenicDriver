# TODO

## Done
- [x] Extract heuristic labeling into shared module (`src/heuristics/labeler.py`).
- [x] Add report generator with histogram + heatmap + side panel (`src/heuristics/report.py`).
- [x] Add CLI to run heuristic labeling + report (`scripts/heuristic_report.py`).
- [x] Add local report server (`scripts/heuristic_report_server.py`).
- [x] Update regression notebook to use shared labeler (`notebooks/regression.mo.py`).
- [x] Add tests for labeler (pairing, determinism, parsing) (`tests/test_heuristics_labeler.py`).
- [x] Add region helper CLI (`scripts/heuristic_report_region.py`).
- [x] Update data docs (`data/README.md`).

## Next Steps
- [ ] Run per-region reports for: `rocky_mountains`, `big_sur`, `olympic_peninsula`, `philadelphia`.
- [ ] Verify classifier loads with `uv run` and checkpoint `models/classifier/best_model.pt`.
- [ ] Optionally add "cluster view" for multi-region heatmap (group by region).
- [ ] Add small troubleshooting section to `data/README.md` (timm/pandas, uv run).
- [ ] Add new region pipeline (download → label → terrain features → train).
- [ ] Routing: stabilize OSM ingest (osmnx version compatibility, bbox validation, smaller demo).
- [ ] Routing: add cached graph builder output under `data/processed/` with a deterministic run name.
- [ ] Routing: implement scenic edge scoring (tile score → edge score aggregation).
- [ ] Routing: add CLI for best-route (start/end, scenic weight, output GeoJSON).
- [ ] Routing: wire route overlay into heuristic viewer (auto-load `route.geojson` if present).
