# Archive Manifest

Purpose: preserve non-primary files while keeping the active workspace focused on the current training + S3 ingestion path.

## Archive Policy
- Active path lives in `notebooks/train.mo.py`, `notebooks/regression.mo.py`, `scripts/download_bbox_tiles.py`, `scripts/heuristic_report.py`, `scripts/rebuild_report_from_labels.py`, and `scripts/annotate_scenic_web.py`.
- Non-primary scripts/models are moved to `archive/` instead of deleted.
- Archived files are restorable with `mv` commands listed below.

## Archived Scripts
Moved to `archive/scripts/`:
- `build_graph_from_osm.py`
- `classify_demo.py`
- `download_naip_bbox.py`
- `download_naip_state.py`
- `download_resisc45.py`
- `download_sample_tiles.py`
- `extract_terrain_features.py`
- `heuristic_report_server.py`
- `route_demo_geojson.py`
- `route_demo_graph_json.py`

Reason: these are not in the current primary workflow and can be restored when routing/NAIP phases are resumed.

## Archived Model Checkpoints
Moved to `archive/models/`:
- `checkpoint.pt`
- `scenic_multitask_best.pt`

Reason: not used by the current default path (`models/classifier/best_model.pt`).

## Restore Commands
Restore archived scripts:

```bash
mv archive/scripts/*.py scripts/
```

Restore archived model checkpoints:

```bash
mv archive/models/*.pt models/
```

## Notes
- This archive is intended to reduce confusion and operational risk, not to remove project history.
- Before deleting archived files, confirm corresponding TODO phase is complete and replacement path is stable.
