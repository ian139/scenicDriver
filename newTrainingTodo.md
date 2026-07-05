# New Training TODO

## Goal

Improve scenic scoring without blocking on new human labeling. Use the current promoted v6 Vast model as the baseline, expand weakly supervised training data, test stronger geospatial embeddings, and promote only if the candidate improves on the existing human benchmark.

## Current Baseline

- Active model: `models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt`
- Registry source of truth: `data/processed/regression/model_registry.json`
- Known v6 corrected split metrics from `docs/research/ml-research-log.md`:
  - MAE: `0.2197`
  - RMSE: `0.4248`
  - Corr: `0.8851`
- Existing feature recipe:
  - RESISC45 ViT embedding
  - terrain features
  - RESISC45 classifier logits
  - mixed human/heuristic scenic labels
  - sample weighting, usually `human=4.0`, `heuristic=1.0`

## Decision

Do **not** make more human labeling the next bottleneck.

Recommended path:

1. Finish pulling more satellite + terrain tiles.
2. Generate weak heuristic labels for the expanded region/data set.
3. Train a larger weak/mixed-supervision candidate using the current feature pipeline.
4. Add a geospatial foundation embedding experiment.
5. Compare all candidates against v6 on the existing human benchmark.
6. Promote only if benchmark behavior improves, not just weak-label validation metrics.

## Phase 1: Finish Data Ingest

- Keep tile layout canonical:
  - `raw/images/satellite/z{zoom}/{region}/{x}_{y}.png`
  - `raw/images/terrain/z{zoom}/{region}/{x}_{y}.png`
- Prefer S3-first mode for large runs:
  - `SCENIC_S3_BUCKET=scenicdriver-data`
  - `SCENIC_S3_ONLY=1`
- Preserve active working sets:
  - `masswhites` at `z14` as primary
  - `amherst_ma` at `z16` as secondary/local validation

Acceptance:

- Satellite and terrain tiles exist for the target region.
- Paths match `data/README.md` canonical layout.
- Missing satellite/terrain pairs are either fixed or intentionally excluded.

## Phase 2: Generate Weak Labels + Reports

Generate or refresh heuristic labels for the expanded tile set.

Outputs should live under `data/processed/`, not git-tracked source paths.

Acceptance:

- New labels CSV has at least:
  - `image_path`
  - `scenic_score`
  - `class_id` if classifier signal is available
- A report run exists for visual QA.
- Obvious bad regions/classes are noted before training.

## Phase 3: Build Mixed/Weak Training Labels

If current `data/raw/labels_human.csv` applies to the tile set, overlay it. Do not require new annotations yet.

Command shape:

```bash
uv run python scripts/modeling/build_mixed_labels.py \
  --heuristic-labels <heuristic-labels.csv> \
  --annotations-csv data/raw/labels_human.csv \
  --output data/processed/regression/labels_<region>_z<zoom>_mixed_vNext.csv \
  --aggregate mean
```

Acceptance:

- Mixed labels include audit columns when human labels overlap:
  - `human_annotation_count`
  - `human_annotator_count`
  - `scenic_human_std`
  - `scenic_human_mean`
  - `scenic_human_median`
- Label source is preserved so sample weighting works.
- Record the labels file hash in run notes before training.

## Phase 4: Candidate A — Current Feature Pipeline, More Data

Export the current feature set:

```bash
uv run python scripts/modeling/export_regression_dataset.py \
  --labels-csv data/processed/regression/labels_<region>_z<zoom>_mixed_vNext.csv \
  --raw-dir s3://scenicdriver-data/raw \
  --output data/processed/regression/features_<region>_z<zoom>_mixed_vNext_h4.npz \
  --label-source-column label_source \
  --human-weight 4.0 \
  --heuristic-weight 1.0
```

Train:

```bash
uv run python scripts/modeling/train_regression_baseline.py \
  --dataset data/processed/regression/features_<region>_z<zoom>_mixed_vNext_h4.npz \
  --output models/scenic_regression_baseline_<region>_z<zoom>_mixed_vNext_weighted_h4.pt \
  --use-sample-weights
```

Evaluate:

```bash
uv run python scripts/modeling/evaluate_regression_baseline.py \
  --dataset data/processed/regression/features_<region>_z<zoom>_mixed_vNext_h4.npz \
  --checkpoint models/scenic_regression_baseline_<region>_z<zoom>_mixed_vNext_weighted_h4.pt \
  --metrics-json data/processed/regression/baseline_metrics_<region>_z<zoom>_mixed_vNext_weighted_h4.json
```

Acceptance:

- Candidate trains successfully.
- Metrics JSON is written.
- Dataset hash and labels hash are recorded.
- Candidate is compared to v6 on existing benchmark before promotion.

## Phase 5: Candidate B — Geospatial Foundation Embeddings

Test whether Earth-observation embeddings beat the RESISC-only visual representation.

Preferred first model:

- `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` or `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`

Experiment options:

1. Replace `vit_embeddings` with geospatial embeddings.
2. Concatenate geospatial embeddings with current RESISC ViT embeddings.
3. Keep terrain features and RESISC logits in both variants for a fair comparison.

Acceptance:

- Feature export writes a separate NPZ, e.g. `features_<region>_z<zoom>_mixed_vNext_prithvi_h4.npz`.
- The candidate uses the same labels and train/eval split as Candidate A.
- Any architecture change is minimal and documented in the run notes.
- Comparison isolates embedding quality, not label or split changes.

## Phase 6: RESISC45 Classifier Refresh — Optional, Not First

RESISC45 is useful, but scenic quality is not the same as scene class. Keep it as an auxiliary signal unless benchmark evidence says classifier errors dominate.

Possible dataset source:

- Hugging Face: `jonathan-roberts1/NWPU-RESISC45`

Only do this after Candidate A/B unless current classifier predictions are visibly poor.

Acceptance:

- New classifier checkpoint is evaluated independently.
- Regression features are re-exported with the new classifier checkpoint.
- Downstream scenic benchmark improves versus the current classifier checkpoint.

## Phase 7: Benchmark Comparison Gate

Use the existing human benchmark as the promotion gate. More weak labels alone are not proof.

Compare:

- current promoted v6
- Candidate A: larger current-feature weak/mixed run
- Candidate B: geospatial embedding run
- optional RESISC refresh candidate

Acceptance:

- Candidate beats or matches v6 on human benchmark metrics.
- Candidate does not collapse score range in reports.
- Route-level QA looks sane: scenic route should improve scenic score without absurd detours.

## Phase 8: Promotion

Promote only after benchmark + report + route QA.

Promotion tasks:

- Update `data/processed/regression/model_registry.json`.
- Save candidate checkpoint under `models/`.
- Save metrics and run notes under `data/processed/regression/`.
- Regenerate learned report.
- Run a route comparison smoke check.

Acceptance:

- Registry points to the promoted checkpoint.
- Report uses the promoted checkpoint.
- Route compare uses the promoted checkpoint.
- Previous v6 baseline remains available as a control; v4 remains available as a historical control.

## When To Add More Human Labels

Skip new labeling for now if Candidate A/B still gives useful movement.

Resume labeling when:

- candidates improve weak-label validation but not the human benchmark;
- benchmark confidence is too low for promotion;
- model behavior differs across regions and needs calibration;
- subjective scenic examples repeatedly disagree with heuristic scores.

Next human-label target when needed:

- grow from ~500 to `1000–1500` labels;
- keep overlap batches for agreement stats;
- stratify by class/score/region so labels cover edge cases, not just easy tiles.

## Open Questions

- Which expanded region should become the first vNext training set?
- Should geospatial embeddings use RGB Mapbox tiles only, or should we add Sentinel/HLS data for a cleaner Prithvi input path?
- Should route-level QA become a formal promotion metric after the next candidate run?
