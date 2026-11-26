# ML Research Log: Learned Scoring v1 through v6

This canonical ML research log preserves the `masswhites` learned-scoring checkpoint history and the rationale for promoting `v6` in that era. The historical measurements below are not metrics for the current deployment.

## Current deployment identity

[`config/app_regions.json`](../../config/app_regions.json) defines the current default region as `new_england_north`, using Prompt Two exp02 run `prompt_two_candidate_exp02_fresh_test20_20260810` and checkpoint `data/processed/regression/checkpoints/0a165e429c8ac050524c7da409dd533d2de8849600c5dd8605aa1a1024d823e9.pt`. The route-enabled `northeast_expanded` entry uses run `prompt_two_candidate_exp02_expanded_20260810`; that entry declares no region-local checkpoint override.

The configured `masswhites` entry retains run `masswhites_z14_learned_h4_v2` and the v6 checkpoint recorded below, with `route_planning: false`. That legacy regional entry does not make v6 the current default deployment model. [README](../../README.md) contains current workflow commands; [FIXES.md](../../FIXES.md) records current engineering contracts.

## Historical Masswhites scope

All three versions below use:

- Region: `masswhites` (`z14`)
- Training size: `5000` mixed labels
- Weighting: `human_weight=4.0`, `heuristic_weight=1.0`
- Split seed: `42` (`val_split=0.15`)

## Version Summary

| Version | Dataset | Checkpoint | MAE | RMSE | Corr |
|---|---|---|---:|---:|---:|
| v1 | `data/processed/regression/features_masswhites_z14_mixed5000_weighted_h4.npz` | `models/scenic_regression_baseline_masswhites_z14_mixed5000_weighted_h4.pt` | `0.4820` | `0.6426` | `0.7472` |
| v2 | `data/processed/regression/features_masswhites_z14_mixed5000_v2_weighted_h4.npz` | `models/scenic_regression_baseline_masswhites_z14_mixed5000_v2_weighted_h4.pt` | `0.2743` | `0.3686` | `0.9227` |
| v3 | `data/processed/regression/features_masswhites_z14_mixed5000_v3_h4.npz` | `models/scenic_regression_baseline_masswhites_z14_mixed5000_v3_h4.pt` | `0.4820` | `0.6426` | `0.7472` |
| v4 | `data/processed/regression/features_masswhites_z14_mixed5000_v4_h4.npz` | `models/scenic_regression_baseline_masswhites_z14_mixed5000_v4_weighted_h4.pt` | `0.1879` | `0.2646` | `0.9609` |
| v6 | `data/processed/regression/features_masswhites_z14_mixed5000_v4_h4.npz` | `models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt` | `0.2197` | `0.4248` | `0.8851` |

Metrics source files:

- `data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_weighted_h4.json`
- `data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v2_weighted_h4.json`
- `data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v3_h4.json`
- `data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v4_weighted_h4.json`
- `data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v6_vast_weighted_h4.json`
- `data/processed/regression/baseline_metrics_masswhites_z14_mixed5000_v4_weighted_h4.local_verify.json`

## What Changed

## v1 (baseline mixed supervision)

- Labels file: `data/processed/regression/labels_masswhites_z14_mixed5000.csv`
- Human overrides existed, but no overlap-aware aggregation metadata.
- Label composition:
  - `4501` heuristic
  - `499` human override
- Feature export produced:
  - SHA256: `f66957307615f1413505c9552258667dcdfdc33c13ab35b3ddfc551f9a0fed9b`

## v2 (overlap-aware label aggregation)

- Labels file changed to: `data/processed/regression/labels_masswhites_z14_mixed5000_v2.csv`
- Built via `scripts/modeling/build_mixed_labels.py` by merging `labels_human.csv` and aggregating duplicate tile annotations.
- New columns added for auditability:
  - `human_annotation_count`
  - `human_annotator_count`
  - `scenic_human_std`
  - `scenic_human_mean`
  - `scenic_human_median`
- Same tile set and same 499 human overrides, but overlap tiles were aggregated:
  - Human override tiles: `499`
  - Tiles with 2 annotators: `200`
  - Tiles that changed score vs v1: `112` (all from overlap group)
  - Mean absolute score change on changed tiles: `0.5759`
- Feature export SHA256 changed:
  - `513036c572791421dd316f73cf1bb5f18bead68c2134572aa5e9b4ece21def3b`
- Result: major quality improvement (best version at that iteration).

## v3 (accidental v1-equivalent rerun)

- Intended as a new iteration, but export used `labels_masswhites_z14_mixed5000.csv` again (not `_v2.csv`).
- Output dataset is content-identical to v1:
  - `features_masswhites_z14_mixed5000_v3_h4.npz` SHA256 matches v1:
  - `f66957307615f1413505c9552258667dcdfdc33c13ab35b3ddfc551f9a0fed9b`
- Therefore v3 metrics match v1 exactly and underperform v2.

## Report-Level Behavior (5000-tile learned reports)

- v1 report (`masswhites_z14_learned_h4_5k`) and v3 report (`masswhites_z14_learned_h4_v3`) have matching summary distributions.
- v2 report (`masswhites_z14_learned_h4_v2`) is broader and better calibrated (higher max, wider spread), aligned with better validation metrics.

## Historical v6 promotion decision

The decision in the Masswhites v6 era was to use `v6` as the active model:

- `models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt`
- Registry source of truth at that time: `data/processed/regression/model_registry.json`

`v1`, `v2`, `v3`, `v4`, and `v5` served as baselines/controls. The v6 promotion compared against v4 using the corrected training/evaluation validation split in `baseline_metrics_masswhites_z14_mixed5000_v4_weighted_h4.local_verify.json`.

## Guardrail for Next Iteration

Before training the next iteration, verify the labels input and dataset hash:

1. Build labels with overlap-aware aggregation (`build_mixed_labels.py`).
2. Export NPZ from the intended labels CSV.
3. Record `sha256sum` of labels + NPZ in the run notes.
4. Train/eval only after hash check passes.
