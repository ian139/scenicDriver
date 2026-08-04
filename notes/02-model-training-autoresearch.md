# OMP Autoresearch Prompt 2 — Scenic Model Training, Evaluation, and Optimization

Use this entire document as the second OMP autoresearch prompt from the Scenic Drive repository root, only after Prompt 1 has produced a valid stage-one handoff.

## Mission

Consume the immutable stage-one handoff, build a deterministic model-training autoresearch harness, run bounded experiments, and retain the simplest candidate that improves human-grounded regional performance without regressing the existing New England benchmark, report distribution, or route-level behavior. Promote a candidate only through explicit reproducible gates.

This is an implementation and execution task. Do not stop at a plan or training scaffold. Build missing orchestration, run the feasible experiment loop, preserve complete evidence, reject failed candidates cleanly, and leave the active registry unchanged unless every promotion gate passes.

## Required input

Locate exactly one intended stage-one artifact:

`data/processed/active_learning/<run_name>/stage1_handoff.json`

Prefer an explicit path supplied with the invocation. If none is supplied, accept only one unambiguous handoff whose hashes validate and whose `ready_for_stage2` flag is true. Fail closed rather than selecting the newest filename by guesswork.

Before any expensive work, validate:

- schema version and every referenced path;
- SHA-256 hashes and row/array counts;
- `data_complete`, `annotations_valid`, `splits_valid`, `benchmark_valid`, and `ready_for_stage2`;
- no train/validation/test geographic leakage or duplicate/adjacent tile leakage;
- fixed baseline registry identity and readable checkpoint;
- human-label, weak-label, region, class/scene, and score distributions;
- missing imagery/features and sample-weight validity;
- secrets are not embedded in manifests or generated commands.

If the handoff is not ready, finish any reachable validation/reporting, write an exact blocker report, and do not train on a guessed or partial dataset.

## Operating rules

- Follow `AGENTS.md`. Main owns architecture, experiment policy, shared contracts, integration, visible browser/route QA, promotion, and final decisions.
- Map the critical path and experiment dependency graph before implementation. Dispatch genuinely independent harness, feature, evaluation, and reporting slices in one parallel wave using several `task` workers and normally at most one `task-high` for the hardest correctness-sensitive invariant. Use isolated worktrees when ownership permits.
- Workers may run narrow owned checks but must skip project-wide suites, shared live-browser manipulation, and registry promotion. Main integrates and runs final gates once.
- Use `uv`, existing modeling CLIs, marimo notebooks, existing dependencies, ignored data/model paths, and explicit model-registry promotion.
- OMP sessions may use required `.env` credentials without additional confirmation, but must never print, commit, persist in artifacts, or unnecessarily share them.
- Preserve the current active v6 model as the immutable product baseline until a candidate passes all gates.
- Never use weak-label validation alone as promotion evidence. Never tune on the held-out human test set.
- Never loosen thresholds, change the benchmark, remove difficult rows, or alter a control after seeing candidate results.

## Canonical contracts to inspect and reuse

- `scripts/modeling/build_mixed_labels.py`
- `scripts/modeling/export_regression_dataset.py`
- `scripts/modeling/train_regression_baseline.py`
- `scripts/modeling/evaluate_regression_baseline.py`
- `scripts/modeling/compare_regression_on_benchmark.py`
- `scripts/modeling/promote_regression_model.py`
- `scripts/modeling/run_regression_v4_pipeline.py`
- `src/scenic_scorer/regression.py`
- `notebooks/learned_scoring.mo.py`
- `data/processed/regression/model_registry.json`
- `docs/research/ml-research-log.md`
- `docs/roadmap.md`
- deterministic-loop conventions in `autoresearch.sh` and `scripts/routing/autoresearch_multidetour.py`

Current exported NPZ inputs include `vit_embeddings`, `terrain_features`, `class_logits`, and `scenic_scores`, with optional `class_probs`, `class_ids`, `sample_weights`, and `image_paths`. Confirm actual contracts in source. Do not silently rewrite an old artifact in place.

## Required foundation

### 1. One-command stage-two orchestrator

Implement a canonical grouped modeling/autoresearch entry point that accepts the stage-one handoff and run name, then performs:

1. validation and immutable baseline capture;
2. feature/dataset preparation or reuse by verified hash;
3. deterministic experiment planning;
4. candidate training with isolated output paths;
5. validation and geographic-slice evaluation;
6. fixed human benchmark comparison;
7. calibration and report-distribution QA;
8. route-level QA;
9. control comparison;
10. keep/reject decision;
11. optional explicit promotion only after all gates;
12. durable run summary and hashes.

Support dry-run, resume, status, and clean rejection. A crashed run must not corrupt the active registry or make a partial checkpoint appear promotable.

Dataset export and evaluation must preserve exact row identity. If missing inputs are skipped, emit and hash the filtered labels/index artifact used by the NPZ; never compare an NPZ against an unfiltered labels CSV. Validate equal row counts and ordered `image_paths` before every benchmark.

Resume semantics must be truthful. Existing checkpoints that lack optimizer, scheduler, epoch, and RNG state may be reused only as completed immutable candidates or restarted from a declared initialization point; they cannot be described as resuming interrupted training. Extend the checkpoint contract if exact training continuation is required.

### 2. Deterministic experiment identity

Store each run under a unique ignored path such as:

`data/processed/modeling_autoresearch/<run_name>/`

Each experiment needs:

- stable experiment ID derived from dataset/split/feature/config hashes;
- parent/baseline identity;
- exact command/config, seeds, environment facts relevant to reproducibility, device, and dependency lock identity;
- input/output SHA-256 hashes;
- start/end state and failure reason;
- metrics by global and geographic/scene slice;
- resource/runtime observations;
- keep/reject decision and rationale;
- immutable checkpoint and metrics paths.

Adopt deterministic environment conventions where compatible: UTC, fixed locale, `PYTHONHASHSEED`, fixed numerical thread counts for evaluation, frozen dependencies, and fixed seeds. Do not claim bitwise GPU determinism unless measured. Separate deterministic evaluation from potentially nondeterministic training and record the distinction.

### 3. Fixed splits and leakage control

Use the stage-one fixed geographic assignments. Update existing training code if necessary so it accepts explicit train/validation/test membership rather than recreating a random row split. Preserve:

- New England benchmark as an immutable regression control;
- expanded-region human challenge validation/test partitions;
- adjacency/geographic-group exclusions;
- fixed random-control subset;
- enough per-slice support before reporting or gating a slice.

Training may use weak and human labels with recorded weights. Validation/test promotion metrics must be human-grounded. Pairwise supervision, if stage one produced it, may be tested as a separate candidate objective only after its ranking conversion and split isolation validate.

### 4. Baseline reproduction

Before optimizing, reproduce the active baseline on all available fixed gates using the exact stage-two datasets and harness. Record why reproduced values may differ from historical metrics—for example corrected geographic splits or new-region data—without overwriting historical evidence.

Fail the loop if:

- active checkpoint identity differs from the handoff;
- benchmark membership/hash changed;
- baseline inference is nondeterministic beyond a defined tolerance;
- required rows cannot be matched by `image_path`;
- metrics contain NaN/inf or sample counts differ unexpectedly.

### 5. Experiment ladder

Run candidates in increasing complexity. Keep the experiment budget explicit and bounded before results are observed. Suggested ladder:

#### Controls

- active feature recipe and model with explicit geographic splits;
- human/heuristic weighting controls, including the existing human=4/heuristic=1 reference;
- seed repeat for any candidate near the retention threshold.

#### Data and objective candidates

- region-balanced and scene-balanced sampling/weights;
- calibrated human/heuristic weighting;
- uncertainty-aware weighting only when uncertainty has a defensible source;
- robust regression loss versus current weighted MSE;
- optional pairwise-plus-absolute objective if pairwise evidence exists;
- calibration fitted only on validation data.

#### Feature candidates

- current RESISC-derived feature recipe;
- a well-maintained geospatial foundation embedding evaluated on exactly the same labels and splits, as required by the roadmap;
- simple feature fusion or ablation only when it tests a specific hypothesis.

#### Model/hyperparameter candidates

- learning rate, weight decay, batch size, and early stopping;
- small head capacity/dropout changes;
- no broad architecture search until data/split/feature baselines are sound.

Do not invent complex architecture merely to increase experiment count. Prefer the simplest retained model. Reuse cached embeddings keyed by source image and model hash.

### 6. Metrics and slices

At minimum compute, with sample counts and confidence intervals where practical:

- MAE, RMSE, Pearson correlation, and rank correlation;
- calibration slope/intercept or reliable calibration error appropriate to continuous scores;
- performance by geographic region/block, score band, scene/class proxy, urban/rural proxy if available, and human-confidence tier;
- low/middle/high scenic ranking quality relevant to route selection;
- pairwise accuracy/ranking metrics when pairwise labels exist;
- worst supported slice and spread between regions;
- repeated-seed variance for finalists.

Do not average away a severe regional regression. Mark low-support slices descriptive rather than gating.

### 7. Report-distribution QA

Automate comparison of baseline and candidate learned-score reports over fixed representative tile corpora. Record:

- count, missing/error rate, min/max, mean/std, quantiles, saturation/ties;
- distribution by region and scene;
- baseline/candidate deltas and outlier examples;
- monotonic/calibration sanity checks;
- visual contact sheet or structured sample evidence for extremes and largest disagreements.

A candidate fails if it collapses score spread, saturates, produces invalid scores, materially worsens supported-region calibration, or only “improves” by shifting the scale incompatibly with routing.

### 8. Route-level QA

Run fixed representative route comparisons using baseline and candidate reports/model outputs. Main owns any shared browser. Validate:

- API and report compatibility;
- route completion and existing routing invariants;
- no invalid or missing scenic values;
- fastest-route control remains unchanged where required;
- scenic alternatives remain within configured detour constraints;
- candidate changes are explainable by score changes rather than artifact mismatch;
- representative New England and expanded-region routes receive qualitative review.

Model training must not modify routing logic to make the candidate look better. Route-level regressions reject the candidate unless the user explicitly changes product policy.

## Retention and promotion gates

Before experiments, encode numeric thresholds in the run manifest. Derive them from current benchmark variance and product tolerance; do not choose them after seeing candidate output. At minimum require:

1. **Integrity:** every input/output hash and sample count validates; no leakage or invalid predictions.
2. **Expanded-region human benchmark:** candidate improves the declared primary metric and does not exceed allowed MAE/RMSE regression.
3. **New England control:** no material regression against the immutable existing human benchmark.
4. **Worst supported slice:** no material geographic/scene collapse.
5. **Calibration/distribution:** passes fixed bounds and visual evidence review.
6. **Route QA:** passes all correctness and product invariants.
7. **Stability:** finalist improvement survives a confirmation seed/run when near threshold.
8. **Complexity:** added dependency, runtime, storage, and operational costs are justified by measured gain.

Use `promote_regression_model.py` only after the full compound gate passes. Do not use negative or permissive thresholds to force promotion. Promotion must be atomic or guarded against concurrent registry changes, append complete history, preserve the prior active record, and verify the newly active checkpoint after writing. Rejection must leave the active pointer byte-for-byte unchanged.

## Autoresearch loop

For each experiment:

1. State one hypothesis and bounded change.
2. Materialize an isolated config/artifact identity.
3. Train or reuse a hash-valid result.
4. Run cheap fixed validation first.
5. Reject immediately on integrity, correctness, leakage, or clear metric failure.
6. Run expensive human benchmark, distribution, route, and confirmation gates only for viable candidates.
7. Compare against the immutable baseline and current retained candidate on the same denominator.
8. Keep only a reproducible improvement; otherwise record and revert source changes.
9. Never mutate benchmark data, stage-one handoff, baseline checkpoint, or historical metrics.

Emit concise machine-readable `METRIC key=value` lines for orchestration plus a complete JSON result. Maintain an experiment table with hypothesis, change, hashes, metrics, decision, and reason.

Stop when any declared condition occurs:

- experiment budget exhausted;
- no meaningful improvement across the declared patience window;
- all justified candidates evaluated;
- data quality/benchmark support is the limiting factor;
- resource or external dependency blocker prevents valid continuation.

If data is limiting, return a concrete next active-learning batch specification for Prompt 1 rather than overfitting the existing benchmark.

## Verification

Run focused modeling tests plus the actual orchestrated smoke/experiment path. Prove:

- false/missing stage-one readiness fails before training;
- explicit geographic splits are honored and leakage detection catches a plausible adjacent duplicate;
- resume does not rerun a hash-valid completed experiment;
- corrupt/partial artifacts are rejected;
- baseline and candidate comparisons use identical rows and denominators;
- rejected runs leave the active registry record/pointer unchanged; an explicit rejection-history append is allowed and must be verified;
- a synthetic passing fixture can reach the promotion decision without changing the real registry;
- real promotion, if warranted, is followed by registry/checkpoint inference verification;
- report-distribution and route gates are part of the decision rather than prose-only checks;
- generated artifacts remain ignored and secrets are absent.

Do not run a project-wide suite while parallel edits are in flight. After integration, run repository-level verification appropriate to the changed modeling surfaces and one end-to-end candidate smoke scenario.

## Required final artifacts

At minimum produce:

- `data/processed/modeling_autoresearch/<run_name>/run_manifest.json`
- `data/processed/modeling_autoresearch/<run_name>/experiments.jsonl`
- baseline reproduction metrics;
- per-candidate configs, metrics, hashes, and checkpoints;
- fixed human benchmark comparisons;
- slice/calibration/distribution reports;
- route-level QA report;
- promotion decision JSON;
- final summary with retained/rejected experiments;
- updated `docs/research/ml-research-log.md` only for completed, observed evidence;
- registry update only if every gate passes.

## Completion response

Return:

1. architecture and files changed;
2. exact command to rerun/resume the stage-two loop;
3. validated stage-one handoff identity;
4. baseline reproduction and experiment table;
5. human benchmark, regional slice, distribution, and route evidence;
6. retained candidate and why it won, or why no candidate was retained;
7. promotion status and exact registry/checkpoint identity;
8. resource/runtime costs and reproducibility limitations;
9. unresolved risks;
10. if data-limited, the exact next annotation-batch request to feed back into Prompt 1.

Do not claim completion from a successful training process alone. Completion requires fixed human-grounded evaluation, product-level QA, an explicit keep/reject decision, and verified registry state.
