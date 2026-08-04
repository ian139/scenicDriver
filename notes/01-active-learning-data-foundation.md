# OMP Autoresearch Prompt 1 — Active-Learning Data and Annotation Foundation

Use this entire document as the first OMP autoresearch prompt from the Scenic Drive repository root.

## Mission

Build and exercise the complete foundation that turns an expanded regional tile pool into deterministic, auditable, human-reviewed training data. Improve the existing browser annotator into a fast, safe, keyboard-first tool. Finish with a machine-readable handoff that Prompt 2 can consume without guessing paths, schemas, hashes, splits, or readiness.

This is an implementation task, not a proposal. Inspect the repository before changing it, reuse canonical workflows, implement missing pieces, run focused checks, and exercise the visible annotator through the main-session browser.

## Fixed operating assumptions

- Repository policy in `AGENTS.md` is mandatory.
- Main owns architecture, decomposition, shared schemas, visible browser state, integration, and final decisions.
- Fan out independent bounded work in one wave: use several `task` workers, normally zero or one `task-high`, optional `scout`; use isolated worktrees when ownership permits. Workers must not manipulate the shared browser.
- Use `uv`, marimo, canonical grouped `scripts/` directories, and existing dependencies before adding packages.
- OMP sessions may read and use `.env` values and credentials required for the requested work. Never print, persist, commit, or unnecessarily share secret values.
- Large tiles, features, checkpoints, caches, reports, and generated datasets stay in ignored canonical paths. Never add them to Git or image layers.
- Preserve `data/README.md` as the data-layout contract and `data/raw/labels_human.csv` as the backward-compatible absolute human-label source.
- Hard expansion budget: at most 740,000 combined zoom-14 rasters—370,000 satellite tiles plus the matching 370,000 Terrain-RGB tiles. This total includes reusable tiles already downloaded or scanned in the current region. Inventory and validate existing tile/feature/label artifacts first, reuse every hash-valid compatible result, and acquire or process only the missing delta. Count unique `(z,x,y,style)` objects before acquisition and fail closed above 740,000.
- Target geography: preserve the current New England North coverage and expand only west and south—never north or east. Favor contiguous US land and useful road corridors, generally avoiding ocean tiles and duplicate coverage. Prefer a deterministic land/state mask or collection of adjoining regions over a wasteful giant rectangle; record any unavoidable water coverage explicitly.
- Do not train or promote the final candidate model in this prompt. Current-model inference, embedding extraction, weak labeling, batch selection, and small smoke training are allowed only to validate the data path.

## Current contracts to preserve

Inspect them rather than assuming they are unchanged:

- Web annotator: `scripts/annotation/annotate_scenic_web.py`
- Marimo annotator: `notebooks/annotate_scenic.mo.py`
- Absolute annotation columns: `image_path, scenic_human, confidence, skip, annotator_id, timestamp, notes`
- Weak labels: `data/raw/labels.csv`; manual labels: `data/raw/labels_human.csv`
- Mixed-label builder: `scripts/modeling/build_mixed_labels.py`
- Benchmark builders: `scripts/annotation/build_human_benchmark.py` and `build_overlap_batch.py`
- Tile acquisition: `scripts/ingest/download_bbox_tiles.py` and canonical `raw/images/{satellite|terrain}/z{zoom}/{region}/{x}_{y}.png` layout
- Active model registry: `data/processed/regression/model_registry.json`
- Current promoted checkpoint is the immutable baseline; stage one must never change the active registry pointer.

## Required decomposition before edits

Map and record:

1. Critical path from region definition through acquisition, weak labels, inference/embeddings, candidate selection, annotation, benchmark splits, mixed labels, and handoff.
2. Independent ownership boundaries for acquisition/manifest logic, selection logic, annotation persistence/API, HTML UX, and focused tests.
3. Shared schemas and invariants before workers start. Schema changes are main-owned.
4. Real dependencies and the minimum runnable slice needed before the visible browser smoke test.

## Deliverables

### 1. Canonical run and region manifest

Implement a deterministic, resumable run format under:

`data/processed/active_learning/<run_name>/`

At minimum persist a versioned region/run manifest containing:

- run name, schema version, UTC creation time, repository revision when available;
- zoom, styles, tile budget, geographic source/mask, included/excluded jurisdictions, and normalized geometry digest;
- exact unique tile counts by style and region plus missing/present/failed counts;
- source URLs/providers without credentials;
- all seeds, model/checkpoint identity, dataset inputs, and SHA-256 hashes;
- stage state and resumable checkpoints;
- explicit failure reasons and readiness flags.

Tile enumeration must be deterministic and dry-runnable. It must deduplicate overlaps, reject invalid geometry, enforce the total budget before network work, and support resuming without redownloading valid tiles. If a robust land mask cannot be sourced from existing dependencies/data, use a documented deterministic collection of state/region polygons or adjoining bboxes and record the limitation; do not silently call a giant rectangle “land-aware.”

### 2. Acquisition and integrity path

Create or extend canonical grouped CLIs so one command can:

1. estimate counts and storage without downloading;
2. enumerate the immutable tile manifest;
3. acquire satellite and Terrain-RGB pairs using repository credentials;
4. resume partial work;
5. validate content type/decodability, dimensions, pair completeness, and hashes;
6. emit a concise machine-readable report.

Use bounded concurrency, retry only transient failures with backoff, and retain deterministic failure records. Never log access tokens. Do not begin acquisition until the dry run proves the 740,000-tile combined cap, reports how many existing satellite/Terrain-RGB tiles and derived artifacts can be reused, and identifies the exact missing delta. Reuse valid cached imagery, weak labels, predictions, and embeddings by content/model hash instead of redownloading or rescanning them. Exercise at least a small representative acquisition or existing-cache validation; do not claim the full 740,000-tile corpus exists unless observed.

### 3. Weak labels, current-model inference, and embeddings

Build a resumable pipeline over the enumerated pool that records, per tile where available:

- canonical tile/image identity, region, z/x/y, latitude/longitude;
- heuristic score and useful heuristic components;
- active-model prediction and checkpoint hash;
- feature/embedding identity and dimensionality;
- class ID/probability or scene metadata;
- uncertainty signal with a precise definition;
- error/missing status.

Never relabel the active model’s prediction as human truth. Store large arrays in NPZ/Parquet or another existing efficient project format, not expanded JSON/CSV blobs. Avoid recomputing stable embeddings; cache by image and model hash.

### 4. Deterministic active-learning batch selector

Implement a reusable CLI that selects high-information, geographically balanced batches from the candidate pool. It must combine, with configurable recorded weights:

- model-versus-heuristic disagreement;
- legitimate model uncertainty (do not fabricate confidence if the model lacks it);
- embedding diversity/cluster representation;
- geographic coverage and minimum spatial separation;
- underrepresented regions/classes/land-cover proxies;
- calibrated sampling of low, middle, and high scenic predictions;
- limited random controls and repeat-overlap QA tiles.

Prevent adjacent near-duplicate tiles from dominating. Exclude already completed absolute labels except intentional overlap. Given identical inputs and seed, output byte-stable ordering or a documented stable canonical ordering. Each selected row must include `selection_reason`, component scores, rank, batch/run identity, region, and the fields needed by the existing annotator. Emit selection diagnostics proving geographic, score, cluster, and reason distributions.

### 5. Annotation schemas

Keep the seven-column absolute-label contract compatible. Add versioned auxiliary artifacts rather than breaking existing consumers:

- a batch manifest with stable batch ID, run ID, mode, ordering, selection metadata, and input hashes;
- a separate pairwise-event schema if pairwise comparison is implemented, with left/right image IDs, winner/tie/skip, confidence, annotator, timestamp, and stable pair ID;
- progress/session state that is recoverable but contains no credentials;
- explicit unusable reasons such as missing imagery, corrupted image, cloud/obstruction, excessive water, duplicate, and other.

Define how pairwise results become a ranking/continuous target (for example Bradley–Terry) and validate it on a tiny fixture, but do not force pairwise labels into `labels_human.csv` prematurely. Absolute labels remain the promotion benchmark source until the downstream contract is deliberately updated and tested.

### 6. Extremely user-friendly HTML annotator

Turn `scripts/annotation/annotate_scenic_web.py` into the canonical fast annotation UI while keeping the marimo workflow valid. The resulting tool must include:

- localhost-safe default binding; explicit opt-in for remote exposure;
- a simple first-run screen with sensible defaults and advanced paths collapsed;
- clear batch identity, selection reason, region, progress, completed/remaining counts, and saved state;
- large responsive imagery, optional satellite/Terrain-RGB or relevant context display, loading skeleton, useful error state, and next-image prefetch;
- keyboard-first absolute scoring, confidence, skip/unusable reasons, previous/next, save-and-advance, and shortcut help;
- optional pairwise mode with equally fast keyboard controls;
- one action per decision, immediate visible save confirmation, disabled duplicate submit, retry on recoverable failure, and no silent data loss;
- restore prior annotation when revisiting; preserve unsaved drafts across accidental navigation/reload where safe;
- resumable progress by stable batch and annotator identity;
- accessible focus states, labels, contrast, responsive layout, and reduced-motion behavior;
- calibration/anchor examples and periodic QA repeats without revealing the prior answer;
- session summary with throughput, skips, confidence distribution, coverage, and overlap consistency—never rankings of annotators by “quality” without a defined metric;
- atomic or otherwise concurrency-safe persistence. Payloads must not be allowed to silently impersonate another configured annotator;
- path traversal protection, output-path validation, no secret values in browser payloads, and no unnecessary raw metadata disclosure.

Do not add a heavy frontend framework unless the existing stack cannot meet these requirements cleanly. Prefer maintainable boring HTML/CSS/JS with a separated template/static structure if the current embedded page has become unmanageable.

### 7. Geographic benchmark and leakage-resistant splits

Create deterministic train/validation/test assignments by geographic groups or blocks, not random neighboring rows. Keep the existing New England human benchmark immutable as a regression set. Produce a new-region challenge benchmark that:

- spans representative geography and scene types;
- is human-labeled only;
- records overlap/agreement evidence;
- cannot leak the same or adjacent tile into training;
- includes enough samples per reported slice to avoid misleading metrics;
- reports score distribution, region/class coverage, annotator overlap, and agreement limitations.

Do not claim generalization from weak-label validation alone.

### 8. Stage-one handoff contract

The final required artifact is:

`data/processed/active_learning/<run_name>/stage1_handoff.json`

It must contain schema version and exact paths/hashes for:

- region and tile manifests;
- validated available imagery and pair-completeness report;
- weak labels, predictions, embeddings, and selection diagnostics;
- absolute annotations and optional pairwise events;
- benchmark/challenge splits and leakage audit;
- mixed-label CSV ready for export;
- fixed train/validation/test assignments;
- baseline model registry/checkpoint/metrics identity;
- every seed and material configuration value;
- incomplete work, failed tiles, and blockers;
- readiness booleans for `data_complete`, `annotations_valid`, `splits_valid`, `benchmark_valid`, and `ready_for_stage2`.

`ready_for_stage2` must be false unless all required inputs exist, hashes match, schemas validate, and minimum human benchmark requirements are met. Stage 2 must fail closed on a false or missing readiness flag.

## Autoresearch execution loop

Use short evidence-driven iterations:

1. Establish deterministic baseline behavior and fixtures before changing it.
2. Implement the smallest coherent candidate.
3. Run narrow contract tests and a real smoke path.
4. Keep only changes that improve correctness, usability, recoverability, throughput, or data quality without weakening invariants.
5. Record experiment/candidate, evidence, decision, and remaining risk in a run log under `data/processed/active_learning/<run_name>/`.
6. Revert rejected candidate changes cleanly; never weaken a check to make a candidate pass.

For UI work, the main session must launch the real tool and use OMP browser/CDP to complete a representative flow: load batch, annotate by keyboard, navigate back, edit, reload/resume, exercise skip/error handling, and verify persisted CSV/event output. Inspect desktop and narrow viewport appearance. Shared browser manipulation remains main-session only.

## Verification gates

Before completion, prove:

- deterministic tile count and budget enforcement;
- resume and deduplication behavior;
- schema validation and idempotent/atomic annotation persistence;
- active-selection determinism and diversity/geographic diagnostics;
- spatial split leakage checks;
- annotator keyboard flow, prefetch, save/retry, revisit, and reload recovery in the browser;
- compatibility with `build_mixed_labels.py` and `build_human_benchmark.py`;
- stage-one handoff hash verification and fail-closed readiness;
- no active model registry change;
- no secret or large generated artifact entered tracked source.

Run focused checks and the real smoke scenario, not a project-wide suite.

## Completion response

Return:

1. concise architecture and decisions;
2. files changed;
3. commands to dry-run/acquire/select/serve/annotate/finalize;
4. observed browser and data-pipeline evidence;
5. exact `stage1_handoff.json` path and readiness flags;
6. dataset/tile/human-label counts actually observed, clearly distinguishing planned from complete;
7. unresolved risks or blockers;
8. the exact command or prompt invocation that should start Prompt 2.

Do not stop at scaffolding, placeholders, or a written plan. Complete every reachable implementation and accurately mark unavailable external data or unfinished annotation work as not ready.
