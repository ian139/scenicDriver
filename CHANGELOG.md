# Changelog

All notable user-visible contract changes to Scenic Drive. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 2026-10-06 — Wave 6 audit fixes (test lanes, CI, dependency model, region single-source)

### Changed

- The dependency model is split: one dev group (plain `uv sync` installs the full
  environment), a lean core API/routing set (`uv sync --no-dev`), and explicit `ml`/
  `ingest` extras for training, notebooks, raster, and S3 tooling. Pydantic, Starlette,
  anyio, botocore, and networkx are declared at their usage tier. Install commands in
  README/AGENTS were updated to match.
- CI enforces lint (`ruff check`), a package smoke (wheel build, native C source
  present, install-and-import outside the checkout), a runtime image smoke (lean
  ~114 MB beta API image, `--network none` assertions), a tracked-artifact policy
  check, Python 3.11+3.12 on ubuntu, a macOS native lane, and a Playwright browser
  lane for the viewer. Pytest markers `native`/`integration`/`slow` are registered.
- `config/app_regions.json` is the single source for app-region bounds, release
  identities, and coverage probes: the planner's built-in region spec is deleted —
  planning requires an explicit `--region-spec` (execution/discovery consume their
  frozen plans and do not), the viewer takes its bounds from `/v1/regions`, and
  `check_beta_artifacts.py` no longer has embedded release fallback identities.
- Historical `notes/` prompts are frozen (non-executable banners; no live prompt
  text), `.gitignore` now covers `.todos/`, `.worktrees/`, `*.pem`, `*.key`, and the
  research log's "current promoted model" claims are scoped to their era.



## 2026-10-06 — Wave 5 audit fixes (runtime efficiency and honest health)

### Changed

- Route responses use one canonical representation: `routes[kind]` carries metrics,
  segment identity, and geometry; GeoJSON feature properties are reduced to the
  viewer-consumed `route_kind`/`snapped_start`/`snapped_end` (a measured ~47%
  payload reduction on long routes). Validated public responses are cached by
  graph/report/request identity; repeated identical requests skip planning,
  validation, and projection entirely. GeoJSON consumers reading the removed rich
  properties must switch to `routes[kind]`.
- `/v1/healthz` separates liveness (`ok`) from routing readiness:
  `routing.{ready, assets_ready, supervisor_alive, execution_mode, ready_regions}`;
  it returns 503 when the routing supervisor or default-region assets are
  unavailable. The supervisor starts independently of preloaded assets (lazy
  loading) and cold starts fail closed instead of planning without hard-kill
  isolation.
- Heavy artifact hashing and response validation/projection run off the event
  loop under the request deadline; heatmap PNGs and grid extents are cached by
  report identity.
- The default New England North region runs on the compact graph manifest (SQLite
  remains the audit/build source; local development requires bootstrapping the
  ignored artifacts). Sidecar/graph generations are validated once at admission —
  repeated searches no longer rescan the score payload — with an fstat identity
  guard preserving post-admission mutation detection.
- Scenic route search adds request-scaled adaptive multipliers beside the original
  four scalarizations (measured oracle gap mean 0.080→0.041 on the generated
  corpus), and the large-graph certificate is explicitly labeled
  `approximate-universal-ceiling` — a universal objective ceiling, not a
  per-request near-optimality certificate. The production benchmark reports
  before/after oracle gaps with a non-regression acceptance invariant.
- Active-learning pool scoring uses one invocation-scoped streaming pipeline
  (bounded-memory source queue with acquisition overlapped on inference, persistent
  workers) and keeps classifier features on-device through the regression head.



## 2026-10-06 — Wave 4/5 audit fixes (terrain + water semantics, ingest efficiency)

### Changed

- Terrain slope is physical and resolution-aware: `compute_terrain_features` requires
  `pixel_spacing_m` (derived from the tile's z/x/y and latitude; missing spacing is a
  hard error), slope is reported in degrees, and the terrain feature contract is
  version 2. Feature datasets and score caches produced under contract v1 are
  superseded and must be regenerated before the next training round.
- The heuristic labeler and active scoring share one canonical water estimator
  (local land-only texture, terrain-validity mask, sea fill classified as water); the
  labeler's whole-image texture gate and `texture` output field are removed, and
  labeler run metadata records the water-metric and terrain-contract identities.
- Raster ingest reads only the destination tile's source window (plus resampling
  halo) instead of entire source rasters per output tile, and reuses the acquisition
  cache's verified content hashes instead of rehashing source files per tile.
- Contributor annotation storage moved from whole-file CSV rewrites under a global
  lock to transactional SQLite (WAL) with uniqueness constraints and indexed
  lookups; legacy CSVs import once on first open and CSV export is an explicit
  snapshot operation. Public API payloads and authentication are unchanged.

## 2026-10-06 — Wave 4 audit fixes (ML contracts)

### Changed


- Classifier checkpoints carry their full preprocessing contract
  (`normalization`, mean/std, image size, class order; schema
  `scenicdriver.classifier` v1). Every consumer — feature export, active
  scoring, the heuristic labeler, and the classifier notebook — derives its
  transform from the loaded checkpoint; `--use-resisc45-stats`-style knobs are
  replaced by an optional `--normalization {imagenet,resisc45}` assertion that
  fails on conflict. Bare legacy state-dict checkpoints are rejected with a
  documented migration (`python -m scenicdriver.classifier.model migrate`) or
  retraining; the default normalization for new training is `resisc45`.
- All checkpoint loading in active evaluation, active scoring, and active
  training uses constrained `weights_only` deserialization (malicious-pickle
  rejection extended to those paths), and every checkpoint/registry/export
  write is atomic (temp file + fsync + rename). Training resume restores
  Python/NumPy/Torch/CUDA/MPS RNG state and the loader generator instead of
  reseeding. Failed feature exports can no longer publish an empty or partial
  NPZ at the destination path.
- Active-learning score caches key on the real transform configuration, a
  versioned terrain-feature contract, and explicit content identities for
  injected preprocessors — implementation or parameter changes can no longer
  reuse stale derived features under an unchanged digest.
- Geographic splits are stable across rounds: an append-only per-region
  registry freezes physical-block→split assignments (new blocks get seeded
  stable assignments independent of pool membership), each run binds an
  immutable digest-verified registry snapshot into its batch manifest and
  stage-one handoff, and stage-two admission rejects handoffs whose splits
  disagree with their registry snapshot. Baseline training derives validation
  splits from geographic blocks (benchmark tiles + buffer protected, not just
  filenames) and records a run manifest binding checkpoint, split, feature
  NPZ, labels, and classifier digests; the evaluator requires
  `--run-manifest`, `--labels-csv`, and `--classifier-checkpoint`, and
  `vast_train.py run` gains `--labels-key` and `--classifier-checkpoint-key`
  (plus assertion-only `--use-resisc45-stats`).
- Skipped annotation decisions require exactly one supported unusable reason at
  the shared save boundary (web annotator and marimo notebook alike);
  finalization remains the backstop for previously persisted rows.
- The route-comparison service no longer accepts or reports a scenic weight it
  never applied: `scenic_weight`/`applied_scenic_weight` are removed from the
  HTTP schema, service contract, CLI, saved-artifact validation, and viewer;
  the production benchmark's q sweep is labeled as an explicit planner
  parameter axis with separate fixed-scenic-priority service probes.

## 2026-10-06 — Waves 2–3 audit fixes (trust boundary, deployment identity)

### Security

- Contributor identity is authenticated: profiles are created via an explicit
  `POST /v1/contrib/profile` that returns a secret `contributor_token` exactly
  once (only its SHA-256 is stored). Every mutating contributor operation —
  profile update, task issuance, label submission — requires the
  `X-Contributor-Token` header and derives the contributor server-side;
  `contributor_id` is removed from mutation request schemas (sending it is a
  422, a missing/unknown token is 401). `GET /v1/contrib/profile` is now
  read-only (unknown contributor → 404); profiles written before this change
  have no token and cannot mutate until an admin reset.
- `/v1/search/suggest` and `/v1/search/retrieve` enforce a per-client-IP token
  bucket (default 30/min, burst 30; `SCENIC_SEARCH_RATE_PER_MIN`,
  `SCENIC_SEARCH_BURST`) returning 429 with `Retry-After`. Route comparison has
  bounded admission (`SCENIC_ROUTE_MAX_PENDING`, default 8) returning 503
  `routing_capacity_exceeded` when saturated. CORS origins come from
  `SCENIC_ALLOWED_ORIGINS` (default `http://localhost:3000`,
  `http://127.0.0.1:3000`) instead of `*`. An explicit `run_name` must match
  the region's configured released run (422 `score_run_not_released`).

### Changed

- Compact score sidecars are content-addressed:
  `sha256(json{report_sha256, graph_sha256, zoom, fallback, normalization})[:16]`
  replaces the absolute-path + stat identity, so prebuilt sidecars are portable
  across hosts. Sidecar manifests record `tile_scores_sha256`; the
  host-dependent `report_signature`/`tile_scores_source` fields are gone. A
  missing sidecar on a read-only graph directory fails admission with a clear
  asset error instead of attempting a rebuild. Graph/report/tile loads publish
  only when the file identity is unchanged across the read (one in-deadline
  retry, then `RouteAssetChangedError`); the beta artifact manifest's sidecar
  destinations now use the content-derived names (digests unchanged).
- Remote training runs a pinned, reproducible runtime: the default image is the
  publish-once `ian139/scenicdriver-remote-training:torch2.10.0-cu128-20261006`
  (digest-pinned pytorch 2.10.0/cuda12.8 base aligned with the lock), the
  unversioned `--upgrade torch` fallback is removed (unavailable pinned wheels
  fail the run), the detached remote script runs under a GNU-timeout deadline
  derived from `--timeout-seconds`, and every run artifact/state records its
  image/torch/CUDA/source-digest identity. Vast.ai exposes no instance-expiry
  knob, so deadlines stop training but not rental billing (documented).
- The viewer publishes route results atomically (Share disabled until the
  current result is on the map — no more sharing the previous route's request),
  replays shared URLs by planning the restored request, shows route progress and
  errors publicly beside the form, renders the heatmap image independently of
  the score inspector, and starts the basemap and all metadata fetches
  concurrently under 15-second request deadlines.
- The package namespace is `scenicdriver.*`: all packages moved from `src/<pkg>`
  to `src/scenicdriver/<pkg>`, the uvicorn target is
  `scenicdriver.app_api.main:app`, and setuptools discovery is configured
  explicitly (native C source shipped as package-data). No `src.*` import
  aliases remain.
- The hosted beta topology is reproducible from the repository:
  `compose.beta.yml` (FastAPI + nginx static frontend, read-only artifact
  mounts, readiness-gated startup, loopback ports behind a documented outer
  HTTPS proxy), `Dockerfile.beta-api` and the frontend image build. Frontend
  asset caching is content-stamped — `scripts/deploy/stamp_frontend_version.py`
  rewrites the `v=` query from `sha256(viewer.js + styles.css)` and the image
  build fails on a stale stamp, replacing the hand-edited version string.
  Docker build contexts exclude operator state and host-built native/bytecode
  artifacts, and both images compile `_compact_search_c` for the target
  platform in a build stage with a load assertion in the runtime stage.

## 2026-10-06 — Wave 1 audit fixes

First remediation wave from the 2026-10-06 repository audit (tracked in `FIXES.md`).

### Fixed

- Contributor `agreement_score` is now a count-weighted mean of approved-label
  agreements instead of the previous `(previous + agreement) / 2` recursion that
  started at 0 (a first perfect label scored 0.5 and later labels were
  exponentially decayed). Profiles persist a private `agreement_count`; scores
  written by the old formula restart from the next approval. Public API payload
  shapes are unchanged.
- Persisted road-graph edges slower than 1 km/h are no longer silently clamped
  to 1 km/h during duration computation (`Edge.travel_time_minutes`, native CSR
  construction, compact graph serialization and loading). Validated positive
  speeds are used exactly; a 0.5 km/h edge now yields its true travel time
  instead of half of it.
- `notebooks/regression.mo.py` masks unknown-class targets (`-1`) emitted by the
  heuristic labeler before the classification loss; batches with no valid class
  rows contribute zero classification loss instead of crashing training.
- `notebooks/annotate_scenic.mo.py` requires an unusable reason (sourced from
  `src.active_learning.finalize.UNUSABLE_REASONS`) when saving a skipped tile
  and persists `unusable_reason`, so notebook annotations satisfy the Stage-One
  finalizer's skip contract instead of being rejected at finalization.

## 2026-10-04 — Unreleased

Documentation and data-contract alignment following the repository audit sweep.
Every entry below reflects behavior that is implemented and enforced in the
current tree.

### Changed

- `scripts/modeling/build_mixed_labels.py` now requires
  `--geographic-splits-csv`. The run's fixed `image_path`/`split` assignments
  are mandatory and never inferred; rows outside the split file are dropped and
  the split file is validated (one split per image, one split per
  `geographic_block`, `split` in `train`/`val`/`test`).
  `scripts/annotation/build_human_benchmark.py` applies the same required flag.
- `scripts/modeling/train_regression_baseline.py` now requires
  `--benchmark-split-csv`, and `scripts/modeling/run_regression_pipeline.py`
  requires both `--benchmark-split-csv` and `--baseline-checkpoint`. Fixed
  benchmark validation/test identities are excluded from training before the
  internal split, and the candidate is compared against an explicit baseline.
- Baseline regression training and evaluation now share one explicit
  validation-membership derivation
  (`scripts/modeling/regression_split.py`) over the filtered training
  population. `train_regression_baseline.py` persists it as a split artifact
  (`--split-artifact`, default `<output stem>.split.json`), and
  `evaluate_regression_baseline.py` requires `--split-artifact` (its
  `--val-split`/`--seed` options are removed) to score exactly the trained
  model's validation rows; any membership drift is a hard error.
- `scripts/remote/vast_train.py run` now requires `--benchmark-split-key`, the
  exact S3 key of the benchmark split CSV whose validation/test identities are
  excluded from training. The wrapper downloads it with `--required`, so a
  missing key fails the run before training starts.
- `scripts/remote/vast_train.py` now guards the training launch boundary with a
  persisted `launch_attempted` flag set before the detached remote command is
  fired. When the launch call raises afterwards (lost ack or dropped channel),
  the outcome is ambiguous: the run records `launch_outcome: "unknown"` and
  keeps the host (never `failed_destroyed`, never a replacement rental) under
  every policy, including `--keep-on-failure` and `--no-destroy`. Only
  confirmed-never-launched allocations are still destroyed before re-renting.
- `notebooks/learned_scoring.mo.py` requires the per-run `cfg.benchmark_split_csv`
  value (no default); training raises without it. It also wires
  `cfg.split_artifact` through training and evaluation so evaluation scores
  exactly the trained model's validation rows.
- `scripts/annotation/annotate_scenic_web.py` accepts repeatable
  `--trusted-authority <host[:port]>`. Public DNS names, wildcard binds, and
  reverse-proxy deployments must declare every served authority or requests are
  rejected as `untrusted_host`.
- `scripts/annotation/build_source_shift_batch.py` accepts `--image-root`, the
  shared satellite tile root. `image_path` is emitted relative to it, the
  summary records it as `image_root`, and the annotator `raw_dir` must equal
  that root so one annotator serves both sources.
- `scripts/reports/heuristic_report.py` accepts `--learned-run-json`, attested
  learned run metadata that supplies missing checkpoint identities. Declared
  identities must match the loaded artifacts; nothing is inferred.
- The Stage-Two autoresearch entry point is
  `scripts/modeling/run_active_scenic_autoresearch.py`. It requires
  `--control-dataset` (a control feature dataset NPZ disjoint from the expanded
  prepared dataset) alongside `--expanded-benchmark-csv`,
  `--control-benchmark-csv`, and `--route-qa-json`, and `--thresholds-json` for
  every non-dry execution. The root `autoresearch.sh` now execs the Northeast
  Expanded routing benchmark and is no longer a modeling launcher.

### Fixed

- The beta deployment contract now ships each graph's `run.json` readiness
  gate. `deploy/beta_artifacts.json` and `deploy/beta_artifacts.sha256` list
  `data/processed/road_graphs/<run>/run.json` for both deployed graphs;
  `scripts/deploy/bootstrap_beta_artifacts.py` downloads them like every other
  manifest destination, and `scripts/routing/check_beta_artifacts.py` now
  rejects a deployment whose gate is missing, stale, or names build-machine
  paths — exactly the deployments the API refuses to route. Previously a
  documented bootstrap produced a service with zero routeable regions.
- `scripts/routing/build_graph_from_osm.py` records project-root-relative
  artifact paths in `run.json` (`output_graph`, `output_dir`, `compact_graph`,
  `runtime_graph`) whenever the published files live under the project root.
  Absolute build-machine paths could never validate on another host, so
  readiness gates must be project-root-relative.

### Added

- `SCENIC_ROUTE_DEADLINE_SECONDS` overrides the per-region
  `route_deadline_seconds` from `config/app_regions.json` for every region. It
  must be a finite non-negative number.
- `CONTRIB_ADMIN_TOKEN` enables `POST /v1/admin/contrib/review/run`, guarded by
  the `X-Admin-Token` header (`503` when unset, `401` missing, `403` mismatch).
- `northeast_expanded` is a documented entry in `config/app_regions.json`, with
  its own `route_deadline_seconds` and `tile_scores_path`, alongside
  `new_england_north`, `masswhites`, `philadelphia`, and `pittsfield`.
- The compact road-graph runtime documents its per-report score sidecar
  (`<graph>.score.<report_signature>.z<zoom>.fb<fallback>.n<normalization>.json`
  plus `.bin`), which stores clamped scenic costs keyed by canonical edge rank.

### Removed

- `OMP_RUN_AUTORESEARCH` and the `bash autoresearch.sh --handoff ...` modeling
  invocation. `autoresearch.sh` no longer parses modeling arguments.
- `data/processed/regression/model_registry.json` is no longer tracked in Git.
  It remains a runtime deployment artifact: `deploy/beta_artifacts.json` ships
  it, `scripts/routing/check_beta_artifacts.py` resolves the active checkpoint
  through it, and `scripts/modeling/promote_regression_model.py` reads and
  writes it. The registry is generated and ignored like other `data/processed/`
  output.
- `scripts/ingest/download_bbox_tiles.py`. Open-data acquisition is
  `scripts/ingest/plan_active_learning_region.py` with
  `scripts/ingest/naip_3dep_workflow.py`. The archive manifest no longer lists
  the deleted CLI and now records `archive/` as a frozen superseded snapshot.

### Regenerated

- `docs/assets/research/routing/*.svg` are rewritten from
  `docs/assets/research/routing/figure-data.json` by
  `uv run python scripts/reports/generate_routing_research_figures.py`, so the
  committed figures match the derived figure claims.
