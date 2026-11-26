# FIXES.md — Audit Fix Guide

> **Status 2026-10-06: Waves 1–6 complete.** All P0/P1/P2 items and the documented P3
> docs items are implemented and verified (full suite green, viewer lane green, ruff
> clean). Remaining below: the P2/P3 parking lot and the explicit user decisions listed
> at the end of this banner.
>
> Open user decisions: (1) delete the frozen `archive/` snapshots per the clean-cutover
> policy, or keep the curated archive with a documented exception; (2) lint debt —
> 16 E402/E731 outside archive (pay now or keep the rule ignores); (3) publish the pinned
> remote-training image (`ian139/scenicdriver-remote-training:torch2.10.0-cu128-20261006`)
> before the next rental; (4) regenerate feature datasets/score caches under terrain
> contract v2 and migrate legacy classifier checkpoints (`python -m
> scenicdriver.classifier.model migrate`) before the next training round; (5) verify the
> shipped sidecar JSONs carry `tile_scores_sha256` on the artifact host before release.

Source: 2026-10-06 full-repo deep-dive audit (four read-only slices: frontend, backend +
routing, ML/data, systems/organization). Every item carries its original evidence so it can
be acted on without re-derivation. This file is the canonical tracking artifact for the
remediation effort; update status as work lands.

- **Status:** ☐ not started · ◐ in progress · ✅ done (verified)
- **Severity:** P0 correctness/security/data-integrity · P1 high urgency · P2 quality drag · P3 polish
- **Effort:** S (< 1h) · M (hours–day) · L (multi-day)
- Waves are ordered by urgency; items inside a wave are independent unless noted.

---

## Wave 1 — Immediate correctness patches (small, independent)

| ID | Sev | Eff | Fix | Status |
|----|-----|-----|-----|--------|
| W1.1 | P0 | S | Sub-1 km/h speeds silently clamped to 1.0 | ✅ |
| W1.2 | P0 | S | Contributor agreement score is a decay, not a mean | ✅ |
| W1.3 | P0 | S | Regression notebook feeds `-1` class sentinel into cross-entropy | ✅ |
| W1.4 | P0 | S/M | Marimo annotator saves skips the stage-one finalizer rejects | ✅ |
| W1.5 | P1 | S | Feature export runs the ViT backbone twice per batch | ✅ |
| W1.6 | P2 | S | Untrack generated `model_registry.json` from Git index | ✅ |

### W1.1 — Sub-1 km/h speeds silently clamped
- **Evidence:** validation accepts any finite positive speed and promises exact preservation
  (`src/route_planner/graph.py:4207-4223`), but `Edge.travel_time_minutes` computes with
  `max(speed, 1.0)` (`graph.py:121-124`), repeated by native CSR construction
  (`src/route_planner/planner.py:624-625`). A 0.5 km/h edge is planned at 2× speed.
- **Change:** use the validated speed exactly in both sites; grep for any other speed clamps.
- **Verify:** extend persisted-speed round-trip tests (`tests/test_graph_sqlite.py:516-542`
  area) with a sub-1 speed asserting exact duration (1 km @ 0.5 km/h → 120 min) plus
  scalar/CSR parity below 1.

### W1.2 — Contributor agreement aggregate is not a mean

- **Evidence:** new profiles start `agreement_score=0`
  (`src/app_api/contrib_repo.py:182-184`); each approved label sets
  `(previous_score + agreement)/2` (`:392-395`). First perfect label scores 0.5; later
  labels exponentially decayed.
- **Change:** count-weighted mean — first approved label initializes the score, then
  `new = (score*count + agreement)/(count+1)`. Persist an agreement count alongside the
  score. Profiles stored by the old formula are superseded output, not a compatibility
  case (clean cutover): a missing count re-initializes from the next approval.
- **Verify:** first label agreement 1.0 → score 1.0; sequence (1.0, 0.5) → 0.75; leaderboard
  reflects the mean. Tests beside existing contributor tests (`tests/test_app_api.py:891-1062`).

### W1.3 — Notebook CE sentinel
- **Evidence:** labeler intentionally emits class `-1` when no classifier is available
  (`src/heuristics/labeler.py:530-540,640-643`); `notebooks/regression.mo.py:277-278` loads
  it unfiltered and `:449-452,482-485` uses ordinary cross-entropy — training crashes on its
  own fallback path.
- **Change:** mask `class_target == -1` rows; compute classification loss only on valid
  rows; skip gracefully when no valid rows exist.
- **Verify:** `py_compile` the notebook; logic review of the masked loss path.

### W1.4 — Marimo skip contract
- **Evidence:** notebook offers a plain Skip checkbox and persists no `unusable_reason`
  (`notebooks/annotate_scenic.mo.py:205-207,269-279`); the web annotator accepts an absent
  reason (`scripts/annotation/annotate_scenic_web.py:988-997`); stage-one finalization
  blocks skips lacking exactly one supported reason (`src/active_learning/finalize.py:405-422`).
- **Change:** expose the supported reason selector in the notebook (source the list from
  the finalize module, not a copy) and require it when Skip is checked; persist
  `unusable_reason` on save.
- **Verify:** `py_compile`; saved payload includes the reason; skip blocked without one.

### W1.5 — Exporter double forward
- **Evidence:** `scripts/modeling/export_regression_dataset.py:245-248` calls `model(x)`
  then `model.get_features(x)` — the backbone runs twice; per-sample CPU copies at
  `:250-255`. Combined forward already exists
  (`src/classifier/model.py:184-208`, used at `src/active_learning/scoring.py:1302-1306`).
- **Change:** single `forward_with_features(x)` call; whole-batch CPU transfer once.
- **Verify:** exporter tests if present (grep `tests/` for the script name), else
  `py_compile` + array-identity review.

### W1.6 — Registry untracking ✅
- `git rm --cached data/processed/regression/model_registry.json` (2026-10-06). Completes
  the removal `CHANGELOG.md:101-106` already documents; `.gitignore:32` covers the path.
  Follow-up: add a tracked-artifact-policy CI check (see W6.4).

Wave-1 results (2026-10-06): W1.1 also removed two additional silent clamps in
compact serialization and compact traversal-weight validation (`graph.py:2538`,
`graph.py:3291`); persisted loaders validate speeds as finite and strictly
positive, so no new admission checks were required. W1.2 stores
`agreement_count` as a private CSV column excluded from every API response.
W1.5 added a one-backbone-per-batch export regression in
`tests/test_export_prediction_heatmap.py`. Targeted suites green:
`tests/test_graph_sqlite.py` (23 passed), `tests/test_app_api.py -k
"contrib or agreement"` (59 passed), exporter regressions (15 passed), notebook
`py_compile` clean.

---

## Wave 2 — Trust boundary + user-facing correctness

| ID | Sev | Eff | Fix | Status |
|----|-----|-----|-----|--------|
| W2.1 | P0 | M | Contributor identity is caller-controlled | ✅ |
| W2.2 | P1 | M | No admission/rate control on paid search + long route jobs | ✅ |
| W2.3 | P0 | M | UI Share can copy the previous route's request | ✅ |
| W2.4 | P1 | M | Shared links display the default route, not the shared one | ✅ |
| W2.5 | P1 | S/M | Route errors hidden inside collapsed dev panel | ✅ |
| W2.6 | P1 | S/M | Inspector-grid failure discards a valid heatmap image | ✅ |
| W2.7 | P1 | M | Serial startup API waterfall, no request deadlines | ✅ |

### W2.1 — Authenticate contributor identity
- **Evidence:** `src/app_api/main.py:2238-2278` accepts `contributor_id` from the request;
  `contrib_repo.py:178-179` allows renaming an existing contributor; `:259-289` issues
  tasks to any supplied ID. Leaderboard/profile responses expose the IDs. Only admin QA
  authenticates (`main.py:2280-2297`).
- **Change:** derive identity from an authenticated session/token; until that exists,
  disable contributor mutation endpoints in the public beta. Keep issued-task binding and
  replay protection.

### W2.2 — Admission control
- **Evidence:** wildcard CORS + body-size only (`main.py:1621-1634`); search proxies every
  query to Mapbox with the server token (`main.py:1693-1717,1735-1744`); one supervisor
  slot (`supervisor.py:1082-1205`); 120 s route budget (`config/app_regions.json:95-99`).
- **Change:** per-client search limits; bounded route queue with 429/503; restrict public
  score-run selection to released pairs; explicit allowed origins.

### W2.3 — Atomic result publication in the viewer
- **Evidence:** publication defers on `isStyleLoaded()` (`viewer.js:788-797`) but the
  dialog renders immediately (`:1503-1508`); `displayedRouteRequest` updates only on render
  (`:781`); Share reads it (`:1561-1571`); button always enabled (`index.html:194-196`).
- **Change:** publish geometry + metrics + share identity + status as one transition;
  disable Share until publishable; add a regression test (new response while rendering
  deferred, then Share).

### W2.4–W2.7 — Viewer startup/error correctness
- Shared link should plan its restored request after readiness, not install the validated
  artifact (`viewer.js:1526-1559,1695-1723`).
- Route progress/errors belong beside the public form with `aria-describedby`, not only in
  `<details class="dev-panel">` (`index.html:62-116`; `viewer.js:1427-1430,1460-1463,1510-1514`).
- Separate heatmap-image availability from inspector-grid availability (`viewer.js:538-584`).
- Build the map promptly; fetch metadata/health/heatmap concurrently with bounded
  deadlines (`viewer.js:1695-1725,528-557`).

Wave-2 results (2026-10-06): contributor identity is now a server-issued secret token
(`X-Contributor-Token`; SHA-256 stored, plaintext returned once at explicit profile
creation; `contributor_id` removed from all mutation schemas — 401 without a valid token,
legacy tokenless rows cannot mutate until admin reset). Search endpoints carry a per-IP
token bucket (default 30/min, `SCENIC_SEARCH_RATE_PER_MIN`/`SCENIC_SEARCH_BURST`, 429 +
Retry-After), route compare has bounded admission (`SCENIC_ROUTE_MAX_PENDING`, default 8,
503 `routing_capacity_exceeded`), CORS uses the `SCENIC_ALLOWED_ORIGINS` allowlist
(default localhost:3000), and explicit `run_name` requests are restricted to the configured
released run (422 `score_run_not_released`). Viewer publishes route results atomically
(Share disabled until publish), replays shared URLs as an actual plan, surfaces errors
publicly with aria wiring, decouples the score inspector from the heatmap image, and starts
the basemap + all metadata fetches concurrently under 15 s AbortController deadlines.
Verification: `tests/test_app_api.py` 173 passed; viewer suite 30 passed.

---

## Wave 3 — Deployment portability

| ID | Sev | Eff | Fix | Status |
|----|-----|-----|-----|--------|
| W3.1 | P0 | M | Sidecar identity is host-dependent; breaks read-only beta mounts | ✅ |
| W3.2 | P0 | M | Load-time file replacement can poison cache identity | ✅ |
| W3.3 | P1 | S/M | Docker context admits host `.so` + operator state; no native build stage | ✅ |
| W3.4 | P1 | L | `src.*` imports vs setuptools package-root mismatch | ✅ |
| W3.5 | P1 | M | Remote training runtime not pinned (mutable `:latest`, unversioned upgrades) | ✅ |
| W3.6 | P1 | M | Remote rental has no provider-side expiry if orchestrator dies | ✅ |
| W3.7 | P1 | M | No reproducible hosted frontend deployment; manual cache version string | ✅ |

### W3.1 — Content-addressed sidecars
- **Evidence:** score identity hashes absolute report path + `(device,inode,size,mtime,ctime)`
  (`src/route_planner/service.py:269-271,500-512,892-901`) and names the sidecar file from it
  (`graph.py:3637-3652`); a copied report never matches its prebuilt sidecar, and the miss
  tries to write beside the graph (`graph.py:4065-4105`) — failing on the documented
  read-only mount (`docs/setup/deployment.md:3`).
- **Change:** key sidecars by report content digest + graph digest + zoom/fallback/
  normalization; resolve admitted sidecars from the deployment manifest; generate offline;
  fail startup admission if absent. Keep stat identity for process-local invalidation only.

### W3.2 — Load-signature equality
- **Evidence:** `_load_cached_graph` captures a signature before loading, then publishes
  under a freshly observed one without comparison (`service.py:694-696,734-749`; tiles
  `:791-821`; SQLite variant `graph.py:1955-1985`).
- **Change:** require before/after identity equality before publication; retry or reject a
  changed load within the deadline. Add a swap-during-read regression test.

W3.1/W3.2 results (2026-10-06): sidecar identity is now
`sha256(json{report_sha256, graph_sha256, zoom, fallback, normalization})[:16]` —
content-derived filenames, portable across hosts; manifests record `tile_scores_sha256`
(host-dependent `report_signature`/`tile_scores_source` fields removed); missing sidecar on
a read-only graph directory fails admission (`RouteAssetError`) instead of attempting a
rebuild. Graph/report/tile loads publish only when before/after identity matches, with one
in-deadline retry (`RouteAssetChangedError` otherwise). `deploy/beta_artifacts.json` +
`.sha256` sidecar destinations updated to the content-derived names (digests unchanged).
Verification: promotion/graph_sqlite/compact suites 344 passed; bootstrap 9 passed.
**Release follow-ups:** (1) shipped sidecar JSONs must contain `tile_scores_sha256` equal
to the report digest — verify on the artifact host or regenerate before release; (2) the
remote-training pin `ian139/scenicdriver-remote-training:torch2.10.0-cu128-20261006`
(digest-pinned base `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b855…`) is a
recipe — build+push before the next rental; (3) vast.ai exposes no instance-expiry knob, so
`provider_expiry` is null: the GNU-timeout deadline stops training but not rental billing;
retained-host exposure is documented in `docs/setup/vast-training.md`.

### W3.3 — Docker/native determinism
- **Evidence:** `.dockerignore:1-37` misses `.worktrees/` (11M), `.todos/`, `.omp/`,
  `*.egg-info`; `Dockerfile.remote-training:26` copies `src/` wholesale including a newer
  macOS arm64 `_compact_search_c.so` that suppresses recompilation
  (`_compact_search.py:63-82`); no compiler/build stage in the image.
- **Change:** exclude host native outputs and operator state from the context; build the
  native library for the target platform during image construction; assert it loads in a
  smoke.

### W3.4 — Package root
- **Evidence:** setuptools discovers top-level packages (`app_api`, …) while all supported
  imports are `src.*` (`pyproject.toml:1-45`; egg-info `top_level.txt`); checkout cwd masks
  the conflict; resources resolve via `__file__.parents[2]` (`main.py:60-64`).
- **Change:** pick one namespace (prefer `src/scenicdriver/`), configure discovery
  explicitly, migrate imports/entrypoints, and add an installed-package smoke outside the
  checkout.

Wave-3 completion (2026-10-06): `.dockerignore` now excludes operator state (`.worktrees/`,
`.todos/`, `.omp/`, egg-info/build/dist) and native/bytecode artifacts globally; both
Dockerfiles build `_compact_search_c` in a target-platform stage and assert
`compact_search_available()` at image build (compiler stays out of the runtime stage).
W3.4: all packages now live under `src/scenicdriver/` with imports `scenicdriver.*`
(uvicorn target `scenicdriver.app_api.main:app`); setuptools discovery configured
(`where=["src"]`, `include=["scenicdriver*"]`, C source shipped as package-data); the only
moved-package `parents[]` fix is `app_api/main.py` (2→3); 29 script bootstraps add repo
root + `src` to sys.path. W3.7: canonical beta topology is `compose.beta.yml` (API +
nginx frontend, six read-only artifact mounts, loopback ports, outer HTTPS proxy
documented); frontend caching is content-stamped via
`scripts/deploy/stamp_frontend_version.py` (nginx build fails on stale stamps).
Verification: full suite 2248 passed / 1 skipped; viewer 30/30; docker compose config
validated; docker-context + stamp tests 23 passed; editable + non-checkout namespace import
smokes added. **Note:** FIXES.md/archive keep historical `src.*` evidence paths — they are
audit history, not live references. Wheel resource-root policy for non-editable
installations remains a documented follow-up (current deployments are checkout-mounted).

### W3.5/W3.6 — Remote runtime pinning + lease watchdog
- **Evidence:** locked torch 2.10 locally (`uv.lock:1940-1942`) vs base 2.5.1 image
  (`Dockerfile.remote-training:10`) installed `--no-deps` (`:22-23`); mutable
  `ian139/scenicdriver-remote-training:latest` (`scripts/remote/vast_train.py:50`); fallback
  `--upgrade torch torchvision` unversioned (`:202-205`); enforcement is local polling only
  (`:347-355,397-405`) with `nohup` detach — orchestrator death leaves spend running.
- **Change:** digest-pinned image variants; fail instead of unversioned upgrades; record
  image/torch/CUDA identities in run artifacts; add provider-side expiry or an independent
  watchdog; surface remaining lease exposure.

### W3.7 — Hosted frontend recipe
- **Evidence:** `nginx.beta.conf` expects `/usr/share/nginx/html` + `api:8080`, but no
  frontend image/compose/topology is tracked; `immutable, max-age=31536000` caching keyed on
  the hand-edited `v=viewer-hardening-v1` string (`index.html:8,203`).
- **Change:** one canonical static/API deployment definition (image or complete recipe)
  covering HTTPS termination, proxy timeouts, health/readiness, rollback; content-derived
  asset versioning or drop `immutable`.

---

## Wave 4 — ML contract unification

| ID | Sev | Eff | Fix | Status |
|----|-----|-----|-----|--------|
| W4.1 | P0 | M | Preprocessing contract missing from classifier checkpoints | ✅ |
| W4.2 | P0 | M | Unrestricted `torch.load` in active paths | ✅ |
| W4.3 | P1 | M | Two incompatible water estimators (labeler vs active scoring) | ✅ |
| W4.4 | P1 | M/L | Terrain slope is resolution-unaware | ✅ |
| W4.5 | P1 | M | Geographic splits unstable across AL rounds; baseline splits by path | ✅ |
| W4.6 | P1 | S/M | Skip-reason contract enforced only in web annotator (extends W1.4) | ✅ |
| W4.7 | P1 | S/M | Non-atomic checkpoint overwrites (classifier + baseline) | ✅ |
| W4.8 | P1 | M | Resume omits RNG state; active trainer omits MPS RNG | ✅ |
| W4.9 | P1 | S/M | Failed feature export still publishes an invalid NPZ | ✅ |
| W4.10 | P1 | M | Score-cache identity ignores preprocessing implementation | ✅ |
| W4.11 | P1 | M | Legacy baseline artifacts lack provenance manifest | ✅ |
| W4.12 | P2 | M | `applied_scenic_weight` reported but never applied | ✅ |

Key evidence: W4.1 `src/classifier/train.py:399,483-489` (ImageNet default) vs
`scripts/modeling/export_regression_dataset.py:50-54` + `src/active_learning/scoring.py:1015-1020`
(RESISC45); W4.2 `active_evaluation.py:42-49`, `scoring.py:901-907`,
`active_training.py:1608-1618` (`weights_only=False`; safe pattern exists at
`heuristics/labeler.py:575-582`); W4.3 `labeler.py:217-235` vs `active_learning/water.py:29-82`;
W4.4 `terrain/features.py:55-83` (no ground spacing; labeler z16 vs baseline z14); W4.5
`selection.py:850-854,1431-1455` + `regression_split.py:36-64`; W4.7 `classifier/train.py:599-616`,
`regression.py:316-330` (atomic writer at `active_training.py:245-261`); W4.8
`train.py:442-443,543-553`, `active_training.py:1558-1605`; W4.9
`export_regression_dataset.py:310-339`; W4.10 `scoring.py:1087-1109`; W4.12
`service.py:1659-1667,1632-1633`.

Wave-4 batch-A results (2026-10-06): classifier checkpoints now carry a preprocessing
contract (`normalization`/mean/std/image_size/class_names; schema `scenicdriver.classifier`
v1) and every consumer (exporter, active scorer, heuristic labeler, notebooks) derives its
transform from the checkpoint; bare legacy state-dicts are rejected with a
`python -m scenicdriver.classifier.model migrate` path. All previously unsafe
`torch.load(weights_only=False)` sites use constrained loading; malicious-pickle tests
extended. One shared atomic writer (temp+fsync+replace) covers classifier/baseline/active/
evaluation/registry writes; exports validate fully before atomic publication. Resume
captures python/numpy/torch/CUDA/MPS RNG + loader generator (no reseed on resume). Score
cache digests bind the real transform config + a versioned terrain contract + injected
preprocessor identity. Geographic splits come from an append-only per-region registry
(stable block assignments independent of pool membership; digest-bound immutable snapshot
in batch_manifest + handoff; stage2 admission verifies the snapshot binding and joins
splits↔registry via `active_training._validate_split_registry_binding`). Baseline splits
derive from geographic blocks with benchmark tile+buffer protection; a run manifest binds
checkpoint/split/NPZ/labels/classifier digests; the evaluator requires `--run-manifest`,
`--labels-csv`, `--classifier-checkpoint`; vast `run` gained `--labels-key`/
`--classifier-checkpoint-key` (+ assertion-only `--use-resisc45-stats`). Skip⇒reason is
enforced at the shared save boundary (web + marimo, finalize as backstop). The unapplied
scenic weight was removed across service/HTTP schema/CLI/benchmark/viewer/saved-artifact
checks (benchmark q axes are honestly planner-scoped; frontend stamp regenerated).
Verification: full suite 2304 passed / 1 skipped; stage2 suites 161 passed; shared-fixture
suites 155 passed. **Regeneration note:** legacy classifier checkpoints need `migrate`
(or retrain); feature datasets produced before the terrain-contract bump (batch B) are
superseded.

Wave-4 batch-B results (2026-10-06): one canonical water estimator (`WATER_METRIC` v1 —
local 3×3 land-only texture, terrain-validity land mask, outside-land counts as water)
with `weak_score_water_components` shared by the labeler and active scorer; the labeler's
whole-image texture gate and `texture` output field are gone, and run metadata records the
water-metric + terrain-contract identities. Terrain slope is physical:
`compute_terrain_features` requires `pixel_spacing_m` (hard error when missing; derived
from tile z/x/y via new `web_mercator.tile_pixel_ground_spacing_m`), slope in degrees,
`TERRAIN_FEATURE_CONTRACT` bumped to v2 (invalidates the scoring cache by design). Same
ramp at z14/43°N and z16/30°N now yields identical features (rel 1e-4).
**Regeneration contract:** feature datasets/score caches produced before contract v2 are
superseded — regenerate via the scoring/export commands before the next training round;
deployed beta serves precomputed artifacts and is unaffected.

W5.4/W5.12 results (2026-10-06): raster ingest reads only the tile-intersection window
plus a scale-aware Lanczos halo through a full-coordinate lazy VRT (no full-source
allocations; disjoint sources read nothing) — 130 fixtures match the frozen full-read
oracle exactly; verified acquisition hashes are carried into QC instead of per-tile
rehashing. Contributor storage is transactional SQLite (WAL) with task/contributor/image
binding constraints, indexed lookups, one-time legacy CSV import, and explicit
`export_csvs()` snapshots; API payloads, token auth, and agreement semantics unchanged
(`tests/test_app_api.py` 175 passed).

---

## Wave 5 — Performance (after correctness waves)

| ID | Sev | Fix | Status |
|----|-----|-----|--------|
| W5.1 | P1 | Whole-sidecar validation scan on every native call (~2.7 GB/request) — admit immutable generations once | ✅ |
| W5.2 | P1 | Default region still SQLite→Python+CSR (~42 B/traversal); switch to supported compact manifest | ✅ |
| W5.3 | P1 | Multi-GB hashing + response validation on the event loop | ✅ |
| W5.4 | P1 | Ingest rereads/rehashes full source rasters per output tile — windowed reads | ✅ |
| W5.5 | P1 | Pool scoring rebuilds DataLoader workers per batch — streaming pipeline | ✅ |
| W5.6 | P1 | Cache hit still unpickle→IPC→validate→deep-copy→serialize; segment table duplicated in GeoJSON | ✅ |
| W5.7 | P2 | Classifier features round-trip CPU between the two model heads | ✅ |
| W5.8 | P2 | Heatmap PNG rebuilt per request — cache by report identity | ✅ |
| W5.9 | P1 | Health stays green when the routing supervisor is dead — readiness vs liveness | ✅ |
| W5.10 | P1 | Best-effort startup drops hard-kill isolation | ✅ |
| W5.11 | P1 | Production scenic search: 4 fixed scalarizations, certificate always 1.0 — real acceptance oracle | ✅ |
| W5.12 | P1 | Contributor storage: full-table CSV under global flock → SQLite | ✅ |

Wave-5 batch-A results (2026-10-06): sidecar/graph validation is admit-once —
`activate_report_scores` is the single sidecar admission point (manifest/size/digest +
one-time finiteness scan via `_admit_score_values`); the native entry's per-call O(T+E)
scan is deleted, replaced by an O(1) fstat identity guard plus per-consumed-edge fault
codes (-1 rank, -4 non-finite); scored-graph cache hits perform no report rescan; nonnative
multiplier weights use one refilled arena (bitwise-equal to the old eager arrays). The
default New England North region now uses the compact manifest pair, proven
representation-only by a tiny-fixture SQLite↔compact parity test; local dev requires
bootstrapping the ignored artifacts (documented). Main-runtime: artifact
hashing/validation/projection moved off-loop into cancellable AnyIO threads under the
request deadline; heatmap PNG + extents cached by report identity; health splits liveness
(`ok`) from routing readiness (`routing.{ready,assets_ready,supervisor_alive,
execution_mode,ready_regions}`, HTTP 503 when routing is unavailable); the supervisor
starts independently of preload with lazy asset loading and fails closed rather than
planning in the parent. Verification: app_api 182 passed; region-config focused 9 +
consumers 119 passed.

Wave-5 batch-B results (2026-10-06): scoring runs one invocation-scoped streaming
pipeline (bounded source queue, producer thread overlapping acquisition with inference,
persistent spawn workers, device tensors held through classifier→regression with whole-batch
NumPy export at the end; explicit NumPy branch kept for injected predictors). The public
route representation is canonical: `routes[kind]` carries metrics+geometry, GeoJSON keeps
only `route_kind`/`snapped_start`/`snapped_end` (the three viewer-consumed properties) —
measured 47% response-size reduction on a 100-segment fixture — validated public bytes are
cached by graph/report/request identity (hits bypass IPC/planning/validation), redaction is
one alias-safe walk, and the service cache dropped pickle round-trips. The scenic search
now uses request-scaled adaptive multipliers alongside the original four (mean oracle gap
0.080→0.041, p95 0.291→0.221, zero-gap 6→8 of 9 fixtures) and the large-graph certificate
is explicitly labeled `approximate-universal-ceiling` / `upper_bound_kind: universal-ceiling`
— a ceiling, not a per-request certificate; the benchmark reports before/after oracle gaps
with a non-regression acceptance invariant. Verification: service+api 288 passed; viewer
30/30; scoring 64 passed; routing benchmark+planner 217 passed.

---

## Wave 6 — Test lanes, CI, and repo hygiene

| ID | Sev | Fix | Status |
|----|-----|-----|--------|
| W6.1 | P1 | Real lifespan/supervisor/read-only-artifact integration fixture | ✅ |
| W6.2 | P1 | Real-browser viewer lane (startup, deferred-render Share, URL replay, keyboard autocomplete, mobile) | ✅ |
| W6.3 | P2 | Pytest marker taxonomy (native/slow/integration); replace sleep-based supervisor tests | ✅ |
| W6.4 | P2 | CI: ruff gate, wheel build + install smoke, runtime-image smoke, tracked-artifact-policy check, platform matrix | ✅ |
| W6.5 | P2 | Consolidate the two `dev` dependency sets; split ML/raster deps from the API set; declare pydantic/starlette/botocore/networkx directly | ✅ |
| W6.6 | P2 | Commit `.todos/`, `.worktrees/` ignores to `.gitignore` | ✅ |
| W6.7 | P2 | Region config has 4 independent sources — make the pinned spec canonical | ✅ |
| W6.8 | P2 | Freeze executable `notes/01-04` prompts referencing deleted scripts | ✅ |
| W6.9 | P3 | Docs drift: promoted-model doc, Mapbox note, notebook count, install commands | ✅ |

Wave-6 results (2026-10-06): `tests/test_api_integration.py` exercises real lifespan +
real supervisor + tiny admitted compact artifacts over HTTP (6 tests: cold lazy load,
swap-during-load, dead-supervisor health, admission saturation, sub-1 speeds, PNG cache);
`tests/browser/` is a self-contained Playwright lane with a deterministic fake API and a
clean skip when browsers are unavailable (CI job installs it). Markers `native`/
`integration`/`slow` registered with `--strict-markers` (conftest applies by module);
supervisor sleep-inference tests now use explicit synchronization. CI has 5 jobs: tests
(ubuntu, py3.11+3.12, ruff + pytest + node viewer), browser, native-macos (`-m native`),
package (wheel build + C-source check + outside-checkout install/import/native smoke),
image (docker build + `--network none` runtime assertions — lean core image ~114 MB).
Dependencies: one dev group (default `uv sync` = everything), core API/routing set vs
`ml`/`ingest` extras, pydantic/starlette/anyio/botocore/networkx declared at their usage
tier; Dockerfile.beta-api on the lean core, remote-training exports `--all-extras`.
Lint enforced (ruff E4/E7/E9/F; archive excluded): debt is 16 E402/E731 outside archive
(listed in the lint note). W6.6: `.todos/`, `.worktrees/`, `*.pem`, `*.key` now in the
committed `.gitignore`. W6.7: `config/app_regions.json` is the single app-region source
(coverage probes included); planner built-in spec deleted — planning requires explicit
`--region-spec` (execute/discovery stay flag-free); viewer bounds come from `/v1/regions`;
release checker has no embedded fallback identities. W6.8: `notes/01-04` + the historical
work-context carry frozen/non-executable banners with no live prompts. W6.9: docs drift
fixed (promoted-model scope, CARTO basemap, install commands synced to the new dependency
model, notebook count, host `uv run python`). Verification: region/beta focused 190 passed;
viewer 33/33; ruff clean.

---

## Parking lot (P2/P3, no wave yet)

- Scripts: the per-script `sys.path` bootstrap (repo root + `src`) is duplicated in ~34
  script/CLI files after the namespace migration — consolidate into one importable helper
  (respecting the grouped-CLI layout) when convenient. Duplicate detector flagged 34 blocks.

- UI: blob URL revocation; binary-grid header presence/positivity validation; camera
  padding from actual dock bounds; attribution occlusion on mobile; touch-target sizes and
  faint-text contrast (4.06:1); explicit `<label for>`; heatmap legend/scale; CSP/integrity
  for CDN MapLibre; ArrowUp initial-index bug + scroll-into-view; `viewport-fit=cover`.
- API: typed response models instead of `dict[str, Any]` (`main.py:1682-2261`); structured
  no-route reasons instead of message-string matching (`main.py:118-125,1912-1916`);
  GET endpoints that mutate contributor storage (`main.py:2249-2258`); consistent
  project-root path resolution (`main.py:133-139,383-385`); shutdown cache release
  (`main.py:719-725`).
- ML: classifier validation loss batch-mean weighting (`train.py:376-383`); explicit
  geometry CRS instead of magnitude inference (`raster_processing.py:243-259`); truthful
  terrain feature names (`features.py:24-40`); label CSV `default_lat/lon` 0/0 fallback
  (`labeler.py:273-280`); disagreement component uses separately min-max normalized scores
  (`selection.py:533-545`); score-domain validation in `ScenicScoreDataset`
  (`regression.py:97-115`); S3 download identity pinning (`s3.py:55-128`); metered-transport
  injected-client retry validation (`metered_transport.py:1247-1251`); Vast baseline jobs
  lack training continuation (`vast_train.py:274-283`); notebook `Path.cwd()` roots;
  classifier notebook trains on open (`classifier.mo.py:42-67`).
- Systems: `.pem`/`.key` gitignore alignment; `autoresearch.sh` location/naming; stale
  cleanup-metric allowlists (`autoresearch_cleanup_metric.py:27-35`); Brandkit worktree
  ownership decision.

## Verification commands

```bash
uv run pytest -q                                  # full Python gate (CI parity)
uv run pytest tests/test_graph_sqlite.py -q       # W1.1
uv run pytest tests/test_app_api.py -q -k contrib # W1.2 (adjust -k to actual names)
uv run python -m py_compile notebooks/regression.mo.py notebooks/annotate_scenic.mo.py  # W1.3/W1.4
node --test tests/test_new_england_north_viewer.mjs                                  # viewer gate
```
