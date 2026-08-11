# OMP Prompt — Implement and Prove the Nationwide Streaming Scenic-Score Map

Execute this task end to end from the Scenic Drive repository root. This is an implementation, benchmark, model-validation, cost-validation, product-QA, and evidence task—not a design-only exercise. Read and obey `AGENTS.md`, [`docs/research/nationwide-streaming-score-map.md`](../docs/research/nationwide-streaming-score-map.md), and [`notes/03-open-data-naip-3dep-transition.md`](03-open-data-naip-3dep-transition.md) in full before changing code.

## Mission

Implement a deterministic, resumable, metered workflow that can produce a complete contiguous-United-States (CONUS) z14 scenic-score surface without retaining a nationwide satellite or Terrain-RGB image archive:

```text
exact CONUS z14 land inventory
  -> hash-pinned NAIP and USGS 3DEP catalogs
  -> source-asset-partitioned COG range reads
  -> aligned in-memory 512x512 NAIP RGB and float DEM
  -> exact in-memory Terrain-RGB
  -> 768 satellite embedding + 45 class logits + 6 terrain features
  -> active ScenicRegressionModel
  -> atomic score-only Parquet partitions
  -> identity-compatible PMTiles heatmap
```

Do not stop after adding interfaces, a CLI scaffold, synthetic-only tests, or a successful inference process. Implement the complete bounded workflow, run every feasible offline and authorized pilot, retain evidence, and make an explicit keep/reject decision about nationwide execution and the sub-$100 hypothesis.

The economic target is a research hypothesis:

- promotion target: conservative projected complete cost `<= $75.00`;
- absolute national-run ceiling: `< $100.00` across network, compute, temporary storage, durable storage, and artifact operations;
- no nationwide execution merely because a mean-cost estimate passes;
- no claim that the target is viable until the geographically stratified 10,000-tile benchmark reconciles all costs and projects conservative p95/p99 behavior.

## Required predecessor

The checkout MUST contain commit:

```text
4b1e1e867be7846357a4cdfb732772b8d6b35962
```

That commit is the completed NAIP/3DEP migration and the required implementation predecessor. Record the current HEAD, branch, worktree, dirty state, source-tree digest, active registry SHA-256, classifier checkpoint SHA-256, regression checkpoint SHA-256, and relevant ignored artifact identities before editing.

Fail closed if:

- the predecessor is not an ancestor;
- unrelated user changes would be overwritten;
- required checkpoints or registry entries are missing or malformed;
- the active registry changes before an explicit promotion decision;
- existing ignored data cannot be distinguished from run-owned output.

Treat all existing data, catalogs, model weights, reports, ledgers, registries, and caches as user artifacts. Never delete or overwrite them without exact run/plan ownership proof.

## Geographic scope

The canonical initial scope is CONUS land coverage only.

- Grid: Web Mercator XYZ, exactly zoom 14.
- Target raster: exactly 512×512 pixels.
- Geometry: hash-pinned official CONUS state/land geometry under the repository's geometry contract.
- Admission: reuse the vectorized exact land-intersection policy in `src.data_pipeline.region_planning`.
- Inventory truth: the exact observed planned coordinate set and digest, not an estimated national tile count.
- Alaska: excluded because the current NAIP contract does not provide complete coverage. Literal 50-state completion requires a separately selected imagery source, source-shift benchmark, and preprocessing/model contract.
- Hawaii and territories: separate explicit contracts; never silently include them in CONUS evidence.

Do not raise or remove the current 370,000-coordinate limit globally. Create a versioned nationwide planning contract whose larger bound is derived from the observed exact inventory and whose resource limits are explicit.

## Current baseline and reusable contracts

The current canonical acquisition workflow is already NAIP/3DEP-only. It is not an active Mapbox downloader.

Relevant current behavior:

- `config/data_sources/naip_3dep_v1.json`
  - z14, 512×512, EPSG:3857;
  - `naip-visualization` RGB imagery;
  - an `allowed_states` list limited to the current eastern/Midwestern target region and enforced separately by `src.data_pipeline.naip.ALLOWED_STATES`;
  - USGS 3DEP 1/3 arc-second elevation;
  - `download_mode: whole_object`;
  - persistent satellite and Terrain-RGB PNG outputs;
  - explicit Requester-Pays authorization boundary.
- `scripts/ingest/naip_3dep_workflow.py`
  - discovery and metadata authorization;
  - immutable execution plans;
  - whole-object acquisition;
  - ownership-safe atomic outputs and resume;
  - request/byte/cost ledgers.
- `src/data_pipeline/region_planning.py`
  - exact deterministic tile enumeration and vectorized land intersection.
- `src/data_pipeline/web_mercator.py`
  - canonical XYZ coordinate and target-grid math.
- `src/data_pipeline/metered_transport.py`
  - explicit LIST, HEAD, and range GET operations;
  - `NetworkCaps`, `SharedMeteredBudget`, `MeteredLedger`;
  - reservation-before-dispatch semantics and retry accounting.
- `src/data_pipeline/naip.py` and `src/data_pipeline/usgs_3dep.py`
  - normalized catalogs, source validation, deterministic asset selection.
- `src/data_pipeline/raster_processing.py`
  - canonical imagery and DEM reprojection, QC, no-data policy, and Terrain-RGB encoding;
  - currently requires local paths for production processing.
- `src/terrain/features.py`
  - canonical Terrain-RGB decode and six-feature extraction;
  - accepts PIL images and RGB arrays, so disk serialization is unnecessary.
- `src/active_learning/scoring.py`
  - canonical classifier/regressor loading and model identities;
  - current full-manifest path accumulates candidate rows and feature arrays and is not a national streaming executor.
- `src/scenic_scorer/regression.py`
  - regression contract: 768-dimensional ViT embedding, six terrain features, and 45 class logits.
- `scripts/modeling/export_prediction_heatmap.py`
  - current hash-bound prepared-feature heatmap export;
  - currently consumes complete NPZ/metadata inputs rather than score partitions.
- `scripts/annotation/build_source_shift_batch.py` and `scripts/modeling/evaluate_source_shift.py`
  - canonical source-shift benchmark preparation and evaluation.

Reuse these contracts. Do not create duplicate coordinate math, source identities, model loaders, terrain formulas, budget ledgers, annotation schemas, or promotion policy.

## Required architecture

Create one canonical orchestration CLI:

```text
scripts/modeling/run_streaming_score_map.py
```

It must support deterministic, resumable modes through explicit subcommands or mutually exclusive flags:

1. offline plan;
2. provider/catalog authorization and discovery;
3. bounded parity pilot;
4. bounded cost benchmark;
5. status/resume;
6. authorized national execution;
7. validate/finalize;
8. PMTiles export.

The exact CLI design may follow existing repository conventions, but there must be one documented command that plans, resumes, and completes a fixed run without manual file surgery.

Canonical run root:

```text
data/processed/streaming_score_maps/conus_z14_streaming_v1/
```

Generated large artifacts remain ignored. `data/README.md` must define their final layout after implementation.

Potential new active modules may separate only real concerns:

- remote COG window access;
- streaming batch inference;
- score-partition artifacts and union.

Do not retain obsolete or parallel implementations after the clean cutover.

## Non-negotiable invariants

### Grid and tile inventory

- Every coordinate is a standard XYZ `(14, x, y)` tile.
- Target bounds, transform, CRS, pixel centers, orientation, width, and height come from the canonical Web Mercator module.
- `x` increases east and `y` increases south.
- Satellite and terrain use the exact same target grid.
- Tile inventory ordering and partition assignment are deterministic.
- Every admitted coordinate appears exactly once in the immutable execution plan.
- Geometry, coordinate list, partition list, and per-partition coordinate digests are hash-bound.
- Water/coast exclusions and geometry limitations are counted and reported.

### Source catalogs and provider selection

Evaluate at least these eligible NAIP access paths on the same bounded sample:

1. AWS `naip-visualization` Requester-Pays COGs;
2. Microsoft Planetary Computer NAIP STAC/COGs, if current terms and bulk access permit the intended use.

USGS 3DEP remains mandatory unless a replacement passes the existing terrain contract.

For every provider:

- verify current official documentation, license, attribution, access date, region, request policy, egress policy, and bulk/rate constraints;
- pin normalized catalog bytes and SHA-256;
- use canonical object/item/asset identities;
- never store SAS signatures, AWS credentials, session tokens, or expiring signed URLs as durable identities;
- measure actual range behavior and COG organization;
- reject silent source fallback;
- fail and re-plan under a new source contract when provider or bytes change.
- create one separately versioned nationwide imagery source contract after coverage validation, migrate the NAIP adapter's regional allowlist into that contract, and reject any state not proven by its pinned catalog;

Provider choice is an observed benchmark decision, not a predetermined preference.

### Metered COG ranges

Implement a production COG window interface that reads only required headers, overviews, compressed blocks, and metadata through auditable range operations.

Requirements:

- all billable or network operations reserve caps before dispatch;
- every attempt, including SDK/library retries, is counted;
- expected ETag/version/checksum is verified before accepting bytes;
- short, long, malformed, or mismatched range responses fail;
- TIFF header/IFD and block caching is bounded and content-addressed;
- adjacent tiles are grouped by source assets/spatial locality to improve block reuse;
- cache keys include canonical source identity and byte range;
- cache hits record zero network bytes and the source content identity;
- concurrent reads for one object/range use single-flight deduplication;
- network timeouts and worker deadlines are explicit;
- no hidden client retry loop or hidden remote raster I/O may bypass the ledger.

Prefer an established COG/TIFF library over implementing TIFF, compression, or geospatial codecs. It is acceptable only if it can consume an explicit metered range-backed source or if every hidden operation is authoritatively reconciled. A convenient GDAL `/vsicurl` success path with unobservable GET/retry behavior does not satisfy the contract.

### In-memory imagery preprocessing

Refactor `src.data_pipeline.raster_processing` so local-path and streaming callers share one canonical array/dataset core.

- Existing deterministic local behavior must remain the bounded parity oracle.
- New production interfaces accept opened datasets/windows or validated arrays.
- NAIP selection, mosaic ordering, resampling, fill, low-variance checks, and land masks remain contract-bound.
- Output in memory is deterministic uint8 RGB, shape `(3, 512, 512)` or one explicitly canonical equivalent.
- Do not serialize streaming satellite PNGs.
- Do not apply contrast, sharpening, histogram normalization, or undocumented color changes.

Use LSP references before changing exported symbols and migrate all callers. Do not leave deprecated aliases or duplicate preprocessing paths.

### In-memory terrain and Terrain-RGB

Both satellite and terrain are mandatory for a production score.

The terrain sequence is exact:

```text
3DEP byte ranges
  -> validated float elevation and source metadata
  -> required vertical datum transform
  -> deterministic target-grid reprojection
  -> zero no-data on valid land
  -> outside-land fill after masking only
  -> exact in-memory Terrain-RGB in encoder layout `(3, H, W)`
  -> explicit transpose to feature layout `(H, W, 3)`
  -> src.terrain.features.compute_terrain_features
  -> release DEM and Terrain-RGB arrays
```

Encoding for finite elevation `h` in meters:

```text
q = floor((h + 10000.0) * 10.0 + 0.5)
require 0 <= q <= 16777215
R = (q >> 16) & 255
G = (q >> 8) & 255
B = q & 255
```

- Use float64 for quantization.
- Reject NaN/Inf on valid land.
- Reject rather than clamp out-of-domain elevations.
- Require decode round-trip absolute error `<= 0.0500001 m`.
- Preserve source and target vertical datum, units, no-data, grid-resource hashes, and transform provenance.
- Never resample encoded Terrain-RGB; reproject float elevation first.
- Do not bypass Terrain-RGB and compute directly from float DEM unless a future separately versioned parity/model change explicitly proves equivalence. Initial implementation must use the exact in-memory Terrain-RGB path.

The regression terrain vector remains the canonical six values, in order:

1. normalized slope variation;
2. elevation change divided by 1000;
3. water proximity;
4. vegetation density;
5. coastal indicator;
6. lake-or-river indicator.

### Streaming model inference

Add a public bounded inference primitive that accepts in-memory satellite and Terrain-RGB batches and returns score/QC rows without accumulating a complete pool.

Reuse:

- `load_scoring_models` and registry/checkpoint validation;
- the canonical RESISC45 classifier transform;
- ViT-B/16 classifier inference;
- `compute_terrain_features`;
- `ScenicRegressionModel`.

Per successful tile, the model input is exactly:

```text
768-dimensional satellite embedding
+ 6-dimensional terrain vector
+ 45 raw class logits
= 819 dimensions
```

Requirements:

- load models once per worker process;
- explicit bounded batch size, prefetch, worker count, and GPU queue;
- no unbounded result list, embedding matrix, logits matrix, or full-pool NPZ;
- release image arrays and tensors after each settled batch;
- record peak CPU/GPU memory and throughput;
- deterministic ordering independent of worker completion order;
- per-tile failures produce explicit status/error evidence rather than disappearing.

Record normalized classifier entropy as `class_uncertainty`. Do not call entropy, `1 - entropy`, raw logits, regression output, or heuristic score calibrated confidence. A confidence field requires a separately trained and hash-bound calibration artifact.

### Score partitions

Publish atomic Parquet partitions and one immutable run manifest.

Minimum row schema:

```text
z: uint8
x: uint32
y: uint32
jurisdiction_id: dictionary integer
scenic_score: nullable float32
class_uncertainty: nullable float32
slope_variation: nullable float32
elevation_change_km: nullable float32
water_proximity: nullable float32
vegetation_density: nullable float32
coastal: nullable bool
has_lake_or_river: nullable bool
water_fraction: nullable float32
terrain_sea_level_fraction: nullable float32
status: dictionary enum
error_code: nullable dictionary enum
source_asset_set_id: nullable dictionary integer
acquisition_identity_sha256: fixed 32-byte value
```

The run manifest stores, without per-row string duplication:

- schema versions;
- source providers, catalogs, object/item tables, and hashes;
- geometry and coordinate digests;
- source and preprocessing contracts;
- classifier, regression, and calibration identities;
- repository revision and source-tree digest;
- commands, environment, dependency lock, seeds, and devices;
- partition plan, ownership, hashes, row counts, coordinate digests, and score/failure summaries;
- ledgers and aggregate cost evidence.

A partition is valid only after temporary write, schema validation, exact coordinate membership check, uniqueness check, row-count check, finite/range checks, SHA-256, and atomic install. Do not treat file existence as completion.

Do not retain full national embeddings or class logits by default. A deterministic bounded diagnostic sample is allowed. Any feature sidecar is separately planned, budgeted, and identified.

### Resume, ownership, and concurrency

- Partition assignment is immutable and hash-bound.
- Workers claim partitions atomically or through expiring leases with owner identity.
- One global shared budget covers all workers.
- Workers cannot enlarge caps, modify the plan, or substitute sources.
- Resume reconstructs completion only from currently validated partitions and checkpoint records.
- Corrupt run-owned files are quarantined before replacement.
- Foreign/unowned outputs are immutable and fail the run.
- Duplicate coordinate output is rejected unless byte/score/identity identical under a declared union rule.
- Final union is independent of worker completion order.
- Spot/preemptible termination must preserve truthful pending/completed state.

### PMTiles heatmap

Extend or cleanly replace the complete-NPZ heatmap path for nationwide score partitions.

- Validate every partition and compatibility identity before export.
- Encode z14 scenic scores and required display metadata without embedding source imagery.
- A lower-zoom visualization pyramid may be derived only through one declared deterministic aggregation rule.
- PMTiles is a visualization artifact, not human/model promotion evidence.
- Measure archive size, build runtime, browser requests, and client rendering behavior.
- Drive the canonical web app in a browser and verify representative low/high score regions, zoom transitions, missing/error rendering, and metadata attribution.

## Cost and resource authority

### No implicit authorization

Planning, catalog construction, provider benchmarking, parity acquisition, cost benchmarking, national execution, and durable upload are distinct authorization boundaries.

If required network or compute caps are absent:

- make no network call;
- launch no paid instance;
- emit an immutable authorization request with exact operations, conservative reservations, pinned rate cards, maximum instance price/count/runtime/storage, expected outputs, and an exact resume command;
- stop at that boundary without generating downstream evidence.

Discovery authorization does not authorize pilot data. Pilot authorization does not authorize national execution. National execution does not authorize registry promotion.

### Aggregate cost model

Calculate and retain:

```text
network requests = settled requests by operation and provider
network transfer = settled billable bytes by provider/region
compute = instance price * observed wall time for every instance
storage = temporary and durable byte-hours at pinned rates
artifact operations = upload requests + transfer + retained storage
complete cost = fixed discovery/export cost + all partitions + contingency
```

`NetworkCaps` is necessary but insufficient because it does not cap GPU/CPU instance charges or durable storage. Add an aggregate run budget that reserves before:

- network dispatch;
- instance rental or worker expansion;
- temporary disk growth;
- durable artifact write/upload.

The run must declare:

- maximum network requests and transfer;
- maximum Requester-Pays spend;
- maximum instance unit price;
- maximum concurrent instances;
- maximum total instance-hours;
- maximum temporary bytes;
- maximum output bytes;
- maximum upload bytes and storage commitment;
- absolute deadline.

Destroy every cloud/Vast instance on success, cap, deadline, keyboard interruption, or terminal failure. Verify zero surviving instances before yielding.

## Implementation and experiment sequence

### Phase 0 — preflight

1. Validate predecessor ancestry, branch/worktree, dirty state, ignored artifact ownership, dependency lock, and credentials availability without exposing secrets.
2. Record registry and checkpoint identities. Confirm no mutation occurs.
3. Re-read current provider documentation and pin access dates/rates/terms.
4. Inventory current tests and reusable symbols. Use LSP references before exported-symbol changes.
5. Create an immutable implementation run manifest and exact resume command.

### Phase 1 — offline national plan

1. Build the hash-pinned CONUS land geometry contract.
2. Enumerate the exact z14 tile set with the canonical land policy.
3. Produce state/slice counts, coordinate digest, partition plan, and storage bounds.
4. Confirm no duplicate, missing, foreign, or water-only coordinates under the declared policy.
5. Emit catalog/provider discovery authorization if network is required.

### Phase 2 — metered COG window foundation

1. Implement explicit range-backed COG access with bounded header/block caching.
2. Integrate ETag/version/checksum validation and metered retry behavior.
3. Add deterministic source-asset selection and window-to-byte provenance.
4. Test malformed ranges, changed identity, caps, retry accounting, concurrency, and cache behavior.
5. Prove no hidden network path through tests and a bounded ledger reconciliation smoke.

### Phase 3 — shared in-memory preprocessing

1. Refactor imagery and DEM processing into shared dataset/window/array cores.
2. Preserve local persistent-pipeline results.
3. Implement exact in-memory Terrain-RGB and six-feature path.
4. Add alignment, datum, no-data, mosaic, resampling, corruption, and round-trip tests.
5. Remove duplicated or obsolete processing code.

### Phase 4 — bounded streaming scorer

1. Implement one batch inference API and model lifecycle.
2. Implement score/QC rows without full-pool accumulation.
3. Add bounded memory/backpressure.
4. Implement partition writer, validation, hashes, claims, checkpoint, and resume.
5. Add deterministic union and status reporting.

### Phase 5 — offline and synthetic verification

Run focused tests for every changed subsystem, then the complete suite. Add a synthetic two-source COG smoke that exercises:

- range reads;
- aligned satellite/DEM windows;
- Terrain-RGB;
- terrain features;
- classifier/regression batch;
- atomic partition;
- interruption/resume;
- score union.

The smoke output is proof of mechanics only, not cost or model validity.

### Phase 6 — 500-tile parity gate

Choose a fixed sample spanning jurisdictions, source edges, mosaics, vintages/resolutions, terrain, coast, urban/rural land, and nodata risks. Use identical pinned source objects for both paths.

Compare the canonical retained-raster oracle against streaming:

- selected asset sets;
- target grids and masks;
- satellite RGB arrays;
- Terrain-RGB arrays;
- QC decisions;
- six terrain features;
- embeddings, logits, uncertainty, and scenic score;
- source/preprocessing/model identities.

Require byte identity where the same deterministic codec path promises it. Otherwise declare and justify numeric tolerances before viewing results. Reject any unexplained discrepancy.

### Phase 7 — human-grounded source-shift gate

Build a fixed matched-source annotation batch using the canonical annotation schema. Ensure sufficient independent annotators, confidence coverage, fixed geographic validation/test splits, and leakage audits.

Evaluate:

- human MAE/RMSE and rank behavior;
- calibration where a real calibration artifact exists;
- state/region, terrain, scene, water/coast, source-vintage, and failure slices;
- report score distribution and tails;
- supported route ordering and behavior;
- control benchmark non-regression.

Do not promote, recalibrate, or modify the registry unless every compound gate passes. If data-limited, emit the exact next annotation-batch request and keep the baseline.

### Phase 8 — 10,000-tile cost/reliability gate

Use a deterministic geographically and technically stratified sample. Record:

- requests and bytes per provider/operation;
- TIFF header/overview/block amplification;
- block-cache and source-asset reuse;
- decode, reprojection, feature, classifier, regression, and write timings;
- CPU/GPU utilization and peak memory;
- retries, throttling, timeouts, missing coverage, and failures;
- temporary/output bytes;
- exact instance, network, and storage prices;
- projected national runtime and p50/p95/p99 cost.

Pass only when:

- every operation and cost reconciles;
- projected complete conservative cost is `<= $75.00`;
- absolute national authorization can remain below `$100.00`;
- no hidden provider/rate/egress assumption remains;
- failure and retry rates are operationally acceptable;
- no data or model gate has failed.

Otherwise reject the current design and retain the evidence. Do not average away expensive source/layout slices.

### Phase 9 — crash/resume and deterministic union

Interrupt workers before request, after reservation, during range read, after inference, during temporary write, and after atomic install. Resume and prove exact coordinate completeness, no invalid skip, truthful spend, ownership safety, and deterministic union.

### Phase 10 — national authorization boundary

If and only if all prior gates pass, produce a national execution authorization package containing:

- exact coordinate and partition counts/digests;
- provider/catalog/source/preprocessing/model identities;
- expected requests, bytes, compute, storage, upload, runtime, and conservative cost;
- aggregate hard caps and deadline;
- exact command to execute/resume;
- expected durable artifacts and teardown plan;
- proof that registry state remains protected.

Do not infer authorization from this prompt. A positive bounded authorization for the observed package is required before national network/compute execution.

### Phase 11 — authorized national run and product QA

After explicit authorization:

1. execute/resume all partitions under the immutable plan;
2. reconcile every worker ledger and resource charge;
3. validate coordinate completeness and all partition hashes;
4. build and validate the compatible PMTiles artifact;
5. drive the web layer in a browser;
6. run supported route QA without claiming coverage where no graph exists;
7. upload only authorized durable artifacts;
8. verify remote size, hashes, metadata, and restore/readability;
9. destroy all instances and verify none remain;
10. make an explicit final keep/reject and promotion decision.

## Test requirements

Add behavior-focused tests for at least:

- exact range reservation, response validation, and retry accounting;
- range-backed COG header/block cache identity;
- single-flight concurrent ranges;
- source selection and mosaic determinism;
- remote/local preprocessing parity;
- imagery and DEM target-grid alignment;
- Terrain-RGB exact encoding, rejection, and round-trip;
- no-data on land and outside-land fill;
- bounded inference memory and ordering;
- class uncertainty naming/definition;
- Parquet schema, atomic install, hashes, and finite/range validation;
- partition claim/lease, crash, quarantine, and resume;
- foreign output refusal;
- duplicate/missing coordinate rejection;
- union compatibility and order invariance;
- aggregate network/compute/storage cap arithmetic;
- no output from failed preflight/authorization;
- PMTiles compatibility and representative browser rendering.

Run focused tests while developing. After integration run changed-file formatting/lint, the complete repository suite, the synthetic end-to-end smoke, the 500-tile parity gate, and every authorized empirical benchmark.

## Clean-cutover requirements

- Keep one coordinate implementation.
- Keep one Terrain-RGB formula and feature implementation.
- Keep one model loader and registry authority.
- Keep one source identity and provenance framework.
- Keep one budget/ledger framework extended to all costs.
- Keep one production national streaming orchestrator.
- Preserve retained raster generation only as the bounded acquisition/annotation/parity workflow; it must share preprocessing cores and must not remain the national execution path.
- Remove obsolete helpers, duplicate code, deprecated aliases, compatibility shims, stale comments, and superseded docs.
- Update `README.md`, `data/README.md`, architecture, setup/deployment guidance, roadmap/research status, and the archive manifest if files are removed or archived.

## Required final evidence

The final report must include:

1. architecture and exact files/symbols changed;
2. exact plan, pilot, benchmark, resume, validation, PMTiles, and authorized execution commands;
3. predecessor, repository, source-tree, dependency, geometry, catalog, source, preprocessing, model, calibration, and artifact hashes;
4. exact CONUS tile and partition counts;
5. persistent-versus-streaming parity table;
6. human benchmark and source-shift results;
7. regional/scene/calibration/distribution/route evidence;
8. 10,000-tile request, byte, throughput, failure, resource, and cost table;
9. p50/p95/p99 national runtime and cost projection;
10. explicit keep/reject decision for the `<$100` hypothesis;
11. registry state before and after, with exact active checkpoint identity;
12. PMTiles size, hash, browser evidence, and compatibility identity;
13. durable upload location, size, hash, and verification, if authorized;
14. cloud/Vast lifecycle and proof of zero surviving instances;
15. unresolved risks and exact next action;
16. if data-limited, the exact next human annotation request.

## Stop and rejection conditions

Reject or stop at the appropriate authorization boundary when any of these occurs:

- exact CONUS inventory exceeds declared planning/resource bounds;
- required NAIP or 3DEP coverage is absent;
- source identity, ETag/version/checksum, datum, units, or no-data metadata is unknown;
- remote operations cannot be fully metered/reconciled;
- streamed arrays/features/scores fail parity;
- human-grounded source-shift or control gates fail;
- projected conservative total cost exceeds `$75` or cannot fit below the `$100` hard ceiling;
- provider terms/rate limits do not support the run;
- resume loses, duplicates, or incorrectly skips work;
- PMTiles union mixes incompatible identities;
- route or distribution behavior regresses;
- registry or baseline identity changes without every promotion gate;
- durable artifacts cannot be hash-verified;
- compute teardown cannot be confirmed.

A rejection with complete evidence is a valid result. Fabricated confidence, cost, route QA, provider capacity, national completeness, or promotion evidence is not.

Do not declare completion from a green test suite, successful 500-tile pilot, or finished national inference alone. Completion requires the full set of feasible gates, an explicit authorization-safe boundary for anything not authorized, a final keep/reject decision, protected registry state, and zero surviving paid resources.
