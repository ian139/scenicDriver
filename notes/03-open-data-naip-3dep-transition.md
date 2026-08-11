# OMP Prompt — Migrate Scenic Drive Acquisition from Mapbox to NAIP + USGS 3DEP

Execute this task end to end from the Scenic Drive repository root. This is an implementation, validation, pilot-data, model-transition, and evidence task—not a design-only exercise. Read and obey `AGENTS.md` and all applicable repository guidance before changing code.

## Mission

Replace the active ML tile-acquisition path with a deterministic, source-versioned open-data pipeline:

```text
NAIP visualization COG ──► exact XYZ z14 grid ──► 512×512 RGB PNG ───────┐
                                                                         ├─► existing inventory, scoring, evaluation, and heatmap workflow
USGS 3DEP DEM ───────────► same exact XYZ grid ──► 512×512 Terrain-RGB PNG ┘
```

Preserve the downstream operational contract, but do not claim that NAIP-derived predictions are numerically comparable to predictions from the Mapbox-trained model until human-grounded source-transition gates pass.

The clean long-term contract is:

- imagery source: NAIP 3-band RGB Cloud Optimized GeoTIFFs;
- terrain source: a pinned USGS 3DEP DEM product with complete target coverage;
- target grid: Web Mercator XYZ zoom 14, exactly 512×512 pixels;
- source-versioned local/S3 roots and manifests;
- source-versioned preprocessing and model identities;
- no Mapbox network acquisition in the active ML pipeline;
- no cross-source cache reuse;
- no mixed-source heatmap presented as one calibrated score surface;
- no active model or registry promotion until every human, control, distribution, and route gate passes.

## Repository baseline

Begin from `main` commit:

```text
1bb4ac846e568e652d73a1f0772ff97a9fec22a4
```

Fail closed if the checkout is not based on that commit or if unrelated user changes would be overwritten. Treat existing ignored data, model checkpoints, reports, and registries as user artifacts. Never delete or overwrite them without explicit proof that the file is owned by this run.

Current relevant contracts include:

- `scripts/ingest/plan_active_learning_region.py`
  - currently hard-codes `MapboxTileSource`, `mapbox.satellite`, and `mapbox.terrain-rgb`;
  - writes `tile_manifest.csv` and `acquisition_preflight.json`;
  - supports local and S3 inventory;
  - defaults to z14 and a 370,000-coordinate cap.
- `src/data_pipeline/region_planning.py`
  - currently owns XYZ coordinate enumeration and simple polygon filtering;
  - imports coordinate helpers from the Mapbox module;
  - the built-in expanded region still includes bounding-box water and foreign land.
- `src/data_pipeline/tile_inventory.py`
  - already accepts a configurable image root;
  - validates paired satellite and terrain PNGs.
- `src/active_learning/scoring.py`
  - already records source/content SHA-256 values;
  - consumes manifest paths and computes model inputs;
  - must be strengthened so cache and candidate identities include the source/preprocessing contract, not only mutable paths.
- `src/terrain/features.py`
  - decodes Terrain-RGB as:

  \[
  h=-10000+0.1(R\cdot256^2+G\cdot256+B)
  \]

  - computes terrain and vegetation features from aligned satellite/terrain tiles.
- `scripts/modeling/export_prediction_heatmap.py`
  - exports deterministic reports from prepared features;
  - any union/join must reject incompatible source, preprocessing, model, grid, or region contracts.
- `scripts/annotation/` and `src/active_learning/`
  - contain the canonical annotation and human-benchmark workflow; reuse it rather than creating a second labeling format.
- `src/scenic_scorer/active_evaluation.py`
  - contains the canonical compound evaluation gates; extend or reuse those gates instead of inventing a parallel promotion system.
- `data/README.md`
  - remains the canonical data layout contract and must describe the final source-versioned layout.

Before modifying exported symbols, use LSP references and migrate every caller. Do not leave compatibility aliases, deprecated wrappers, duplicate coordinate math, or dead Mapbox acquisition paths.

## Authoritative external sources

Use primary sources and record the exact URLs, access date, object identities, and licenses in run provenance:

1. NAIP on AWS: <https://registry.opendata.aws/naip/>
   - use the `naip-visualization` 3-band RGB COG collection unless a source audit proves another NAIP product is required;
   - it is JPEG-compressed RGB, quality 85, with 512×512 internal tiles and overviews;
   - the bucket is Requester Pays in `us-west-2`;
   - the registry states “Public Domain with Attribution”;
   - the catalog currently covers varying state vintages, generally 2010–2023, and each state updates on a different cycle.
2. USGS 3DEP: <https://www.usgs.gov/3d-elevation-program>
   - 3DEP products are free of charge and without use restrictions;
   - discover and pin the exact DEM product and objects used;
   - prefer the seamless 1/3 arc-second product for consistent complete regional coverage unless evidence supports a different uniform product;
   - do not mix 1 m, 3 m, 10 m, and 30 m products within one preprocessing contract without an explicit, tested policy;
   - the public National Map bucket exposes elevation products below `s3://prd-tnm/StagedProducts/Elevation/`.
3. State/coast clipping: U.S. Census 2025 cartographic boundary files: <https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html>
   - pin the exact 2025 state boundary archive and SHA-256;
   - prefer the highest practical official state/coastline representation, currently the 1:500,000 national state file, rather than handwritten rectangles;
   - quantify simplification and coastal-island limitations in the run report.

Do not use expiring signed URLs as durable identities. Store canonical `s3://` keys or stable HTTPS object paths, object ETag/version/checksum when available, and source metadata. Never write AWS credentials, request signatures, session tokens, or secret query parameters to logs or manifests.

## Non-negotiable invariants

### Data and geometry

- Zoom is exactly 14.
- Every output tile is exactly 512×512.
- Satellite output is deterministic 8-bit RGB PNG with no alpha channel.
- Terrain output is deterministic 8-bit RGB PNG using the repository’s Terrain-RGB formula.
- Satellite and terrain outputs for a coordinate use the exact same bounds, CRS, affine transform, width, height, pixel centers, and orientation.
- `x` increases east; `y` increases south; latitude is not vertically flipped.
- Tile bounds use standard slippy-map/Web Mercator semantics in EPSG:3857.
- Do not derive target bounds by rounding lat/lon. Compute them mathematically from `(z, x, y)`.
- Do not apply per-tile auto-contrast, histogram equalization, sharpening, or color normalization. Any future source normalization must be versioned, trained, and evaluated as a model change.
- Preserve source pixels as faithfully as the reprojection permits.
- Satellite resampling must use one declared deterministic kernel appropriate for imagery, such as Lanczos or cubic. DEM resampling must use a declared deterministic continuous-elevation kernel, normally bilinear. Never use RGB resampling on already encoded Terrain-RGB.
- Decode DEM values first, transform vertical datum when required, reproject/resample float elevation, handle no-data, and only then encode Terrain-RGB.
- The hashed source/preprocessing contract must declare one exact target vertical CRS/datum, including EPSG/PROJ identity and the SHA-256 of any geoid/grid resource. Every selected DEM must either already match that identity or be deterministically transformed to it before mosaicking. Unknown source datum, unavailable transform grids, mixed unnormalized datums, or unverified units fail the plan.
- Record horizontal CRS, source and target vertical datums, units, no-data value, source resolution, output resolution, and every transformation resource. Do not infer datum from filenames.
- Terrain-RGB encoding is byte-exact: for finite elevation `h` in meters, compute `q = floor((h + 10000.0) * 10.0 + 0.5)` using float64, i.e. half steps round toward positive infinity after the positive offset. Require `0 <= q <= 16777215`; reject rather than clamp out-of-domain values. Encode `R=(q>>16)&255`, `G=(q>>8)&255`, and `B=q&255`. Reject NaN/Inf on valid land. Decode with the repository formula and require absolute round-trip error at most `0.0500001 m`. Bind this algorithm and version into the preprocessing hash.
- Construct a target-grid land mask using the pinned land geometry and nearest-neighbor mask rasterization. The allowed DEM no-data fraction for output pixels whose centers are on land is exactly `0.0`. Any missing land pixel is a hard failure. Ocean/outside-land fill may use 0 m only after masking and must be counted separately from measured 0 m elevation.
- Detect and reject all-zero, all-black, all-white, truncated, corrupt, wrong-dimension, or implausibly low-variance satellite tiles unless explicitly classified as valid by a predeclared rule in the hashed preprocessing contract.
- Measure no-data fraction, valid-pixel fraction, source overlap, and output coverage per tile.

### Source selection

- NAIP source selection must be deterministic.
- Discover all COGs intersecting the target tile, then choose using a predeclared ordering. At minimum include state, acquisition year/date, product type, valid coverage fraction, resolution, stable asset ID, and stable tie-break ordering.
- Prefer a single recent statewide vintage where possible to reduce mosaicking seams. If multiple assets are required for one output tile, mosaic in a deterministic order and record every contributing asset and pixel fraction.
- Never silently substitute an older year, different state product, or different NAIP collection. Record substitutions as explicit warnings and include them in promotion gates.
- Choose and pin one uniform 3DEP product contract for the run. Record every backing object/window.
- Asset discovery results must be cached with a catalog/version identity and invalidated when the catalog contract changes.

### Source-versioned storage

The relative downstream file shape must remain:

```text
satellite/z14/<region>/<x>_<y>.png
terrain/z14/<region>/<x>_<y>.png
```

However, do not overwrite legacy Mapbox-derived files or rely on source-blind roots. Use a source-versioned root for acquisition and scoring, for example:

```text
data/raw/sources/naip_3dep_v1/images/satellite/z14/<region>/<x>_<y>.png
data/raw/sources/naip_3dep_v1/images/terrain/z14/<region>/<x>_<y>.png
```

The planner, inventory, scorer, benchmark builder, and exporter must accept the explicit root through a manifest or CLI option. The normalized relative path contract remains unchanged beneath that root.

Never mix Mapbox and NAIP bytes under one source-blind cache key. Define output ownership independently from content: a file is run-owned only when its adjacent run record contains the exact immutable run ID and execution-plan SHA-256. For the same run/plan, quarantine and audit the old bytes before atomically replacing a corrupt or content-mismatched output. Treat a file owned by another run/plan, or a file with no provable owner, as immutable and fail rather than overwrite it. Write replacements to a temporary file, validate them completely, fsync where the repository’s atomic helpers do so, and atomically install only run-owned outputs.

### Heatmap/model compatibility

- A report or heatmap is valid only for one source contract, one preprocessing contract, one model/checkpoint, and one grid contract.
- Do not append NAIP-scored Southern New England tiles to the existing Mapbox-derived expanded heatmap and present the result as one calibrated score field.
- A deterministic union tool may join independently processed chunks only when all of these match exactly:
  - source contract SHA-256;
  - preprocessing contract SHA-256;
  - tile zoom/size/grid semantics;
  - classifier checkpoint SHA-256;
  - regression checkpoint SHA-256;
  - score schema version;
  - label schema/version;
  - calibration identity;
  - non-overlap or byte/score-identical duplicate policy.
- If the open-data source transition passes, regenerate every region intended for one unified open-data heatmap using the same NAIP/3DEP model contract. Until then, expose source-specific experimental layers only.
- Keep route planning disabled for any heatmap region without a compatible road graph. Do not claim route QA where no route graph exists.

### Cost and network safety

- Dry-run planning is the default.
- No network acquisition may begin merely because `--acquire` was supplied.
- Treat source discovery as network execution: NAIP LIST/HEAD/catalog operations may themselves be Requester Pays. Prefer a pinned, hash-verified local catalog snapshot or a stable non-billable official index when one exists. Never assume discovery is free.
- If no adequate local/non-billable catalog exists and no positive discovery caps are authorized, do not contact the bucket. Produce `discovery_authorization_request.json` containing target geometry/count, planned catalog prefixes/operations, per-operation byte reservation, conservative request/byte/cost caps, rate-card source/date, and the exact resume command. This pre-discovery package is the first valid authorization boundary.
- After discovery-only caps are authorized, meter catalog operations through the same transport, pin the resulting catalog snapshot/hash, and then build the exact immutable acquisition execution plan below. Discovery authorization does not authorize imagery/DEM acquisition.
- Build an immutable execution plan containing:
  - exact coordinate count;
  - exact missing satellite/terrain counts;
  - unique NAIP assets;
  - unique 3DEP assets;
  - estimated COG range requests and upper bound;
  - expected bytes and conservative upper bound;
  - local and S3 reuse counts;
  - expected local storage;
  - Requester Pays status;
  - rate-card source/date;
  - estimated request, transfer, and total cost;
  - explicit hard caps;
  - source/catalog/preprocessing/geometry hashes.
- Execution must require the plan path plus its expected SHA-256. Recompute discovery and fail if the plan drifts.
- Require explicit positive caps for request count, transferred bytes, local bytes, and requester-pays spend. Zero means no paid/requester-pays network execution, not unlimited.
- Add a paid-source kill switch that defaults to disabled. The active ML pipeline must refuse Mapbox and any unknown billable provider. NAIP Requester Pays must require a separate explicit `--allow-requester-pays` acknowledgement plus a maximum spend.
- Route every remote catalog list, HEAD, GET, range GET, retry, and error response through one metered transport that reserves capacity before dispatch and records actual response bytes. A request attempt counts even when it fails; every retry counts separately; catalog and metadata traffic count; transferred bytes are bytes actually read from the response. Reserve the declared maximum range length before an indeterminate range request, reconcile after completion, and stop before dispatch when reservation would exceed any cap.
- Do not let GDAL `/vsicurl` or `/vsis3` perform unobservable remote I/O. The preferred design is a controlled local range-cache/downloader or an explicitly proven interception proxy through which every GDAL operation passes; rasterio/GDAL then opens only the metered local/controlled source. A test must demonstrate that hidden HEAD, directory-listing, retry, and range operations cannot bypass counters. If enforceable interception cannot be proven, fail closed rather than claim runtime cap enforcement.
- Enforce request, byte, disk, and cost caps during execution, not only during preflight. Rate-limit and bound concurrency. Retry only transient failures with capped exponential backoff and jitter. Never retry authentication, authorization, malformed-object, checksum, or cap failures.
- This prompt grants no financial authorization and intentionally provides no numeric Requester Pays cap. Do not invent authorization. With neither a usable pinned non-billable catalog nor discovery authorization, stop at the verified `discovery_authorization_request.json` boundary. After discovery is authorized/completed but acquisition caps are absent, stop at the immutable execution plan, its SHA-256, conservative cost bound, and exact resume command. A real one-tile smoke and pilot become required only after their complete positive caps are supplied.
- Emit request/byte/cost counters throughout execution without logging credentials or signed URLs.
- Prefer execution near the source region when that materially reduces transfer cost, but do not add cloud infrastructure unless it reduces total complexity and is reproducible.
- A terminal failure must leave a resumable, truthful manifest. Never mark incomplete tiles reusable.

## Required architecture

Implement the smallest coherent architecture that satisfies the above contracts. Reuse existing code; do not create a second planning/scoring framework.

### 1. Decouple XYZ math from Mapbox

Move provider-independent tile math out of `src/data_pipeline/mapbox.py` into a provider-neutral module such as `src/data_pipeline/web_mercator.py`.

Required functions, names adjusted to repository convention:

- `lat_lon_to_tile(lat, lon, zoom)`;
- `tile_to_lat_lon_center(x, y, zoom)`;
- `tile_bounds_wgs84(x, y, zoom)`;
- `tile_bounds_web_mercator(x, y, zoom)`;
- `tile_transform_web_mercator(x, y, zoom, width=512, height=512)`.

Use LSP references to migrate every caller. Remove duplicated/provider-specific copies.

### 2. Add source models and adapters

Add typed immutable source/provenance models. A source asset record must be serializable without secrets and include at least:

```text
provider
collection
asset_id
canonical_uri
state_or_region
acquisition_year
capture_date_or_range
published_at if available
license
attribution
horizontal_crs
vertical_datum when applicable
native_resolution
band contract
no_data
etag/version/checksum when available
metadata_sha256
```

Add a NAIP adapter responsible for deterministic asset discovery and reading imagery windows. Add a 3DEP adapter responsible for deterministic product discovery and reading float-elevation windows. Keep provider-specific catalog parsing and request semantics inside those adapters.

Use `rasterio`/GDAL for COG window reads, CRS transforms, WarpedVRT/reprojection, and no-data handling unless a source-verified existing dependency offers a simpler equivalent. Use `uv` for dependency changes. Add `shapely` only if necessary for robust geometry; do not hand-roll multipolygon/holes/intersection logic.

Source adapters must be dependency-injectable so tests use tiny local GeoTIFF fixtures and fake catalogs without network access.

### 3. Add deterministic raster processing

Create one canonical processing layer that:

- constructs the target EPSG:3857 transform once;
- reads/mosaics the source window;
- reprojects directly onto that transform;
- validates coverage and no-data;
- emits deterministic RGB arrays;
- encodes float DEM to Terrain-RGB;
- validates encode/decode round trip;
- writes deterministic PNGs atomically;
- returns output SHA-256, byte count, pixel statistics, and full source contribution provenance.

Satellite and terrain must share the exact target-grid object, not independently recompute approximately equal transforms.

### 4. Refactor regional planning and clipping

Refactor `src/data_pipeline/region_planning.py` and `scripts/ingest/plan_active_learning_region.py` to support authoritative multipolygon geometry with holes.

For the Southern New England source-transition target:

- include Connecticut, Rhode Island, and the missing Massachusetts area;
- subtract coordinates already covered by the intended target run only after source identity is considered;
- use a pinned state boundary to select jurisdictions and a separately verified, pinned land/coast geometry for land area if the state geometry contains jurisdictional water;
- perform tile/land area calculations in EPSG:5070 with no additional simplification after loading the pinned geometry;
- include a tile when its center is on land **or** its positive-area land intersection divided by tile area is at least `0.05`; boundary touching with zero area does not qualify;
- apply the same rule to islands and coastal slivers; record which branch admitted each tile;
- exclude ocean-only tiles;
- put the area CRS, `0.05` threshold, center-on-land rule, boundary semantics, geometry versions, and SHA-256 values inside the hashed region specification;
- record per-state counts, intersection-fraction distribution, center-rule admissions, excluded-water counts, and exact boundary/land hashes.

Do not call a jurisdiction polygon a land mask without proving its water semantics. Quantify coastline generalization and inspect representative Cape/island/coastal cases. Do not retain the current simple ray-casting implementation if `shapely` becomes the canonical geometry engine. Cleanly migrate callers and tests.

### 5. Refactor the planner CLI

The planner must have explicit source selection and default to open data. A representative final interface is:

```bash
uv run python scripts/ingest/plan_active_learning_region.py \
  --run-name sne_naip_3dep_pilot_20260810 \
  --region-spec config/data_sources/regions_v1.json \
  --region-source config/app_regions.json \
  --imagery-source naip-visualization \
  --terrain-source usgs-3dep-13as \
  --source-contract config/data_sources/naip_3dep_v1.json \
  --image-root data/raw/sources/naip_3dep_v1/images \
  --output-root data/processed/active_learning \
  --budget 500
```

Planning writes a run directory and performs no acquisition. A separate explicit execution mode must bind to the immutable plan, for example:

```bash
uv run python scripts/ingest/plan_active_learning_region.py \
  --execute-plan data/processed/active_learning/sne_naip_3dep_pilot_20260810/execution_plan.json \
  --expected-plan-sha256 <sha256> \
  --allow-requester-pays \
  --max-source-requests <positive integer> \
  --max-transfer-bytes <positive integer> \
  --max-local-bytes <positive integer> \
  --max-requester-pays-usd <positive decimal> \
  --workers <bounded integer>
```

Exact flags may follow existing conventions, but the two-step immutable-plan contract and runtime caps are mandatory.

Remove Mapbox acquisition from this planner. Existing legacy Mapbox artifacts may remain readable as immutable historical inputs, but the active planner must never make a Mapbox request. Delete obsolete bulk-download scripts or paths if they have no remaining legitimate caller. Do not leave a hidden fallback based on `MAPBOX_ACCESS_TOKEN`.

### 6. Extend manifests and cache identities

Version the manifest schema and add per-row fields for both sources:

```text
source_contract_sha256
preprocessing_contract_sha256
boundary_geometry_sha256
satellite_provider
satellite_collection
satellite_asset_ids
satellite_acquisition_year
satellite_capture_date
satellite_license
satellite_attribution
satellite_source_checksums
satellite_output_sha256
satellite_valid_fraction
terrain_provider
terrain_collection
terrain_asset_ids
terrain_vertical_datum
terrain_native_resolution
terrain_license
terrain_source_checksums
terrain_output_sha256
terrain_valid_fraction
mosaic_contributions
processing_version
```

Use deterministic compact JSON strings for one-to-many fields in CSV, or move nested provenance to a row-addressable JSONL sidecar while keeping scalar manifest columns. Do not create ambiguous comma-joined fields.

Use layered identities; do not make immutable acquisition identity depend on a model:

```text
acquisition_tile_identity
  = (z, x, y, region)
  + exact target-grid contract
  + exact satellite and terrain output SHA-256
  + source contract SHA-256
  + preprocessing contract SHA-256

embedding_feature_identity
  = acquisition_tile_identity
  + classifier preprocessing identity
  + classifier checkpoint SHA-256
  + terrain feature schema/version

prediction_report_identity
  = embedding_feature_identity
  + regression checkpoint SHA-256
  + calibration type/config/artifact SHA-256
  + score schema version
  + label schema/version
```

Update `source_identity`, `tile_identity`, feature cache keys, prepared-dataset identity, report identity, and model registry records according to those layers. Any mismatch at a layer must recompute that layer and every downstream layer rather than reacquire valid upstream bytes unnecessarily. Add a negative test proving Mapbox-derived embeddings cannot be reused for NAIP bytes at the same `(z, x, y)` and path shape. Add another proving two calibration artifacts cannot collide.

### 7. Deterministic resume and audit

A resumed run must:

- validate run name and immutable plan identity;
- validate every existing output PNG and recorded SHA-256;
- validate source and preprocessing contract hashes;
- skip only exact valid matches;
- reacquire corrupt or mismatched run-owned outputs;
- never trust existence or non-zero size alone;
- append structured attempt/failure records without losing prior history;
- keep final manifests deterministically sorted by `(region, z, x, y)`;
- produce the same completed manifest/report hashes when inputs and provider objects are unchanged, excluding explicitly isolated wall-clock metadata.

## Pilot acquisition and source-transition evaluation

Do not begin full Southern New England acquisition immediately.

### Pilot design

Use a deterministic, capped two-stage pilot. Do not depend on scores or terrain features before acquiring the bytes needed to compute them.

1. Build a metadata-only acquisition frame of at most 500 coordinates using state, geographic macroblock, coast distance, NAIP catalog vintage, and predicted single-asset/mosaic status. Freeze and hash this frame before network execution.
2. Acquire and score the entire authorized frame under the immutable request/byte/disk/cost plan. If authorization cannot cover the complete frame, stop and replan; do not silently shrink it.
3. From the acquired frame, select a 300-tile human-label pilot with deterministic seed `42`, stratifying on observed scene proxy, active-model prediction, terrain relief, state, source year, seam status, and geography. Retain unused acquired frame tiles as source-QC data, list them explicitly, and exclude them from human/model evidence unless a later hashed protocol admits them.
4. Use one independent confirmation seed only for bounded model confirmation, not to redraw the sealed test set.

The final label pilot must cover Connecticut, Rhode Island, southern Massachusetts, urban/suburban/rural/forest/agriculture/coast/inland-water/elevated terrain, available NAIP vintages, seams and interiors, low/medium/high baseline predictions, low/medium/high relief, and geographic macroblocks.

Before looking at labels, freeze a geographic split. For 300 tiles, target approximately 60% train, 20% validation, and 20% sealed test, subject to geographic grouping. No duplicate or adjacent coordinate may cross splits. Hash the acquisition-frame, pilot-selection, and split manifests.

### Human annotation

Reuse the repository’s canonical annotation UI and schema. Do not invent a second rating scale.

- Human-grade every pilot tile used for test or promotion evidence.
- Double-annotate at least 20% across states, scenes, years, and score strata.
- Keep annotators blind to model predictions, source candidate decisions, and split role where practical.
- Record unusable/ambiguous labels and reasons.
- Measure inter-annotator agreement and confidence.
- Do not coerce unusable labels into numeric targets.
- Seal test labels from training and model-selection code.
- Run duplicate/adjacency/leakage audits before training and again before final evaluation.

If human labels are not yet available, finish implementation, acquisition, deterministic batch construction, and all non-human validation, then stop only at the explicit annotation boundary. State the exact batch path/count/hash and command needed to resume. Never fabricate labels or promotion evidence.

### Baseline source-shift evaluation

Run the unchanged active model on the NAIP/3DEP pilot and report:

- MSE, RMSE, MAE;
- Pearson and Spearman correlations;
- calibration error using the repository’s fixed bins;
- prediction/target mean, standard deviation, min, max, quantiles, unique ratio, and spread ratio;
- state slices;
- urban/rural/coast/forest/agriculture/water/elevation slices where supported;
- NAIP year/vintage slices;
- single-asset versus mosaic/seam slices;
- worst supported slice;
- bootstrap confidence intervals with fixed seeds;
- paired residual plots and representative success/failure tiles;
- source no-data and preprocessing QC correlated with residuals.

Treat this as a source-shift diagnosis, not evidence that the old model is acceptable merely because it executes.

### Bounded adaptation ladder

Run the smallest-first bounded ladder, preserving one immutable record per candidate:

1. unchanged active model;
2. output-only affine calibration fitted on train and selected on validation;
3. monotonic/isotonic calibration only if sample support is sufficient and complexity is justified;
4. regression-head fine-tune with frozen image encoder;
5. limited last-block image-encoder fine-tune only if simpler candidates fail and the data volume supports it.

Do not unfreeze the full vision model on a 100–500-tile pilot. Do not search broad hyperparameter grids. Predeclare a small bounded candidate set, fixed seeds, maximum epochs, early stopping, and wall-clock cap. Use validation only for selection. Evaluate the sealed test once for the selected candidate and its unchanged baseline.

If GPU compute is used, retain exact image, command, seed, instance identity, runtime, and cost. Destroy every rented instance immediately after completion or terminal failure. CPU is acceptable for calibration and small regression-head experiments.

### Promotion gates

A source-versioned candidate may be retained only if all applicable canonical gates pass. Reuse `src/scenic_scorer/active_evaluation.py`; do not weaken its current defaults silently. Current canonical defaults include:

```text
min_expanded_corr = 0.80
max_expanded_mse = 2.0
min_expanded_mse_improvement = 0.0
max_control_mse_regression = 0.05
min_control_corr = 0.75
max_worst_slice_mse = 2.5
max_calibration_error = 1.5
min_supported_slice_samples = 5
```

For the NAIP transition, add source-specific gates without removing legacy control gates:

- all data/provenance/integrity checks pass;
- no split leakage or adjacent cross-split pairs;
- human benchmark sample counts are sufficient and disclosed;
- selected candidate improves or matches the unchanged baseline on sealed NAIP test MSE;
- selected candidate does not materially regress rank correlation;
- calibration passes;
- every supported state/year/scene slice passes the declared worst-slice threshold;
- prediction distribution is non-collapsed and stays in valid score bounds;
- confirmation seed does not reverse the decision;
- original canonical control benchmark does not regress; if that control cannot be validly and lawfully evaluated with intact provenance, registry promotion is blocked. An experimental NAIP candidate may still be retained without activation, but the gate is never silently “not applicable”;
- report generation is deterministic;
- route evidence passes only for regions with compatible graphs and real route requests;
- complexity is the minimum needed;
- source/model registry identities are complete;
- active registry remains unchanged until explicit promotion.

If existing canonical defaults are unattainable because the pilot is smaller or the original benchmark measures a different source, do not quietly lower thresholds. Record the failed gate and reject. Propose a larger annotation batch with exact stratification as the next action.

## Full acquisition gate

Proceed beyond the pilot only when:

- the pilot source data contract passes all raster/provenance checks;
- human source-shift evidence exists;
- a source-versioned model/calibration decision is explicit;
- the active registry is either intentionally unchanged with an experimental model, or promotion passed every gate;
- the full execution plan has exact bounds, counts, assets, byte/request/cost upper bounds, and hashes;
- the user-provided caps permit execution.

Full acquisition must be chunked into deterministic geographic units that can resume independently. Every chunk must share the exact same source/preprocessing/model contract. Build a deterministic union/report tool if one does not already exist. It must reject mismatched or overlapping chunks unless duplicates are output- and score-identical.

Do not merge the new Southern New England open-data report with the old Mapbox report. Either:

1. keep the new report as an explicitly experimental source-specific layer, or
2. regenerate the complete intended Northeast coverage with NAIP + 3DEP and the retained source-versioned model before publishing one unified layer.

## Product and route validation

After a valid source-specific report exists:

- expose it through the existing API/viewer only as an explicit experimental run/source;
- show source name, acquisition years, model identity, and experimental status in diagnostics;
- preserve `graph_exists`/route-capability truthfulness;
- keep route submission disabled for regions without a compatible graph;
- inspect representative heatmap locations in the real browser;
- verify tile placement, north/south orientation, coastline alignment, seam behavior, score lookup, and run selection;
- sample pixels/coordinates against manifest entries;
- run real route comparisons only where the graph and score coverage both exist;
- record route distance, overlap, score distribution, and invariant evidence;
- never fabricate route QA for the Southern New England heatmap if the road graph is absent.

## Tests

Write focused tests before implementation where the repository workflow requires it. Tests must defend observable behavior, not source text.

Required focused coverage:

1. XYZ/Web Mercator bounds and affine transform against known coordinates.
2. Exact 512×512 satellite/terrain grid equality.
3. North/south and x/y orientation.
4. Local synthetic GeoTIFF window reads without network.
5. Deterministic NAIP discovery ordering and multi-asset mosaicking.
6. Deterministic 3DEP product selection.
7. Satellite resampling and RGB output contract.
8. DEM float reprojection before Terrain-RGB encoding.
9. Terrain-RGB encode/decode round trips within 0.1 m.
10. DEM no-data handling over land versus ocean.
11. Multipolygon, holes, coast, islands, and land-intersection filtering.
12. Plan SHA binding and plan-drift rejection.
13. Request, byte, disk, and cost cap enforcement before and during execution.
14. Paid/unknown source refusal and explicit Requester Pays acknowledgement.
15. Bounded retry classification.
16. Atomic writes and interruption-safe resume.
17. Existing output hash validation and corrupt-file repair.
18. Secret-free manifests/logs.
19. Source/proprocessing version in cache identity.
20. Negative cache test: same coordinate, different source bytes cannot reuse embeddings.
21. Manifest deterministic ordering and stable hashes.
22. Deterministic chunk union and incompatible-source rejection.
23. Scoring pipeline consumption of source-versioned roots.
24. End-to-end synthetic two-tile plan → acquire → inventory → score/report smoke.
25. Existing focused inventory, scoring, evaluation, API, viewer, and exporter tests remain green.

Do not place live paid network calls in the test suite. After tests pass, run real source steps only at the applicable authorization boundary: discovery under discovery-only caps, one real tile under acquisition caps, then the bounded pilot under its complete immutable plan and caps.

## Verification sequence

At minimum:

1. Always run focused unit/integration tests for each changed subsystem.
2. Always run Ruff format/check only on changed Python files.
3. Always run the existing focused inventory/scoring/export/evaluation/API/viewer tests affected by the cutover.
4. Always run a synthetic end-to-end two-tile smoke with local fixtures.
5. If a valid pinned non-billable catalog is locally available or discovery-only caps are authorized, run the planner-only Southern New England discovery/preflight and inspect counts, bounds, assets, operations, and costs. Otherwise, produce and verify the pre-discovery authorization package and stop network work there.
6. After authorized discovery, verify the immutable acquisition plan and every available artifact SHA-256.
7. Only when explicit acquisition caps authorize it, run one real capped tile acquisition.
8. After step 7, validate that pair with `tile_inventory.py` and `compute_terrain_features()` and visually inspect satellite pixels, decoded DEM, orientation, and overlay alignment.
9. Only when caps authorize the complete frozen acquisition frame, run the capped pilot. Do not run a partial frame as promotion evidence.
10. After the pilot exists, build and freeze the annotation batch.
11. Only after human labels exist, run source-shift evaluation and bounded adaptation.
12. Only after a valid source-specific report exists, inspect its heatmap in the browser.
13. Run route QA only when a compatible graph and score coverage exist.
14. Always verify the active model registry hash before and after; it remains unchanged unless every promotion gate passed.
15. If a durable S3 prefix is configured, upload required ignored data/run artifacts, verify remote length/checksum by independently reading the uploaded objects, and preserve a release manifest. If no durable prefix is configured, record `not_configured` plus local paths/hashes; local-only evidence is acceptable for implementation and authorization/annotation boundaries, but production activation and full acquisition completion are blocked until durable storage is configured and verified.
16. Always confirm no temporary compute instance or acquisition process remains.

At an authorization boundary, clearly label the result “implementation/preflight complete; network acquisition not authorized.” Do not present unit tests or a preflight package as completed data migration. Once authorization is present, exercise every newly authorized real source, processing, scoring, and browser step before claiming that corresponding phase complete.

## Artifacts

Keep large/generated artifacts out of Git. A completed run must retain locally and, where configured, in durable S3:

```text
source_contract.json
preprocessing_contract.json
boundary_geometry metadata + SHA-256
region_spec.json
catalog_snapshot.json or stable catalog identity
execution_plan.json
execution_plan.sha256
tile_manifest.csv
nested_provenance.jsonl if used
acquisition_preflight.json
request_transfer_ledger.jsonl
inventory_report.json
failures.jsonl
pilot_split.csv
leakage_audit.json
annotation_batch.csv
human_benchmark.csv
baseline_source_shift.json
candidate experiment records
selected_candidate decision.json
checkpoint/calibration artifact + SHA-256
report/labels.csv
report/report.json
report/run.json
heatmap provenance
browser screenshots
route QA where supported
release manifest and remote verification
```

Every artifact must bind to exact inputs, commands, code commit, environment/lock identity, seeds, source objects, preprocessing contract, models, and hashes.

## Git and documentation cutover

Commit only source, configuration, tests, and concise canonical documentation. Do not commit imagery, DEMs, generated PNGs, prepared features, checkpoints, reports, caches, or credentials.

Update:

- `data/README.md` with the source-versioned root and provenance contract;
- relevant setup/deployment docs with Requester Pays and cap requirements;
- canonical commands for plan, execute, resume, pilot, score, evaluate, and export;
- any beta artifact manifests only if a new artifact is actually activated and verified.

Remove obsolete Mapbox acquisition code and documentation from active surfaces. Do not remove unrelated viewer basemaps or historical data records merely because they mention Mapbox; distinguish acquisition/training use from map display. Preserve immutable historical evidence, but do not retain executable legacy acquisition fallbacks.

## Stop and failure conditions

Fail closed and preserve truthful evidence when any of these occur:

- repository baseline mismatch;
- dirty conflicting worktree;
- unpinned or drifting catalog/boundary/source object;
- unavailable or ambiguous source license;
- missing AWS Requester Pays acknowledgement;
- unknown/conservative cost upper bound;
- cap would be exceeded;
- missing source coverage;
- source metadata/CRS/datum ambiguity that affects output correctness;
- satellite/terrain target-grid mismatch;
- invalid/corrupt output;
- any DEM no-data on a target pixel whose center is inside the pinned land mask (allowed fraction exactly `0.0`);
- cache identity mismatch;
- split leakage;
- insufficient human evidence;
- failed calibration/control/slice/distribution/route gate;
- registry drift;
- failed S3 verification;
- surviving rented compute.

Do not substitute a smaller unrepresentative pilot, lower a gate, silently choose another product year, reuse stale Mapbox embeddings, or publish a mixed-source report to make the task appear successful.

## Required final report

Return a concise but complete report containing:

1. architecture and files changed;
2. exact source collections/products and why they were selected;
3. exact source, boundary, preprocessing, model, and code hashes;
4. exact plan/acquire/resume/score/evaluate/export commands;
5. planned versus observed requests, bytes, storage, runtime, and cost;
6. pilot coordinates/counts and geographic/scene/year distribution;
7. raster QC and alignment evidence;
8. human annotation counts, agreement, leakage audit, and split hashes;
9. unchanged-model source-shift metrics;
10. bounded candidate table with rejection reasons;
11. calibration, regional/year/scene slices, distribution, and route evidence;
12. retained candidate or explicit rejection;
13. full-acquisition decision and exact reason;
14. heatmap publication status and whether it is experimental;
15. active registry before/after SHA-256 and checkpoint identity;
16. S3 artifact URIs and independent verification results when durable storage is configured; otherwise the explicit `not_configured` status, local paths/hashes, and the resulting activation blocker;
17. confirmation that no Mapbox acquisition occurred and no rented instance remains;
18. unresolved risks;
19. if data-limited, the exact next human annotation batch request.

## Completion definition

This task is complete only to the furthest authorized boundary. The unconditional implementation boundary requires a deterministic, tested, fail-closed NAIP/3DEP path, local-fixture smoke, verified registry state, and no live paid process. Because this prompt supplies no financial authorization, a cold checkout with no adequate pinned non-billable catalog must stop with a verified discovery-authorization package, conservative caps, and exact resume command. Authorized discovery must produce a pinned catalog and immutable acquisition plan/hash; absent acquisition caps, that plan is the next safe boundary. Once acquisition caps are supplied, completion additionally requires one real capped source pair and the complete bounded pilot when its frame is authorized. Human transition evidence is required when labels are available. Every reached phase needs an explicit keep/reject/blocked decision and hash-bound artifacts. Production activation or full-acquisition completion additionally requires every promotion gate plus configured, independently verified durable storage.

A successful download, a passing test suite, or a rendered heatmap alone is not completion. Same downstream file shape is required. Same model semantics must be demonstrated, recalibrated, or explicitly rejected—not assumed.
