# Release and deployment next steps

## Release boundary

The current release is a **source preview**, not an open-source distribution. The
repository remains private and not for distribution; `LICENSE` is an
all-rights-reserved notice. Do not publish a source archive, model weights,
datasets, generated reports, credentials, or deployment images without written
permission from the copyright owner.

The source preview contains the application code, configuration, tests, locked
Python dependencies, and the static New England North viewer. It intentionally
does not contain the ignored runtime graph/report/checkpoint artifacts listed
below. A clean checkout can run the API and viewer shell, but a complete learned
heatmap and route comparison require the artifact set.

## Reproduce the source preview

Use Python 3.11+ and `uv`:

```bash
uv sync --frozen --extra dev
uv run pytest -q
node --test tests/test_new_england_north_viewer.mjs
```

To view the shell locally, start the API and static viewer in separate
terminals:

```bash
uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080
cd apps/new_england_north && python3 -m http.server 3000
```

Then open `http://localhost:3000`. With no ignored artifacts, leave
`SCENIC_ROUTE_PRELOAD` unset (best-effort mode) or set it to `off`; the API will
still start, while artifact-backed heatmap and route data remain unavailable.
Address search also requires a runtime `MAPBOX_ACCESS_TOKEN`.

## Required ignored artifacts for the hosted beta

The hosted beta uses read-only mounts from `data/processed/` and `models/`.
These paths are ignored by Git and must be supplied out of band:

- `data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3`
- `data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/report.json`
- `data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/route.geojson`
- `data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/route_metrics.json`
- `data/processed/regression/model_registry.json`
- `models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt`

The registry's active checkpoint entry is authoritative. The checkpoint must
exist at the registry destination and match the promoted model expected by the
configured New England North region; do not rename or silently substitute it.
Raw imagery, processed feature arrays, classifier weights, and other generated
reports may be needed by training workflows but are not required by the beta
runtime unless a manifest explicitly adds them.

Use the checked-in manifest and bootstrap/checksum tool before starting a host:

```bash
SCENIC_S3_BUCKET=scenicdriver-data \
SCENIC_S3_PREFIX=releases/routeOptimizer/75ee0431/ \
  uv run python scripts/deploy/bootstrap_beta_artifacts.py
SCENIC_S3_BUCKET=scenicdriver-data \
SCENIC_S3_PREFIX=releases/routeOptimizer/75ee0431/ \
  uv run python scripts/deploy/bootstrap_beta_artifacts.py --check-only
```

The bootstrap command downloads only the manifest destinations, and verifies
SHA-256 and byte size for every artifact. Keep AWS credentials and the `.env.beta`
file out of Git; provide credentials through the runtime environment or an
external credentials file.

The large SQLite graph and JSON/GeoJSON objects are stored as `.gz` in S3
to make bootstrap transfer practical; the script decompresses them and
verifies the uncompressed destination digest and size recorded in the manifest.
The plain checksum list is also checked in at `deploy/beta_artifacts.sha256`
for standard `sha256sum` tooling; keep it synchronized with the JSON manifest.

## Remote full-bbox benchmark

The canonical benchmark can run on a high-memory Vast CPU host with durable
JSONL checkpoints. Keep `.secrets/aws.env` outside Git and set the S3
destination before using the state-backed runner:

```bash
chmod 600 .secrets/aws.env
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_S3_PREFIX=outputs/vast/new-england-north-full-bbox-v1

uv run python scripts/remote/vast_route_benchmark.py run full-bbox-v1 \
  --s3-bucket "$SCENIC_S3_BUCKET" \
  --s3-prefix "$SCENIC_S3_PREFIX" \
  --local-secrets-env-file .secrets/aws.env \
  --workers 2 --group-size 64
uv run python scripts/remote/vast_route_benchmark.py status full-bbox-v1
uv run python scripts/remote/vast_route_benchmark.py recover full-bbox-v1
uv run python scripts/remote/vast_route_benchmark.py cleanup full-bbox-v1 \
  --destroy --yes
```

The runner validates the canonical graph/report artifacts before launch,
uploads only validated checkpoint snapshots, resumes by a content fingerprint,
and refuses final S3 publication unless all planned case IDs are unique and
persisted. See [`docs/setup/vast-training.md`](docs/setup/vast-training.md) for
resource sizing, dry-run syntax, and interruption recovery.

## Hosted beta deployment

A hosted deployment is distinct from the source preview. It requires Docker,
the ignored artifacts above, and a Mapbox token supplied only at runtime:

```bash
cp .env.beta.example .env.beta
# Edit .env.beta and set MAPBOX_ACCESS_TOKEN; do not commit .env.beta.
docker compose --env-file .env.beta -f compose.beta.yml up --build
```

`compose.beta.yml` sets `SCENIC_ROUTE_PRELOAD=required`, which makes missing or
invalid default-region graph and report assets fail startup instead of silently
serving an incomplete beta.
The API mounts `data/processed/` and `models/` read-only; model files and
credentials must never be copied into the image layers. Stop the deployment
with:

```bash
docker compose --env-file .env.beta -f compose.beta.yml down
```

## Routing performance delivery

- [x] Deadline propagation and cancellation — integrated at `dc84492f`.
  One absolute `RoutingDeadline` now reaches API validation, service loading and
  scoring, endpoint resolution, graph and planner searches, path evaluation,
  benchmark cases, and the persistent hard-stop worker. In-flight cancellation
  crosses the fork boundary; expired, cancelled, and late IPC results stop and
  reap the disposable worker without returning partial routes. Gate:
  `349 passed` across the focused routing/API/benchmark suite, plus zero Pyright
  errors for the cancellation/supervisor boundary.
- [x] Zero-copy scenic endpoint access — integrated at `6e1fb38c`.
  Scenic endpoint graphs now share base nodes, edges, and adjacency storage,
  own only frozen request-local additions, and serialize same-planner requests.
  A 10,000-extra-node allocation probe fell from 976,560 bytes for the former
  clone to 12,095 bytes (12,863 bytes for the small structural overlay). Gate:
  `280 passed` across planner/oracle/cancellation/objective/service/API coverage,
  Ruff passed, and graph Pyright diagnostics stayed at the same seven
  pre-existing errors.
- [x] Compact reverse CSR and target-bounded bidirectional search — integrated
  at `ca5a7c30`. Reverse traversal now uses numeric CSR rows and forward-edge
  positions instead of Python predecessor lists. Production-sized built-in
  searches run sparse, target-bounded bidirectional Dijkstra before any
  full-source SciPy path, including base-CSR plus request-local endpoint edges.
  Directed, tied, reverse, unreachable, mutation, compactness, sparse-state,
  and prewarmed endpoint behavior are covered. Gate: `284 passed` across the
  integrated routing/service/API suite, Ruff passed, and an independent review
  found no correctness blocker.
- [x] Versioned persisted spatial edge index with corruption recovery —
  integrated at `d3d7872a`; the canonical beta sidecar was published and added
  to the artifact manifest at `76572f20`. The version-2 `SCENEDGE` sidecar is
  graph-hash bound, mmap read-only, projection-stamp invalidated, and rebuilt
  lazily after missing, stale, truncated, or corrupt loads. The 10,792,528-edge
  production sidecar is 508,427,024 bytes; generation took 56.4 seconds after a
  147.2-second graph load and peaked at 7,476,871,168 resident bytes. Gate:
  `178 passed`, focused artifact/benchmark coverage passed, Pyright was clean
  for the index implementation, and independent review found no blocker.
- [x] Re-ran the unchanged 10-second production benchmark from implementation
  revision `bb78048b` as Vast run `full-bbox-v1-r7`: 2 isolated workers,
  group size 64, 23 pairs/2,256 cases, the canonical graph/report, and the
  original q/kappa/highway matrix. Command:
  `uv run python scripts/remote/vast_route_benchmark.py run full-bbox-v1-r7
  --offer-id 44333585 --s3-bucket scenicdriver-data --s3-prefix
  outputs/vast/new-england-north-full-bbox-v1-r7/ --output
  data/processed/routing_benchmarks/production_artifact_benchmark_r7.json
  --workers 2 --group-size 64 --case-timeout-seconds 10`.
  Against revision `75ee0431`/run r5, completed cases rose 3→1,281 and timeouts
  fell 2,253→975; fixed-denominator median fell 10,546.3→5,611.5 ms (46.8%)
  and p95 fell 14,956.5→11,551.4 ms (22.8%). All 1,281 completed cases stayed
  below 10 seconds (median 5.21 ms), but only 56.8% of all cases did, so the
  all-case SLA still failed. Cold isolated-worker preload was 207.6 seconds and
  peak RSS was 16,935,841,792 bytes; the sidecar reported version 2,
  `bvh-spherical-lb`, 10,792,528 edges, 508,427,024 bytes, `state=loaded`, and
  read-only mmap with no invalid reason. The fresh manifest bootstrap log
  contains no fallback build.
- [x] Evaluated CCH versus MLD; neither is justified yet. A targeted
  60-second profile of a timed-out Burlington→Montpelier fastest-route query
  spent 59.78 seconds across 24 legal endpoint-access pair searches and
  55.15 seconds in target-bounded bidirectional traversal (4.60 million heap
  pops), while direct start/end projections took 0.70/0.23 ms. This measures
  multiplicative dispatch in `_large_graph_fastest_route`, not one irreducibly
  slow scalar query. CCH would favor static or preload-customized scalar
  metrics and faster queries, but requires shortcut preprocessing, storage,
  customization, deterministic unpacking, and persisted invalidation. MLD
  better accommodates frequent localized metric updates, but needs
  partition/overlay infrastructure; both hierarchies can accept query-local
  multi-access endpoints only if current tie, direction, and reconstruction
  semantics remain explicit. Neither solves the non-additive duration-capped
  scenic frontier, and frequent-update requirements are not present.
  The exact ranked multi-access query is now implemented and covered by the
  routing oracle/API tests. The fixed production benchmark remains pending a
  reachable Vast host; do not reconsider CCH/MLD until that unchanged matrix
  is rerun and profiling shows one scalar traversal is still dominant.

## Ordered next steps

1. Keep the complete, versioned beta artifact set in the private artifact store;
   the current checked-in manifest uses
   `releases/routeOptimizer/75ee0431/` as its S3 prefix.
2. On the intended host, run the bootstrap download command and then repeat it
   with `--check-only` before building Docker images.
3. Run the strict-preload beta smoke check, including API health, heatmap data,
   route comparison, and address search with a valid Mapbox token.
4. Keep model/token credentials and all generated artifacts in external secret
   and artifact stores; rotate tokens independently of source releases.
5. Treat `.github/workflows/ci.yml` as the clean-runner gate: it installs the
   frozen project plus locked `dev` extra, then runs both required test commands.
6. Before any broader release, obtain explicit distribution permission and
   replace this private preview boundary with an approved license and artifact
   publication policy.
