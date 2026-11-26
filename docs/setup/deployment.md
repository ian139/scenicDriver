# Beta Deployment

The hosted beta is distinct from a clean source checkout. It requires the canonical processed graph, learned report, model registry, and active checkpoint. These artifacts are ignored by Git and mounted read-only at runtime.

## Required artifacts

The authoritative release manifest is [`deploy/beta_artifacts.json`](../../deploy/beta_artifacts.json); [`deploy/beta_artifacts.sha256`](../../deploy/beta_artifacts.sha256) is its command-line checksum companion.

The current beta expects:

- `data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3`
- `data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3.edge_projection_index`
- `data/processed/heuristic_runs/prompt_two_candidate_exp02_fresh_test20_20260810/report/report.json`
- `data/processed/heuristic_runs/prompt_two_candidate_exp02_expanded_20260810/report/report.json` (derived visualization; not promotion evidence)
- `data/processed/heuristic_runs/prompt_two_candidate_exp02_fresh_test20_20260810/report/route.geojson`
- `data/processed/heuristic_runs/prompt_two_candidate_exp02_fresh_test20_20260810/report/route_metrics.json`
- `data/processed/regression/model_registry.json`
- `data/processed/regression/checkpoints/0a165e429c8ac050524c7da409dd533d2de8849600c5dd8605aa1a1024d823e9.pt`
- `data/processed/road_graphs/new_england_north_full_bbox_v1/run.json` (graph readiness gate)
- `data/processed/road_graphs/northeast_expanded_scored_tiles_v1/run.json` (graph readiness gate)

Do not rename or substitute the active checkpoint without activating it through the model workflow and updating the release manifest.

## Region configuration

`config/app_regions.json` keeps `new_england_north` as the default region and
routes it through
`data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.compact.json`.
The manifest selects its paired `road_graph.compact.bin`; both destinations are
pinned by `deploy/beta_artifacts.json`. The SQLite graph remains the audit/build
source (and is required by the readiness gate), not the routing representation.
The existing `northeast_expanded` region already uses its compact manifest and
is unchanged.

The complete artifact set includes the projection index, `run.json`, learned
report, and the content-addressed score sidecar pair. For the default region the
sidecars are `road_graph.score.710bbec5c1803a32.z14.fbnull.nlinear-v1.{json,bin}`,
as pinned by the release manifest; do not rename them based on a local path or
mtime. Missing compact artifacts fail admission rather than falling back to
SQLite. A clean local checkout must first run
`uv run python scripts/deploy/bootstrap_beta_artifacts.py` with the AWS
environment shown in [Bootstrap and validate](#bootstrap-and-validate), then run
the check-only bootstrap and routing checker before starting the API.

## Graph readiness gate

`scripts/routing/build_graph_from_osm.py` publishes `run.json` beside every graph
it builds (`<run_dir>/run.json`) as the run's readiness gate. The API validates
it before routing anything: the gate records the graph identity
(`output_graph`, `graph_sha256`, `graph_size_bytes`) and, when the compact
manifest is the runtime graph, `runtime_graph` and `compact_manifest_sha256`.
A missing, stale, or unresolvable gate means the region refuses to route and its
assets are never preloaded — a deployment without `run.json` serves zero
routeable regions.

Gate paths must be project-root-relative: the runtime resolves them against its
own project root, so records carrying absolute build-machine paths can never
validate on a deployment host. The builder records relative paths automatically
for artifacts published under the project root; keep builds' `--output-root`
inside the checkout so the published gate is portable. Each deployed graph
directory therefore ships its own `run.json`, and the release manifest pins its
digest like every other artifact.

## Score sidecars

A compact runtime graph scores a learned report through a precomputed
scenic-cost sidecar beside the graph:
`road_graph.score.<identity>.z<zoom>.fb<fallback>.n<normalization>.{json,bin}`.
`<identity>` is the first 16 hex digits of a SHA-256 over the report's content
digest, the compact graph's source SQLite digest, the zoom, the fallback score,
and the normalization version. It contains no path or file-stat data, so
byte-identical artifacts resolve the same sidecar on any host, even after a
copy or rename. The sidecar manifest's `tile_scores_sha256` and
`source.sha256` must match the live report and graph content.

The release manifest pins the admitted sidecars at their content-derived
destinations (`*.compact.score.manifest` / `*.compact.score.payload`). At
startup a missing sidecar is generated only when the graph directory is
writable. On the read-only beta mount, a missing sidecar fails admission for
that region with a route asset error, and nothing is recomputed or written.
Generate sidecars offline (preload the region once against a writable copy of
the graph directory) and publish them with the release manifest.

A graph or report replaced while it is being read is never cached. The runtime
compares the file identity before and after each load, retries once, and then
fails with a route asset error.

## Bootstrap and validate

Provide AWS credentials through the runtime environment or an external credentials file. Never store them in the repository or image.

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_S3_PREFIX=releases/routeOptimizer/prompt-two-exp02-20260810/

uv run python scripts/deploy/bootstrap_beta_artifacts.py
uv run python scripts/deploy/bootstrap_beta_artifacts.py --check-only
uv run python scripts/routing/check_beta_artifacts.py --project-root .
```

The bootstrap downloads only manifest destinations (including both `run.json` readiness gates), decompresses packaged objects when required, and verifies their uncompressed sizes and SHA-256 digests. The routing checker validates the graph schema, configured bounding box, coverage probes, and report/model linkage, and it admits each graph's `run.json` readiness gate: presence plus the recorded identity values, so a deployment the API would treat as not-ready fails the check before activation.


## Build the regional graph

Source PBFs and OSMnx intermediates belong under `cache/`; generated graph files and metadata belong under `data/processed/`. The canonical build uses dated, checksum-verified extracts:

```bash
uv run --with 'osmnx==2.1.0' python scripts/routing/build_graph_from_osm.py \
  --min-lat 42.488301979602255 --min-lon -73.5205078125 \
  --max-lat 47.50235895196859 --max-lon -66.796875 \
  --network drive --run-name new_england_north_full_bbox_v1 \
  --graph-format sqlite3 \
  --cache-folder cache/osmnx/new_england_north_full_bbox_v1 \
  --require-source-checksums \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/new-york-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/vermont-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/new-hampshire-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/maine-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/massachusetts-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/quebec-260716.osm.pbf \
  --source-pbf cache/osm-pbf/new_england_north_full_bbox_v1/new-brunswick-260716.osm.pbf \
  --coverage-probe rutland_usps 43.60784414 -72.98226538 \
  --coverage-probe lisbon_police 44.02516775 -70.10003245 \
  --coverage-probe burlington 44.475884 -73.214003 \
  --coverage-probe bangor 44.801616 -68.771305
```

Run the artifact checker before activation. For rollback, restore the previous graph, region configuration, and matching manifest as one versioned set; validate it before restarting the API.

## Hosted Compose release

[`compose.beta.yml`](../../compose.beta.yml) is the canonical hosted topology:
`api` runs `uvicorn scenicdriver.app_api.main:app` on container port 8080 using
[`Dockerfile.beta-api`](../../Dockerfile.beta-api); `frontend` serves the checked
viewer from `/usr/share/nginx/html` on container port 80 using
[`apps/new_england_north/Dockerfile`](../../apps/new_england_north/Dockerfile).
Nginx forwards `/api/` to `api:8080` and keeps the existing route/search limits.
The frontend waits for API readiness before starting.

The API image installs the full frozen `uv.lock` environment, including torch.
It is intentionally heavyweight until W6.5 supplies a separate lean API
dependency path; do not manually prune dependencies from this image. The native
compact-search library is compiled in a matching target-platform build stage
and copied into the runtime image, which needs no compiler.

### Artifact and credential boundary

Compose mounts these release directories/files read-only at the same
project-relative paths below `/app`:

- `data/processed/road_graphs/new_england_north_full_bbox_v1/`
- `data/processed/road_graphs/northeast_expanded_scored_tiles_v1/`
- `data/processed/heuristic_runs/prompt_two_candidate_exp02_fresh_test20_20260810/report/`
- `data/processed/heuristic_runs/prompt_two_candidate_exp02_expanded_20260810/report/`
- `data/processed/regression/model_registry.json`
- `data/processed/regression/checkpoints/0a165e429c8ac050524c7da409dd533d2de8849600c5dd8605aa1a1024d823e9.pt`

Graph directories include every manifest-pinned compact payload, projection
index, content-derived score sidecar, and `run.json`; report directories include
the report and route files. Bootstrap and validate the complete manifest, not
just the abbreviated required-artifact list above. Missing mount sources fail
startup rather than creating empty directories. `SCENIC_ARTIFACT_ROOT` may point
to a versioned external directory with the same contents as `data/processed/`;
the default is `./data/processed`. Never update mounted artifacts in place:
stage and validate a new release directory before switching it.

Contributor state is separate: the named `contrib-data` volume is writable and
survives container replacement. It is not a model/report artifact mount; back it
up separately and do not use `docker compose down --volumes` during release or
rollback.

Compose passes `MAPBOX_ACCESS_TOKEN`, `SCENIC_ALLOWED_ORIGINS`,
`CONTRIB_ADMIN_TOKEN`, and `SCENIC_ROUTE_DEADLINE_SECONDS` from the runtime
environment, not image layers. Set the Mapbox token on the host (never in the
viewer), and set the CORS allowlist to the actual HTTPS beta origin. Keep the
admin token unset unless contributor administration is intended. The deployment
forces `SCENIC_ROUTE_PRELOAD=required` so a missing default graph/report prevents
API startup.

### Stamp, build, and start

Run from the repository root after bootstrapping and validating artifacts:

```bash
# Set runtime secrets via the deployment host's secret manager/environment.
: "${MAPBOX_ACCESS_TOKEN:?Set the server-side Mapbox access token}"
export SCENIC_ALLOWED_ORIGINS=https://beta.example.com
export SCENIC_RELEASE_TAG=20261006-01  # unique per source/artifact release
# Optional: export SCENIC_IMAGE_PREFIX=registry.example.com/scenicdriver
# Optional: export SCENIC_ARTIFACT_ROOT=/srv/scenic/releases/20261006-01/processed

uv run python scripts/deploy/stamp_frontend_version.py
uv run python scripts/deploy/stamp_frontend_version.py --check
docker compose -f compose.beta.yml config --quiet
docker compose -f compose.beta.yml build
docker compose -f compose.beta.yml up -d --no-build --wait --wait-timeout 900
docker compose -f compose.beta.yml ps
curl --fail http://127.0.0.1:80/index.html
curl --fail http://127.0.0.1:80/api/v1/healthz
```

The stamp is SHA-256 of `viewer.js` followed by `styles.css`; both asset queries
receive that digest. Commit the stamped `index.html` with the asset change.
Verification fails after either asset changes until it is re-stamped, and the
frontend Docker build performs the same check before copying any static files.
Nginx never caches `index.html` and caches stamped JS/CSS for a year; no manual
version bump is needed.

The API healthcheck requires `ok`, a mounted registry, and both
`new_england_north` and `northeast_expanded` in
`route_preload.loaded_regions`. HTTP 200 alone is only liveness, not release
readiness. Investigate `docker compose -f compose.beta.yml logs api` on failure;
do not disable preload or the check to admit incomplete artifacts.

### TLS and rollback

Compose serves plain HTTP, bound to host loopback ports `:80` (frontend) and
`:8080` (API). An outer HTTPS reverse proxy on the deployment host must terminate
TLS and forward the public beta origin to `127.0.0.1:80`, including `/api/`.
Do not expose the API port directly or publish this HTTP topology to the
Internet. If the outer proxy is containerized, explicitly join its network or
adapt the loopback binding; its container's loopback is not the host's. Public
HTTPS/CORS policy belongs to that outer boundary; nginx's upstream connection
is still HTTP.

Retain both image tags and the corresponding source/configuration, manifest,
and artifact directory for each release; never overwrite an existing release
tag. To roll back, restore the previous release's source/configuration and
validated artifact set, then select its retained images without rebuilding:

```bash
export SCENIC_RELEASE_TAG=20261005-01
export SCENIC_ARTIFACT_ROOT=/srv/scenic/releases/20261005-01/processed
docker compose -f compose.beta.yml config --quiet
docker compose -f compose.beta.yml up -d --no-build --pull never --wait --wait-timeout 900
curl --fail http://127.0.0.1:80/api/v1/healthz
```

Repeat the readiness inspection above before returning public traffic. Keep
runtime secrets and the contributor volume independent of image rollback.


## Service environment

Runtime variables are read at request time; credentials stay outside source
and image layers:

- `MAPBOX_ACCESS_TOKEN` enables server-side address search. Do not send it to
  the browser.
- `SCENIC_ALLOWED_ORIGINS` is the comma-separated CORS origin allowlist; set it
  to the public HTTPS origin(s).
- `SCENIC_ROUTE_DEADLINE_SECONDS` overrides the per-region `route_deadline_seconds`
  from `config/app_regions.json` for every region. It must parse as a finite,
  non-negative number; an unset or empty value falls back to the region's
  configured deadline, or to the planner default when the region sets none.
- `CONTRIB_ADMIN_TOKEN` enables the contributor QA admin review endpoint
  `POST /v1/admin/contrib/review/run`. Callers must send the token as the
  `X-Admin-Token` header. When the variable is unset the endpoint returns `503`
  instead of running promotion; a missing header returns `401` and a mismatched
  header returns `403`. Treat the token as a credential: set it only in the
  runtime environment, never in the repository or image.

## Release boundary

A clean checkout contains application source, configuration, tests, locked dependencies, and the static viewer. It intentionally excludes model weights, source imagery, feature arrays, generated reports, road graphs, and credentials. A working hosted beta is therefore not a claim that the source preview is self-contained.
