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

## Service environment

Two runtime environment variables adjust service behavior; both are read at
request time and neither is required:

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
