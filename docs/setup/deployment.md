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

Do not rename or substitute the active checkpoint without activating it through the model workflow and updating the release manifest.

## Bootstrap and validate

Provide AWS credentials through the runtime environment or an external credentials file. Never store them in the repository or image.

```bash
export SCENIC_S3_BUCKET=scenicdriver-data
export SCENIC_S3_PREFIX=releases/routeOptimizer/prompt-two-exp02-20260810/

uv run python scripts/deploy/bootstrap_beta_artifacts.py
uv run python scripts/deploy/bootstrap_beta_artifacts.py --check-only
uv run python scripts/routing/check_beta_artifacts.py --project-root .
```

The bootstrap downloads only manifest destinations, decompresses packaged objects when required, and verifies their uncompressed sizes and SHA-256 digests. The routing checker validates the graph schema, configured bounding box, coverage probes, and report/model linkage.


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

## Release boundary

A clean checkout contains application source, configuration, tests, locked dependencies, and the static viewer. It intentionally excludes model weights, source imagery, feature arrays, generated reports, road graphs, and credentials. A working hosted beta is therefore not a claim that the source preview is self-contained.
