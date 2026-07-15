# Compact Graph Backend TODO

## Wave 1 gate status (2026-07-14)

No compact-graph TODO items are closed yet. A versioned MessagePack prototype passed parity checks but was rejected for integration because production `RoadGraph.load()` regressed from 9.54s (JSON) to 17.40s (compact), despite a 43.7% artifact-size reduction. The remaining runtime gate requires a packed fixed-width/CSR or mmap-backed layout with measured cold-load and RSS improvement.

No routing-policy TODO item is closed yet. The soft highway-policy draft remains isolated on `wave1-routing` because its production scenic path exceeded the bounded test window and its policy-aware objective is not shared consistently across oracle, frontier, and certification paths.

## Design

- [ ] Define a versioned, checksummed compact graph artifact format.
- [ ] Store float64 node coordinates and edge distances/scores; preserve node and edge IDs in string tables.
- [ ] Store directed traversal in CSR adjacency arrays, preserving insertion and adjacency order.
- [ ] Persist the exact nearest-node KD index using the existing raw lat/lon metric and insertion-order ties.
- [ ] Define immutable base-graph and per-request learned-score overlay boundaries.

## Conversion

- [ ] Add an atomic `road_graph.json` → compact-artifact converter.
- [ ] Validate node/edge counts, IDs, endpoint references, one-way defaults, arc counts, checksums, and geodesic-bound metadata.
- [ ] Record artifact schema, source hash, counts, and conversion metadata in `run.json`.
- [ ] Build and retain a compact artifact for the 4.6M-node regional graph.

## Runtime

- [ ] Make `RoadGraph.load()` detect and open the compact artifact while retaining JSON fallback.
- [ ] Traverse packed CSR arcs without allocating reverse `Edge` objects on every expansion.
- [ ] Materialize public `Node`/`Edge` adapters only where callers require them.
- [ ] Load compact graph bases through mmap and reuse immutable bases safely across API requests.
- [ ] Apply learned tile scores as isolated per-request overlays; do not mutate a shared base graph.
- [ ] Keep custom/mutable cost functions on the existing object-backed fallback path.

## API and Migration

- [ ] Teach region graph resolution to prefer the compact artifact, then fall back to JSON/GeoJSON.
- [ ] Preserve `RouteRequest.graph_geojson` compatibility during the rollout.
- [ ] Keep the generated route GeoJSON, metrics, snapping, route IDs, and tie behavior unchanged.
- [ ] Archive or retire JSON only after compact-artifact parity is accepted.

## Verification

- [ ] Differential-test JSON versus compact loading for attributes, one-way defaults, reverse views, IDs, and adjacency order.
- [ ] Verify exact nearest-node selection and insertion-order ties.
- [ ] Verify learned-score mapping and overlay isolation across reports.
- [ ] Verify constrained scenic routes against exhaustive feasible optima on small graphs.
- [ ] Verify fallback behavior for unsafe manual distances and custom cost functions.
- [ ] Benchmark cold load, RSS, nearest lookup, baseline route, and constrained scenic route on the 1.04M-node corridor and 4.6M-node graph.
