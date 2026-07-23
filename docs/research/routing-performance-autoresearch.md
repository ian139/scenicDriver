# Routing Performance Autoresearch Log

## Decision

Retain the bounded scenic scalar traversal implemented in `86bbfde3` and merged to `main` by `ba5cf73b`. On the fixed 2,256-case production workload it reduced all-case median latency by 80.80%, reduced p95 by 63.55%, and raised the under-10-second rate from 68.22% to 99.91% without changing any route completed by both revisions.

Do not retain the later incumbent-bound, precomputed-sum, or sparse-transpose experiments. Do not begin frontier pruning, CCH, or MLD work without new profiler evidence that the retained implementation still has a material scalar-traversal bottleneck.

## Scope and revisions

| Role | Revision | Description |
|---|---|---|
| Product baseline | `522d1baa` | Endpoint-aware routing and benchmark accounting before this autoresearch loop |
| Immutable experiment baseline | `0d8c4949fc41` | Product baseline plus the fixed autoresearch harness |
| Retained candidate | `86bbfde3` | Bounded scenic scalar Dijkstra traversal |
| Main integration | `ba5cf73b` | No-conflict merge of the harness and retained candidate |

The retained `src/route_planner/planner.py` SHA-256 at integration is `e4e514c0230f9b6b319f37746bf3906b38f3fbdbb691f895bd6bd51ade0c1a8c`.

## Hypothesis and implementation

Profile evidence on the retained Vast CPU host showed that repeated scalar CSR traversal dominated endpoint-aware scenic planning. For `medium_burlington_montpelier|q=0.9|kappa=1.8|avoid=false`, `_large_graph_scenic_search` consumed 4.246 seconds of 5.129 seconds and `_large_graph_multi_access_path` ran five times. The repeated compiled scalar traversal, not endpoint projection or frontier expansion, was the measured bottleneck.

Hypothesis: the already-computed fastest route is a feasible path for every scenic Lagrangian multiplier. Its scalar cost is therefore a valid upper bound for that multiplier's shortest-path traversal. Passing that bound to SciPy `dijkstra(limit=...)` should avoid exploring nodes that cannot improve the scalar optimum while preserving route selection.

The retained implementation:

- computes the fastest route's scalar cost for each multiplier;
- uses SciPy's bounded Dijkstra for scenic scalar traversals;
- keeps fastest-path traversal unbounded when no incumbent exists;
- adds a `1e-9` relative tie margin and one `nextafter` step;
- falls back to the previous unbounded traversal if the aggregate scalar bound is non-finite; and
- leaves endpoint rank, direction, reconstruction, highway filtering, deadline, and cancellation behavior unchanged.

## Fixed workload

| Setting | Value |
|---|---|
| Corpus | `scripts/routing/production_benchmark_pairs.json` |
| Corpus SHA-256 | `b92cfcbe67b9ce864752c62364b0c550ca5567eabe39af0018b98f2a60c59d6a` |
| Pair count | 23 |
| Planned denominator | 2,256 cases |
| `q` values | `0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0` |
| `kappa` values | `1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 3.0` |
| Highway settings | `avoid_highways=false` and `true` |
| Extra activation cases | `checked_in_default_reproduction` and `full_bbox_rutland_lisbon` at `q=0.8`, `kappa=1.8`, `avoid_highways=false` |
| Case timeout | 10 seconds |
| Workers / group size | 2 / 64 |
| Seed | none; routing benchmark uses no RNG |
| CPU class | AMD EPYC 9655, 48 effective cores allocated |
| Cache policy | fresh process, application caches cleared, then planner prewarmed |
| Graph | `data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3` |
| Graph SHA-256 | `48c5b052deaf23a5e0fee29262fd384caec859a6317f6f7eb7a90eef07ae09b8` |
| Scenic report | `data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/report.json` |
| Report SHA-256 | `249fc55085b6fe17396fa022ff550100e7accbaca8902813cc6f53374dc88b39` |
| Projection sidecar SHA-256 | `a2e271c78fcb4791067bff804161f72df575283a672d3e448eed98d602fdbd08` |

The denominator is $23 \times 7 \times 7 \times 2 + 2 = 2{,}256$: the full pair/settings matrix plus the two explicit activation cases.

Both measurements used the same host, graph, report, projection sidecar, corpus, denominator, timeout, worker configuration, and application cache boundary.

## Reproduction command

Use Vast for this graph-scale benchmark; do not run the full workload locally.

```bash
uv run --offline --frozen python scripts/routing/production_benchmark.py \
  --corpus scripts/routing/production_benchmark_pairs.json \
  --graph data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3 \
  --report data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/report.json \
  --output data/processed/routing_benchmarks/production_artifact_benchmark.json \
  --case-timeout-seconds 10.0 \
  --workers 2 \
  --group-size 64
```

The benchmark validates the fixed matrix, unique case IDs, checkpoint fingerprint consistency, all-case persistence, route invariants, cache boundaries, and strict-service/direct-planner parity before publishing a final artifact.

## Production benchmark result

| Metric | Baseline `0d8c4949fc41` | Integrated candidate `86bbfde3` | Change |
|---|---:|---:|---:|
| Planned/persisted cases | 2,256 / 2,256 | 2,256 / 2,256 | unchanged |
| Completed routes | 1,568 | 2,058 | +490 |
| Timeouts | 492 | 2 | -490 |
| No-route results | 196 | 196 | unchanged |
| All-case median | 6,449.169 ms | 1,238.558 ms | -80.80% |
| All-case p95 | 12,274.275 ms | 4,473.816 ms | -63.55% |
| Under-10-second count | 1,539 | 2,254 | +715 |
| Under-10-second rate | 68.2181% | 99.9113% | +31.6932 percentage points |
| Completed-route p95 | not used for acceptance | 4,560.098 ms | protected by fixed-denominator metrics |
| Process peak RSS | same host boundary | 16,925,442,048 bytes | no memory gate regression observed |

The first candidate measurement produced 2,059 completions and one timeout. After an independent review added the non-finite-bound fallback, the exact integrated-checkout rerun above produced 2,058 completions and two timeouts. The median and p95 remained far beyond the 5% acceptance threshold, and all protected checks passed.

## Correctness and behavior checks

| Check | Result |
|---|---|
| All candidate rows persisted | pass, 2,256/2,256 |
| Candidate `q0_fastest` | pass, 2,058/2,058 completed routes |
| Baseline presence | pass, 2,058/2,058 |
| Duration cap | pass, 2,058/2,058 |
| Prohibited highways | pass, 2,058/2,058 |
| Objective recomputation | pass, 2,058/2,058 |
| Route metric recomputation | pass, 2,058/2,058 |
| Certification consistency | pass, 2,058/2,058 |
| Diagnostics consistency | pass, 2,058/2,058 |
| UI reproduction parity | pass |
| Python focused suite | 235 passed after merge |
| Python full suite | 523 passed after merge |
| New England North viewer suite | 14 passed after merge |
| Highway-toggle API smoke | HTTP 200 for both settings; diagnostics reported the requested value |

For the 1,568 cases completed by both baseline and candidate:

- all 1,568 complete baseline and scenic route snapshots were identical;
- all 1,568 non-timing evaluation fields were identical;
- compared data included canonical edge IDs, traversal IDs, segment identities, geometry digest, duration, distance, scenic and objective values, settings, execution mode, and invariants;
- the candidate added 490 completions and lost no baseline completion.

The artifact schema does not persist endpoint access-rank integers. Exact selected segment identity plus requested/snapped endpoint checks were compared instead. This limitation should remain explicit if access-rank persistence is added later.

## Rejected follow-up experiments

The local four-case gate used a retained baseline median of 4,042.518 ms. It is a coarse screening tool, not production acceptance evidence.

| Experiment | First result | Confirmation/result | Decision |
|---|---:|---:|---|
| Cheapest discovered feasible path as multiplier bound | median 3,617.168 ms; p95 10,025.139 ms | median 4,787.685 ms; p95 11,105.326 ms | Reject: gain reversed; median +18.43% and p95 +11.00% vs baseline |
| Precompute fastest-path scenic/duration sums | median 3,979.655 ms; p95 10,700.782 ms | no confirmation warranted | Reject: median improved only 1.55%; p95 regressed 6.96% |
| Reuse sparse transpose within traversal | median 3,765.867 ms; p95 10,475.260 ms | median 4,158.136 ms; p95 10,577.233 ms | Reject: initial gain reversed; p95 regressed in both runs |

No-change repeats ranged from 3,709.062 to 4,931.237 ms median, with timeout-dominated p95 values from 10,437.908 to 10,936.657 ms. This exceeded 20% median variation. Any future candidate near the 5% threshold requires confirmation and same-host fixed-denominator Vast validation.

All rejected source changes were automatically reset to `86bbfde3`; the final rollback run preserved the four-case denominator, three completions, one timeout, `q0_fastest`, and zero correctness failures.

## Evidence durability

### Durable artifacts

Baseline:

```text
s3://scenicdriver-data/outputs/vast/new-england-north-full-bbox-v1-autoresearch-baseline-0d8c4949/full-bbox-v1-autoresearch-baseline-0d8c4949.json
```

- verified present after merge;
- size: 5,408,286,595 bytes;
- S3 ETag: `b1196fa9c03486016ef2343c6b7ff1f3-645`;
- server-side encryption: AES-256.

Integrated candidate:

```text
s3://scenicdriver-data/outputs/vast/new-england-north-full-bbox-v1-autoresearch-candidate-limit-integrated-v1/full-bbox-v1-autoresearch-candidate-limit-integrated-v1.json
```

- verified present after merge;
- size: 5,975,072,526 bytes;
- S3 ETag: `2d947e036f36fc2ced58ce764a308f96-713`;
- checkpoint fingerprint: `e896854d8276973813ba65708dcddee50203fab7f531a9366dfaa082689469cd`;
- server-side encryption: AES-256.

### Machine-local evidence

These records are saved but intentionally Git-ignored and are not portable repository history:

- OMP run logs under `~/.omp/autoresearch/.../runs/`;
- local profile JSON files under `data/processed/routing_benchmarks/autoresearch_profile_runs/`;
- other local benchmark checkpoints and outputs under `data/processed/routing_benchmarks/`; and
- raw files on a Vast instance, which are not durable after teardown unless uploaded.

The scripts, fixed corpus, validation logic, and this summary are tracked in Git. The large canonical results are durable in S3, not Git.

## Detour-policy boundary

The retained 2,256-case Vast evidence predates the route-control correction
that defines the max-detour factor against the unrestricted fastest route,
even when `avoid_highways=true`. That correction intentionally changes the
1,128 avoid-highways cases: their displayed baseline remains unrestricted and
their scenic route must satisfy the unrestricted-baseline duration cap. The
published performance and exact-output comparisons remain valid historical
evidence for the scalar-bound implementation under the prior policy, not
validation of the corrected avoid-highways workload. Focused planner,
benchmark-validator, and API checks cover the new contract; performance
retention requires a new fixed-denominator Vast run before making current
2,256-case latency claims.


## Next research gate

The scalar incumbent bound removed the measured dominant traversal at production scale. Do not infer that frontier pruning, CCH, or MLD is now beneficial. Before another architectural optimization:

1. profile the retained revision on the fixed workload;
2. identify a new dominant operation;
3. define one bounded hypothesis and a fixed acceptance threshold;
4. screen locally only for large effects; and
5. retain only after the same-host 2,256-case benchmark and exact route-output comparison pass.
