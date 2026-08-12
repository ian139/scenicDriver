# Routing Performance Autoresearch Log

## Decision

Retain the bounded scenic scalar traversal implemented in `86bbfde3` and merged to `main` by `ba5cf73b`. On the fixed 2,256-case production workload it reduced all-case median latency by 80.80%, reduced p95 by 63.55%, and raised the under-10-second rate from 68.22% to 99.91% without changing any route completed by both revisions.

Do not retain the later incumbent-bound, precomputed-sum, or sparse-transpose experiments. Do not begin frontier pruning, CCH, or MLD work without new profiler evidence that the retained implementation still has a material scalar-traversal bottleneck.

## Northeast Expanded exact warm-request study

### Decision

Retain the native compact-search heap improvements and the bounded exact
`plan_routes` response cache from the 2026-08-12 Burlington→Pittsburgh study.
For five repeated identical warm requests, the final median was 45.740 ms
versus the fresh 89.344-second baseline: a 99.9488% reduction and 1,953.3×
speedup with the complete semantic response unchanged.

This is a repeated-request cache-hit result, not a general route-search latency
claim. The last measured cache-miss candidate median was 80.937 seconds, a
9.41% improvement from baseline. Requests with new parameters, changed graph
or score artifacts, cache eviction, or a fresh process still execute the exact
search. Further cache-miss improvement remains the routing performance gate.

### Fixed request and artifacts

| Setting | Value |
|---|---|
| Graph | `data/processed/road_graphs/northeast_expanded_scored_tiles_v1/road_graph.compact.json` |
| Graph SHA-256 | `26c5a61392a83056729848f3f12cf898e2a1f5a2e5cb71ecd909f588ff8b4195` |
| Scenic report | `data/processed/heuristic_runs/prompt_two_candidate_exp02_expanded_20260810/report/report.json` |
| Report SHA-256 | `eb656ca4abf5e1cc1b9b53849ddf3f94a3e3d323b81cf0609a89a999dd0b51ff` |
| Start | `(44.475884, -73.214003)` |
| End | `(40.44062, -79.99589)` |
| Scenic weight | `0.8` |
| Max detour factor | `1.8` |
| Avoid highways | `false` |
| Include baseline | `true` |
| Max snap distance | `1.0 km` |
| Per-request deadline | 120 seconds |
| Host | Apple M2 Max, 12 cores, 32 GiB RAM |
| OS / compiler | macOS 26.5.1 / Apple clang 17.0.0 |

The deterministic harness preloads the graph and report once, performs one
untimed warm-up, then measures complete `plan_routes` calls. Preload remains
separate and visible; the final confirmation preload took 19.635 seconds.
Every timed response is compared with the baseline semantic oracle after
removing only elapsed-time and cache-hit diagnostics.

### Result

| Metric | Fresh baseline | Final |
|---|---:|---:|
| Timed warm calls | 3 | 5 |
| Wall times | `89.1021, 89.8382, 89.3437 s` | `0.04574, 0.08158, 0.02737, 0.02411, 0.16294 s` |
| Median | 89.3437 s | 0.045740 s |
| Minimum / maximum | 89.1021 / 89.8382 s | 0.024115 / 0.162944 s |
| Median CPU time | 88.0178 s | 0.044303 s |
| Peak RSS | 11.190 GiB | 12.178 GiB |
| Semantic fingerprint | `226c52550ba6c0a91ad6ca54969422e2a946a1b3cb7189b7715542388896ef28` | unchanged |

All five final measurements were response-cache hits. The cache is
process-local, bounded to eight entries, and keyed by the complete normalized
request, graph and report file signatures, and frontier time-limit policy.
Cached values are immutable, process-generated pickle bytes; no external or
persistent pickle payload is loaded. Each hit deserializes a fresh mutable
response, so one caller cannot mutate later responses. Cache publication
checks cancellation and deadlines after serialization and again under the
cache lock.

### Experiment ledger

| Experiment | Median | Decision |
|---|---:|---|
| Fresh baseline | 89.344 s | Baseline |
| Floyd bottom-up binary heap pop | 89.590 s | Reject: 0.28% regression |
| Four-ary native heap | 91.252 s | Reject: 2.14% regression |
| `-mcpu=native` compilation | 89.885 s | Reject: 0.61% regression |
| Branch-reduced strict heap comparator | 83.794 s | Retain: 6.21% improvement |
| Naturally aligned 32-byte heap record | 80.937 s | Retain: 3.41% over prior candidate, 9.41% from baseline |
| Bounded exact response cache using `deepcopy` | 0.221 s | Retain, then supersede clone mechanism |
| Post-publication deadline check | 0.221 s | Retain correctness fix |
| Tailored recursive response clone | 0.280 s | Reject: 27.0% cache-hit regression |
| Compact JSON response payload | 0.286 s | Reject: 29.5% cache-hit regression |
| Process-private pickle response payload | 0.044 s | Retain: 80.0% over `deepcopy` cache hits |
| Direct tuple request key | 0.045 s | Reject: 1.92% slower and within noise |

The configured 12-experiment cap stopped the loop. Rejected source changes
were reset completely. Retained revisions are `5f9acd14`, `0fa2d2b7`,
`42eff630`, `2906b613`, `3f384070`, and `14136c95`.

### Correctness and reproduction

The oracle covers ordered route edge and traversal IDs, selected endpoint
access, route costs and metrics, certified bounds and exactness, score mapping,
tie ordering, and the complete non-timing response. Focused cache, deadline,
and cancellation verification passed 11 tests. The affected routing, compact
runtime, API, and artifact run passed 436 tests; one timing-sensitive frontier
stress case failed in the shared run and passed in isolation. Native C compiled
with `-O3 -shared -fPIC -Wall -Wextra -Werror`. Independent exact-diff review
passed after verifying cancellation-safe cache publication.

With the ignored production artifacts mounted, reproduce the fixed warm
request with:

```bash
uv run python scripts/routing/northeast_expanded_autoresearch_benchmark.py \
  --timed-runs 5
```

The harness writes its complete local oracle and latest result under
`data/processed/routing_benchmarks/northeast_expanded_autoresearch/`. Those
outputs and OMP experiment ledgers are intentionally Git-ignored; the harness,
artifact hashes, protocol, distributions, and decisions above are the durable
repository evidence.

### Remaining gate

The remaining bottleneck is an uncached exact compact Lagrangian search; the
profile attributed about 89.6% of samples to native compact search and 57.5%
of total samples to heap pop. Exact landmarks, partitions, or customizable
hierarchies may improve misses, but require a separately approved graph or
deployment artifact-format change and production preprocessing. Do not reduce
the candidate set, multiplier set, endpoint access, certified search space, or
deterministic ordering to improve the miss result.


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

### Best-effort highway-avoidance correction

The route-control correction now treats highway avoidance as a preference:
the planner first attempts a strict highway-free route under the
unrestricted-fastest cap, then retries under the same cap and request deadline
with a highway-exposure penalty when strict avoidance is infeasible. The
unrestricted baseline remains unchanged. The benchmark validator independently
recomputes the fallback penalty and verifies the reported fallback mode,
trigger, highway count, preference, cap, and detour reference.

Local verification passed 533 Python tests and 14 viewer tests. A
live Augusta→Lewiston request that previously returned HTTP 422 now returns
HTTP 200 in `best_effort_fallback` mode: the scenic route uses 42 highway
segments versus 719 on the unrestricted baseline, stays within the
unrestricted-fastest cap (`68.859 <= 99.403` minutes), and passes every
production-benchmark response invariant after API-shape adaptation. Turning
the preference off returns the 719-highway-segment baseline route, confirming
that the control materially changes selection. This is correctness evidence,
not fixed-denominator performance evidence; the 2,256-case revalidation gate
remains open.


## Next research gate

The scalar incumbent bound removed the measured dominant traversal at production scale. Do not infer that frontier pruning, CCH, or MLD is now beneficial. Before another architectural optimization:

1. profile the retained revision on the fixed workload;
2. identify a new dominant operation;
3. define one bounded hypothesis and a fixed acceptance threshold;
4. screen locally only for large effects; and
5. retain only after the same-host 2,256-case benchmark and exact route-output comparison pass.
