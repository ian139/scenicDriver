# Roadmap

This is the canonical forward plan. Completed experiment details belong in the research logs; operational procedures belong in setup guides.

## Product

- Add optional signed-in contributor annotation mode with credits and a shadow QA queue.
- Expand the region selector and validate additional configured regions end to end.
- Keep the responsive New England North web app as the canonical client.

## Model quality

The promoted v6 learned regressor remains the baseline. Its metrics and provenance are recorded in the [ML research log](research/ml-research-log.md).

- Expand weakly supervised regional training data without making new human labels the immediate bottleneck.
- Compare the current RESISC-based feature recipe with a geospatial foundation embedding on the same labels and split.
- Evaluate every candidate against the existing human benchmark; weak-label validation alone is not a promotion gate.
- Promote only after benchmark, report-distribution, and route-level QA pass.
- Resume broader human labeling when candidates fail to improve the human benchmark or regional calibration remains weak.

## Data and reporting

- Add a repeatable region pipeline from tile acquisition through labels, reports, and deployment metadata.
- Extend visual QA and learned reports to additional representative regions.
- Keep data provenance, hashes, and generated outputs under the canonical paths in [`data/README.md`](../data/README.md).
- Evaluate the [nationwide streaming score-map proposal](research/nationwide-streaming-score-map.md) through parity, human source-shift, and metered cost pilots before any CONUS execution.

## Routing

The bounded scenic scalar traversal, compact-search heap improvements, and bounded exact warm-response cache are retained. Their production evidence and policy boundaries are recorded in the [routing performance log](research/routing-performance-autoresearch.md).

- Revalidate the fixed production workload after the corrected detour and best-effort highway policies.
- Reduce exact Northeast Expanded cache-miss latency; the sub-50 ms result applies only to repeated identical warm requests.
- Keep future routing optimization evidence-driven: profile first, preserve route identity and correctness invariants, then compare on a fixed workload.
- Consider exact precomputed indexes only with an approved graph/deployment artifact change and production preprocessing plan.

## Release

- Keep the complete beta artifact set versioned in the private artifact store.
- Bootstrap and checksum-verify artifacts before building or starting the beta.
- Keep models, generated data, credentials, and tokens outside Git and image layers.
- Treat GitHub Actions as the clean-checkout source gate and the [deployment runbook](setup/deployment.md) as the hosted-beta gate.
