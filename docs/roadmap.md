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

The bounded scenic scalar traversal is the retained implementation. Its production benchmark and policy boundary are recorded in the [routing performance log](research/routing-performance-autoresearch.md).

- Revalidate the fixed production workload after the corrected detour and best-effort highway policies.
- Keep future routing optimization evidence-driven: profile first, preserve route identity and correctness invariants, then compare on a fixed workload.
- Revisit a compact, mmap-backed graph artifact only if it improves cold load time and resident memory on the production graph while preserving IDs, adjacency order, nearest-node ties, overlays, and route output.

## Release

- Keep the complete beta artifact set versioned in the private artifact store.
- Bootstrap and checksum-verify artifacts before building or starting the beta.
- Keep models, generated data, credentials, and tokens outside Git and image layers.
- Treat GitHub Actions as the clean-checkout source gate and the [deployment runbook](setup/deployment.md) as the hosted-beta gate.
