# Roadmap

## Phase 0: MVP App (Priority)

- [ ] Add optional signed-in contributor annotation mode with credits + shadow QA queue.
- [ ] Add region selector/data registry in app and validate at least: `pittsfield`, `philadelphia`.

## Phase 1: Model Quality (Priority)

- [ ] Retrain classifier on a larger/higher-quality dataset (or expanded labels) to improve domain fit and reduce class mislabeling in Northeast regions.
- [ ] Keep classifier signal but shift it to model features/auxiliary loss instead of fixed manual class weights.
- [ ] Replace heuristic class-weight scoring with a learned scenic regressor that uses satellite embeddings + terrain features (and optional class probabilities) as inputs.
- [ ] Expand manual labels from 500 -> 1000-1500 and rerun mixed training/eval.

## Phase 2: Data + Reporting

- [ ] Add new region pipeline (download → label → terrain features → train).
- [ ] Run per-region reports for: `rocky_mountains`, `olympic_peninsula` (`philadelphia` done, Big Sur done).
- [ ] Optionally add "cluster view" for multi-region heatmap (group by region).

## Phase 3: Routing

- [ ] Expand annotations to 1000+ and rerun benchmark + mixed training from the active registry model.
- [x] Retain the bounded scenic scalar traversal; see the [routing performance autoresearch log](research/routing-performance-autoresearch.md) for fixed-workload metrics, correctness evidence, rejected trials, and durable artifact locations.
