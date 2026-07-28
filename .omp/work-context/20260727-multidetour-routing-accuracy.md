# Multi-detour routing accuracy

- Status: stopped after the current run at user direction
- Last updated: 2026-07-27
- Originating OMP session: unavailable
- Subsequent OMP sessions: unavailable

## Authoritative objective

Improve compiled scenic-route planning accuracy for one route containing multiple detours through a fixed, deterministic autoresearch loop: one measurable hypothesis per experiment, automatic accept/reject/rollback, then central integration review.

## Constraints and non-goals

- Keep the benchmark command, fixtures, metric denominator, and baseline immutable within each experiment segment.
- Primary metric: compiled mean objective regret in percentage points.
- Protect exact/oracle match, frontier recall, multi-detour recall, correctness, cancellation, and latency gates.
- Hard-bound candidate state growth and preserve routing deadlines and graph-stamp checks.
- Do not claim production-scale performance; the 2,256-case production run remains blocked and unverified.
- The single-scalar-source repair remains an explicitly approximate strategy; no claim of global exactness.
- Generated data, models, caches, and large artifacts remain outside Git.

## Acceptance criteria

- AC1 [done]: The fixed 18-case segment includes all four central-review adversaries and is frozen at revision `f39cf85c`, digest `843ab3a7f6df0378233620f5402783cceae7ed2c88037b23c20d53820b64511a`.
- AC2 [not met]: No corrected bounded candidate passed both the expanded accuracy suite and the protected latency gate before autoresearch stopped.
- AC3 [not applicable]: No candidate was retained for central code review.
- AC4 [not applicable]: No final candidate revision was retained for integrated verification.
- AC5 [done]: Every completed experiment and the final rollback/stop decision are recorded; rejected planner changes were restored.

## Decisions

- 2026-07-27: Reject commits `facd6431`, `18198564`, and `85b9153d` despite benchmark acceptance. Central review proved two missing benchmark invariants: scenic-priority beam ranking omitted the distance tie-break, and numeric winner materialization aborted on a cyclic recombination instead of continuing to the next feasible simple state.
- 2026-07-27: Preserve the 64-state bound, the 48 comparator-ranked plus 16 shortest-duration tranche, deadline checks, graph-stamp checks, successful-repair accounting, and failed-source identity deduplication in any corrected design.
- 2026-07-27: Freeze revision `d4c4a7ad`, denominator `16`, and digest `6ac7ede8dc6be0aa16411f49e5a55583d675b1602aebcfa576276de33cf36a5f` as the corrected segment boundary after independent reviewer approval.
- 2026-07-27: Candidate retention requires lower primary regret, all correctness/recall controls, and compiled p95 no more than 10% above the immutable `1793` µs baseline (`1972.3` µs).
- 2026-07-27: Reject and roll back the first corrected frontier hypothesis. It reached zero regret and every accuracy gate, but compiled p95 was `2194` µs, `22.37%` above the immutable baseline and outside the 10% latency gate.
- 2026-07-27: Next hypothesis: replace per-state `PathEvaluation` construction with a flat numeric state frontier and compute only the active policy's comparator components. The distance adversary creates 128 transient states, so removing allocation and unused objective work targets the measured hotspot without changing search breadth or semantics.
- 2026-07-27: Retain optimized candidate `bdc2d73f` for central review. It makes the full 16-case oracle/review/recall matrix perfect while holding compiled p95 to `1942` µs, `8.31%` over baseline and inside the 10% limit.
- 2026-07-27: Central review rejects `bdc2d73f`; revert `895e31b6` restored the segment baseline. The 48-state ranked tranche used bitmask order instead of canonical edge identity for numeric ties, and subtractive aggregate deltas could rank a materialized route incorrectly. Both defects require fixed adversaries before another candidate.
- 2026-07-27: Expand the immutable boundary to 18 cases. The canonical-cutoff fixture forces the comparator winner through predecessor mask `112`, and the rounded-aggregate fixture forces direct path-order accumulation; rejected candidate `bdc2d73f` matches only `2/4` review cases.
- 2026-07-27: Freeze revision `f39cf85c`, denominator `18`, review denominator `4`, and digest `843ab3a7f6df0378233620f5402783cceae7ed2c88037b23c20d53820b64511a`. Candidate latency must not exceed 110% of the formal `1660` µs p95 baseline (`1826` µs).
- 2026-07-27: Reject and roll back the path-order prefix frontier. It eliminated all regret and passed every correctness control, but compiled p95 was `2540` µs, `53.01%` above the immutable baseline and outside the `1826` µs gate.
- 2026-07-27: Next hypothesis: keep flat delta expansion, but derive numeric and canonical comparator ranks from direct path-order summaries only at a `>64` cutoff (at most 128 transient states) and final selection (at most 64 retained states). Unlike an ill-conditioning heuristic, this covers every subtractive inversion by construction without accumulating every expansion.
- 2026-07-27: Reject and roll back cutoff/final direct summarization. It passed every accuracy/control gate but compiled p95 was `2541` µs; profiling attributed `22` ms of `61` ms helper time to 2,980 memoized `summarize` calls over 90 compiled runs.
- 2026-07-27: Next hypothesis: carry only exact path-order numeric prefixes during expansion, keep cap viability as one precomputed suffix addition, and construct canonical identities lazily at cutoffs. This removes repeated full-path summaries plus the prior per-state `fsum`/ULP bound while preserving exact comparator totals.
- 2026-07-27: Reject and roll back the lazy-canonical prefix frontier. It passed every accuracy/control gate with zero regret, but compiled p95 was `2339` µs, `40.90%` above the immutable `1660` µs baseline and outside the `1826` µs ceiling.
- 2026-07-27: Stop autoresearch after this run at user direction. No candidate is retained; the fixed harness remains at `f39cf85c` and production routing remains on the restored baseline.

## Progress

### Completed

- Built and validated the 14-fixture comparator-aware benchmark segment, digest `72a11c49b549791702bfa6b637068d901e61648090a2c37740715b0401655074`.
- Ran segment experiments 34–42 and retained the single-source frontier plus bookkeeping/dedup corrections before central review.
- Ran a central review of revision `85b9153d`; outcome: reject for the two correctness counterexamples above.
- Restored uncommitted cleanup tests before rollback.
- Verified repository state: clean worktree at `85b9153df08a` on `autoresearch/continue-the-project-from-the-latest-engineering-20260727`.
- Reverted rejected commits with new revisions `cf63abd7`, `d531311f`, and `a76a7968`; comparator-aware benchmark commit `344ea73f03b7` remains in history.
- Confirmed the rolled-back 14-case benchmark: mean objective regret `47.480323804377` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1482` µs.
- Expanded the harness to 16 fixtures with graph-serialized digest `6ac7ede8dc6be0aa16411f49e5a55583d675b1602aebcfa576276de33cf36a5f`; both exact and frontier modes match the exhaustive oracle on every case.
- Reapplied rejected candidate `facd6431` without committing: it still matched the original 14 fixtures but matched `0/2` review fixtures, proving both adversaries close the central-review gap; restored `planner.py` afterward.
- Independent harness review approved the latest oracle, digest coverage, denominators, endpoint behavior, and isolation of both defects with confidence `0.98`.
- Committed the frozen measurement boundary as `d4c4a7ad` and recorded its formal baseline: mean regret `42.021473805021` pp, review match `0.0`, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1793` µs.
- Measured the corrected eager-frontier hypothesis: all accuracy, review, recall, exact/frontier, q0, and correctness gates passed; latency failed, so `planner.py` was restored and the rollback benchmark reconfirmed baseline behavior.
- Profiled the rejected reference repair against the rollback. `distance_tie_beam_pruning` dominated at median `2350` versus `1715.5` µs; cProfile attributed `83` ms of 40 runs to `_large_graph_detour_candidate`, including `30` ms in 5,120 per-state `summarize` calls.
- Measured and retained the flat policy-ranked frontier as `bdc2d73f`: mean/worst objective and scenic regret `0`, every match/recall rate `1.0`, correctness failures `0`, compiled p95 `1942` µs.
- Two independent central reviewers rejected `bdc2d73f` with confidence `0.99`: an eight-island 70-way canonical tie drops the required predecessor mask, and rounded whole-path totals plus subtractive deltas can invert the actual materialized comparator. Revert `895e31b6` and the 16-case rollback benchmark completed.
- Added canonical-cutoff and rounded-aggregate fixtures. The expanded rollback has denominator `18`, review denominator `4`, digest `843ab3a7f6df0378233620f5402783cceae7ed2c88037b23c20d53820b64511a`, and exact/frontier oracle match `1.0`.
- Temporarily reapplied `bdc2d73f`: it fails exactly the two new review fixtures, with overall match `16/18` and review match `2/4`; restored `planner.py` afterward.
- Independent `ExpandedHarnessReview` approved the 18-case boundary with confidence `0.99` and no findings.
- Committed the approved 18-case harness as `f39cf85c` and recorded its formal baseline: mean objective regret `44.019087826685` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1660` µs.
- Measured the path-order prefix frontier: all accuracy/review/recall/control gates passed, but compiled p95 was `2540` µs, so the candidate was rejected and `planner.py` restored.
- Profiled the rejected candidate across 90 compiled runs: `_large_graph_detour_candidate` consumed `59` ms cumulative, including `12` ms in `can_complete`, `9` ms in `append_segment`, and `7` ms in edge premeasurement.
- Confirmed rollback on the immutable segment: baseline mean regret `44.019087826685` pp and compiled p95 `1627` µs.
- Measured cutoff/final direct summarization: zero regret and every accuracy/control rate `1.0`, but compiled p95 was `2541` µs, so the candidate was rejected and restored.
- Profiled the rejected helper at `61` ms over 90 compiled runs; `summarize` consumed `22` ms across 2,980 calls despite bounded memoization.
- Confirmed rollback again: immutable mean regret `44.019087826685` pp and compiled p95 `1567` µs.
- Measured the lazy-canonical prefix frontier: zero objective/scenic regret and every accuracy/control rate `1.0`, but compiled p95 was `2339` µs, so the candidate was rejected.
- Profiled the rejected helper at `56` ms over 90 calls; `rank_states` consumed `13` ms across 105 calls, `state_identity` `10` ms across 4,580 calls, and edge premeasurement `7` ms across 180 calls.
- Restored `src/route_planner/planner.py` and confirmed the immutable rollback: mean objective regret `44.019087826685` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1598` µs.

### Current

- Autoresearch stopped after completing, rejecting, and rolling back the current run.

### Next

1. None. Resume only on a new explicit user instruction.

## Blockers and unconfirmed operations

- Autoresearch is intentionally stopped at user direction; no implementation blocker is being carried forward.
- Production-scale 2,256-case performance remains unconfirmed and outside this fixed-suite experiment.

## Touched files and symbols

- `src/route_planner/planner.py`: rejected bounded-repair experiments; restored to the retained production baseline.
- `scripts/routing/autoresearch_multidetour.py`: fixed compiled multi-detour benchmark.
- `autoresearch.sh`: canonical benchmark command.
- `.omp/work-context/20260727-multidetour-routing-accuracy.md`: durable task state.

## Verification evidence

- `git status --short --branch && git rev-parse --short=12 HEAD` → clean worktree, branch `autoresearch/continue-the-project-from-the-latest-engineering-20260727`, revision `85b9153df08a`.
- Central reviewer outcome for `85b9153d` → reject; concrete distance-tie beam and cyclic-winner counterexamples recorded above.
- `git revert --no-edit 85b9153d 18198564 facd6431` → three clean revert commits ending at `a76a7968`.
- `bash autoresearch.sh` on the rollback → denominator `14`, digest `72a11c49b549791702bfa6b637068d901e61648090a2c37740715b0401655074`, mean regret `47.480323804377` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1482` µs.
- `bash autoresearch.sh` on the expanded rollback → denominator `16`, digest `6ac7ede8dc6be0aa16411f49e5a55583d675b1602aebcfa576276de33cf36a5f`, mean regret `42.021473805021` pp, review match `0.0`, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1738` µs.
- Temporary `git cherry-pick --no-commit facd6431` plus that expanded benchmark → overall oracle match `0.875`, review match `0.0`, mean regret `0.476190476190` pp, compiled p95 `2225` µs; `git restore --staged --worktree src/route_planner/planner.py` left only the intended harness/work-record changes.
- `uv run ruff check scripts/routing/autoresearch_multidetour.py` → `OK`.
- Pyright diagnostics for `scripts/routing/autoresearch_multidetour.py` → `OK`.
- Independent `HarnessReview` verdict → correct/safe to freeze, confidence `0.98`, no findings.
- `git commit` froze the harness at `d4c4a7ad`.
- Formal `bash autoresearch.sh` baseline at `d4c4a7ad` → mean regret `42.021473805021` pp, review match `0.0`, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1793` µs.
- Rejected corrected-frontier benchmark → mean/worst regret `0`, overall/review/multi-detour/recall/exact/frontier match `1.0`, correctness failures `0`, compiled p95 `2194` µs (`+22.37%` versus `1793`).
- `git restore --staged --worktree src/route_planner/planner.py` followed by `bash autoresearch.sh` → baseline mean regret `42.021473805021` pp and compiled p95 `1616` µs, confirming rollback.
- Per-fixture 12-run profile: rejected reference distance adversary median/p95 `2350/2525` µs; rollback median/p95 `1715.5/2421` µs. Other candidate medians were at most `1517.5` µs.
- cProfile over 40 rejected-reference distance runs → helper cumulative `83` ms, `summarize` cumulative `30` ms across `5,120` calls; the state-evaluation allocation path is the bounded hotspot.
- Accepted optimized-frontier benchmark → mean/worst objective and scenic regret `0`, overall/review/multi-detour/detour-recall/exact/frontier match `1.0`, q0 pass `1`, correctness failures `0`, compiled p95 `1942` µs (`+8.31%` versus `1793`).
- `uv run ruff check src/route_planner/planner.py` → `OK`.
- `git commit` retained the measured candidate at `bdc2d73f`.
- `RetainedFrontierReview` and `FrontierInvariantChallenge` → unsafe for retention, confidence `0.99`; both independently reproduced canonical-cutoff and rounded-aggregate rank inversions.
- `git revert --no-edit bdc2d73f` → rollback revision `895e31b6`.
- Rollback `bash autoresearch.sh` → digest `6ac7ede8dc6be0aa16411f49e5a55583d675b1602aebcfa576276de33cf36a5f`, baseline mean regret `42.021473805021` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1612` µs.
- Expanded rollback `bash autoresearch.sh` → denominator `18`, review denominator `4`, digest `843ab3a7f6df0378233620f5402783cceae7ed2c88037b23c20d53820b64511a`, mean objective regret `44.019087826685` pp, worst regret `75` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1791` µs.
- Temporary `git cherry-pick --no-commit bdc2d73f` on the expanded harness → overall oracle match `0.888888888889`, review match `0.5`, mean/worst regret `2.962962962963/53.333333333333` pp, compiled p95 `3032` µs; restoring `planner.py` recovered the rollback.
- `uv run ruff check scripts/routing/autoresearch_multidetour.py` and Pyright diagnostics → `OK`.
- `ExpandedHarnessReview` → correct/safe to freeze, confidence `0.99`, no findings.
- `git commit` froze the 18-case harness at `f39cf85c`.
- Formal `bash autoresearch.sh` at `f39cf85c` → digest `843ab3a7f6df0378233620f5402783cceae7ed2c88037b23c20d53820b64511a`, denominator `18`, review denominator `4`, mean objective regret `44.019087826685` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1660` µs; the 10% retention ceiling is `1826` µs.
- Rejected path-order frontier `bash autoresearch.sh` → zero objective/scenic regret, every match/recall/control rate `1.0`, correctness failures `0`, compiled p95 `2540` µs (`+53.01%` versus `1660`).
- cProfile on the rejected frontier → helper cumulative `59` ms over 90 compiled calls; `can_complete` `12` ms, `append_segment` `9` ms, and edge premeasurement `7` ms.
- `git restore --staged --worktree src/route_planner/planner.py` followed by `bash autoresearch.sh` → immutable rollback digest/denominators unchanged, mean regret `44.019087826685` pp, compiled p95 `1627` µs.
- Rejected cutoff/final direct-ranking `bash autoresearch.sh` → zero regret, every match/recall/control rate `1.0`, correctness failures `0`, compiled p95 `2541` µs (`+53.07%` versus `1660`).
- cProfile → helper cumulative `61` ms over 90 calls; memoized `summarize` cumulative `22` ms across 2,980 calls.
- `git restore --staged --worktree src/route_planner/planner.py` followed by `bash autoresearch.sh` → immutable rollback mean regret `44.019087826685` pp, compiled p95 `1567` µs.
- Rejected lazy-canonical prefix `bash autoresearch.sh` → zero objective/scenic regret, every match/recall/control rate `1.0`, correctness failures `0`, compiled p95 `2339` µs (`+40.90%` versus `1660`).
- cProfile → helper cumulative `56` ms over 90 calls; `rank_states` `13` ms across 105 calls, `state_identity` `10` ms across 4,580 calls, and edge premeasurement `7` ms across 180 calls.
- `git restore --staged --worktree src/route_planner/planner.py` followed by `bash autoresearch.sh` → digest and denominators unchanged, baseline mean objective regret `44.019087826685` pp, exact/frontier match `1.0`, correctness failures `0`, compiled p95 `1598` µs.
