# Changelog

All notable user-visible contract changes to Scenic Drive. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 2026-10-04 — Unreleased

Documentation and data-contract alignment following the repository audit sweep.
Every entry below reflects behavior that is implemented and enforced in the
current tree.

### Changed

- `scripts/modeling/build_mixed_labels.py` now requires
  `--geographic-splits-csv`. The run's fixed `image_path`/`split` assignments
  are mandatory and never inferred; rows outside the split file are dropped and
  the split file is validated (one split per image, one split per
  `geographic_block`, `split` in `train`/`val`/`test`).
  `scripts/annotation/build_human_benchmark.py` applies the same required flag.
- `scripts/modeling/train_regression_baseline.py` now requires
  `--benchmark-split-csv`, and `scripts/modeling/run_regression_pipeline.py`
  requires both `--benchmark-split-csv` and `--baseline-checkpoint`. Fixed
  benchmark validation/test identities are excluded from training before the
  internal split, and the candidate is compared against an explicit baseline.
- Baseline regression training and evaluation now share one explicit
  validation-membership derivation
  (`scripts/modeling/regression_split.py`) over the filtered training
  population. `train_regression_baseline.py` persists it as a split artifact
  (`--split-artifact`, default `<output stem>.split.json`), and
  `evaluate_regression_baseline.py` requires `--split-artifact` (its
  `--val-split`/`--seed` options are removed) to score exactly the trained
  model's validation rows; any membership drift is a hard error.
- `scripts/remote/vast_train.py run` now requires `--benchmark-split-key`, the
  exact S3 key of the benchmark split CSV whose validation/test identities are
  excluded from training. The wrapper downloads it with `--required`, so a
  missing key fails the run before training starts.
- `scripts/remote/vast_train.py` now guards the training launch boundary with a
  persisted `launch_attempted` flag set before the detached remote command is
  fired. When the launch call raises afterwards (lost ack or dropped channel),
  the outcome is ambiguous: the run records `launch_outcome: "unknown"` and
  keeps the host (never `failed_destroyed`, never a replacement rental) under
  every policy, including `--keep-on-failure` and `--no-destroy`. Only
  confirmed-never-launched allocations are still destroyed before re-renting.
- `notebooks/learned_scoring.mo.py` requires the per-run `cfg.benchmark_split_csv`
  value (no default); training raises without it. It also wires
  `cfg.split_artifact` through training and evaluation so evaluation scores
  exactly the trained model's validation rows.
- `scripts/annotation/annotate_scenic_web.py` accepts repeatable
  `--trusted-authority <host[:port]>`. Public DNS names, wildcard binds, and
  reverse-proxy deployments must declare every served authority or requests are
  rejected as `untrusted_host`.
- `scripts/annotation/build_source_shift_batch.py` accepts `--image-root`, the
  shared satellite tile root. `image_path` is emitted relative to it, the
  summary records it as `image_root`, and the annotator `raw_dir` must equal
  that root so one annotator serves both sources.
- `scripts/reports/heuristic_report.py` accepts `--learned-run-json`, attested
  learned run metadata that supplies missing checkpoint identities. Declared
  identities must match the loaded artifacts; nothing is inferred.
- The Stage-Two autoresearch entry point is
  `scripts/modeling/run_active_scenic_autoresearch.py`. It requires
  `--control-dataset` (a control feature dataset NPZ disjoint from the expanded
  prepared dataset) alongside `--expanded-benchmark-csv`,
  `--control-benchmark-csv`, and `--route-qa-json`, and `--thresholds-json` for
  every non-dry execution. The root `autoresearch.sh` now execs the Northeast
  Expanded routing benchmark and is no longer a modeling launcher.

### Fixed

- The beta deployment contract now ships each graph's `run.json` readiness
  gate. `deploy/beta_artifacts.json` and `deploy/beta_artifacts.sha256` list
  `data/processed/road_graphs/<run>/run.json` for both deployed graphs;
  `scripts/deploy/bootstrap_beta_artifacts.py` downloads them like every other
  manifest destination, and `scripts/routing/check_beta_artifacts.py` now
  rejects a deployment whose gate is missing, stale, or names build-machine
  paths — exactly the deployments the API refuses to route. Previously a
  documented bootstrap produced a service with zero routeable regions.
- `scripts/routing/build_graph_from_osm.py` records project-root-relative
  artifact paths in `run.json` (`output_graph`, `output_dir`, `compact_graph`,
  `runtime_graph`) whenever the published files live under the project root.
  Absolute build-machine paths could never validate on another host, so
  readiness gates must be project-root-relative.

### Added

- `SCENIC_ROUTE_DEADLINE_SECONDS` overrides the per-region
  `route_deadline_seconds` from `config/app_regions.json` for every region. It
  must be a finite non-negative number.
- `CONTRIB_ADMIN_TOKEN` enables `POST /v1/admin/contrib/review/run`, guarded by
  the `X-Admin-Token` header (`503` when unset, `401` missing, `403` mismatch).
- `northeast_expanded` is a documented entry in `config/app_regions.json`, with
  its own `route_deadline_seconds` and `tile_scores_path`, alongside
  `new_england_north`, `masswhites`, `philadelphia`, and `pittsfield`.
- The compact road-graph runtime documents its per-report score sidecar
  (`<graph>.score.<report_signature>.z<zoom>.fb<fallback>.n<normalization>.json`
  plus `.bin`), which stores clamped scenic costs keyed by canonical edge rank.

### Removed

- `OMP_RUN_AUTORESEARCH` and the `bash autoresearch.sh --handoff ...` modeling
  invocation. `autoresearch.sh` no longer parses modeling arguments.
- `data/processed/regression/model_registry.json` is no longer tracked in Git.
  It remains a runtime deployment artifact: `deploy/beta_artifacts.json` ships
  it, `scripts/routing/check_beta_artifacts.py` resolves the active checkpoint
  through it, and `scripts/modeling/promote_regression_model.py` reads and
  writes it. The registry is generated and ignored like other `data/processed/`
  output.
- `scripts/ingest/download_bbox_tiles.py`. Open-data acquisition is
  `scripts/ingest/plan_active_learning_region.py` with
  `scripts/ingest/naip_3dep_workflow.py`. The archive manifest no longer lists
  the deleted CLI and now records `archive/` as a frozen superseded snapshot.

### Regenerated

- `docs/assets/research/routing/*.svg` are rewritten from
  `docs/assets/research/routing/figure-data.json` by
  `uv run python scripts/reports/generate_routing_research_figures.py`, so the
  committed figures match the derived figure claims.
