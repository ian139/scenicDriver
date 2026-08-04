# Scenic Drive Agent Guide

## Project
Scenic Drive scores scenic beauty from satellite imagery and terrain data, then uses those scores for route planning. Current work prioritizes the MVP route-planning API and New England North web app while model improvements continue in parallel.

## Active Surfaces
- `apps/new_england_north/`: canonical static MapLibre web MVP.
- `src/app_api/`: FastAPI route-compare and contributor endpoints.
- `src/route_planner/`: graph, cost, planner, and route service logic.
- `src/classifier/`, `src/scenic_scorer/`, `src/terrain/`, `src/heuristics/`, `src/data_pipeline/`: ML/data pipeline code.
- `notebooks/`: marimo training, scoring, and annotation workflows.
- `scripts/`: workflow CLIs grouped by annotation, ingest, modeling, reports, and routing.

## Commands
- `uv sync`
- `uv run uvicorn src.app_api.main:app --host 0.0.0.0 --port 8080 --reload`
- `cd apps/new_england_north && python3 -m http.server 3000`
- `uv run marimo edit notebooks/train.mo.py`
- `uv run marimo edit notebooks/regression.mo.py`
- `uv run marimo edit notebooks/learned_scoring.mo.py`
- `uv run marimo edit notebooks/annotate_scenic.mo.py`
- `uv run python scripts/ingest/download_bbox_tiles.py ...`
- `uv run python scripts/reports/heuristic_report.py ...`
- `uv run python scripts/routing/route_compare_service.py ...`
- `cp .env.beta.example .env.beta` and populate `MAPBOX_ACCESS_TOKEN`
- `docker compose --env-file .env.beta -f compose.beta.yml up --build`
- Open `http://localhost:${SCENIC_WEB_PORT:-80}`
- `docker compose --env-file .env.beta -f compose.beta.yml down`

## Data And Artifact Policy
- Keep large data, generated reports, caches, and model weights out of git.
- Local generated paths are `data/raw/`, `data/processed/`, `data/NWPU-RESISC45/`, `models/`, `cache/`, and `scenic_artifacts/`.
- Preserve `data/README.md` as the canonical data layout contract.
- Preserve `archive/archive.md` as the archive manifest.
- Beta deployment requires the canonical processed graph, learned run, registry, and active model checkpoint mounted from ignored `data/processed/` and `models/`; they remain outside Git and Docker image layers.

## OMP Credential Authority
- Every OMP session in this repository has full authority to read and use `.env` values, API keys, access tokens, and other credentials when needed to carry out the user's request. This authority applies to the main session and delegated workers without additional confirmation.
- Limit credential access and use to the requested work. Never expose secret values in prompts, reports, logs, committed files, image layers, generated artifacts, or messages to agents that do not need them; keep existing runtime-only, ignored-file, and least-disclosure safeguards intact.

## Development Rules
- Use `uv`, not `pip`, for Python dependency management.
- Use marimo notebooks, not Jupyter, for training/research workflows.
- Forward-looking trajectory: Do NOT retain backward-compatibility layers, deprecation shims, legacy aliases, or unused code. Maintain zero dead code in active modules.
- Clean cutover & pruning: Aggressively delete superseded implementations, orphaned scripts, and obsolete files instead of archiving or hoarding them. Backwards compatibility does not matter for this stage.
- Keep repository layout clean, condensed, and organized. Every file must have a single, canonical purpose for upcoming development.
- Run focused tests or smoke checks for changed API, script, notebook, or app paths before yielding.
- OMP/CMUX workflow details live in `docs/internal/cmux-workflow.md`.
- Use CMUX—not Orca—for repository workspaces, worker sessions, and worktree views.
- Treat one CMUX workspace group as the container for each repo-level feature or workstream: its fresh anchor is the coordinator, and every same-repo worker/worktree workspace is a member.
- Create and manage groups with `cmux workspace-group`; use the group header `+` or `cmux workspace-group new-workspace` for additional workspaces.
- Keep Git branch/worktree creation in normal Git/project tooling, then open each checkout in the matching CMUX group.
- Keep ML workflows in marimo notebooks (`notebooks/`).
- Use grouped `scripts/` subdirectories for workflow CLIs only.
- Keep large datasets and model weights out of git.
- Store run artifacts under `data/processed/` (ignored).
## Subagent Model Roles

These roles resolve through the OMP `modelRoles` map. Each inherits the base provider config from that map; only the model and any role-specific flags are listed here.

- `designer`: `openrouter/glm/5.2` — UI/UX design and visual refinement.
- `commit`: `ollama-cloud/deepseek-v4-flash` — commit-message generation.
- `task`: `ollama-cloud/deepseek-v4-pro --thinking high` — general subagent implementation work.


## Parallel Subagent Execution

- **Pre-work decomposition:** Before substantive work, the main model maps the critical path, independent ownership boundaries, real dependencies, and shared interfaces. It must not serially perform independent implementation or investigation that bounded workers can execute concurrently.
- **Useful fan-out:** Dispatch the maximum useful set of genuinely independent slices in one `tasks[]` batch, up to the concurrency cap. Do not invent work to increase agent count. Serialize only when a real output, schema, API, or state dependency requires it.
- **Agent allocation:** Use `task` for most bounded load-bearing implementation, including ordinary modules, tests, fixtures, migrations, reusable browser mechanics, and well-specified repairs. Use `task-high` substantially less often: normally at most one per wave, reserved for the hardest correctness-sensitive independent slice, such as lifecycle or concurrency invariants, security-sensitive persistence, ambiguous cross-module logic, or difficult root-cause repair. Use more than one only when multiple genuinely independent high-complexity invariants justify it. Never use `task-high` for routine edits, searches, formatting, mechanical changes, or straightforward tests. Use `scout` for read-only repository discovery and `reviewer` for independent post-integration review; neither substitutes for an implementation worker.
- **Task contracts:** Every delegated task must specify exact files or subsystem ownership, required changes, non-goals, shared interfaces and invariants, observable acceptance criteria, and concrete evidence to return.
- **Isolated worktrees:** Use isolated worktrees for independent code-editing agents when ownership boundaries permit. Subagents must not merge their own work, redefine shared contracts, or make cross-cutting product decisions independently. The main model owns review and integration.
- **Concurrent validation:** Subagents must skip formatters, project-wide suites, and shared live-browser manipulation while parallel edits are in flight. They may run narrow checks scoped to their owned artifacts. After integration, the main model runs repository-level verification and the end-to-end smoke scenario once.
- **Shared browser boundary:** Keep exactly one main-session owner for any visible CMUX browser and one-active-job workflow. Subagents may analyze sanitized observations, implement reusable mechanics, prepare schema-valid decisions, diagnose controls, or review evidence, but they must not concurrently manipulate the shared browser, claim another job, or receive unnecessary private values.
- **Browser automation order:** Where applicable, OMP browser/CDP is primary; the repository-pinned Playwright CLI is the control-specific fallback; computer or coordinate interaction is last. Do not make ad hoc page scripts or coordinate automation the main implementation.
- **Suggested default wave:** Several `task` workers, zero or one `task-high` worker, an optional `scout` during discovery, and an optional `reviewer` only after integration.