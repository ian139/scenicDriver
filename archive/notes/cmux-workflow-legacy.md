# Historical OMP + Orca workflow (archived, non-active)

> This file is retained solely as project history. Do not execute its commands
> or treat it as current workflow; use `docs/internal/cmux-workflow.md` instead.

## Always-on rules

- Start from the repository root so OMP sees `AGENTS.md` automatically.
- Treat `AGENTS.md` as mandatory context for every prompt, worker assignment, and review.
- Start every non-trivial feature, bug fix, refactor, or research task with `/plan`.
- Use test-first development: write or update the focused failing test before implementation.
- Prefer functional core, explicit service boundaries, typed data, and side effects at service edges.
- Keep containers as the default runtime; host execution is developer convenience only.
- Run final verification through the containerized path before marking work ready to push.
- Never use model review as a substitute for tests, smoke checks, or container evidence.

## Coordinator startup

Create one feature worktree and one advisor-enabled OMP parent for a new feature or risky fix:

```bash
orca worktree create --name "<short-feature-name>" --json
orca terminal create \
  --worktree "<short-feature-name>" \
  --title "coordinator" \
  --command 'omp --advisor --model "openai-codex/gpt-5.5" --thinking medium' \
  --json
orca terminal wait --terminal <handle> --for tui-idle --timeout-ms 60000 --json
```

Use `orca-dev` instead of `orca` when operating an Orca development build. The OMP command stays the same unless the dev environment explicitly requires another binary.

## Long-running workstream branches

Keep exactly three long-running Scenic Drive workstream branches beyond `main`:

- `Ian139/RemoteTraining`: remote GPU lifecycle, S3-backed training execution, training CLIs/notebooks, and modeling infrastructure needed to run training remotely.
- `Ian139/UI-Fixes`: Figma-driven web/mobile UI rebuild, app shell work, and UI/API contract changes needed by the web or mobile interfaces.
- `Ian139/S3Management`: S3 data movement, bucket layout, lifecycle policy, data acquisition, and S3-aware reporting/download paths.

Create feature-sized child worktrees from the matching parent stream when work begins; do not create additional long-running top-level branches unless the task is truly cross-cutting and temporary.

## Planning checklist

During `/plan`, the coordinator must record:

1. User goal and acceptance criteria.
2. Affected files, services, commands, and container targets.
3. The test to write or update first.
4. File ownership before any worker edits.
5. Whether execution is coordinator-only, same-worktree workers, sub-worktree workers, or mixed.
6. Risks, service contracts, environment variables, and verification commands.

## Worker model default

Use DeepSeek V4 Pro through Ollama Cloud for implementation workers by default:

```bash
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

Use GPT-5.5 only when the task needs advisor-level coordination, unusually deep architecture work, or the DeepSeek/Ollama Cloud path is unavailable.

## Same-worktree workers

Use same-worktree workers only when file ownership is disjoint and conflicts are unlikely:

```bash
orca terminal create \
  --worktree active \
  --title "<specific-subtask>" \
  --command 'omp --model "ollama-cloud/deepseek-v4-pro" --thinking high' \
  --json
orca terminal wait --terminal <handle> --for tui-idle --timeout-ms 60000 --json
orca terminal send --terminal <handle> --text '<narrow task prompt>' --enter --json
```

## Sub-worktree workers

Use sub-worktree workers when isolation, competing implementations, risky tests, or independent verification improves quality:

```bash
orca worktree create --name "<parent-feature>-<specific-subtask>" --parent-worktree active --json
orca terminal create \
  --worktree "<parent-feature>-<specific-subtask>" \
  --title "<specific-subtask>" \
  --command 'omp --model "ollama-cloud/deepseek-v4-pro" --thinking high' \
  --json
orca terminal wait --terminal <handle> --for tui-idle --timeout-ms 60000 --json
orca terminal send --terminal <handle> --text '<narrow task prompt>' --enter --json
```

Sub-worktree output is a patch proposal. The coordinator must inspect the worker diff, reject unrelated edits, integrate the useful patch into the parent worktree, and rerun focused verification in the parent before final verification.

## Worker prompt contract

Every worker prompt must include:

```text
Context:
AGENTS.md is mandatory project policy and is already available in this workspace. Follow it without asking the user to restate it.

Target:
<exact files/symbols>.

Change:
<specific behavior to add/fix>.

Non-goals:
Do not edit <files/subsystems>.
Do not do broad cleanup.
Do not create a new worktree unless explicitly assigned a sub-worktree.

Ownership:
You own only <files>.
Do not touch anything else.

Development rule:
Add or update the focused failing test first, then implement the smallest passing patch.

Acceptance:
<focused checks or observable behavior>.

Verification:
Run <specific focused test or command>. Do not run project-wide gates unless assigned.

Report:
1. Files changed
2. Test-first evidence
3. Verification run
4. Result
5. Unresolved risks
```

## Verification gate

For completed feature work, run the focused test and the container path that covers the changed service:

```bash
docker compose build
docker compose up -d
# docker compose exec <service> <focused test command>
docker compose ps
docker compose down
```

If the repository uses another container runner, use the project-native equivalent. If container verification cannot run, state why and do not present the change as ready to push.