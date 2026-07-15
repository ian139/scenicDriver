# OMP + CMUX development workflow

Use this workflow for every OMP coordinator, CMUX worker, and CMUX development session in this repository. `AGENTS.md` is the source of truth and is auto-loaded for agents launched from this workspace; do not ask the user to restate it.

## Always-on rules

- Start from the repository root so OMP sees `AGENTS.md` automatically.
- Treat `AGENTS.md` as mandatory context for every prompt, worker assignment, and review.
- Start every non-trivial feature, bug fix, refactor, or research task with `/plan`.
- Use test-first development: write or update the focused failing test before implementation.
- Prefer functional core, explicit service boundaries, typed data, and side effects at service edges.
- Keep containers as the default runtime; host execution is developer convenience only.
- Run final verification through the containerized path before marking work ready to push.
- Never use model review as a substitute for tests, smoke checks, or container evidence.

## Initial one-time integration

Install the OMP integration hook and enable agent hibernation once per machine:

```bash
cmux hooks setup omp
cmux agent-hibernation on
```

## Coordinator startup and workspace groups

Create one CMUX workspace group for every repo-level feature or long-running
workstream. The group always creates a fresh anchor workspace; use that anchor
as the coordinator. Keep every worker, terminal view, and Git worktree for that
effort inside the same group.

```bash
cmux workspace-group create \
  --name "<repo>: <short-feature-name>" \
  --cwd "$PWD" \
  --json
cmux workspace-group list --json
```

Focus the returned group, then start the advisor-enabled OMP parent in its
anchor workspace. Use the group header `+` or `cmux workspace-group
new-workspace <group-id>` for additional workspaces; do not create ungrouped
repo workspaces.

CMUX owns the session view and grouping. Git worktrees are created or opened by
normal Git/project tooling outside CMUX, then their workspaces are added to the
matching group. Closing a group anchor dissolves the group, so do not close the
anchor while the workstream is active. Use `ungroup` to preserve member
workspaces or `delete` only when every member should be closed.

## Long-running workstream branches

Keep exactly three long-running Scenic Drive workstream branches beyond `main`:

- `Ian139/RemoteTraining`: remote GPU lifecycle, S3-backed training execution, training CLIs/notebooks, and modeling infrastructure needed to run training remotely.
- `Ian139/UI-Fixes`: Figma-driven web/mobile UI rebuild, app shell work, and UI/API contract changes needed by the web or mobile interfaces.
- `Ian139/S3Management`: S3 data movement, bucket layout, lifecycle policy, data acquisition, and S3-aware reporting/download paths.

Open or create these branches with normal Git/project tooling. Create or focus
one persistent CMUX workspace group for each workstream, then add every
feature-sized child worktree workspace to that group:

```bash
cmux workspace-group create \
  --name "Scenic Drive: <workstream-name>" \
  --cwd "<absolute-worktree-path>" \
  --json
cmux workspace-group new-workspace <group-id> --placement end
```

Do not create additional long-running top-level branches unless the task is
truly cross-cutting and temporary. CMUX groups organize the whole repository's
session topology; they do not create Git branches or worktrees.

## Planning checklist

During `/plan`, the coordinator must record:

1. User goal and acceptance criteria.
2. Affected files, services, commands, and container targets.
3. The test to write or update first.
4. File ownership before any worker edits.
5. Whether execution is coordinator-only, same-worktree workers, isolated worker workspaces, or mixed.
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
cmux new-split right --focus true
cmux send 'omp --model "ollama-cloud/deepseek-v4-pro" --thinking high'
cmux send-key Enter
```

After OMP starts, send the worker prompt contract below as the narrow task prompt.

## Isolated worker workspaces

Use isolated worker workspaces when isolation, competing implementations,
risky tests, or independent verification improves quality. Create the Git
worktree outside CMUX first. Open that checkout as a CMUX workspace, then add
the returned workspace ID to the matching feature/workstream group:

```bash
cmux new-workspace \
  --name "<parent-feature>-<specific-subtask>" \
  --cwd "<absolute-worktree-path>" \
  --focus false
cmux workspace-group add --group <group-id> --workspace <workspace-id>
```

Start the worker in that member workspace. Never leave same-repository worker
workspaces ungrouped.

Isolated worker output is a patch proposal. The coordinator must inspect the
worker diff, reject unrelated edits, integrate the useful patch into the parent
worktree, and rerun focused verification in the parent before final
verification.

## Markdown plan/viewer

Open the approved plan or handoff document in CMUX when it helps keep context visible:

```bash
cmux markdown open local://<slug>-plan.md --focus true
```

## Diff viewer

Use CMUX to keep the active diff visible during review:

```bash
cmux diff --source unstaged --cwd "$PWD" --title "<short-feature-name> diff"
```

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
Do not create or open a new CMUX workspace unless explicitly assigned an isolated workspace.

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

## CMUX worktree viewer

Install and open the OMP worktree sidebar with:

```bash
mkdir -p ~/.config/cmux/sidebars
cmux sidebar validate omp-worktrees
cmux sidebar select omp-worktrees
cmux sidebar open omp-worktrees
```

The sidebar shows live CMUX workspaces. A git worktree appears there after it has been opened as a CMUX workspace.

## Legacy persisted-state migration

Older remote lifecycle runs may have state under `.orca-vast/state`. The active
CMUX host implementation reads those files only when a matching CMUX state file
does not exist, normalizes legacy workspace keys in memory, and writes the next
state update under `.cmux-vast/state`. Active writes never contain legacy keys,
and no legacy runtime setup, status, pairing, or worktree command is invoked.

## Opening an active workspace

Normal `scripts/remote/vast-start-task.sh` (the `start-task` subcommand) allocates
and bootstrap-checks the Vast host, then creates and registers its CMUX workspace
through the v2 JSON-RPC call `cmux rpc workspace.create`. The wrapper persists the
returned workspace reference and UUID separately in
`.cmux-vast/state/<task-name>.json` as `cmux_workspace_ref` and
`cmux_workspace_id`. `scripts/remote/vast-watch.sh` uses that recorded identity
when it observes `cmux workspace list --json`; it does not infer a workspace from
the name or current directory.

If workspace creation or registration fails, `start-task` clears the identity,
leaves the state `workspace_pending`, records the error, and returns failure.
Rerun the same command to retry registration safely.

The explicit `--manual` exception prints a `cmux new-workspace` command and
leaves the state `workspace_pending` and unregistered, with no identity recorded.
Creating that workspace manually does not make it watchable until it is
explicitly paired. Legacy migrated Orca state is likewise unregistered until an
explicit pairing/registration action; migration only preserves descriptive
metadata and never creates a CMUX workspace or infers CMUX identity from Orca
fields.

Do not create or open a workspace during bulk state migration, and do not start
OMP automatically for imported state.
## Repository-wide group operations

These group rules apply to every Scenic Drive task, branch, worktree, and
worker—not only UI or remote-training work.

- Inspect topology with `cmux workspace-group list --json`.
- Add an existing workspace with `cmux workspace-group add --group <group-id> --workspace <workspace-id>`.
- Remove a workspace without closing it with `cmux workspace-group remove --workspace <workspace-id>`.
- Use `collapse`/`expand`, `pin`/`unpin`, `set-color`, and `set-icon` to keep long-running workstreams legible.
- Use `focus <group-id>` before spawning a member so `⌘N` and the header `+` inherit the correct group.
- Use `ungroup <group-id>` when the container should disappear but member workspaces must survive.
- Use `delete <group-id>` only when the anchor and every child workspace should close; deletion is irreversible.
- Never promote an existing workspace to a newly created group anchor. CMUX always creates a fresh anchor by design.

Per-group defaults and context-menu actions belong under `workspaceGroups` in
`~/.config/cmux/cmux.json`, keyed by the anchor working directory. Repository
guidance must use the documented `cmux workspace-group` CLI and must not depend
on Orca runtime state.

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
