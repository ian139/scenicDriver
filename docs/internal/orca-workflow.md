# OMP + cmux development workflow

Use this workflow for every OMP coordinator, cmux-managed worktree/session, and OMP subagent in this repository. `AGENTS.md` is the source of truth and is auto-loaded for agents launched from this workspace; do not ask the user to restate it.

## Always-on rules

- Start from the repository root so OMP sees `AGENTS.md` automatically.
- Treat `AGENTS.md` as mandatory context for every prompt, subagent assignment, and review.
- Start every non-trivial feature, bug fix, refactor, or research task with `/plan`, then run it through `workflowz`: decompose, dispatch subagents, monitor, integrate, and verify.
- Use test-first development: write or update the focused failing test before implementation.
- Prefer functional core, explicit service boundaries, typed data, and side effects at service edges.
- Keep containers as the default runtime; host execution is developer convenience only.
- Run final verification through the containerized path before marking work ready to push.
- Never use model review as a substitute for tests, smoke checks, or container evidence.

## Coordinator startup

Create one feature worktree/session in cmux and run one advisor-enabled OMP parent for a new feature or risky fix:

```bash
omp --advisor --model "openai-codex/gpt-5.5" --thinking xhigh
```

Use cmux as the viewing and management surface for worktrees, panes, and long-lived sessions. Run the exact OMP command above inside the cmux-managed coordinator context. If a task specifically requires an Orca-managed worktree, the Orca CLI remains a fallback/legacy surface for that task only.

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
4. File ownership before any subagent edits.
5. Whether execution is coordinator-only, same-worktree subagents, sub-worktree subagents, or mixed.
6. Risks, service contracts, environment variables, and verification commands.

## Subagent model default

Use DeepSeek V4 Pro through Ollama Cloud for implementation subagents by default:

```bash
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

Use GPT-5.5 only when the task needs advisor-level coordination, unusually deep architecture work, or the DeepSeek/Ollama Cloud path is unavailable.

## Same-worktree subagents

Use same-worktree subagents only when file ownership is disjoint and conflicts are unlikely. Open a cmux-managed pane in the active worktree, then run the implementation OMP command inside it:

```bash
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

Fallback/legacy Orca-only path, for tasks that specifically require an Orca-managed terminal:

```bash
orca terminal create \
  --worktree active \
  --title "<specific-subtask>" \
  --command 'omp --model "ollama-cloud/deepseek-v4-pro" --thinking high' \
  --json
orca terminal wait --terminal <handle> --for tui-idle --timeout-ms 60000 --json
orca terminal send --terminal <handle> --text '<narrow task prompt>' --enter --json
```

## Sub-worktree subagents

Use sub-worktree subagents when isolation, competing implementations, risky tests, or independent verification improves quality. Create or open the child worktree through cmux, run the implementation OMP command inside the cmux-managed context, and have the coordinator integrate only the reviewed patch:

```bash
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

Fallback/legacy Orca-only path, for tasks that specifically require an Orca-managed child worktree:

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

Sub-worktree output is a patch proposal. The coordinator must inspect the subagent diff, reject unrelated edits, integrate the useful patch into the parent worktree, and rerun focused verification in the parent before final verification.

## Subagent prompt contract

Every subagent prompt must include:

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

## Vast GPU cmux host workflow

Publish Docker images before renting runtime GPU hosts. Do image publishing from a local checkout or a cmux-managed child worktree such as `vast-image-publish`; runtime Vast sessions only pull already-published images. `scripts/remote/vast-start-task.sh` performs SSH-gated allocation, repo upload, uv bootstrap, container smoke, and Orca server setup; do not use it to build or push container images.

```bash
# In a cmux-managed child worktree named vast-image-publish:
cd /Users/ian/orca/workspaces/Scenic\ Drive/vast-image-publish
docker buildx build --platform linux/amd64 -f Dockerfile.remote-training -t ian139/scenicdriver-remote-training:latest --push .
docker buildx build --platform linux/amd64 -f Dockerfile.remote-training -t ian139/scenicdriver-remote-training:smoke --push .
docker buildx build --platform linux/amd64 -f Dockerfile.remote-training -t ian139/scenicdriver-remote-training:heavy --push .
docker pull ian139/scenicdriver-remote-training:latest
docker pull ian139/scenicdriver-remote-training:smoke
docker pull ian139/scenicdriver-remote-training:heavy
```

For the automatic training lifecycle, start the head orchestrator Goal before
dispatching planning or execution subagents:

```bash
/goal vast-auto-training-lifecycle
# Objective: start Vast, validate S3/GPU smoke, train regression model, sync outputs, destroy instance

# Head orchestrator terminal
omp --advisor --model "openai-codex/gpt-5.5" --thinking xhigh

# Planning/execution subagents
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

Start each runtime task on a fresh Vast instance. The default `--allocation-attempts 3` attaches the configured SSH key, waits for the endpoint, reboots, verifies SSH, destroys bad hosts, and retries before any agent or expensive bootstrap runs.

```bash
scripts/remote/vast-start-task.sh scenic-orca-prebuilt-smoke 'Infrastructure smoke using the prebuilt remote-training image.' \
  --agent none \
  --allocation-attempts 3 \
  --disk-gb 32 \
  --image ian139/scenicdriver-remote-training:smoke \
  --timeout-seconds 1800 \
  --offer-query 'dlperf_usd>=180 dph<2 num_gpus=1 verified=true direct_port_count>=1 rentable=true'
```

For actual training runs through `scripts/remote/vast_lifecycle.py init`, omitting `--image` pulls `ian139/scenicdriver-remote-training:latest`. Pass `--containerfile` only when intentionally building a one-off image on the remote host; pass `--image` explicitly if that one-off build should use a different tag.

For train-and-close model runs, use the dedicated training lifecycle command
instead of `vast-watch.sh`. It pulls
`ian139/scenicdriver-remote-training:latest`, runs
`scripts/remote/provision_vast.sh` first, trains with the exact dataset key, syncs
S3 outputs, copies local artifacts, and destroys the instance by default:

```bash
scripts/remote/vast-train.sh run <task-name> \
  --train-dataset-key <key> \
  --epochs 1 \
  --batch-size 64

scripts/remote/vast-train.sh status <task-name>

scripts/remote/vast-train.sh cleanup <task-name> --copy-artifacts --destroy --yes
```

Use `--epochs 1 --batch-size 64` only for cost-controlled smoke training. If the
local orchestrator dies before its cleanup path runs, the cleanup command resumes
from `.orca-vast/state/<task-name>.json`, copies the recorded remote output root
when reachable, and destroys the recorded `instance_id`.

Manual SSH preflight is for debugging or exact-offer testing. Normal `vast-start-task.sh` runs rely on `--allocation-attempts` and fail closed before worktree setup if no SSH-ready host is found:

```bash
ssh-keygen -y -f ~/.ssh/id_ed25519
vastai attach ssh <instance-id> "$(ssh-keygen -y -f ~/.ssh/id_ed25519)"
vastai ssh-url <instance-id>
ssh -i ~/.ssh/id_ed25519 -p <ssh-port> -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@<ssh-host> echo vast-ssh-ok
```

If `vastai attach ssh` says the key is already associated but SSH returns `Permission denied (publickey)`, destroy that instance and rerent; the remote image did not accept the key. If SSH returns `Connection reset by peer` or hangs, wait briefly and retry the preflight; if it continues, rerent on another host before starting cmux/OMP pairing.

Save artifacts and destroy explicitly for cmux-managed worktree tasks:

```bash
scripts/remote/vast-down.sh scenic-orca-prebuilt-smoke --copy-artifacts --destroy --yes
```

Run this in the long-lived orchestrator terminal for automatic artifact copy and
cost control after cmux worktrees close:

```bash
scripts/remote/vast-watch.sh --interval-seconds 60 --destroy --yes
```

`scripts/remote/vast-down.sh <task-name> --copy-artifacts --destroy --yes`
remains a fallback for cmux-managed worktree tasks, not the primary training
cleanup command.

GPU selection defaults to on-demand pricing. Prefer on-demand for agent-backed tasks because Vast interruptible/bid instances can stop before the cmux worktree is completed. Use interruptible only for checkpointed training jobs that can tolerate eviction and resume cleanly.

For exact host selection, inspect offers first and pass the selected id while still using the prebuilt image and SSH-gated allocation:

```bash
vastai search offers 'dlperf_usd>=180 dph<2 num_gpus=1 verified=true direct_port_count>=1 rentable=true' -o 'dlperf_usd-' --raw
scripts/remote/vast-start-task.sh scenic-orca-prebuilt-smoke 'Infrastructure smoke using the prebuilt remote-training image.' --agent none --offer-id <offer-id> --allocation-attempts 3 --disk-gb 32 --image ian139/scenicdriver-remote-training:smoke
```

Required local and remote credentials:

- Local `vastai` authenticated.
- Local cmux/OMP coordinator available; use `orca status --json` only for fallback/legacy Orca-managed tasks.
- Docker Hub token with push access for `ian139/scenicdriver-remote-training`.
- `.secrets/aws.env` containing `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `SCENIC_S3_BUCKET`, and `SCENIC_S3_ONLY`.
- `~/.ssh/id_ed25519` and `~/.ssh/id_ed25519.pub` attachable to Vast.
- Remote agent CLI credentials are required only when `vast-start-task.sh` runs with `--agent codex` or another agent. Use `--agent none` for infrastructure smoke tests.

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