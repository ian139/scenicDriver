# Job Application Assistant: What We're Building

We are building a local job-application assistant that prepares applications from a SQLite backlog, opens the apply URL, observes the live form, proposes answers, fills safe fields, and stops before final submission.

This repo is intentionally centered on local dry-run preparation, review, and guarded execution. It is not a mass auto-apply bot. Keep the workflow deterministic, testable, and policy-first.

## Product Contract
- The assistant helps prepare job applications locally.
- It may open application pages, inspect forms, fill known safe fields, upload the configured resume, and advance through approved non-final navigation.
- It must stop before any final submit action.
- It must stop when a field, workflow, or page state requires human judgment.
- It must never become board-specific automation glued together with brittle Playwright templates.

## Target Workflow
1. Read the SQLite job backlog.
2. Open the apply URL with Playwright.
3. Run the deterministic observer over the DOM and frames.
4. Produce a normalized page snapshot.
5. Send the snapshot, profile, resume, facts, job description, and policies to the resolver.
6. Receive strict JSON answers, navigation choices, submit candidates, and uncertainty flags.
7. Execute only guarded generic actions.
8. Repeat `observe -> resolve -> fill -> advance` until the run reaches a terminal result.
9. Save the run result for review.

## Normalized Snapshot Contract
The observer emits structured page state only. It does not decide answers.

Fields should include:
- `id`
- `kind`
- `label`
- `required`
- `options`
- `value`
- `visible`
- `frame`
- `selector`

Buttons should include:
- `id`
- `text`
- `type`
- `disabled`
- `finalSubmitCandidate`
- `visible`
- `frame`
- `selector`

Snapshots may also include:
- errors
- blockers
- unsupported controls
- sign-in or CAPTCHA indicators
- upload state
- navigation state

## Run Results
- `dry_run_ready`: The assistant reached the final submit state and stopped without submitting.
- `needs_review`: The page requires an unknown, sensitive, legal, or manual answer.
- `blocked`: The assistant hit sign-in, CAPTCHA, no form, job gone, weird upload behavior, or an unsupported workflow.
- `failed`: The run failed due to browser, LLM, parser, executor, or navigation failure.

## Safety Policy
Hard rules:
- Never mass-submit applications.
- Never click final submit.
- Never answer sensitive, legal, demographic, eligibility, disability, veteran, sponsorship, background-check, or identity fields by inference.
- Never bypass sign-in, CAPTCHA, assessments, identity checks, or human verification.
- Upload only the configured resume file unless the user explicitly changes policy.
- Treat destructive database cleanup as requiring a clear user instruction.

Soft preferences:
- Use minimal Playwright R&D.
- Prefer deterministic observation and guarded generic execution.
- When a board fails, collect representative samples and reasons before adding targeted policies or fixtures.
- Add targeted behavior only when it is testable and improves the generic pipeline.

## Applicant Reference
Use these default applicant facts for local dry-run preparation and resolver context. Do not infer sensitive or legal answers from them.

- Resume file: `Main_Resume.pdf`
- LinkedIn: `https://www.linkedin.com/in/ianrapko`
- Personal site: `https://immemorized.com`

## Architecture Split
Keep observer, resolver, and executor responsibilities separate. Do not blur these boundaries for convenience.

### Observer
The observer answers: what exists on the page?

Input:
- browser page state
- frame state

Output:
- normalized page snapshot

Rules:
- No profile facts.
- No resume facts.
- No answer decisions.
- No browser mutation beyond safe inspection.
- Deterministic and testable from static HTML fixtures where possible.

### Resolver
The resolver answers: what should be filled, clicked, or escalated?

Input:
- normalized snapshot
- profile
- resume
- facts
- job description
- policies

Output:
- strict JSON answers
- next-button choice
- submit-button identification
- uncertainty flags
- `needsReview` decision

Rules:
- Refuse unknown fields instead of guessing.
- Refuse sensitive or legal fields instead of inferring.
- Never perform browser actions.
- May choose safe initial `Apply` / `Start` navigation only from observed buttons.
- The in-program DeepSeek/Ollama resolver may load `skills/SKILL.md` as operational guidance for live proof/navigation while still returning strict JSON for guarded execution.

### Executor
The executor answers: which approved browser actions are allowed now?

Input:
- normalized snapshot
- resolver JSON

Allowed actions:
- fill text-like fields
- select dropdown options
- check or uncheck allowed boxes
- upload the configured resume
- click policy-approved non-final `Apply`, `Next`, or `Continue` navigation

Stop conditions:
- final submit candidate
- CAPTCHA
- sign-in
- unsupported control
- sensitive field
- legal field
- unknown required field
- weird upload flow
- manual assessment

Rules:
- Use reusable generic Playwright primitives only.
- Never click final submit.
- Never silently skip required fields.
- Never invent board-specific templates as the default solution.

## Project Structure (Current)
- `scraper/src/theirstack/`: TheirStack query and client code.
- `scraper/src/sync/`: SQLite backlog sync and dedupe.
- `scraper/src/db/schema.sql`: backlog and application-run schema.
- `scraper/src/apply_pipeline/`: application-assistant contracts and pure helpers.
- `scraper/tests/`: unit tests for contracts, sync, query behavior, and application-pipeline policy.
- `old/`: archived code only. Active entrypoints and tests do not belong here.

## Commands (Primary)
For normal Python changes in `scraper/`:

```bash
.venv/bin/python -m pytest
```

For TheirStack preview changes, also run a safe preview:

```bash
.venv/bin/job-sync dry-run --call-api --posted-at-max-age-days 2
```

For paid fetches, run only after explicit user approval because returned jobs can spend credits:

```bash
ENABLE_PAID_FETCH=true JOB_SYNC_DB_PATH=data/job_sync_test.sqlite3 \
.venv/bin/job-sync sync-once --limit <preview_total> --max-pages 1 --posted-at-max-age-days 2
```

When containers are in scope, default container verification is:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose down
```

Never claim a command was run unless it was actually executed.

## Development Principles
- Keep business logic as pure functions over typed data where possible.
- Keep side effects at boundaries: CLI, browser, filesystem, HTTP, SQLite, queues, and network calls.
- Add or update a focused failing test before implementation.
- Add tests for every new branch in observer, resolver, executor, and policy logic.
- Prefer boring schemas and explicit JSON contracts.
- Do not add a second convention beside an existing one.
- Prefer measurable behavior over subjective review.
- Prefer small, composable modules over class hierarchies.
- If a class is unavoidable, keep it as a thin adapter around functional code.

## OMP + Orca Workflow
Use `OMP_ORCA_WORKFLOW.md` as the reusable operating workflow for OMP coordinators, Orca workers, and Orca dev sessions in this repository.

Mandatory invariants:
- Launch OMP from the repository root so `AGENTS.md` is auto-loaded.
- Treat this file as always-on policy for every coordinator, worker, and review prompt.
- Use test-first development.
- Use DeepSeek V4 Pro through Ollama Cloud for implementation workers by default:

```bash
omp --model "ollama-cloud/deepseek-v4-pro" --thinking high
```

- Use `orca-dev` instead of `orca` only when operating an Orca development build.
- Keep the same worktree, terminal, verification, and handoff workflow across `orca` and `orca-dev`.

## Orchestration Policy
The coordinator owns planning, worker allocation, integration, verification, and final reporting.

Use `OMP_ORCA_WORKFLOW.md` as the canonical workflow for:
- planning
- worker allocation
- Orca worktrees
- launch commands
- task dispatch
- handoff review
- parent/coordinator checklists

For non-trivial tasks, the default coordinator flow is:
1. Run `/plan`.
2. Decide whether parallel work would reduce risk or cycle time.
3. Create worker terminals or worktrees when useful.
4. Assign explicit ownership to each worker.
5. Monitor worker progress.
6. Review all worker output.
7. Integrate changes.
8. Run the required verification.
9. Report only what was actually changed and checked.

Assume permission for reversible development operations unless the user explicitly restricts them.

## Service Boundaries
Prefer a functional, service-oriented design over object-oriented architecture.

Candidate service boundaries:
- job ingestion / TheirStack sync
- job description normalization
- resume, profile, and facts loading
- page observation
- answer resolution
- guarded execution
- failure sampling / review queue
- report and UI serving
- persistence, cache, and queue infrastructure

Each real service boundary needs:
- documented input/output contract
- focused tests
- container or CLI execution path
- smoke check or health check for long-running processes

Split responsibilities only when the boundary is independently testable and useful.

## Containerization Contract
Everything should be able to run in containers by default. Host execution is a developer convenience, not the only verified path.

For every service or CLI entrypoint added or changed, ensure:
- dependencies are declared in project files
- required environment variables have examples or safe defaults
- tests can run from a clean checkout
- long-running services have a health check or smoke check
- Docker or compose wiring is updated when the service boundary needs it

Before merge or push, when containers are in scope, run:
- `docker compose build`
- `docker compose up -d`
- `docker compose ps`
- `docker compose down`

If containerized verification cannot be run, explain why and do not present the work as ready to push.

## Review Policy
Prefer executable evidence over vague review.

Useful review findings cite:
- failing tests
- missing tests
- broken contracts
- unclear ownership boundaries
- containerization gaps
- security-sensitive logic
- measurable complexity
- diffs outside scope

Do not ask for vague "looks good" reviews. The coordinator owns scope review and verification.

## Verification Contract
Every completed task must include:
- commands actually run
- container build/start commands actually run
- tests actually run inside containers when applicable
- files modified
- files intentionally not modified
- services added, removed, or changed
- service contracts added or changed
- known risks
- remaining TODOs

Never claim verification unless the command was executed.

## Task Completion Report
At the end of a feature task, the parent/coordinator must report:
- summary of behavior changed
- files changed
- tests/checks run
- known risks or skipped checks
- suggested next patch, if any

## Archive Policy
Archived code belongs under `old/`.

Rules:
- Do not move active entrypoints or tests into `old/`.
- Do not restore archived code unless it directly supports the current architecture.
- Before introducing a second workflow path, check whether the repo already has a convention.
- If archived code is restored, document why in the task report or commit message.

## Non-Goals
- No mass auto-apply system.
- No final-submit automation.
- No CAPTCHA, sign-in, identity, or assessment bypass.
- No board-specific Playwright template library as the primary architecture.
- No inference of sensitive, legal, demographic, eligibility, or identity answers.
- No destructive database cleanup without clear user instruction.
