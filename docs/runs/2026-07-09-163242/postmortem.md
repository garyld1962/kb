# Postmortem

## Run reference
- **Plan:** `docs/plans/2026-07-08-kb-v1-capture-search.md`
- **Verdict:** WARN  Gate: manually-closed-pending-adversarial-review
- **Branch / base / head:** `feat/kb-v1-capture-search` / `081d986` / `37d941e`
- **Trigger:** WARN verdict + accepted-risk deviation (`--postmortem` auto rules), plus operator-terminated run
- **Mode:** full

## What happened

kb v1 (capture + search core) was built end-to-end across 11 tasks, 35
review findings fixed at four gates (two milestones, checkpoint, PR
boundary), and 2 deviations properly resolved via plan amendments. The
run needed 5 resumes (a 6th was manually stopped) because of two
plan-authoring gaps this session discovered live — missing dependency
declarations and missing `write_scope` entries — and one real bug in the
`execute-plan` runtime itself (`gateFindings` blocking on any open
finding, not just critical/major). The operator stopped the run once the
remaining blocker became a whack-a-mole over a non-deterministic
deviation ID rather than a real code problem; this report and the
verdict were compiled manually from the journal and git history.

## What worked

- Task agents correctly refused to edit outside their `write_scope`
  (Task 9 stopping cold on missing FastAPI/httpx rather than silently
  patching `pyproject.toml`) — exactly the boundary discipline the
  co-tenancy instruction is meant to produce.
- The milestone and PR-boundary review gates earned their keep: 35 real
  findings (10 critical/major bugs like `no-runtime-wiring`,
  `path-traversal-destination`, `reindex-scheduler-never-started`) were
  caught and fixed before this ever reached a human reviewer.
- Plan-alignment correctly caught every one of this session's manual
  out-of-band edits (the dependency fixes, the runtime bugfix, the
  missing test files) — nothing slipped through undocumented.
- Recording ambiguity resolutions as plan Closed Decisions + recomputed
  Plan-SHA (the sanctioned recovery path) durably fixed two deviation
  categories on the next alignment pass — it works when the deviation's
  identity is stable.

## What broke down

### Tool and skill usage

- `gateFindings()`'s terminal check (`findings.find(f => f.status ===
  'open')`) scanned *all* findings, including ones deliberately left
  `open` by design (non-blocking minor/nit), instead of only
  fix-cycle-eligible critical/major ones. This directly contradicts the
  disposition rubric this skill's own SKILL.md documents (minor/nit-open
  is a valid WARN, not a FAIL) and caused one full run abort for a single
  minor finding (`orphaned-stale-chunks`). Fixed locally (`a64364a`); not
  yet upstreamed to `savviety-skills`.
- `plan-alignment`'s deviation IDs are free-form LLM-generated slugs with
  no stability guarantee across resumes — the *same* underlying
  deviation (my runtime bugfix commit) was named
  `run-plan-mjs-gatefindings-fix`, then `execute-plan-gatefindings-fix`,
  then `run-plan-mjs-gate-fix` across three resumes. `acceptRisk`'s
  exact-ID matching can't reliably target a moving name, which is what
  ultimately made the operator stop the run.
- Editing the plan's `## Closed Decisions` section busts the agent-call
  cache for *every* task (the block is embedded verbatim in every task's
  prompt), not just the task the edit was actually about. Three
  documentation-only plan amendments each triggered a full 11-task
  re-verification pass (15–55 minutes, ~1–1.9M tokens each) instead of a
  scoped re-check of the one task that needed it.
- Checkpoint's `/verify` finding (`kb ask` timeout) was independently
  rediscovered and re-fixed on two separate resumes
  (`7e6f6e1` then `8819480`) even though the branch already contained the
  first fix as an ancestor — a task-redo triggered by an unrelated
  Closed-Decision cache-bust likely rewrote `app.py`/`cli/main.py` widely
  enough to lose the earlier tuning rather than incrementally patching it.

### Requirements fit

- The plan (authored by this session via `/execute-prd`) named FastAPI,
  httpx, uvicorn, and watchdog as stack/implementation choices in Closed
  Decisions and task bodies, but never listed them in Task 1's dependency
  list — an authoring gap the AERS readiness rubric and `/validate-plan`
  don't currently check for.
- Three tasks' acceptance criteria named unit-test slices (`-k
  inbox_pipeline`, `-k log_pipeline`, `-k reindex`) that require their own
  test files, but the plan only declared the integration-test files in
  `write_scope` — the same class of gap, one level down (acceptance
  implies a file; `write_scope` didn't list it).

## What the process missed

- `orphaned-stale-chunks` (minor, `inbox_pipeline.py`) was never
  independently reverified after later fixes changed the exact
  delete/upsert ordering it complained about. Likely superseded, not
  confirmed — a real coverage gap in this manually-compiled report.
- Adversarial review never ran at all — it's gated behind a clean
  plan-alignment pass, which this run never achieved end-to-end in one
  resume.
- The e2e smoke test against live the GPU host is still manual-only and
  hasn't actually been run once by anyone.

## Recommendations

| # | Target | Type | Summary |
|---|---|---|---|
| 1 | `execute-plan-skill` | `tune-trigger` | Scope the agent-call cache key so a Closed-Decisions-only plan edit doesn't force every already-merged task to re-run; only tasks whose own body changed (or that never ran) should redo. |
| 2 | `execute-plan-skill` | `improve-prompt` | Give plan-alignment deviation IDs a deterministic derivation (e.g. hash of file + category) instead of free-form LLM slugs, so `acceptRisk` can reliably target a recurring deviation across resumes. |
| 3 | `execute-plan-skill` | `relax-gate` | Promote the local `gateFindings()` fix (only block on open critical/major, matching the disposition rubric) from this repo's `.claude/skills/execute-plan/workflows/run-plan.mjs` to the canonical `savviety-skills` copy before the next `--update` refresh overwrites it. |
| 4 | `execute-prd-skill` | `add-rubric-rule` | Cross-check that every library/framework named in a plan's Closed Decisions appears in the scaffolding task's actual dependency list (`pyproject.toml`/`package.json`/etc.) before `/validate-plan` passes. |
| 5 | `plan-template` | `add-rubric-rule` | Cross-check that every test file implied by an acceptance bullet (e.g. `-k <slice>`) has a matching `write_scope` entry on the same task. |
| 6 | `execute-plan-skill` | `tighten-gate` | When a task is redone after an unrelated plan edit busts its cache, instruct the agent to check for and preserve any review-fix commits already applied to the files it's about to rewrite, rather than reimplementing from the task spec alone. |

**Headline:** Editing plan Closed Decisions busts every task's agent-call
cache, turning a one-line documentation fix into a full re-verification
pass — this was the single biggest cost driver across this run's 5 resumes.
