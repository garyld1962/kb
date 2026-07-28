# Execution Report — kb v1 Capture + Search (run 2)

- **Plan:** `docs/plans/2026-07-08-kb-v1-capture-search.md`
- **Plan-SHA (final):** `9b97998a96157e7cb2cb43761db7ae3f8d449296275be1fc04f1729d9938cd1d`
- **Base-SHA:** `081d986d4ed7525186f7002ef7c71ca6d3d18ec4`
- **Branch:** `feat/kb-v1-capture-search`
- **Run ID:** `wf_f762ffeb-80d` (5 resumes: `wpuryoc85`, `wem4f59gg`, `wx31kk3b5`, `w6h7foxz3`, `w33amkw53`, terminated during a 6th, `wb12s45ga`)
- **Verdict:** **WARN** (manually assessed — see "How this report was produced")
- **Compiled:** manually, by the operating session, after the operator stopped the automated run before it reached its Report phase

## How this report was produced

The automated `/execute-plan` runtime never completed its own Report
phase — it was manually terminated (`TaskStop`) during its 6th resume
after 5 prior resumes each spent 15–55 minutes chasing plan-authoring
gaps and one runtime bug. This report is assembled from the run's
`journal.jsonl` and `git log 081d986..HEAD`, not from a runtime-generated
`state` object. Every task and finding below is corroborated by an actual
commit on `feat/kb-v1-capture-search`; nothing here is inferred without
a commit to point at.

## Tasks

| # | Title | Status | Commit |
|---|---|---|---|
| 1 | Project scaffolding + shared core models/config | done | c6bbcfe |
| 2 | Chunker | done | f75f38f (final reverify; earlier: 53fdb37/559d7a6) |
| 3 | Embed + index clients | done | fdb905d (merged 5d071c3) |
| 4 | kb CLI (add, log, readlog, status, config) | done | a7b0405 (merged 8fe5d07; earlier: 33b8c38/627b604) |
| 5 | Project registry | done | 58d0ffc (merged 339360a) |
| 6 | Inbox pipeline | done | 46121d2 (merged bd4644b) |
| 7 | Log pipeline | done | 69a2587 (merged 7addac0) |
| 8 | Re-indexing | done | 7487c6a |
| 9 | Search + Ask HTTP API | done | 589b8ff (merged a41e61b) — **milestone 1** |
| 10 | Error-handling hardening | done | ec25107 |
| 11 | Docker packaging + e2e smoke test | done | 10168a9 — **milestone 2** |

All 11 tasks implemented and merged. Several tasks were re-verified
across resumes after plan edits (Closed Decisions changes bust the
runtime's agent cache for every task); where a task's final commit
differs from its first, the table shows the final one.

## Findings — milestone gate (after task 9)

| ID | Sev | File | Status |
|---|---|---|---|
| logwatcher-registry-crash | major | log watcher | fixed (8140570) |
| log-pipeline-no-error-boundary | major | log_pipeline.py | fixed (6830f37) |
| status-blind-to-dead-watcher | major | app.py | fixed (7a0473f) |
| no-runtime-wiring | critical | app.py | fixed (cb675af) |
| missing-status-route | major | app.py | fixed (919023a) |
| dead-kb-client | major | cli/main.py | fixed (422cd61) |
| path-traversal-destination | critical | inbox_pipeline.py | fixed (386ba73) |
| unguarded-quarantine | major | inbox_pipeline.py | fixed (2fbf497) |
| reindex-delete-before-upsert | major | reindex.py | fixed (da452ed) |
| inbox-non-transient-error-stuck-forever | major | inbox_pipeline.py | fixed (d4e13eb) |
| orphaned-stale-chunks | minor | inbox_pipeline.py | open, non-blocking — see note below |

**Note on `orphaned-stale-chunks`:** first surfaced in an earlier resume
(before the `gateFindings` runtime bug fix, see Postmortem) and never
independently reverified after `reindex-delete-before-upsert` and
`reindex-doc-hash-mismatch-wastes-reembed` changed the same delete/upsert
ordering it complained about. Very likely superseded, not confirmed —
flagged in the postmortem as a residual verification gap.

## Findings — milestone gate (after task 11)

| ID | Sev | File | Status |
|---|---|---|---|
| reindex-scheduler-never-started | critical | app.py | fixed (7ee3fd1) |
| kb-knowledge-collection-never-bootstrapped | critical | app.py | fixed (c1ae589) |
| smoke-test-fails-on-fresh-deploy | major | docs/runbook | fixed (5904ac0) |
| kb-1 (file-path probe guard) | major | source detection | fixed (e2f9ec9) |
| kb-2 (InboxEventHandler echo) | major | inbox watcher | fixed (9dc8ea6) |
| kb-3 (registry register() lock) | major | registry.py | fixed (3d74221) |
| log-pipeline-non-transient-crash | major | log_pipeline.py | fixed (0894f6b) |
| reindex-doc-hash-mismatch-wastes-reembed | major | reindex.py | fixed (428a3b2) |
| kb-log-concurrent-append-lost-update | major | cli/vault.py | fixed (8a9290f) |
| reindex-stale-chunk-delete-failure-unrecoverable | major | reindex.py | fixed (64b0d78) |

## Findings — checkpoint (`/verify`)

| ID | Sev | File | Status |
|---|---|---|---|
| verify-kb-ask-timeout | major | app.py / cli | fixed (7e6f6e1) |
| verify-ask-timeout | major | app.py / cli | fixed (8819480) |

Both findings describe the same underlying issue (`kb ask`'s end-to-end
timeout budget too tight against observed LLM generation latency),
rediscovered across two different resumes and fixed both times — a sign
`/verify`'s checkpoint call is not being cache-replayed identically to
task calls (see postmortem recommendation).

## Findings — PR-boundary gate (`domain-review full`)

| ID | Sev | File | Status |
|---|---|---|---|
| embed-failed-notes-stuck-forever | major | reindex.py | fixed (b3bdcfb) |
| status-endpoint-fake-health-check | major | app.py | fixed (45cadb9) |
| http-error-detail-discarded | minor | app.py | fixed (80a37c7) |
| log-entry-key-collision-drops-content | major | vault.py | fixed (74059bc) |
| concurrency-log-doc-hash-race | major | reindex.py / log_pipeline.py | fixed (4a4b945) |
| correctness-add-file-autodetect-collision | minor | cli/main.py | fixed (f6c517f) |
| resilience-reindex-delete-before-reembed | major | reindex.py | fixed (90f8c82) |
| concurrency-inbox-watcher-reindex-race | major | inbox_pipeline.py / reindex.py | fixed (e9f48a2) |
| cli-log-project-path-traversal | critical | cli/vault.py | fixed (5242b4d) |
| cli-vault-unterminated-frontmatter-crash | major | cli/vault.py | fixed (e149f23) |
| log-entry-heading-injection | major | vault.py | fixed (015a6bd) |
| inbox-destination-collision-race | major | inbox_pipeline.py | fixed (df3753a) |
| missing-negative-tests-cli-vault | minor | tests/unit/test_cli_vault.py | fixed (37d941e) |

## Deviations

| ID | Category | Status |
|---|---|---|
| run-plan-mjs-gate-fix | unmatched | **accepted-risk (manual)** — orchestrator tooling fix, not a plan deliverable; see commit `a64364a` |
| uv-lock-added | lockfile | disagree-with-evidence (auto-accepted) |

Earlier deviations, resolved during the run by amending the plan rather
than accepting risk (so they don't appear in the final list above):

| ID | Resolution |
|---|---|
| pyproject-watchdog-and-kbserver-script-undocumented | Recorded as a Closed Decision (`19bca30`) |
| test-inbox-pipeline-unit / test-log-pipeline-delta / test-reindex-unit / test-watcher-unit | Added to owning tasks' `write_scope` (`23c4c9b`) |

## Adversarial review

**Not run.** The automated pipeline never got past the plan-alignment
gate cleanly enough to reach the adversarial-review step (it's the last
gate before the Report phase). Given the diff touches 20+ files, the
`adversarial: auto` threshold would fire. **Recommended before merge.**

## Manual fixes applied outside the task/review-fix loop

| Commit | Summary |
|---|---|
| `051db5b` | Added `fastapi`/`httpx`/`uvicorn` to `pyproject.toml` — Task 1 named FastAPI as the stack but never declared it as a dependency, blocking Task 9 |
| `a64364a` | Fixed a real bug in `run-plan.mjs`'s `gateFindings()`: it threw on *any* open finding regardless of severity, instead of only `critical`/`major` ones — see postmortem |

## Verification status

- `uv run pytest` — 110 passed, 6 skipped (integration, `RUN_INTEGRATION` gated), 1 deselected (e2e, manual-only)
- `uv run ruff check .` — clean
- `docker build -f docker/Dockerfile .` — succeeds
- End-to-end smoke test against live the GPU host — **not yet run manually**, per plan's Definition of Done this is a manual step
