---
slug: kb-v1-capture-search
source_prd: docs/specs/2026-04-12-kb-design.md
intent: Build kb v1 — CLI capture, kb-server Inbox/Log pipelines, and a Search/Ask HTTP API over the local Obsidian vault, deferring MCP and nightly summarization to a follow-on phase.
type: feature
---

# kb v1 — Capture + Search Plan

**Source:** docs/specs/2026-04-12-kb-design.md

## Closed Decisions

- Stack: Python 3.12 (`uv`-managed), FastAPI (kb-server), Qdrant (vectors),
  Ollama `mxbai-embed-large` (embeddings) — no cloud dependencies.
- Persistence: the Obsidian vault (markdown files) is the sole source of
  truth; Qdrant is a derived index only, kb-server never owns data.
- Package layout: single uv package `kb`, submodules `kb.core` (shared
  models/config, single-owner contract), `kb.cli`, `kb.server` — MCP
  (follow-on) joins later as `kb.mcp`.
- v1 scope: CLI capture + kb-server (Inbox + Log pipelines) + Search/Ask
  HTTP API — MCP and the nightly summarization scheduler are out of scope
  for this plan (see PRD's "Scope (v1 vs Follow-on)").
- Log API scope: `POST /log/entry`/`GET /log/read` serve the MCP layer
  only and are out of scope here, since `kb log`/`kb readlog` write/read
  the vault directly (offline, no the server dependency, per PRD Components).
- No auth on kb-server's HTTP API for v1 (LAN-only, matches existing
  the GPU host/the server trust boundary — PRD Closed Decisions).
- Chunking: hardcoded constants — markdown headers (H1/H2/H3) as primary
  split points, 500-token / 50-token-overlap sliding window fallback
  within long sections, not configurable in v1.
- `_project-registry.md`: YAML frontmatter (`project`, `folder`, `status`,
  `created`) plus free markdown body — kb-server reads frontmatter only.
- Qdrant collections: `kb_knowledge`, `kb_logs`, both 1024-dim vectors.
- Test gating: unit tests always run (no network); integration tests
  (real Qdrant + mock Ollama) gated behind `RUN_INTEGRATION=1`, skipped
  by default; end-to-end smoke test is manual-only against live the GPU host,
  not part of the automated suite.
- Dependencies fix (post-Task-1): `fastapi`, `httpx`, `uvicorn` added to
  `pyproject.toml`'s `[project].dependencies` by the orchestrator after
  Task 9 correctly blocked on their absence rather than editing outside
  its write scope — Task 1 named FastAPI as the stack but omitted it from
  the dependency list.
- Dependencies fix (post-Task-9 review): `watchdog>=4` added to
  `pyproject.toml`'s `[project].dependencies` and a `kb-server =
  "kb.server.app:main"` entry added to `[project.scripts]`, since Task 6
  used `watchdog` without declaring it and Task 11's Dockerfile needs a
  console-script entry point for kb-server.

## Task 1: Project scaffolding + shared core models/config

```yaml
depends_on: []
write_scope:
  - pyproject.toml
  - .python-version
  - src/kb/__init__.py
  - src/kb/core/**
  - tests/unit/test_core_models.py
milestone_end: false
```

Scaffold the uv package: `.python-version` pinned to `3.12`; `pyproject.toml`
with `ruff` and `pytest` as dev deps, `[project.scripts]` entry point
`kb = "kb.cli.main:app"`, `src/kb/` layout (`src` package format). Configure
`[tool.pytest.ini_options]` with `markers = ["e2e: manual-only, requires
live the GPU host services"]` and `addopts = "-m 'not e2e'"` (Task 11 adds the
only test using this marker; registering it here keeps `pyproject.toml`
single-owner).

In `src/kb/core/models.py`, define Pydantic models for:
- Inbox note frontmatter: `id, created, source_machine, source_type
  (url|text|file|stdin|mcp|agent), source_ref, title, tags, project,
  destination, status (inbox|processed|embed_failed|error)`.
- Daily log entry / log file frontmatter: `project, date, status
  (active|closed), entries`.
- Qdrant payload: `doc_id, doc_path, doc_hash, doc_title, chunk_index,
  chunk_of, header_path, content, tags, project, source_type, source_ref,
  created_at, indexed_at`.
- Project registry entry: `project, folder, status, created`.

In `src/kb/core/config.py`, add a loader for `~/.config/kb/config.yaml`
(`vault_path, server_url, machine_name, default_project`), with a
documented default when the file is absent.

Write `tests/unit/test_core_models.py` covering frontmatter round-trip
(parse → serialize → re-parse equality) for each model above, and config
load with missing/present file.

**Acceptance:**
- `uv run pytest tests/unit/test_core_models.py` exits 0
- `uv run ruff check src/kb/core/` exits 0

## Task 2: Chunker

```yaml
depends_on: [1]
write_scope:
  - src/kb/server/chunker.py
  - tests/unit/test_chunker.py
milestone_end: false
```

Implement `chunk(content: str) -> list[Chunk]` in
`src/kb/server/chunker.py`: split on H1/H2/H3 headers first; within any
section longer than 500 tokens, fall back to a 500-token/50-overlap
sliding window (constants, not configurable). Each returned chunk carries
`chunk_index`, `chunk_of`, and `header_path` (e.g.
`"Key Insights > Beyond RAG"`).

**Acceptance:**
- `uv run pytest tests/unit/test_chunker.py` exits 0
- `uv run pytest tests/unit/test_chunker.py -k header_path` exits 0 and
  asserts `header_path == "Key Insights > Beyond RAG"` on a fixture doc
  with that nesting

## Task 3: Embed + index clients

```yaml
depends_on: [1]
write_scope:
  - src/kb/server/embed.py
  - src/kb/server/index.py
  - tests/unit/test_embed_index.py
milestone_end: false
```

`src/kb/server/embed.py`: client for Ollama `mxbai-embed-large` at a
configurable host (from `kb.core.config`, default `http://localhost:11434`),
returning 1024-dim vectors.

`src/kb/server/index.py`: Qdrant client wrapper exposing `upsert_chunks`,
`delete_by_doc_id`, `search(query_vector, collection, limit)`, and
collection bootstrap for `kb_knowledge` / `kb_logs` (1024-dim, created if
absent, idempotent).

Write `tests/unit/test_embed_index.py` with a mocked Ollama client and a
mocked Qdrant client, asserting: payload shape matches
`kb.core.models`'s Qdrant payload model exactly; `delete_by_doc_id`
removes all chunks for a `doc_id`; collection bootstrap is idempotent
(calling it twice does not error or duplicate).

**Acceptance:**
- `uv run pytest tests/unit/test_embed_index.py` exits 0
- `uv run ruff check src/kb/server/embed.py src/kb/server/index.py` exits 0

## Task 4: kb CLI (add, log, readlog, status, config)

```yaml
depends_on: [1]
write_scope:
  - src/kb/cli/**
  - tests/unit/test_cli_vault.py
milestone_end: false
```

Implement the CLI commands from the PRD's Components table:
- `kb add [text|url|file]` / `kb add --stdin` — write a new Inbox note
  (using `kb.core.models`) to `<vault_path>/Inbox/`. No network call.
- `kb log <project> <message>` — append an entry to
  `AI-Daily-Log/<project>/<YYYY-MM-DD>.md`, creating the file/frontmatter
  if absent. No network call.
- `kb readlog <project> [--date=... --range=...]` — read entries directly
  from the vault (no server round-trip).
- `kb status` / `kb search` / `kb ask` — HTTP clients against
  `server_url` (implemented as stubs here; wired to the real API in
  Task 9). On connection failure, print `"kb-server on the server
  unreachable. Captures still work offline."` and exit non-zero.
- `kb config` — view/edit `~/.config/kb/config.yaml`.

Write `tests/unit/test_cli_vault.py` covering: `kb add` writes a
well-formed Inbox note; `kb log` on an unknown project auto-creates the
project folder and prints a warning; `kb readlog` returns entries for a
date range; `kb status`/`kb search` against an unreachable server exits
non-zero with the exact message above.

**Acceptance:**
- `uv run pytest tests/unit/test_cli_vault.py` exits 0
- `uv run ruff check src/kb/cli/` exits 0

## Task 5: Project registry

```yaml
depends_on: [1]
write_scope:
  - src/kb/server/registry.py
  - tests/unit/test_registry.py
milestone_end: false
```

`src/kb/server/registry.py`: read/write `_project-registry.md` (YAML
frontmatter: `project → folder` mapping, `status`, `created`, plus a free
markdown body). Expose `lookup(project) -> folder | None` and
`register(project, folder)` which appends a stub entry when the project
is unknown, matching the CLI's auto-create/warn flow.

Write `tests/unit/test_registry.py` covering: lookup of a known project;
`register` on an unknown project appends a well-formed stub entry and is
idempotent if called twice for the same project.

**Acceptance:**
- `uv run pytest tests/unit/test_registry.py` exits 0
- `uv run ruff check src/kb/server/registry.py` exits 0

## Task 6: Inbox pipeline

```yaml
depends_on: [2, 3, 5]
write_scope:
  - src/kb/server/inbox_pipeline.py
  - src/kb/server/watcher.py
  - tests/integration/test_inbox_pipeline.py
  - tests/unit/test_inbox_pipeline_unit.py
  - tests/unit/test_watcher_unit.py
milestone_end: false
```

`src/kb/server/watcher.py`: `watchdog`-based filesystem watcher on
`Inbox/`, dispatching new/changed files to the inbox pipeline handler.

`src/kb/server/inbox_pipeline.py`: on a new Inbox file — parse
frontmatter, skip if `status != "inbox"`, compute `doc_hash` (SHA-256),
clean/normalize content, chunk (Task 2), embed each chunk (Task 3), upsert
into `kb_knowledge` (Task 3), move `Inbox/note.md` → `<destination>/note.md`,
update frontmatter (`status: processed, processed_at, chunk_count`).
Malformed frontmatter → move to `Inbox/_errors/` with an error log, do not
raise past the handler. Unknown `project` → auto-create folder via the
registry (Task 5) and warn.

Write `tests/integration/test_inbox_pipeline.py`, gated behind
`RUN_INTEGRATION=1` (skipped otherwise), using a real test Qdrant instance
and a mocked Ollama embed client: assert a sample inbox note ends up
chunked/embedded/indexed in `kb_knowledge` with correct payload, the file
moves to its destination, and frontmatter is updated correctly.

**Acceptance:**
- `uv run pytest tests/unit -k inbox_pipeline` exits 0 (pure-logic unit
  slice: frontmatter parsing, error routing, destination resolution)
- `RUN_INTEGRATION=1 uv run pytest tests/integration/test_inbox_pipeline.py`
  exits 0 when a local test Qdrant is reachable

## Task 7: Log pipeline

```yaml
depends_on: [3, 5]
write_scope:
  - src/kb/server/log_pipeline.py
  - tests/integration/test_log_pipeline.py
  - tests/unit/test_log_pipeline_delta.py
milestone_end: false
```

`src/kb/server/log_pipeline.py`: on a change to
`AI-Daily-Log/<project>/<date>.md`, chunk only the entries added since the
last index (delta), embed, and upsert into `kb_logs`. Reuses the chunker
(Task 2) and embed/index clients (Task 3).

Write `tests/integration/test_log_pipeline.py`, gated behind
`RUN_INTEGRATION=1`, using a real test Qdrant instance and a mocked Ollama
embed client: appending two entries to a log file results in exactly the
delta being indexed into `kb_logs` (not a full re-index of prior entries).

**Acceptance:**
- `uv run pytest tests/unit -k log_pipeline` exits 0 (delta-detection
  logic as a pure-logic unit slice)
- `RUN_INTEGRATION=1 uv run pytest tests/integration/test_log_pipeline.py`
  exits 0 when a local test Qdrant is reachable

## Task 8: Re-indexing

```yaml
depends_on: [6, 7]
write_scope:
  - src/kb/server/reindex.py
  - tests/integration/test_reindex.py
  - tests/unit/test_reindex_unit.py
milestone_end: false
```

`src/kb/server/reindex.py`: periodic scan (every 30 min, via a scheduler
hook) over `Knowledge/`, `Archive/`, and `AI-Daily-Log/`. Hash-based change
detection: if `doc_hash` differs from last-indexed, delete old chunks (by
`doc_id`) and re-chunk/re-embed/re-index. Rename detection: same `doc_id`,
new `doc_path` → update payloads in place without re-embedding. Deletion:
file gone → purge chunks by `doc_id`.

Write `tests/integration/test_reindex.py`, gated behind `RUN_INTEGRATION=1`:
editing a file's content triggers re-index with old chunks purged; renaming
a file updates `doc_path` in existing payloads without a new embed call;
deleting a file purges its chunks.

**Acceptance:**
- `uv run pytest tests/unit -k reindex` exits 0 (hash-diff/rename/deletion
  decision logic as a pure-logic unit slice)
- `RUN_INTEGRATION=1 uv run pytest tests/integration/test_reindex.py`
  exits 0 when a local test Qdrant is reachable

## Task 9: Search + Ask HTTP API

```yaml
depends_on: [3, 4]
write_scope:
  - src/kb/server/app.py
  - src/kb/cli/client.py
  - tests/unit/test_search_ask.py
milestone_end: true
```

`src/kb/server/app.py`: FastAPI app exposing `POST /search` (semantic
search over `kb_knowledge` by default, `collection` param to target
`kb_logs`) and `POST /ask` (retrieve top-k chunks, synthesize an answer via
Ollama `qwen3.5:9b`, return with citations). No auth (closed decision).

`src/kb/cli/client.py`: HTTP client used by `kb search`/`kb ask`/`kb status`
(Task 4's stubs) to call the real API; on connection failure, surfaces the
exact message from Task 4 and a non-zero exit.

Write `tests/unit/test_search_ask.py` with FastAPI's `TestClient` and a
mocked embed/index/LLM layer: `POST /search` happy path and malformed-input
4xx; `POST /ask` happy path returns an answer plus citations, and a
malformed request returns 4xx.

**Acceptance:**
- `uv run pytest tests/unit/test_search_ask.py` exits 0
- `uv run ruff check src/kb/server/app.py src/kb/cli/client.py` exits 0

This task ends the first milestone: capture, indexing, and query are all
wired end-to-end at the unit/API layer.

## Task 10: Error-handling hardening

```yaml
depends_on: [6, 7, 8, 9]
write_scope:
  - src/kb/server/inbox_pipeline.py
  - src/kb/server/log_pipeline.py
  - src/kb/server/embed.py
  - src/kb/server/index.py
  - tests/unit/test_error_handling.py
milestone_end: false
```

Add the retry/backoff and quarantine behavior the PRD's Error Handling
section requires but the earlier tasks stub: Ollama/Qdrant unreachable →
exponential backoff retry, then `status: embed_failed` and leave the file
in place for later retry; corrupted/binary content → skip, log, move to
`Inbox/_errors/`; partial embedding is safely retryable because re-index
(Task 8) is `doc_id`-idempotent.

Write `tests/unit/test_error_handling.py` covering: repeated
Ollama-unreachable errors eventually mark `embed_failed` without raising;
corrupted file content is quarantined, not partially indexed; a partial
embed followed by a retry does not duplicate chunks (asserted against the
mocked index client's upsert calls).

**Acceptance:**
- `uv run pytest tests/unit/test_error_handling.py` exits 0
- `uv run pytest tests/unit` exits 0 (no regressions in prior unit suites)

## Task 11: Docker packaging + e2e smoke test

```yaml
depends_on: [9, 10]
write_scope:
  - docker/Dockerfile
  - docker/docker-compose.yml
  - tests/e2e/test_smoke.py
  - docs/runbooks/kb-v1-smoke-test.md
milestone_end: true
```

`docker/Dockerfile` + `docker/docker-compose.yml`: package kb-server as
described in Deployment (port `8090`, mounts `/data/obsidian` read-write,
connects to Qdrant/Ollama on the GPU host over LAN).

`tests/e2e/test_smoke.py`: the PRD's smoke test (`kb add` → wait for
processor → `kb search` finds it), marked `@pytest.mark.e2e` and
registered in `pyproject.toml` (`markers = ["e2e: manual-only, requires
live the GPU host services"]`) with default `addopts = "-m 'not e2e'"` so it
is deselected by the default `uv run pytest` invocation and only runs via
explicit `uv run pytest -m e2e`.

`docs/runbooks/kb-v1-smoke-test.md`: exact manual steps to run the smoke
test against dev the server, including the the GPU host service preconditions
(Qdrant reachable, Ollama with `mxbai-embed-large` and `qwen3.5:9b`
pulled).

**Acceptance:**
- `docker build -f docker/Dockerfile .` exits 0
- `uv run pytest` exits 0 and `uv run pytest --collect-only -q | grep -qv test_smoke` exits 0 (e2e test deselected by default `addopts`)
- `uv run pytest -m e2e --collect-only -q | grep -q test_smoke` exits 0 (e2e test discoverable when explicitly requested)
- `uv run ruff check .` exits 0

This is the final milestone: v1's Definition of Done is met.
