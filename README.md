# kb — Personal Knowledge Base CLI + Server

[![CI](https://github.com/garyld1962/kb/actions/workflows/ci.yml/badge.svg)](https://github.com/garyld1962/kb/actions/workflows/ci.yml)

Capture notes and AI work logs from any machine into a local Obsidian
vault, and semantically search/ask over them via a small server backed by
Qdrant, Voyage AI (embedding + reranking), and an OpenAI-compatible LLM
server such as oMLX (answer generation).

This is **v1: capture + search**. MCP and nightly AI-log summarization are
a planned follow-on, not implemented yet (see
`docs/specs/2026-04-12-kb-design.md`, "Scope (v1 vs Follow-on)").

## How it fits together

- **`kb` CLI** — runs on any machine. `add`/`log`/`readlog` write directly
  to your local Obsidian vault and never touch the network. `status`/
  `search`/`ask` call **kb-server** over HTTP.
- **kb-server** — one instance on an always-on host, watching a vault's
  `Inbox/` and `AI-Daily-Log/` folders. It
  chunks, embeds (Voyage `voyage-4`), and indexes captures into Qdrant.
  `/search` and `/ask` over-fetch candidates from Qdrant and rerank them
  with Voyage `rerank-2.5`; `/ask` then answers through the LLM server.
- **Qdrant + LLM server** — existing services kb-server talks to. Defaults
  are `http://localhost:6333` and `http://localhost:8085/v1`; override with
  `KB_QDRANT_URL` and `KB_LLM_URL` when they run on another host. The
  answer model defaults to `Qwen3.6-35B-A3B-8bit`; override with
  `KB_LLM_MODEL` (use the id the server lists under `/v1/models`).
- **Voyage AI** — hosted embedding and rerank API. kb-server reads the key
  from `VOYAGE_API_KEY` (e.g. `op read 'op://Infra/Voyage API Key/credential'`).

## Requirements

- Python 3.12 (pinned via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/)
- A Voyage AI API key in `VOYAGE_API_KEY`
- A reachable Qdrant instance (and an OpenAI-compatible LLM server, e.g.
  oMLX, if you'll use `kb ask`)

## Install

**As a CLI tool (any machine that only captures/queries):**

```bash
uv tool install --from /path/to/kb kb
```

This installs the `kb` command. It does **not** install `kb-server` —
that's meant to run once, centrally (see below).

**For local development (this repo):**

```bash
git clone <repo> && cd kb
uv sync            # installs kb, kb-server, and dev deps (pytest, ruff)
```

`uv run kb ...` and `uv run kb-server` both work without a separate
install step.

## Configure

kb reads `~/.config/kb/config.yaml` (created automatically on first
`kb config set`, or write it by hand):

```yaml
vault_path: /path/to/your/obsidian/vault   # default: ~/obsidian
server_url: http://localhost:8090          # default: http://localhost:8090
machine_name: workstation-1                # default: this host's hostname
default_project: ""                       # optional shortcut for `kb log`
```

Any field can be viewed/set individually:

```bash
kb config              # print the fully-resolved config
kb config get vault_path
kb config set server_url http://localhost:8091
```

Missing file → kb runs on documented defaults, so capture works with zero
setup (just `vault_path` will land in `~/obsidian`, which you probably
want to change first).

## Running kb-server

**Locally (dev, or a single-machine setup):**

```bash
VOYAGE_API_KEY=... uv run kb-server
```

Runs on port `8090`, backed by whatever `vault_path` your config points
at (or `~/obsidian` by default). If port `8090` is already taken on your
machine, run uvicorn directly on another port instead:

```bash
uv run uvicorn kb.server.app:app --host 0.0.0.0 --port 8091
```

(and set `server_url` in your config to match).

**As a container** (`docker/docker-compose.yml`, mounts `/data/obsidian`;
set `KB_QDRANT_URL`/`KB_LLM_URL` if those services are on another host;
`VOYAGE_API_KEY` is required):

```bash
cd docker
VOYAGE_API_KEY=... docker compose up -d --build
curl http://localhost:8090/status   # {"status": "ok", ...}
```

`/status` also reports `qdrant_ok`, `llm_ok`, `voyage_ok` (key
configured), `embed_failed_backlog`, and `watchers_ok` — useful for confirming a fresh deploy actually wired
up correctly.

The container defaults to UID:GID `999:999` (the `kb` user baked into the
image). If `/data/obsidian` on the host isn't owned by/writable by that
UID, the Inbox → destination move will fail once a note is embedded (see
Troubleshooting). Set `KB_UID`/`KB_GID` to match the vault's actual owner:

```bash
KB_UID=$(id -u) KB_GID=$(id -g) docker compose up -d --build
```

## Using it

```bash
# Capture (offline, no server needed)
kb add "Karpathy's piece on LLM knowledge bases" --tags research,llm
kb add --stdin < notes.txt
kb log baker-street "Refactored tool architecture for parallel execution"
kb readlog baker-street --date 2026-07-08
kb readlog baker-street --range 2026-07-01:2026-07-08

# Query (needs kb-server reachable)
kb status
kb search "knowledge base architecture"
kb search "what did I do with X" --logs      # search AI-Daily-Log instead
kb ask "What did I decide about chunk sizing?"
```

If kb-server is unreachable, query commands print `kb-server unreachable.
Captures still work offline.` and exit non-zero — captures
still land in the vault either way.

## Command reference

| Command | Needs server | Notes |
|---|---|---|
| `kb add [content] [--stdin] [--title] [--tags a,b] [--project] [--destination] [--source-type url\|text\|file\|stdin\|mcp\|agent]` | No | Writes to vault `Inbox/` |
| `kb log <project> <message>` | No | Appends to today's `AI-Daily-Log/<project>/<date>.md`; auto-creates unknown projects with a warning |
| `kb readlog <project> [--date YYYY-MM-DD] [--range START:END]` | No | Reads directly from the vault |
| `kb status` | Yes | Vault/server health, incl. Qdrant/LLM reachability and Voyage key |
| `kb search <query> [--logs]` | Yes | Semantic search; `--logs` targets `kb_logs` instead of `kb_knowledge` |
| `kb ask <question>` | Yes | RAG answer with citations (uses `KB_LLM_MODEL`) |
| `kb config [get\|set] [key] [value]` | No | View/edit `~/.config/kb/config.yaml` |

## Vault layout kb-server expects

```
<vault_path>/
├── Inbox/              # kb add lands here; kb-server moves processed notes out
├── Knowledge/          # default destination for processed captures
├── AI-Daily-Log/<project>/<YYYY-MM-DD>.md
├── Archive/
└── Inbox/_errors/      # malformed/corrupted captures get quarantined here
```

## Migrating an index from the Ollama embedder

Vectors written before the switch to Voyage (`mxbai-embed-large`) are not
comparable with `voyage-4` vectors, and the reindex pass only re-embeds
documents whose content hash changed. On an existing deployment, drop both
collections once and let the next reindex pass rebuild them from the vault:

```bash
curl -X DELETE http://<qdrant>:6333/collections/kb_knowledge
curl -X DELETE http://<qdrant>:6333/collections/kb_logs
```

Then restart kb-server; `ensure_collections()` recreates them empty.

## Troubleshooting

- **`kb-server unreachable...`** — `server_url` doesn't point
  at a running kb-server, or it's down. Check `kb config get server_url`
  and `curl <server_url>/status`.
- **Capture never shows up in search** — check kb-server's logs for
  embed/index errors, and that Qdrant/Voyage are reachable from wherever
  kb-server runs. `kb status`'s `embed_failed_backlog` counts captures
  stuck retrying.
- **Capture shows up in search but the file is still sitting in `Inbox/`
  with `status: inbox`** (containerized deployment) — a permission
  mismatch between the container's user and `/data/obsidian`'s owner
  stopped the final move; the note gets quarantined to `Inbox/_errors/`
  with a `move to destination failed` reason once container permissions
  are fixed and it's reprocessed. Set `KB_UID`/`KB_GID` (see above) to
  match the vault's real owner and restart the container.
- **Full manual smoke test** (`kb add` → wait → `kb search` finds it),
  including exact preconditions: `docs/runbooks/kb-v1-smoke-test.md`.

## More detail

- Full design/architecture: `docs/specs/2026-04-12-kb-design.md`
- What was actually built and how: `docs/plans/2026-07-08-kb-v1-capture-search.md`, `docs/runs/2026-07-09-163242/`

## Tests

```bash
uv run pytest        # 118 tests (117 collected by default; 1 e2e deselected)
uv run ruff check .
```

## License

MIT
