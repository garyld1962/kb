# kb v1 smoke test runbook

Manual-only end-to-end check: `kb add` -> wait for kb-server's processor ->
`kb search` finds it. Not part of the automated default suite (see
`tests/e2e/test_smoke.py`, marked `@pytest.mark.e2e`).

## Preconditions

**the GPU host** (`localhost`, GPU host):

- Qdrant reachable at `http://localhost:6333`. It does not need to have
  `kb_knowledge`/`kb_logs` pre-created — kb-server's startup lifespan calls
  `ensure_collections()` before the watchers start, so a genuinely fresh
  Qdrant instance is bootstrapped automatically.
- Ollama reachable at `http://localhost:11434` with both models pulled:

  ```
  ollama pull mxbai-embed-large
  ollama pull qwen3.5:9b
  ```

**Dev the server** (the machine running kb-server for this test):

- kb-server running per `docker/docker-compose.yml`:

  ```
  cd ~/repos/kb
  docker compose -f docker/docker-compose.yml up -d --build
  ```

  Confirm it's up: `curl http://localhost:8090/status` returns
  `{"status": "ok"}`.
- `/data/obsidian` (the compose mount) is the vault kb-server watches.

**Test machine** (can be the same box as dev the server):

- `kb` installed (`uv tool install --from /path/to/kb kb`, or run via
  `uv run kb` from the repo).
- `~/.config/kb/config.yaml` set so `vault_path` points at the *same* vault
  kb-server watches (i.e. `/data/obsidian` if running the CLI directly on
  dev the server; a Syncthing-replicated copy otherwise) and `server_url`
  points at dev the server's kb-server, e.g.:

  ```yaml
  vault_path: /data/obsidian
  server_url: http://localhost:8090
  ```

## Run

```
cd ~/repos/kb
uv run pytest -m e2e
```

This runs `tests/e2e/test_smoke.py::test_add_then_search_finds_capture`,
which:

1. Writes a uniquely-tagged capture via `kb add` (`kb.cli.vault.add_note`).
2. Polls `POST /search` against the configured `server_url` for up to 60s.
3. Fails if the capture never shows up in search results.

## Manual variant

To watch the pipeline step by step instead of running the pytest wrapper:

```
uv run kb add "kb v1 smoke test $(date +%s)"
# wait a few seconds for kb-server's Inbox watcher to pick it up
uv run kb search "kb v1 smoke test"
```

Confirm the capture appears in the search results and that the source file
moved from `Inbox/` to its `destination` folder in the vault.

## Troubleshooting

- `kb-server on the server unreachable. Captures still work offline.` — kb
  can't reach `server_url`; check the container is up and the port mapping
  in `docker/docker-compose.yml`.
- Capture never appears in search — check kb-server's logs
  (`docker compose -f docker/docker-compose.yml logs -f kb-server`) for
  embed/index errors; confirm Qdrant and Ollama are reachable from inside
  the container (`docker compose exec kb-server curl http://localhost:6333`
  and `http://localhost:11434`).
