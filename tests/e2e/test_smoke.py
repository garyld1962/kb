"""End-to-end smoke test: ``kb add`` -> wait for the processor -> ``kb search``
finds it.

Manual-only (``@pytest.mark.e2e``): requires a live kb-server on the dev host
watching a vault, backed by real Qdrant plus Voyage (embedding/rerank) and oMLX (answers). See
docs/runbooks/kb-v1-smoke-test.md for exact preconditions and run
instructions. Deselected by default (``addopts = "-m 'not e2e'"`` in
pyproject.toml); run explicitly with ``uv run pytest -m e2e``.
"""

from __future__ import annotations

import time
import uuid

import pytest

from kb.cli import vault
from kb.cli.client import KbClient
from kb.core.config import load_config

# How long to wait for kb-server's Inbox watcher + pipeline to process and
# index the capture before giving up.
POLL_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2


@pytest.mark.e2e
def test_add_then_search_finds_capture() -> None:
    config = load_config()
    marker = f"kb-e2e-smoke-{uuid.uuid4().hex[:12]}"
    vault.add_note(config, f"Smoke test capture {marker}", title=marker)

    client = KbClient(config.server_url)
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    found = False
    while time.monotonic() < deadline:
        result = client.search(marker)
        hits = result.get("results", [])
        if any(marker in hit.get("content", "") for hit in hits):
            found = True
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    assert found, (
        f"kb search did not find capture '{marker}' within "
        f"{POLL_TIMEOUT_SECONDS}s of kb add"
    )
