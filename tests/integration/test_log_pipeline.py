"""Integration test for the log pipeline against a real test Qdrant.

Gated behind ``RUN_INTEGRATION=1`` (skipped otherwise). Ollama is mocked; the
Qdrant instance is real (``QDRANT_URL``, default ``http://localhost:6333``).

Proves the core delta guarantee: appending two entries to an already-indexed
log file indexes **exactly the two new entries** into ``kb_logs`` — the prior
entry is neither re-embedded nor re-upserted.

The test isolates itself by a unique per-run ``doc_id`` (unique project name)
and purges those points in teardown, so it never disturbs real ``kb_logs``
data on the shared test collection.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from kb.server.embed import EMBED_DIM
from kb.server.index import LOG_COLLECTION, IndexClient
from kb.server.log_pipeline import LogPipeline, log_doc_id

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="integration test; set RUN_INTEGRATION=1 with a reachable test Qdrant",
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


class FakeEmbedClient:
    """Deterministic 1024-dim embeddings; records every call for delta asserts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # A stable, slightly varied vector keeps points valid without a network
        # round-trip; exact geometry is irrelevant to a delta-count assertion.
        seed = (len(text) % 97) / 100.0
        return [0.001 + seed] * EMBED_DIM


def _render_log(project: str, day: str, entries: list[tuple[str, str, str]]) -> str:
    lines = [
        "---",
        f"project: {project}",
        f"date: {day}",
        "status: active",
        f"entries: {len(entries)}",
        "---",
        "",
        f"# {project} — {day}",
    ]
    for time, title, body in entries:
        lines += ["", f"## {time} — {title}", "*machine: testhost*", "", body, "", "---"]
    return "\n".join(lines) + "\n"


def _count_chunks_for_doc(index: IndexClient, doc_id: str) -> list[dict]:
    """Return payloads in ``kb_logs`` belonging to ``doc_id``.

    ``IndexClient.search`` returns nearest points without a server-side doc
    filter, so over-fetch and filter client-side. The unique per-run project
    keeps the result set tiny and unambiguous.
    """
    hits = index.search([0.001] * EMBED_DIM, LOG_COLLECTION, limit=1000)
    return [h.payload for h in hits if h.payload.get("doc_id") == doc_id]


def test_appending_entries_indexes_only_the_delta(tmp_path: Path):
    project = f"kb-logtest-{uuid.uuid4().hex[:12]}"
    day = "2026-07-08"
    doc_id = log_doc_id(project, day)

    embed = FakeEmbedClient()
    index = IndexClient(url=QDRANT_URL)
    pipeline = LogPipeline(tmp_path, embed, index)

    log_file = tmp_path / "AI-Daily-Log" / project / f"{day}.md"
    log_file.parent.mkdir(parents=True)

    try:
        # First index: single entry.
        log_file.write_text(
            _render_log(project, day, [("09:00", "Alpha", "vector search notes")])
        )
        first = pipeline.handle(log_file)
        assert first.entries_indexed == 1
        assert first.chunks_indexed == 1
        assert len(embed.calls) == 1

        stored = _count_chunks_for_doc(index, doc_id)
        assert len(stored) == 1

        # Append two entries -> only the delta (2) is embedded and upserted.
        log_file.write_text(
            _render_log(
                project,
                day,
                [
                    ("09:00", "Alpha", "vector search notes"),
                    ("10:00", "Beta", "qdrant upsert idempotency"),
                    ("11:00", "Gamma", "ollama embedding dims"),
                ],
            )
        )
        second = pipeline.handle(log_file)
        assert second.entries_indexed == 2
        assert second.chunks_indexed == 2
        # Prior entry was NOT re-embedded: exactly two new embed calls.
        assert len(embed.calls) == 3

        stored = _count_chunks_for_doc(index, doc_id)
        assert len(stored) == 3
        assert {p["chunk_index"] for p in stored} == {0, 1, 2}
        contents = sorted(p["content"] for p in stored)
        assert any("vector search" in c for c in contents)
        assert any("qdrant upsert" in c for c in contents)
        assert any("ollama embedding" in c for c in contents)

        # Re-running with no change is a no-op: no additional embeds/points.
        third = pipeline.handle(log_file)
        assert third.entries_indexed == 0
        assert third.chunks_indexed == 0
        assert len(embed.calls) == 3
        assert len(_count_chunks_for_doc(index, doc_id)) == 3
    finally:
        index.delete_by_doc_id(LOG_COLLECTION, doc_id)
