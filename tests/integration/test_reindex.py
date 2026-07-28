"""Integration test for the periodic reindex pass against a real test Qdrant.

Gated behind ``RUN_INTEGRATION=1`` (skipped otherwise) and additionally
skipped when no Qdrant is reachable at ``QDRANT_URL`` (default
``http://localhost:6333``). Ollama is mocked; Qdrant is real.

Covers the three re-indexing scenarios from the design's Re-indexing section:
editing a file's content triggers a re-index with old chunks purged; renaming
a file updates ``doc_path`` in the existing payloads without a new embed
call; deleting a file purges its chunks.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from kb.server.embed import EMBED_DIM
from kb.server.index import KNOWLEDGE_COLLECTION, IndexClient
from kb.server.reindex import QdrantStateReader, reindex_once

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="integration tests are gated behind RUN_INTEGRATION=1",
)


def _qdrant_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{QDRANT_URL}/collections", timeout=2)
    except (urllib.error.URLError, OSError):
        return False
    return True


@pytest.fixture(autouse=True)
def _require_qdrant():
    if not _qdrant_reachable():
        pytest.skip(f"no Qdrant reachable at {QDRANT_URL}")


class FakeEmbedClient:
    """Deterministic 1024-dim embeddings; records every call for re-embed asserts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        seed = (len(text) % 97) / 100.0
        return [0.001 + seed] * EMBED_DIM


def _note_text(doc_id: str, body: str) -> str:
    return (
        "---\n"
        f"id: {doc_id}\n"
        "created: 2026-04-12T14:32:00-04:00\n"
        "source_machine: workstation-1\n"
        "source_type: text\n"
        "source_ref: null\n"
        "title: Reindex test note\n"
        "tags: [research]\n"
        'project: ""\n'
        "destination: Knowledge/research\n"
        "status: processed\n"
        "---\n"
        f"\n{body}"
    )


def test_content_edit_purges_old_chunks_and_reembeds(tmp_path: Path):
    doc_id = uuid.uuid4().hex
    note = tmp_path / "Knowledge" / "research" / f"{doc_id}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(_note_text(doc_id, "# Original\n\nOriginal content about search.\n"))

    embed = FakeEmbedClient()
    index = IndexClient(url=QDRANT_URL)
    reader = QdrantStateReader(url=QDRANT_URL)

    try:
        first = reindex_once(tmp_path, embed, index, reader=reader)
        assert first.indexed == 1
        first_chunks = reader.doc_chunks(KNOWLEDGE_COLLECTION, doc_id)
        assert len(first_chunks) >= 1
        assert all(p["content"] == "Original content about search." for p, _ in first_chunks)
        calls_after_first = len(embed.calls)

        # Edit the file's content in place (same path, new doc_hash).
        note.write_text(
            _note_text(doc_id, "# Original\n\nCompletely different content about chunking.\n")
        )

        second = reindex_once(tmp_path, embed, index, reader=reader)
        assert second.indexed == 1
        assert len(embed.calls) > calls_after_first, "changed content must be re-embedded"

        second_chunks = reader.doc_chunks(KNOWLEDGE_COLLECTION, doc_id)
        contents = {p["content"] for p, _ in second_chunks}
        assert contents == {"Completely different content about chunking."}
        assert "Original content about search." not in contents
    finally:
        index.delete_by_doc_id(KNOWLEDGE_COLLECTION, doc_id)


def test_rename_updates_doc_path_without_reembed(tmp_path: Path):
    doc_id = uuid.uuid4().hex
    old_path = tmp_path / "Knowledge" / "research" / f"{doc_id}.md"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text(_note_text(doc_id, "# Stable\n\nContent that never changes.\n"))

    embed = FakeEmbedClient()
    index = IndexClient(url=QDRANT_URL)
    reader = QdrantStateReader(url=QDRANT_URL)

    try:
        first = reindex_once(tmp_path, embed, index, reader=reader)
        assert first.indexed == 1
        calls_after_first = len(embed.calls)
        stored_vectors = {
            tuple(vector) for _, vector in reader.doc_chunks(KNOWLEDGE_COLLECTION, doc_id)
        }

        # Rename: same content, new location.
        new_path = tmp_path / "Knowledge" / "research" / f"{doc_id}-renamed.md"
        old_path.rename(new_path)

        second = reindex_once(tmp_path, embed, index, reader=reader)
        assert second.renamed == 1
        assert len(embed.calls) == calls_after_first, "rename must not re-embed"

        chunks = reader.doc_chunks(KNOWLEDGE_COLLECTION, doc_id)
        assert chunks, "chunks must still exist after rename"
        assert all(
            p["doc_path"] == f"Knowledge/research/{doc_id}-renamed.md" for p, _ in chunks
        )
        # Vectors are untouched: reused verbatim from before the rename.
        assert {tuple(v) for _, v in chunks} == stored_vectors
    finally:
        index.delete_by_doc_id(KNOWLEDGE_COLLECTION, doc_id)


def test_deletion_purges_chunks(tmp_path: Path):
    doc_id = uuid.uuid4().hex
    note = tmp_path / "Knowledge" / "research" / f"{doc_id}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(_note_text(doc_id, "# Gone soon\n\nThis file will be deleted.\n"))

    embed = FakeEmbedClient()
    index = IndexClient(url=QDRANT_URL)
    reader = QdrantStateReader(url=QDRANT_URL)

    try:
        first = reindex_once(tmp_path, embed, index, reader=reader)
        assert first.indexed == 1
        assert reader.doc_chunks(KNOWLEDGE_COLLECTION, doc_id)

        note.unlink()

        second = reindex_once(tmp_path, embed, index, reader=reader)
        assert second.purged == 1
        assert reader.doc_chunks(KNOWLEDGE_COLLECTION, doc_id) == []
    finally:
        index.delete_by_doc_id(KNOWLEDGE_COLLECTION, doc_id)
