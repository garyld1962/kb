"""Integration test for the inbox pipeline against a real Qdrant instance.

Gated behind ``RUN_INTEGRATION=1`` (skipped otherwise) and additionally
skipped when no Qdrant is reachable at ``QDRANT_URL`` (default
``http://localhost:6333``). Ollama is mocked; Qdrant is real. Each test uses a
unique ``doc_id`` and purges its own points on teardown so the shared
``kb_knowledge`` collection is left clean.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from kb.server.embed import EMBED_DIM
from kb.server.index import KNOWLEDGE_COLLECTION, IndexClient
from kb.server.inbox_pipeline import process_inbox_file
from kb.server.registry import lookup

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
    """Mock Ollama embed client returning a fixed 1024-dim vector."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.0] * (EMBED_DIM - 1) + [1.0]


def _scroll_by_doc_id(collection: str, doc_id: str) -> list[dict]:
    body = json.dumps(
        {
            "filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
            "with_payload": True,
            "limit": 100,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["result"]["points"]


def _note_text(doc_id: str, *, project: str = "", destination: str = "Knowledge/research") -> str:
    return (
        "---\n"
        f"id: {doc_id}\n"
        "created: 2026-04-12T14:32:00-04:00\n"
        "source_machine: workstation-1\n"
        "source_type: text\n"
        "source_ref: null\n"
        "title: Karpathy on knowledge bases\n"
        "tags: [research, llm, rag]\n"
        f'project: "{project}"\n'
        f"destination: {destination}\n"
        "status: inbox\n"
        "---\n"
        "\n"
        "# Key Insights\n\n"
        "Knowledge bases benefit from structure.\n\n"
        "## Beyond RAG\n\n"
        "Chunking with header context improves retrieval quality.\n"
    )


def _write_note(vault: Path, name: str, text: str) -> Path:
    inbox = vault / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(text)
    return path


def test_inbox_note_chunked_embedded_indexed_and_moved(tmp_path):
    doc_id = uuid.uuid4().hex
    index = IndexClient(url=QDRANT_URL)
    index.ensure_collections()
    embed = FakeEmbedClient()
    note = _write_note(tmp_path, "karpathy-kb.md", _note_text(doc_id))

    try:
        result = process_inbox_file(
            note,
            vault_path=tmp_path,
            embed_client=embed,
            index_client=index,
        )

        assert result.status == "processed"
        assert result.chunk_count >= 1
        assert embed.texts, "embed client was never called"

        # File moved to its destination and gone from Inbox.
        moved = tmp_path / "Knowledge" / "research" / "karpathy-kb.md"
        assert moved.exists()
        assert not note.exists()

        # Frontmatter rewritten in place.
        text = moved.read_text()
        assert "status: processed" in text
        assert f"chunk_count: {result.chunk_count}" in text
        assert "processed_at:" in text

        # Points landed in kb_knowledge with a correct payload.
        points = _scroll_by_doc_id(KNOWLEDGE_COLLECTION, doc_id)
        assert len(points) == result.chunk_count
        payload = points[0]["payload"]
        assert payload["doc_id"] == doc_id
        assert payload["doc_hash"].startswith("sha256:")
        assert payload["doc_path"] == "Knowledge/research/karpathy-kb.md"
        assert payload["doc_title"] == "Karpathy on knowledge bases"
        assert payload["source_type"] == "text"
        assert payload["chunk_of"] == result.chunk_count
        assert payload["content"]
        assert payload["header_path"]
        header_paths = {p["payload"]["header_path"] for p in points}
        assert "Key Insights > Beyond RAG" in header_paths
    finally:
        index.delete_by_doc_id(KNOWLEDGE_COLLECTION, doc_id)


def test_unknown_project_auto_registered(tmp_path):
    doc_id = uuid.uuid4().hex
    index = IndexClient(url=QDRANT_URL)
    index.ensure_collections()
    registry_file = tmp_path / "AI-Daily-Log" / "_project-registry.md"
    note = _write_note(
        tmp_path, "proj-note.md", _note_text(doc_id, project="baker-street")
    )

    try:
        result = process_inbox_file(
            note,
            vault_path=tmp_path,
            embed_client=FakeEmbedClient(),
            index_client=index,
            registry_file=registry_file,
        )

        assert result.status == "processed"
        assert result.project_registered is True
        assert lookup("baker-street", registry_file) is not None
    finally:
        index.delete_by_doc_id(KNOWLEDGE_COLLECTION, doc_id)
