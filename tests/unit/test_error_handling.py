"""Unit tests for Task 10's error-handling hardening.

Covers three behaviors from the PRD's Error Handling section:

1. Ollama/Qdrant unreachable -> the client wrappers retry with exponential
   backoff, then raise a distinguished "unavailable" error; the inbox/log
   pipelines catch that and mark the work ``embed_failed`` instead of
   raising past their own boundary.
2. Corrupted/binary Inbox content is quarantined (moved to ``Inbox/_errors/``)
   rather than partially indexed.
3. A partial embed followed by a retry does not duplicate chunks in the
   index, because nothing is upserted until every chunk in a note has
   embedded successfully.

All network I/O is faked; no real Ollama/Qdrant is required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from kb.core.models import QdrantPayload
from kb.server.embed import EMBED_DIM, MAX_RETRIES, EmbedClient, EmbedUnavailableError
from kb.server.index import (
    LOG_COLLECTION,
    IndexClient,
    IndexUnavailableError,
)
from kb.server.index import MAX_RETRIES as INDEX_MAX_RETRIES
from kb.server.inbox_pipeline import process_inbox_file, split_frontmatter
from kb.server.log_pipeline import LogPipeline


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeEmbedClient:
    """Records every embed call; raises ``exc`` after ``fail_after`` successes.

    ``fail_after=0`` (the default) fails on the very first call.
    """

    def __init__(self, exc: Exception, *, fail_after: int = 0) -> None:
        self.exc = exc
        self.fail_after = fail_after
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if len(self.calls) > self.fail_after:
            raise self.exc
        return [0.0] * (EMBED_DIM - 1) + [1.0]


class SucceedingEmbedClient:
    """Always succeeds; records every embed call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.0] * (EMBED_DIM - 1) + [1.0]


class FakeIndexClient:
    """Records upsert calls in memory; never talks to real Qdrant."""

    def __init__(self) -> None:
        self.upserts: list[tuple[str, list, list]] = []

    def upsert_chunks(self, collection, payloads, vectors) -> None:
        self.upserts.append((collection, payloads, vectors))

    def ensure_collections(self) -> None:
        pass


class FlakyQdrantTransport:
    """In-memory Qdrant transport whose ``upsert`` always raises ``OSError``."""

    def __init__(self) -> None:
        self.calls = 0

    def collection_exists(self, name: str) -> bool:
        return True

    def create_collection(self, name: str, vector_size: int) -> None:
        pass

    def upsert(self, collection: str, points: list[dict]) -> None:
        self.calls += 1
        raise ConnectionRefusedError("qdrant refused connection")

    def delete_by_doc_id(self, collection: str, doc_id: str) -> None:  # pragma: no cover
        pass

    def search(self, collection, vector, limit):  # pragma: no cover
        return []


# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirrors tests/unit/test_inbox_pipeline_unit.py)
# --------------------------------------------------------------------------- #


def _note_text(
    *,
    status: str = "inbox",
    body: str = (
        "# Section One\n\nSome content about retrieval.\n\n"
        "# Section Two\n\nMore content about indexing.\n"
    ),
) -> str:
    return (
        "---\n"
        "id: 01HXYZTEST0000000000000001\n"
        "created: 2026-04-12T14:32:00-04:00\n"
        "source_machine: workstation-1\n"
        "source_type: text\n"
        "source_ref: null\n"
        "title: Sample capture\n"
        "tags: [research]\n"
        'project: ""\n'
        "destination: Knowledge/research\n"
        f"status: {status}\n"
        "---\n"
        f"\n{body}"
    )


def _write_inbox_note(vault: Path, name: str, text: str | bytes) -> Path:
    inbox = vault / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# 1a. Client-level retry/backoff (embed.py / index.py)
# --------------------------------------------------------------------------- #


def test_embed_client_retries_then_raises_unavailable(monkeypatch):
    client = EmbedClient()
    sleeps: list[float] = []
    monkeypatch.setattr("kb.server.embed.time.sleep", lambda s: sleeps.append(s))

    def always_refused(payload):
        raise ConnectionRefusedError("ollama refused connection")

    monkeypatch.setattr(client, "_post", always_refused)

    with pytest.raises(EmbedUnavailableError):
        client.embed("hello")

    # Initial attempt + MAX_RETRIES retries; a backoff sleep between each.
    assert len(sleeps) == MAX_RETRIES
    assert sleeps == sorted(sleeps)  # non-decreasing: exponential backoff


def test_index_client_retries_then_raises_unavailable(monkeypatch):
    transport = FlakyQdrantTransport()
    index = IndexClient(transport=transport)
    sleeps: list[float] = []
    monkeypatch.setattr("kb.server.index.time.sleep", lambda s: sleeps.append(s))

    payload = QdrantPayload(
        doc_id="d",
        doc_path="Knowledge/d.md",
        doc_hash="sha256:x",
        doc_title="d",
        chunk_index=0,
        chunk_of=1,
        header_path="",
        content="hi",
        tags=[],
        project="",
        source_type="text",
        source_ref=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(IndexUnavailableError):
        index.upsert_chunks("kb_knowledge", [payload], [[0.0] * 1024])

    assert transport.calls == INDEX_MAX_RETRIES + 1
    assert len(sleeps) == INDEX_MAX_RETRIES


# --------------------------------------------------------------------------- #
# 1b. Pipeline-level: repeated unreachable errors mark embed_failed, no raise
# --------------------------------------------------------------------------- #


def test_inbox_pipeline_marks_embed_failed_without_raising(tmp_path):
    note = _write_inbox_note(tmp_path, "sample.md", _note_text())
    embed = FakeEmbedClient(EmbedUnavailableError("ollama unreachable"))
    index = FakeIndexClient()

    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=embed,
        index_client=index,
    )

    assert result.status == "embed_failed"
    # Left in place, not moved and not quarantined.
    assert note.exists()
    assert result.path == note
    # No partial index: upsert only ever happens after every chunk embeds.
    assert index.upserts == []

    fm, _ = split_frontmatter(note.read_text())
    assert fm["status"] == "embed_failed"


def test_inbox_pipeline_marks_embed_failed_on_index_unavailable(tmp_path):
    note = _write_inbox_note(tmp_path, "sample.md", _note_text())
    embed = SucceedingEmbedClient()

    class AlwaysUnavailableIndexClient:
        def upsert_chunks(self, collection, payloads, vectors):
            raise IndexUnavailableError("qdrant unreachable")

    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=embed,
        index_client=AlwaysUnavailableIndexClient(),
    )

    assert result.status == "embed_failed"
    assert note.exists()
    fm, _ = split_frontmatter(note.read_text())
    assert fm["status"] == "embed_failed"


def test_inbox_pipeline_retries_embed_failed_note(tmp_path):
    """A note left in embed_failed by an earlier outage is retryable, not stuck."""

    note = _write_inbox_note(tmp_path, "sample.md", _note_text(status="embed_failed"))
    embed = SucceedingEmbedClient()
    index = FakeIndexClient()

    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=embed,
        index_client=index,
    )

    assert result.status == "processed"
    assert not note.exists()
    assert len(index.upserts) == 1


def test_log_pipeline_marks_embed_failed_without_raising(tmp_path):
    embed = FakeEmbedClient(EmbedUnavailableError("ollama unreachable"))
    index = FakeIndexClient()
    pipeline = LogPipeline(tmp_path, embed, index, collection=LOG_COLLECTION)

    log_file = tmp_path / "AI-Daily-Log" / "proj" / "2026-07-08.md"
    log_file.parent.mkdir(parents=True)
    log_file.write_text(
        "---\nproject: proj\ndate: 2026-07-08\nstatus: active\nentries: 1\n---\n"
        "\n# proj — 2026-07-08\n\n## 09:00 — A\n*machine: h*\n\nwork done\n\n---\n"
    )

    result = pipeline.handle(log_file)

    assert result.status == "embed_failed"
    assert result.entries_indexed == 0
    assert index.upserts == []

    # The delta is retried, not lost: a later successful call re-indexes it.
    embed_ok = SucceedingEmbedClient()
    pipeline2 = LogPipeline(tmp_path, embed_ok, index, collection=LOG_COLLECTION)
    result2 = pipeline2.handle(log_file)
    assert result2.status == "ok"
    assert result2.entries_indexed == 1
    assert len(index.upserts) == 1


# --------------------------------------------------------------------------- #
# 2. Corrupted/binary content is quarantined, not partially indexed
# --------------------------------------------------------------------------- #


def test_binary_content_is_quarantined_not_indexed(tmp_path):
    note = _write_inbox_note(tmp_path, "binary.md", b"\x00\x01\xfe\xff not utf-8 \x80")
    index = FakeIndexClient()

    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=SucceedingEmbedClient(),
        index_client=index,
    )

    assert result.status == "error"
    assert not note.exists()
    quarantined = tmp_path / "Inbox" / "_errors" / "binary.md"
    assert quarantined.exists()
    assert (tmp_path / "Inbox" / "_errors" / "binary.md.error.log").exists()
    assert index.upserts == []


# --------------------------------------------------------------------------- #
# 3. Partial embed followed by a retry does not duplicate chunks
# --------------------------------------------------------------------------- #


def test_partial_embed_then_retry_does_not_duplicate_upserts(tmp_path):
    note = _write_inbox_note(tmp_path, "sample.md", _note_text())
    index = FakeIndexClient()

    # First attempt: the second chunk's embed fails after the first succeeds.
    # Nothing is upserted, since upsert only happens once every chunk embeds.
    flaky = FakeEmbedClient(EmbedUnavailableError("ollama unreachable"), fail_after=1)
    first = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=flaky,
        index_client=index,
    )
    assert first.status == "embed_failed"
    assert index.upserts == []
    assert note.exists()

    # Simulate the later retry pass resetting the note back to "inbox" (the
    # out-of-scope retry trigger; Task 10 only guarantees the retry is safe).
    fm, body = split_frontmatter(note.read_text())
    fm["status"] = "inbox"
    note.write_text(f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n{body}")

    reliable = SucceedingEmbedClient()
    second = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=reliable,
        index_client=index,
    )

    assert second.status == "processed"
    # Exactly one upsert call, for the full chunk set: the failed first
    # attempt contributed zero chunks, so nothing is duplicated.
    assert len(index.upserts) == 1
    _, payloads, vectors = index.upserts[0]
    assert len(payloads) == second.chunk_count
    assert len(vectors) == second.chunk_count
