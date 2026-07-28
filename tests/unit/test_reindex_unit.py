"""Pure-logic unit slice for the periodic reindex pass.

Covers the hash-diff/rename/deletion decision logic and the on-disk scanners
with no network: embed/index/state-reader are in-memory fakes. The real-Qdrant
path is exercised by ``tests/integration/test_reindex.py``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kb.server.embed import EMBED_DIM
from kb.server.index import KNOWLEDGE_COLLECTION, LOG_COLLECTION
from kb.server.reindex import (
    IndexedDoc,
    ReindexAction,
    ScannedFile,
    decide_action,
    reindex_once,
    retry_embed_failed_notes,
    scan_knowledge_files,
    scan_log_files,
)


def _scanned(doc_id: str = "doc-1", doc_path: str = "Knowledge/a.md", doc_hash: str = "sha256:aaa") -> ScannedFile:
    return ScannedFile(
        doc_id=doc_id,
        doc_path=doc_path,
        doc_hash=doc_hash,
        doc_title="Title",
        tags=[],
        project="",
        source_type="text",
        source_ref=None,
        created_at=datetime.now().astimezone(),
        body="# Title\n\nSome content.\n",
    )


def _indexed(doc_id: str = "doc-1", doc_path: str = "Knowledge/a.md", doc_hash: str = "sha256:aaa") -> IndexedDoc:
    return IndexedDoc(doc_id=doc_id, doc_hash=doc_hash, doc_path=doc_path)


# --------------------------------------------------------------------------- #
# decide_action: hash-diff / rename / deletion decision logic
# --------------------------------------------------------------------------- #


def test_decide_action_new_file_indexes():
    assert decide_action(_scanned(), None) is ReindexAction.INDEX


def test_decide_action_unchanged_is_noop():
    scanned = _scanned()
    indexed = _indexed()
    assert decide_action(scanned, indexed) is ReindexAction.NOOP


def test_decide_action_hash_changed_reindexes():
    scanned = _scanned(doc_hash="sha256:new")
    indexed = _indexed(doc_hash="sha256:old")
    assert decide_action(scanned, indexed) is ReindexAction.INDEX


def test_decide_action_path_changed_same_hash_renames():
    scanned = _scanned(doc_path="Knowledge/renamed.md")
    indexed = _indexed(doc_path="Knowledge/a.md")
    assert decide_action(scanned, indexed) is ReindexAction.RENAME


def test_decide_action_missing_on_disk_purges():
    assert decide_action(None, _indexed()) is ReindexAction.PURGE


def test_decide_action_neither_scanned_nor_indexed_is_noop():
    assert decide_action(None, None) is ReindexAction.NOOP


def test_decide_action_hash_change_wins_over_path_change():
    """A doc that both changed content and moved is a re-index, not a rename."""

    scanned = _scanned(doc_path="Knowledge/renamed.md", doc_hash="sha256:new")
    indexed = _indexed(doc_path="Knowledge/a.md", doc_hash="sha256:old")
    assert decide_action(scanned, indexed) is ReindexAction.INDEX


# --------------------------------------------------------------------------- #
# Scanning: Knowledge/Archive (InboxNoteFrontmatter-shaped) and AI-Daily-Log
# --------------------------------------------------------------------------- #


def _write_processed_note(vault: Path, rel_path: str, *, doc_id: str) -> Path:
    path = vault / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"id: {doc_id}\n"
        "created: 2026-04-12T14:32:00-04:00\n"
        "source_machine: workstation-1\n"
        "source_type: text\n"
        "source_ref: null\n"
        "title: Sample capture\n"
        "tags: [research]\n"
        'project: ""\n'
        "destination: Knowledge/research\n"
        "status: processed\n"
        "processed_at: '2026-04-12T14:33:00-04:00'\n"
        "chunk_count: 1\n"
        "---\n"
        "\n# Sample\n\nSome content about retrieval.\n"
    )
    return path


def test_scan_knowledge_files_parses_processed_note(tmp_path):
    _write_processed_note(tmp_path, "Knowledge/research/sample.md", doc_id="doc-a")

    files = scan_knowledge_files(tmp_path)

    assert set(files) == {"doc-a"}
    scanned = files["doc-a"]
    assert scanned.doc_path == "Knowledge/research/sample.md"
    assert scanned.doc_hash.startswith("sha256:")
    assert scanned.doc_title == "Sample capture"
    assert "retrieval" in scanned.body


def test_scan_knowledge_files_covers_archive_too(tmp_path):
    _write_processed_note(tmp_path, "Archive/old/sample.md", doc_id="doc-b")

    files = scan_knowledge_files(tmp_path)

    assert set(files) == {"doc-b"}


def test_scan_knowledge_files_skips_unparseable_file(tmp_path):
    path = tmp_path / "Knowledge" / "junk.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("no frontmatter here\n")

    files = scan_knowledge_files(tmp_path)

    assert files == {}


def _write_log_file(vault: Path, project: str, day: str) -> Path:
    path = vault / "AI-Daily-Log" / project / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"project: {project}\n"
        f"date: {day}\n"
        "status: active\n"
        "entries: 1\n"
        "---\n"
        "\n"
        f"# {project} — {day}\n\n"
        "## 09:00 — Alpha\n*machine: testhost*\n\nwork done\n\n---\n"
    )
    return path


def test_scan_log_files_derives_doc_id(tmp_path):
    _write_log_file(tmp_path, "proj", "2026-07-08")

    files = scan_log_files(tmp_path)

    assert set(files) == {"log:proj:2026-07-08"}
    scanned = files["log:proj:2026-07-08"]
    assert scanned.doc_path == "AI-Daily-Log/proj/2026-07-08.md"
    assert scanned.project == "proj"
    assert "Alpha" in scanned.body


def test_scan_log_files_skips_project_registry(tmp_path):
    registry = tmp_path / "AI-Daily-Log" / "_project-registry.md"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("---\nprojects: []\n---\n\n# Project Registry\n")
    _write_log_file(tmp_path, "proj", "2026-07-08")

    files = scan_log_files(tmp_path)

    assert set(files) == {"log:proj:2026-07-08"}


# --------------------------------------------------------------------------- #
# Full reconciliation pass against in-memory fakes
# --------------------------------------------------------------------------- #


class FakeEmbedClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.0] * (EMBED_DIM - 1) + [1.0]


class FakeIndexClient:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list, list]] = []
        self.deletes: list[tuple[str, str]] = []
        self.stale_deletes: list[tuple[str, str, int]] = []
        self.calls: list[str] = []
        self.collections_ready = False

    def ensure_collections(self) -> None:
        self.collections_ready = True

    def upsert_chunks(self, collection, payloads, vectors) -> None:
        self.calls.append("upsert")
        self.upserts.append((collection, payloads, vectors))

    def delete_by_doc_id(self, collection, doc_id) -> None:
        self.calls.append("delete_by_doc_id")
        self.deletes.append((collection, doc_id))

    def delete_stale_chunks(self, collection, doc_id, min_chunk_index) -> None:
        self.calls.append("delete_stale_chunks")
        self.stale_deletes.append((collection, doc_id, min_chunk_index))


class FakeStateReader:
    """In-memory stand-in for :class:`QdrantStateReader`.

    ``store`` maps ``collection -> doc_id -> list[(payload_dict, vector)]``.
    """

    def __init__(self, store: dict[str, dict[str, list[tuple[dict, list[float]]]]]) -> None:
        self.store = store

    def doc_states(self, collection: str):
        result = {}
        for doc_id, chunks in self.store.get(collection, {}).items():
            payload = chunks[0][0]
            result[doc_id] = IndexedDoc(
                doc_id=doc_id,
                doc_hash=payload["doc_hash"],
                doc_path=payload["doc_path"],
                chunk_count=len(chunks),
            )
        return result

    def doc_chunks(self, collection: str, doc_id: str):
        return self.store.get(collection, {}).get(doc_id, [])


def _payload(doc_id: str, doc_hash: str, doc_path: str) -> dict:
    return {
        "doc_id": doc_id,
        "doc_path": doc_path,
        "doc_hash": doc_hash,
        "doc_title": "Sample capture",
        "chunk_index": 0,
        "chunk_of": 1,
        "header_path": "",
        "content": "old content",
        "tags": [],
        "project": "",
        "source_type": "text",
        "source_ref": None,
        "created_at": "2026-04-12T14:32:00-04:00",
        "indexed_at": "2026-04-12T14:33:00-04:00",
    }


def test_reindex_once_content_change_purges_and_reembeds(tmp_path):
    _write_processed_note(tmp_path, "Knowledge/research/sample.md", doc_id="doc-a")
    on_disk_hash = scan_knowledge_files(tmp_path)["doc-a"].doc_hash

    reader = FakeStateReader(
        {
            KNOWLEDGE_COLLECTION: {
                "doc-a": [
                    (_payload("doc-a", "sha256:stale", "Knowledge/research/sample.md"), [0.1] * EMBED_DIM)
                ]
            }
        }
    )
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    summary = reindex_once(tmp_path, embed, index, reader=reader)

    assert summary.indexed == 1
    assert index.deletes == []
    assert len(index.upserts) == 1
    collection, payloads, vectors = index.upserts[0]
    assert collection == KNOWLEDGE_COLLECTION
    assert payloads[0].doc_hash == on_disk_hash
    assert embed.calls, "content change must trigger re-embedding"
    # The new chunks are upserted (visible) before any stale leftover chunks
    # from the previous version are removed, so a mid-sequence failure can
    # never leave the document fully absent from search.
    assert index.calls[0] == "upsert"
    assert index.stale_deletes == [(KNOWLEDGE_COLLECTION, "doc-a", len(payloads))]


def test_reindex_once_noop_retries_leftover_stale_chunk_delete(tmp_path):
    # Simulates a previous pass whose INDEX action upserted new chunks but
    # whose compensating delete_stale_chunks call failed: the on-disk hash
    # now matches what's indexed (so decide_action returns NOOP), but the
    # index still has more chunks stored than the current document produces.
    _write_processed_note(tmp_path, "Knowledge/research/sample.md", doc_id="doc-a")
    on_disk_hash = scan_knowledge_files(tmp_path)["doc-a"].doc_hash

    reader = FakeStateReader(
        {
            KNOWLEDGE_COLLECTION: {
                "doc-a": [
                    (_payload("doc-a", on_disk_hash, "Knowledge/research/sample.md"), [0.1] * EMBED_DIM),
                    (_payload("doc-a", on_disk_hash, "Knowledge/research/sample.md"), [0.2] * EMBED_DIM),
                ]
            }
        }
    )
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    summary = reindex_once(tmp_path, embed, index, reader=reader)

    assert summary.unchanged == 1
    assert not embed.calls, "recovering a leftover stale delete must not re-embed"
    assert index.upserts == []
    assert index.stale_deletes == [(KNOWLEDGE_COLLECTION, "doc-a", 1)]


def test_reindex_once_rename_updates_path_without_reembed(tmp_path):
    _write_processed_note(tmp_path, "Knowledge/research/moved.md", doc_id="doc-a")
    on_disk_hash = scan_knowledge_files(tmp_path)["doc-a"].doc_hash
    stored_vector = [0.42] * EMBED_DIM

    reader = FakeStateReader(
        {
            KNOWLEDGE_COLLECTION: {
                "doc-a": [
                    (_payload("doc-a", on_disk_hash, "Knowledge/research/old-name.md"), stored_vector)
                ]
            }
        }
    )
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    summary = reindex_once(tmp_path, embed, index, reader=reader)

    assert summary.renamed == 1
    assert index.deletes == []
    assert not embed.calls, "rename must not re-embed"
    collection, payloads, vectors = index.upserts[0]
    assert collection == KNOWLEDGE_COLLECTION
    assert payloads[0].doc_path == "Knowledge/research/moved.md"
    assert vectors == [stored_vector]


def test_reindex_once_deletion_purges_chunks(tmp_path):
    # Nothing on disk for doc-a, but it is currently indexed.
    (tmp_path / "Knowledge").mkdir(parents=True, exist_ok=True)

    reader = FakeStateReader(
        {
            LOG_COLLECTION: {
                "log:proj:2026-07-08": [
                    (_payload("log:proj:2026-07-08", "sha256:x", "AI-Daily-Log/proj/2026-07-08.md"), [0.0] * EMBED_DIM)
                ]
            }
        }
    )
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    summary = reindex_once(tmp_path, embed, index, reader=reader)

    assert summary.purged == 1
    assert index.deletes == [(LOG_COLLECTION, "log:proj:2026-07-08")]
    assert index.upserts == []


# --------------------------------------------------------------------------- #
# embed_failed Inbox/ notes are retried, not stuck forever
# --------------------------------------------------------------------------- #


def _write_embed_failed_note(vault: Path, name: str = "stuck.md") -> Path:
    inbox = vault / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(
        "---\n"
        "id: 01HXYZTEST0000000000000009\n"
        "created: 2026-04-12T14:32:00-04:00\n"
        "source_machine: workstation-1\n"
        "source_type: text\n"
        "source_ref: null\n"
        "title: Stuck capture\n"
        "tags: []\n"
        'project: ""\n'
        "destination: Knowledge\n"
        "status: embed_failed\n"
        "embed_failed_reason: 'ollama unreachable'\n"
        "---\n"
        "\n# Stuck\n\nContent that failed to embed earlier.\n"
    )
    return path


def test_retry_embed_failed_notes_reprocesses_stuck_note(tmp_path):
    note = _write_embed_failed_note(tmp_path)
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    retried = retry_embed_failed_notes(tmp_path, embed, index)

    assert retried == 1
    assert not note.exists()
    assert (tmp_path / "Knowledge" / "stuck.md").exists()
    assert len(index.upserts) == 1


def test_retry_embed_failed_notes_ignores_inbox_status_notes(tmp_path):
    inbox = tmp_path / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "fresh.md").write_text("---\nstatus: inbox\n---\n\nnot embed_failed\n")

    retried = retry_embed_failed_notes(tmp_path, FakeEmbedClient(), FakeIndexClient())

    assert retried == 0
    assert (inbox / "fresh.md").exists()


def test_reindex_once_retries_embed_failed_inbox_notes(tmp_path):
    note = _write_embed_failed_note(tmp_path)
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    summary = reindex_once(tmp_path, embed, index, reader=FakeStateReader({}))

    assert summary.embed_failed_retried == 1
    assert not note.exists()
    assert (tmp_path / "Knowledge" / "stuck.md").exists()
