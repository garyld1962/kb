"""Pure-logic unit slice for the inbox pipeline.

Covers frontmatter parsing, error routing, and destination resolution with no
network: the embed/index clients are in-memory fakes. The real-Qdrant path is
exercised by ``tests/integration/test_inbox_pipeline.py``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from kb.core.models import InboxNoteFrontmatter
from kb.server import inbox_pipeline
from kb.server.embed import EMBED_DIM
from kb.server.index import KNOWLEDGE_COLLECTION
from kb.server.inbox_pipeline import (
    MalformedNoteError,
    compute_doc_hash,
    parse_note,
    process_inbox_file,
    quarantine,
    resolve_destination,
    split_frontmatter,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeEmbedClient:
    """Records embed calls and returns a fixed 1024-dim vector."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.0] * (EMBED_DIM - 1) + [1.0]


class FakeIndexClient:
    """Records upsert/delete calls in memory."""

    def __init__(self) -> None:
        self.upserts: list[tuple[str, list, list]] = []
        self.deleted_doc_ids: list[str] = []

    def upsert_chunks(self, collection, payloads, vectors) -> None:
        self.upserts.append((collection, payloads, vectors))

    def delete_by_doc_id(self, collection, doc_id) -> None:
        self.deleted_doc_ids.append(doc_id)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _note_text(
    *,
    status: str = "inbox",
    project: str = "",
    destination: str = "Knowledge/research",
    body: str = "# Sample\n\nSome content about retrieval.\n",
) -> str:
    return (
        "---\n"
        "id: 01HXYZTEST0000000000000000\n"
        "created: 2026-04-12T14:32:00-04:00\n"
        "source_machine: workstation-1\n"
        "source_type: text\n"
        "source_ref: null\n"
        "title: Sample capture\n"
        "tags: [research, rag]\n"
        f'project: "{project}"\n'
        f"destination: {destination}\n"
        f"status: {status}\n"
        "---\n"
        f"\n{body}"
    )


def _write_inbox_note(vault: Path, name: str, text: str) -> Path:
    inbox = vault / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# Frontmatter parsing
# --------------------------------------------------------------------------- #


def test_split_frontmatter_returns_body():
    data, body = split_frontmatter(_note_text())
    assert data["id"] == "01HXYZTEST0000000000000000"
    assert body.lstrip("\n").startswith("# Sample")


def test_parse_note_valid():
    frontmatter, body = parse_note(_note_text(project="baker-street"))
    assert isinstance(frontmatter, InboxNoteFrontmatter)
    assert frontmatter.project == "baker-street"
    assert frontmatter.status == "inbox"
    assert "retrieval" in body


def test_parse_note_missing_frontmatter_raises():
    with pytest.raises(MalformedNoteError):
        parse_note("# no frontmatter here\n\njust text\n")


def test_parse_note_unterminated_frontmatter_raises():
    with pytest.raises(MalformedNoteError):
        parse_note("---\nid: x\ntitle: y\n")


def test_parse_note_bad_yaml_raises():
    with pytest.raises(MalformedNoteError):
        parse_note("---\nid: [unclosed\n---\n\nbody\n")


def test_parse_note_invalid_status_raises():
    with pytest.raises(MalformedNoteError):
        parse_note(_note_text(status="bogus"))


# --------------------------------------------------------------------------- #
# Doc hash + destination resolution
# --------------------------------------------------------------------------- #


def test_compute_doc_hash_prefix_and_stability():
    first = compute_doc_hash("hello world")
    assert first.startswith("sha256:")
    assert first == compute_doc_hash("hello world")
    assert first != compute_doc_hash("hello world!")


def test_resolve_destination_nested():
    frontmatter, _ = parse_note(_note_text(destination="Knowledge/research"))
    dest = resolve_destination(frontmatter, Path("/vault"))
    assert dest == Path("/vault/Knowledge/research")


def test_resolve_destination_defaults_when_blank():
    frontmatter, _ = parse_note(_note_text(destination='""'))
    dest = resolve_destination(frontmatter, Path("/vault"))
    assert dest == Path("/vault/Knowledge")


# --------------------------------------------------------------------------- #
# Error routing
# --------------------------------------------------------------------------- #


def test_quarantine_moves_file_and_writes_log(tmp_path):
    note = _write_inbox_note(tmp_path, "broken.md", "not a note")
    dest = quarantine(note, tmp_path, "malformed frontmatter: boom")

    assert not note.exists()
    assert dest == tmp_path / "Inbox" / "_errors" / "broken.md"
    assert dest.exists()
    log = tmp_path / "Inbox" / "_errors" / "broken.md.error.log"
    assert log.exists()
    assert "boom" in log.read_text()


def test_process_malformed_routes_to_errors(tmp_path):
    note = _write_inbox_note(tmp_path, "junk.md", "no frontmatter at all\n")
    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=FakeEmbedClient(),
        index_client=FakeIndexClient(),
    )
    assert result.status == "error"
    assert not note.exists()
    assert (tmp_path / "Inbox" / "_errors" / "junk.md").exists()


def test_process_skips_non_inbox_status(tmp_path):
    note = _write_inbox_note(tmp_path, "done.md", _note_text(status="processed"))
    index = FakeIndexClient()
    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=FakeEmbedClient(),
        index_client=index,
    )
    assert result.status == "skipped"
    assert note.exists()  # left in place
    assert index.upserts == []


def test_process_missing_file_is_skipped(tmp_path):
    result = process_inbox_file(
        tmp_path / "Inbox" / "ghost.md",
        vault_path=tmp_path,
        embed_client=FakeEmbedClient(),
        index_client=FakeIndexClient(),
    )
    assert result.status == "skipped"


# --------------------------------------------------------------------------- #
# Happy path (network-free, via fakes)
# --------------------------------------------------------------------------- #


def test_process_happy_path_moves_and_updates_frontmatter(tmp_path):
    note = _write_inbox_note(tmp_path, "sample.md", _note_text())
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=embed,
        index_client=index,
    )

    assert result.status == "processed"
    assert result.chunk_count >= 1

    # File moved to destination, gone from Inbox.
    moved = tmp_path / "Knowledge" / "research" / "sample.md"
    assert moved.exists()
    assert not note.exists()

    # Frontmatter updated in place.
    fm, _ = split_frontmatter(moved.read_text())
    assert fm["status"] == "processed"
    assert fm["chunk_count"] == result.chunk_count
    assert "processed_at" in fm

    # Chunks embedded and upserted into kb_knowledge with a valid payload.
    assert len(embed.texts) == result.chunk_count
    assert len(index.upserts) == 1
    collection, payloads, vectors = index.upserts[0]
    assert collection == KNOWLEDGE_COLLECTION
    assert len(payloads) == len(vectors) == result.chunk_count
    payload = payloads[0]
    assert payload.doc_id == "01HXYZTEST0000000000000000"
    assert payload.doc_hash.startswith("sha256:")
    assert payload.doc_path == "Knowledge/research/sample.md"
    assert payload.doc_title == "Sample capture"
    assert payload.chunk_of == result.chunk_count


def test_process_move_failure_rolls_back_upsert_and_quarantines(tmp_path):
    note = _write_inbox_note(tmp_path, "sample.md", _note_text())
    dest_dir = tmp_path / "Knowledge" / "research"
    dest_dir.mkdir(parents=True)
    dest_dir.chmod(0o555)  # read+execute only -> shutil.move raises PermissionError
    embed = FakeEmbedClient()
    index = FakeIndexClient()

    try:
        result = process_inbox_file(
            note,
            vault_path=tmp_path,
            embed_client=embed,
            index_client=index,
        )
    finally:
        dest_dir.chmod(0o755)  # restore so tmp_path teardown can remove it

    assert result.status == "error"
    assert not note.exists()
    assert (tmp_path / "Inbox" / "_errors" / "sample.md").exists()

    # Chunks were embedded and upserted once...
    assert len(index.upserts) == 1
    # ...then rolled back, since the note never actually reached its
    # indexed doc_path.
    assert index.deleted_doc_ids == ["01HXYZTEST0000000000000000"]


def test_process_unknown_project_auto_registers(tmp_path):
    note = _write_inbox_note(
        tmp_path, "proj.md", _note_text(project="baker-street")
    )
    registry_file = tmp_path / "AI-Daily-Log" / "_project-registry.md"

    result = process_inbox_file(
        note,
        vault_path=tmp_path,
        embed_client=FakeEmbedClient(),
        index_client=FakeIndexClient(),
        registry_file=registry_file,
    )

    assert result.status == "processed"
    assert result.project_registered is True
    assert registry_file.exists()
    assert "baker-street" in registry_file.read_text()

    # Second note for the same project does not re-register.
    note2 = _write_inbox_note(
        tmp_path, "proj2.md", _note_text(project="baker-street")
    )
    result2 = process_inbox_file(
        note2,
        vault_path=tmp_path,
        embed_client=FakeEmbedClient(),
        index_client=FakeIndexClient(),
        registry_file=registry_file,
    )
    assert result2.project_registered is False


def test_normalize_strips_html_for_url_source():
    cleaned = inbox_pipeline.normalize_content(
        "<p>Hello <b>world</b></p>\r\nline2", "url"
    )
    assert "<p>" not in cleaned and "<b>" not in cleaned
    assert "Hello world" in cleaned
    assert "\r" not in cleaned


def test_render_processed_note_roundtrips_body():
    frontmatter, body = parse_note(_note_text())
    rendered = inbox_pipeline._render_processed_note(
        frontmatter, body, chunk_count=3, processed_at=datetime.now().astimezone()
    )
    data, new_body = split_frontmatter(rendered)
    assert data["status"] == "processed"
    assert data["chunk_count"] == 3
    assert new_body == body
