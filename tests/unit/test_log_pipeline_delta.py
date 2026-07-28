"""Unit tests for the log pipeline's delta-detection logic.

Pure-logic slice: entry parsing and delta selection are exercised directly,
and a full ``handle`` pass runs against in-memory embed/index fakes (no
network) to prove that appending entries indexes only the delta.
"""

from __future__ import annotations

from pathlib import Path

from kb.core.models import QdrantPayload
from kb.server.embed import EMBED_DIM
from kb.server.index import LOG_COLLECTION, IndexClient
from kb.server.log_pipeline import (
    LogPipeline,
    log_doc_id,
    parse_entries,
    select_delta,
    split_frontmatter,
)


def render_log(project: str, day: str, entries: list[tuple[str, str, str]]) -> str:
    """Render a log file: ``entries`` are ``(time, title, body)`` tuples."""

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


# --- pure parsing / delta selection ---------------------------------------


def test_parse_entries_extracts_headings_and_ignores_h1():
    text = render_log(
        "proj",
        "2026-07-08",
        [("09:00", "Alpha", "alpha body"), ("10:30", "Beta", "beta body")],
    )
    _, body = split_frontmatter(text)
    entries = parse_entries(body)

    assert [e.key for e in entries] == ["00000:09:00 — Alpha", "00001:10:30 — Beta"]
    assert [e.time for e in entries] == ["09:00", "10:30"]
    assert [e.title for e in entries] == ["Alpha", "Beta"]
    # Body text is preserved; the H1 line is not treated as an entry.
    assert "alpha body" in entries[0].markdown
    assert not entries[0].markdown.rstrip().endswith("---")


def test_parse_entries_stops_at_nightly_summary_marker():
    body = (
        "# proj — 2026-07-08\n\n"
        "## 09:00 — Real entry\n*machine: h*\n\nwork done\n\n---\n\n"
        "<!-- === kb-server nightly summary (generated ...) === -->\n"
        "## Daily Summary\n\nSummary prose that must not be indexed.\n"
    )
    entries = parse_entries(body)

    assert [e.key for e in entries] == ["00000:09:00 — Real entry"]
    assert "Summary prose" not in entries[0].markdown


def test_parse_entries_does_not_split_on_escaped_heading_lookalike():
    # ``kb.cli.vault._render_entry`` backslash-escapes any logged-message
    # line that would otherwise collide with the entry-heading pattern; the
    # parser must not treat that escaped line as a boundary.
    body = (
        "# proj — 2026-07-08\n\n"
        "## 09:00 — Real entry\n*machine: h*\n\n"
        "\\## 23:59 — Forged Entry\nsome body text\n\n---\n"
    )
    entries = parse_entries(body)

    assert [e.key for e in entries] == ["00000:09:00 — Real entry"]
    assert "Forged Entry" in entries[0].markdown


def test_select_delta_returns_only_unindexed_entries():
    entries = parse_entries(
        split_frontmatter(
            render_log(
                "proj",
                "2026-07-08",
                [("09:00", "A", "a"), ("10:00", "B", "b"), ("11:00", "C", "c")],
            )
        )[1]
    )
    indexed = {"00000:09:00 — A"}

    delta = select_delta(entries, indexed)

    assert [e.key for e in delta] == ["00001:10:00 — B", "00002:11:00 — C"]


def test_select_delta_empty_when_all_indexed():
    entries = parse_entries(
        split_frontmatter(render_log("proj", "2026-07-08", [("09:00", "A", "a")]))[1]
    )
    assert select_delta(entries, {"00000:09:00 — A"}) == []


def test_parse_entries_disambiguates_same_minute_same_title():
    # Two entries with identical time + title must not collapse to one key,
    # or the second occurrence would be silently dropped from indexing.
    text = render_log(
        "proj",
        "2026-07-08",
        [("09:00", "Standup", "first"), ("09:00", "Standup", "second")],
    )
    _, body = split_frontmatter(text)
    entries = parse_entries(body)

    assert len(entries) == 2
    assert entries[0].key != entries[1].key
    assert "first" in entries[0].markdown
    assert "second" in entries[1].markdown


# --- handle() delta against in-memory fakes -------------------------------


class FakeEmbedClient:
    """Records every embed call so tests can assert exact delta counts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.01] * EMBED_DIM


class FakeQdrantTransport:
    """In-memory Qdrant transport (mirrors the embed/index unit-test fake)."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, vector_size: int) -> None:
        assert name not in self.collections, f"double-create of {name}"
        self.collections[name] = {}

    def upsert(self, collection: str, points: list[dict]) -> None:
        store = self.collections.setdefault(collection, {})
        for point in points:
            store[point["id"]] = {"vector": point["vector"], "payload": point["payload"]}

    def delete_by_doc_id(self, collection: str, doc_id: str) -> None:  # pragma: no cover
        store = self.collections.get(collection, {})
        for pid in [p for p, v in store.items() if v["payload"]["doc_id"] == doc_id]:
            del store[pid]

    def search(self, collection, vector, limit):  # pragma: no cover - unused
        return []


def _payloads_for(transport: FakeQdrantTransport, doc_id: str) -> list[dict]:
    store = transport.collections.get(LOG_COLLECTION, {})
    return [v["payload"] for v in store.values() if v["payload"]["doc_id"] == doc_id]


def test_handle_indexes_only_delta_on_append(tmp_path: Path):
    embed = FakeEmbedClient()
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    pipeline = LogPipeline(tmp_path, embed, index)

    log_file = tmp_path / "AI-Daily-Log" / "proj" / "2026-07-08.md"
    log_file.parent.mkdir(parents=True)

    # First index: one entry -> one chunk -> one embed call.
    log_file.write_text(render_log("proj", "2026-07-08", [("09:00", "A", "alpha")]))
    first = pipeline.handle(log_file)
    assert first.entries_indexed == 1
    assert first.chunks_indexed == 1
    assert len(embed.calls) == 1

    # Append two entries: exactly the two-entry delta is embedded, not a
    # re-index of the prior entry.
    log_file.write_text(
        render_log(
            "proj",
            "2026-07-08",
            [("09:00", "A", "alpha"), ("10:00", "B", "beta"), ("11:00", "C", "gamma")],
        )
    )
    second = pipeline.handle(log_file)
    assert second.entries_indexed == 2
    assert second.chunks_indexed == 2
    assert len(embed.calls) == 3  # +2 only

    doc_id = log_doc_id("proj", "2026-07-08")
    stored = _payloads_for(transport, doc_id)
    assert len(stored) == 3
    # Stored chunks carry the full-document numbering, contract-shaped payload.
    assert {p["chunk_index"] for p in stored} == {0, 1, 2}
    assert all(set(p.keys()) == set(QdrantPayload.model_fields.keys()) for p in stored)
    assert all(p["project"] == "proj" for p in stored)


def test_handle_no_change_indexes_nothing(tmp_path: Path):
    embed = FakeEmbedClient()
    index = IndexClient(transport=FakeQdrantTransport())
    pipeline = LogPipeline(tmp_path, embed, index)

    log_file = tmp_path / "log.md"
    log_file.write_text(render_log("proj", "2026-07-08", [("09:00", "A", "alpha")]))

    pipeline.handle(log_file)
    again = pipeline.handle(log_file)

    assert again.entries_indexed == 0
    assert again.chunks_indexed == 0
    assert len(embed.calls) == 1  # unchanged
