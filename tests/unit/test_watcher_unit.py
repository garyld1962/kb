"""Unit tests for InboxEventHandler's self-write echo debounce.

Regression test for kb-2: rewriting a note in place with
``status: embed_failed`` must not re-trigger processing of the same write via
the watcher's own ``on_modified`` event.
"""

from __future__ import annotations

from pathlib import Path

from watchdog.events import FileModifiedEvent

from kb.server.embed import EmbedUnavailableError
from kb.server.watcher import InboxEventHandler

NOTE_TEXT = (
    "---\n"
    "id: 01HXYZTEST0000000000000000\n"
    "created: 2026-04-12T14:32:00-04:00\n"
    "source_machine: workstation-1\n"
    "source_type: text\n"
    "source_ref: null\n"
    "title: Sample capture\n"
    "tags: [research]\n"
    'project: ""\n'
    "destination: Knowledge/research\n"
    "status: inbox\n"
    "---\n"
    "\n# Sample\n\nSome content.\n"
)


class FailingEmbedClient:
    """Always unavailable, like a down Ollama."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        raise EmbedUnavailableError("ollama unreachable")


class NoopIndexClient:
    def upsert_chunks(self, collection, payloads, vectors) -> None:  # pragma: no cover
        raise AssertionError("upsert should never be reached when embed fails")


def _write_inbox_note(vault: Path, name: str, text: str) -> Path:
    inbox = vault / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(text)
    return path


def test_embed_failed_rewrite_does_not_retrigger_via_watcher(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    path = _write_inbox_note(vault, "note.md", NOTE_TEXT)

    embed_client = FailingEmbedClient()
    handler = InboxEventHandler(vault, embed_client, NoopIndexClient())

    # First dispatch: pipeline attempts embed, fails, rewrites the note in
    # place with status: embed_failed.
    handler.on_modified(FileModifiedEvent(str(path)))
    assert embed_client.calls == 1
    assert "status: embed_failed" in path.read_text()

    # The in-place rewrite fires a second on_modified for the same mtime
    # (the watcher's own echo of its write). It must be skipped, not
    # reprocessed.
    handler.on_modified(FileModifiedEvent(str(path)))
    assert embed_client.calls == 1

    # A genuine subsequent external change (new mtime) is still processed.
    import time

    time.sleep(0.01)
    path.write_text(path.read_text().replace("embed_failed", "embed_failed") + "\n")
    handler.on_modified(FileModifiedEvent(str(path)))
    assert embed_client.calls == 2
