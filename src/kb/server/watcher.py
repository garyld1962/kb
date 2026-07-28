"""``watchdog`` filesystem watchers on the vault's ``Inbox/`` and ``AI-Daily-Log/``.

New or changed markdown files in ``Inbox/`` are dispatched to
:func:`kb.server.inbox_pipeline.process_inbox_file` via :class:`InboxWatcher`.
The inbox watcher is non-recursive: quarantined files under ``Inbox/_errors/``
do not re-trigger processing, and processed notes are moved out of ``Inbox/``
entirely. A note marked ``status: embed_failed`` is rewritten in place
(still inside ``Inbox/``), which would otherwise re-trigger ``on_modified``
against the same file; :class:`InboxEventHandler` remembers the mtime of its
own ``embed_failed`` rewrite and skips the resulting echo event, so no
self-triggering loop forms.

Changed log files under ``AI-Daily-Log/`` are dispatched to
:class:`kb.server.log_pipeline.LogPipeline` via :class:`LogWatcher`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from kb.server.embed import EmbedClient
from kb.server.inbox_pipeline import ERRORS_DIRNAME, process_inbox_file
from kb.server.index import IndexClient
from kb.server.log_pipeline import LogPipeline
from kb.server.registry import registry_path

logger = logging.getLogger("kb.server.watcher")


class InboxEventHandler(FileSystemEventHandler):
    """Dispatch created/modified ``Inbox/*.md`` files to the inbox pipeline."""

    def __init__(
        self,
        vault_path: Path,
        embed_client: EmbedClient,
        index_client: IndexClient,
    ) -> None:
        self._vault_path = Path(vault_path)
        self._embed_client = embed_client
        self._index_client = index_client
        self._registry_file = registry_path(self._vault_path)
        # mtime of the last ``embed_failed`` rewrite this handler made to a
        # given path, so the ``on_modified`` echo of that self-write can be
        # told apart from a genuine external change (see module docstring).
        self._last_self_write: dict[Path, float] = {}

    def _dispatch(self, src_path: str) -> None:
        path = Path(src_path)
        if path.suffix != ".md" or ERRORS_DIRNAME in path.parts:
            return
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            current_mtime = None
        if current_mtime is not None and self._last_self_write.get(path) == current_mtime:
            del self._last_self_write[path]
            return
        result = process_inbox_file(
            path,
            vault_path=self._vault_path,
            embed_client=self._embed_client,
            index_client=self._index_client,
            registry_file=self._registry_file,
        )
        if result.status == "embed_failed":
            try:
                self._last_self_write[result.path] = result.path.stat().st_mtime
            except OSError:
                self._last_self_write.pop(result.path, None)
        else:
            self._last_self_write.pop(path, None)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path)


class InboxWatcher:
    """Own a ``watchdog`` observer over ``<vault>/Inbox/``."""

    def __init__(
        self,
        vault_path: Path,
        embed_client: EmbedClient,
        index_client: IndexClient,
    ) -> None:
        self.vault_path = Path(vault_path)
        self.inbox_dir = self.vault_path / "Inbox"
        self._handler = InboxEventHandler(self.vault_path, embed_client, index_client)
        self._observer = Observer()

    def start(self) -> None:
        """Create ``Inbox/`` if absent and begin watching it (non-recursive)."""

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(self._handler, str(self.inbox_dir), recursive=False)
        self._observer.start()
        logger.info("watching inbox at %s", self.inbox_dir)

    def stop(self) -> None:
        """Stop the observer and wait for its thread to exit."""

        self._observer.stop()
        self._observer.join()

    def is_alive(self) -> bool:
        """Whether the observer thread is still running.

        ``False`` after an unhandled exception has killed the watcher thread
        (or before ``start()``/after ``stop()``), so callers can detect a
        silently-dead watcher instead of assuming it is still indexing.
        """

        return self._observer.is_alive()


class LogEventHandler(FileSystemEventHandler):
    """Dispatch created/modified ``AI-Daily-Log/**/*.md`` files to the log pipeline."""

    def __init__(self, log_pipeline: LogPipeline) -> None:
        self._pipeline = log_pipeline

    def _dispatch(self, src_path: str) -> None:
        path = Path(src_path)
        if path.suffix != ".md" or path.name.startswith("_"):
            return
        self._pipeline.handle(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path)


class LogWatcher:
    """Own a ``watchdog`` observer over ``<vault>/AI-Daily-Log`` (recursive).

    Recursive because entries live under per-project subdirectories
    (``AI-Daily-Log/<project>/<date>.md``), unlike the flat ``Inbox/``.
    """

    def __init__(self, vault_path: Path, log_pipeline: LogPipeline) -> None:
        self.vault_path = Path(vault_path)
        self.log_dir = self.vault_path / "AI-Daily-Log"
        self._handler = LogEventHandler(log_pipeline)
        self._observer = Observer()

    def start(self) -> None:
        """Create ``AI-Daily-Log/`` if absent and begin watching it (recursive)."""

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(self._handler, str(self.log_dir), recursive=True)
        self._observer.start()
        logger.info("watching logs at %s", self.log_dir)

    def stop(self) -> None:
        """Stop the observer and wait for its thread to exit."""

        self._observer.stop()
        self._observer.join()

    def is_alive(self) -> bool:
        """Whether the observer thread is still running (see :class:`InboxWatcher`)."""

        return self._observer.is_alive()
