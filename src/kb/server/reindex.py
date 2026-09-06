"""Periodic re-indexing pass over the vault's persisted documents.

Runs a scan (default every 30 minutes, via :class:`ReindexScheduler`) over
``Knowledge/``, ``Archive/``, and ``AI-Daily-Log/`` to reconcile the Qdrant
index against direct-in-Obsidian edits that bypass the Inbox/watcher paths
(the ones ``inbox_pipeline``/``log_pipeline`` react to live). Each pass also
sweeps ``Inbox/`` for notes left in ``status: embed_failed`` (an earlier
Voyage/Qdrant outage that exhausted the pipeline's own retries) and retries
them, so a note can't get stuck forever waiting for a filesystem event that
may never come. For each document found on disk, its current
``doc_hash``/``doc_path`` is compared against what is currently indexed:

- Unknown to the index -> chunk/embed/index it (:class:`ReindexAction.INDEX`).
- ``doc_hash`` differs from what is indexed -> re-chunk/re-embed/upsert the
  new chunks (reusing point ids, so overlapping chunks are overwritten live),
  then remove any leftover chunks beyond the new chunk count from a
  previous, longer version (also :class:`ReindexAction.INDEX`).
- Same ``doc_id``, ``doc_hash`` unchanged, but a different ``doc_path`` -> a
  rename: update the existing payloads' ``doc_path`` in place, reusing their
  already-stored vectors (:class:`ReindexAction.RENAME`, no re-embed).
- Indexed but no longer present on disk -> purge its chunks
  (:class:`ReindexAction.PURGE`).

``IndexClient`` (Task 3) has no operation to list what is currently indexed,
so this module talks to Qdrant's scroll endpoint directly via
:class:`QdrantStateReader`, independent of ``IndexClient``.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from enum import Enum
from pathlib import Path

from pydantic import ValidationError

from kb.core.models import InboxNoteFrontmatter, LogFileFrontmatter, QdrantPayload
from kb.server.chunker import chunk
from kb.server.index import (
    DEFAULT_QDRANT_URL,
    KNOWLEDGE_COLLECTION,
    LOG_COLLECTION,
    IndexClient,
)
from kb.server.inbox_pipeline import (
    MalformedNoteError,
    compute_doc_hash,
    normalize_content,
    process_inbox_file,
)
from kb.server.inbox_pipeline import split_frontmatter as _split_knowledge_frontmatter
from kb.server.log_pipeline import log_doc_id
from kb.server.log_pipeline import split_frontmatter as _split_log_frontmatter
from kb.server.registry import registry_path

logger = logging.getLogger("kb.server.reindex")

# Vault folders scanned each pass, per the design's Re-indexing section.
KNOWLEDGE_ROOTS = ("Knowledge", "Archive")
LOG_ROOT = "AI-Daily-Log"

# Where embed_failed notes needing a retry live (flat, non-recursive: mirrors
# InboxWatcher, which never watches Inbox/_errors).
INBOX_ROOT = "Inbox"

# Periodic scan interval (30 min, per the plan's closed decision).
REINDEX_INTERVAL_SECONDS = 30 * 60

# Log chunks are AI work-log entries; matches log_pipeline's source_type.
_LOG_SOURCE_TYPE = "agent"


class ReindexAction(str, Enum):
    """The decision made for one ``doc_id`` during a reindex pass."""

    NOOP = "noop"
    INDEX = "index"  # new, or doc_hash changed: (re)chunk/embed/index
    RENAME = "rename"  # doc_hash unchanged, doc_path changed: payload-only
    PURGE = "purge"  # indexed but no longer on disk: delete chunks


@dataclass(frozen=True)
class ScannedFile:
    """One on-disk document found during a scan, ready to be (re)indexed."""

    doc_id: str
    doc_path: str
    doc_hash: str
    doc_title: str
    tags: list[str]
    project: str
    source_type: str
    source_ref: str | None
    created_at: datetime
    body: str


@dataclass(frozen=True)
class IndexedDoc:
    """What is currently stored in Qdrant for one ``doc_id``."""

    doc_id: str
    doc_hash: str
    doc_path: str
    chunk_count: int = 0


@dataclass
class ReindexSummary:
    """Counts of actions taken by one :func:`reindex_once` pass."""

    indexed: int = 0
    renamed: int = 0
    purged: int = 0
    unchanged: int = 0
    errors: int = 0
    embed_failed_retried: int = 0


# --------------------------------------------------------------------------- #
# Pure decision logic (unit-tested without any I/O)
# --------------------------------------------------------------------------- #


def decide_action(scanned: ScannedFile | None, indexed: IndexedDoc | None) -> ReindexAction:
    """Decide what a ``doc_id`` needs given its on-disk and indexed state."""

    if scanned is None:
        return ReindexAction.PURGE if indexed is not None else ReindexAction.NOOP
    if indexed is None:
        return ReindexAction.INDEX
    if scanned.doc_hash != indexed.doc_hash:
        return ReindexAction.INDEX
    if scanned.doc_path != indexed.doc_path:
        return ReindexAction.RENAME
    return ReindexAction.NOOP


# --------------------------------------------------------------------------- #
# Scanning: on-disk documents under Knowledge/, Archive/, AI-Daily-Log/
# --------------------------------------------------------------------------- #


def _iter_markdown_files(root: Path) -> Iterator[Path]:
    """Yield ``*.md`` files under ``root``, skipping ``_``-prefixed metadata files."""

    if not root.exists():
        return
    for path in sorted(root.rglob("*.md")):
        if path.is_file() and not path.name.startswith("_"):
            yield path


def _scan_knowledge_file(path: Path, vault_path: Path) -> ScannedFile | None:
    """Parse one ``Knowledge/``/``Archive/`` file into a :class:`ScannedFile`.

    Returns ``None`` (logging a warning) if the file is unreadable or its
    frontmatter does not match :class:`InboxNoteFrontmatter`.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning("reindex: skipping unreadable file %s: %s", path, exc)
        return None
    try:
        data, body = _split_knowledge_frontmatter(raw)
        frontmatter = InboxNoteFrontmatter.model_validate(data)
    except (MalformedNoteError, ValidationError) as exc:
        logger.warning("reindex: skipping unparseable file %s: %s", path, exc)
        return None
    return ScannedFile(
        doc_id=frontmatter.id,
        doc_path=path.relative_to(vault_path).as_posix(),
        doc_hash=compute_doc_hash(raw),
        doc_title=frontmatter.title,
        tags=frontmatter.tags,
        project=frontmatter.project,
        source_type=frontmatter.source_type,
        source_ref=frontmatter.source_ref,
        created_at=frontmatter.created,
        body=normalize_content(body, frontmatter.source_type),
    )


def _scan_log_file(path: Path, vault_path: Path) -> ScannedFile | None:
    """Parse one ``AI-Daily-Log/`` file into a :class:`ScannedFile`.

    Returns ``None`` (logging a warning) if the file is unreadable or its
    frontmatter does not match :class:`LogFileFrontmatter`.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning("reindex: skipping unreadable file %s: %s", path, exc)
        return None
    try:
        data, body = _split_log_frontmatter(raw)
        frontmatter = LogFileFrontmatter.model_validate(data)
    except (ValidationError, ValueError) as exc:
        logger.warning("reindex: skipping unparseable file %s: %s", path, exc)
        return None
    created_at = datetime.combine(frontmatter.date, dt_time.min).astimezone()
    return ScannedFile(
        doc_id=log_doc_id(frontmatter.project, frontmatter.date.isoformat()),
        doc_path=path.relative_to(vault_path).as_posix(),
        doc_hash=compute_doc_hash(raw),
        doc_title=f"{frontmatter.project} — {frontmatter.date.isoformat()}",
        tags=[],
        project=frontmatter.project,
        source_type=_LOG_SOURCE_TYPE,
        source_ref=None,
        created_at=created_at,
        body=body,
    )


def scan_knowledge_files(vault_path: Path) -> dict[str, ScannedFile]:
    """Scan ``Knowledge/`` and ``Archive/`` into ``{doc_id: ScannedFile}``."""

    vault_path = Path(vault_path)
    files: dict[str, ScannedFile] = {}
    for root_name in KNOWLEDGE_ROOTS:
        for path in _iter_markdown_files(vault_path / root_name):
            scanned = _scan_knowledge_file(path, vault_path)
            if scanned is not None:
                files[scanned.doc_id] = scanned
    return files


def scan_log_files(vault_path: Path) -> dict[str, ScannedFile]:
    """Scan ``AI-Daily-Log/`` into ``{doc_id: ScannedFile}``."""

    vault_path = Path(vault_path)
    files: dict[str, ScannedFile] = {}
    for path in _iter_markdown_files(vault_path / LOG_ROOT):
        scanned = _scan_log_file(path, vault_path)
        if scanned is not None:
            files[scanned.doc_id] = scanned
    return files


def _iter_embed_failed_notes(vault_path: Path) -> Iterator[Path]:
    """Yield ``Inbox/*.md`` files whose frontmatter has ``status: embed_failed``.

    Non-recursive, like :class:`kb.server.watcher.InboxWatcher`: quarantined
    files under ``Inbox/_errors/`` are never visited.
    """

    inbox_dir = vault_path / INBOX_ROOT
    if not inbox_dir.exists():
        return
    for path in sorted(inbox_dir.glob("*.md")):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data, _ = _split_knowledge_frontmatter(raw)
        except (UnicodeDecodeError, OSError, MalformedNoteError):
            continue
        if data.get("status") == "embed_failed":
            yield path


def count_embed_failed_notes(vault_path: Path) -> int:
    """Return how many ``Inbox/`` notes are currently stuck in ``embed_failed``.

    Used by the ``/status`` health check to surface backlog size without
    retrying anything.
    """

    return sum(1 for _ in _iter_embed_failed_notes(Path(vault_path)))


def retry_embed_failed_notes(
    vault_path: Path, embed_client, index_client: IndexClient
) -> int:
    """Retry every ``Inbox/`` note stuck in ``status: embed_failed``.

    ``process_inbox_file`` treats ``embed_failed`` as retryable, but nothing
    re-invokes it once the note is rewritten in place — a filesystem watcher
    may miss the event, or none may ever fire again. Returns the number of
    notes retried (regardless of whether the retry itself succeeded).
    """

    vault_path = Path(vault_path)
    registry_file = registry_path(vault_path)
    retried = 0
    for path in list(_iter_embed_failed_notes(vault_path)):
        process_inbox_file(
            path,
            vault_path=vault_path,
            embed_client=embed_client,
            index_client=index_client,
            registry_file=registry_file,
        )
        retried += 1
    return retried


# --------------------------------------------------------------------------- #
# Qdrant state: what is currently indexed (IndexClient has no scroll/list op)
# --------------------------------------------------------------------------- #


class QdrantStateReader:
    """Reads currently-indexed chunk state directly via Qdrant's scroll API.

    ``IndexClient`` exposes only upsert/delete/search, with no way to
    enumerate what is already indexed. Reindexing needs exactly that to
    detect hash changes, renames, and deletions, so this talks to the Qdrant
    REST API directly rather than extending ``IndexClient``.
    """

    def __init__(self, url: str = DEFAULT_QDRANT_URL) -> None:
        self.url = url.rstrip("/")

    def doc_states(self, collection: str) -> dict[str, IndexedDoc]:
        """One :class:`IndexedDoc` per ``doc_id`` currently in ``collection``.

        ``chunk_count`` is the number of chunks actually stored for that
        ``doc_id`` right now (not just what a well-formed document would
        have) so callers can detect leftover stale chunks even once
        ``doc_hash`` matches again.
        """

        states: dict[str, IndexedDoc] = {}
        counts: dict[str, int] = {}
        for point in self._scroll(collection, with_vector=False):
            payload = point["payload"]
            doc_id = payload["doc_id"]
            counts[doc_id] = counts.get(doc_id, 0) + 1
            if doc_id not in states:
                states[doc_id] = IndexedDoc(
                    doc_id=doc_id,
                    doc_hash=payload["doc_hash"],
                    doc_path=payload["doc_path"],
                )
        for doc_id, state in states.items():
            states[doc_id] = IndexedDoc(
                doc_id=state.doc_id,
                doc_hash=state.doc_hash,
                doc_path=state.doc_path,
                chunk_count=counts[doc_id],
            )
        return states

    def doc_chunks(self, collection: str, doc_id: str) -> list[tuple[dict, list[float]]]:
        """Every ``(payload, vector)`` pair currently stored for ``doc_id``."""

        return [
            (point["payload"], point["vector"])
            for point in self._scroll(collection, doc_id=doc_id, with_vector=True)
        ]

    def _scroll(
        self, collection: str, *, doc_id: str | None = None, with_vector: bool
    ) -> Iterator[dict]:
        query_filter = None
        if doc_id is not None:
            query_filter = {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}
        offset = None
        while True:
            body: dict = {"with_payload": True, "with_vector": with_vector, "limit": 200}
            if query_filter is not None:
                body["filter"] = query_filter
            if offset is not None:
                body["offset"] = offset
            result = self._post(f"/collections/{collection}/points/scroll", body)
            points = result.get("points", [])
            yield from points
            offset = result.get("next_page_offset")
            if not points or offset is None:
                break

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")).get("result", {})


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _embed_document(
    scanned: ScannedFile, embed_client
) -> tuple[list[QdrantPayload], list[list[float]]]:
    """Chunk and embed ``scanned``, returning payloads/vectors ready to upsert.

    Does not touch the index. Embedding is the failure-prone step (transient
    Voyage outages), so callers must finish this before touching any of the
    document's existing chunks — otherwise a failed embed leaves the document
    unindexed until the next reindex pass.
    """

    pieces = chunk(scanned.body)
    now = datetime.now().astimezone()
    payloads = [
        QdrantPayload(
            doc_id=scanned.doc_id,
            doc_path=scanned.doc_path,
            doc_hash=scanned.doc_hash,
            doc_title=scanned.doc_title,
            chunk_index=piece.chunk_index,
            chunk_of=piece.chunk_of,
            header_path=piece.header_path,
            content=piece.content,
            tags=scanned.tags,
            project=scanned.project,
            source_type=scanned.source_type,
            source_ref=scanned.source_ref,
            created_at=scanned.created_at,
            indexed_at=now,
        )
        for piece in pieces
    ]
    vectors = [embed_client.embed(piece.content) for piece in pieces]
    return payloads, vectors


def _rename_document(
    collection: str,
    scanned: ScannedFile,
    reader: QdrantStateReader,
    index_client: IndexClient,
) -> None:
    """Rewrite ``doc_path`` on ``scanned``'s existing chunks; no re-embed."""

    chunks = reader.doc_chunks(collection, scanned.doc_id)
    if not chunks:
        return
    payloads = []
    vectors = []
    for payload_dict, vector in chunks:
        updated = dict(payload_dict)
        updated["doc_path"] = scanned.doc_path
        payloads.append(QdrantPayload.model_validate(updated))
        vectors.append(vector)
    index_client.upsert_chunks(collection, payloads, vectors)


def _reconcile(
    collection: str,
    scanned_files: dict[str, ScannedFile],
    reader: QdrantStateReader,
    embed_client,
    index_client: IndexClient,
    summary: ReindexSummary,
) -> None:
    """Reconcile one collection's scanned files against its indexed state."""

    indexed_docs = reader.doc_states(collection)
    for doc_id in set(scanned_files) | set(indexed_docs):
        scanned = scanned_files.get(doc_id)
        indexed = indexed_docs.get(doc_id)
        action = decide_action(scanned, indexed)
        try:
            if action is ReindexAction.PURGE:
                index_client.delete_by_doc_id(collection, doc_id)
                summary.purged += 1
            elif action is ReindexAction.INDEX:
                payloads, vectors = _embed_document(scanned, embed_client)
                index_client.upsert_chunks(collection, payloads, vectors)
                if indexed is not None:
                    # New chunks reuse the same point ids as their old
                    # counterparts (same doc_id/chunk_index), so the upsert
                    # above already overwrote every overlapping chunk live.
                    # Only chunks beyond the new document's length can still
                    # be stale leftovers from a previous, longer version.
                    index_client.delete_stale_chunks(collection, doc_id, len(payloads))
                summary.indexed += 1
            elif action is ReindexAction.RENAME:
                _rename_document(collection, scanned, reader, index_client)
                summary.renamed += 1
            else:
                # NOOP means doc_hash already matches, but if a *previous*
                # pass's INDEX action upserted new chunks and then failed
                # its compensating delete_stale_chunks call, this doc_id
                # would land here on every pass after that (decide_action
                # only looks at doc_hash), silently leaving those stale
                # chunks in the index forever. Retry the delete whenever
                # more chunks are stored than the current document would
                # produce; chunking (not embedding) is cheap and the delete
                # is a no-op when nothing is actually stale.
                assert scanned is not None and indexed is not None
                current_chunk_count = len(chunk(scanned.body))
                if current_chunk_count < indexed.chunk_count:
                    index_client.delete_stale_chunks(
                        collection, doc_id, current_chunk_count
                    )
                summary.unchanged += 1
        except Exception:  # noqa: BLE001 - one bad doc must not abort the scan
            logger.exception(
                "reindex failed for doc_id=%s collection=%s", doc_id, collection
            )
            summary.errors += 1


def reindex_once(
    vault_path: Path,
    embed_client,
    index_client: IndexClient,
    *,
    reader: QdrantStateReader | None = None,
    log_lock: threading.Lock | None = None,
) -> ReindexSummary:
    """Run one full reindex pass over the vault. Never raises past this call.

    ``log_lock``, when shared with :class:`kb.server.log_pipeline.LogPipeline`
    (see ``kb.server.app``'s wiring), serializes the ``AI-Daily-Log/`` scan
    and reconcile below against ``LogPipeline.handle()``'s own
    read/embed/upsert sequence. ``AI-Daily-Log/`` is scanned fresh *inside*
    the lock (not reused from an earlier snapshot) so that a delta the
    pipeline just indexed can never be scanned-past-then-clobbered by this
    pass's delete-and-rebuild. Defaults to a private lock when not shared,
    which is safe but gives no cross-pipeline protection.
    """

    vault_path = Path(vault_path)
    reader = reader if reader is not None else QdrantStateReader()
    log_lock = log_lock if log_lock is not None else threading.Lock()
    index_client.ensure_collections()
    summary = ReindexSummary()
    _reconcile(
        KNOWLEDGE_COLLECTION,
        scan_knowledge_files(vault_path),
        reader,
        embed_client,
        index_client,
        summary,
    )
    with log_lock:
        _reconcile(
            LOG_COLLECTION,
            scan_log_files(vault_path),
            reader,
            embed_client,
            index_client,
            summary,
        )
    try:
        summary.embed_failed_retried = retry_embed_failed_notes(
            vault_path, embed_client, index_client
        )
    except Exception:  # noqa: BLE001 - one bad sweep must not abort the pass
        logger.exception("reindex: embed_failed retry sweep failed")
        summary.errors += 1
    logger.info(
        "reindex pass complete: indexed=%d renamed=%d purged=%d unchanged=%d "
        "errors=%d embed_failed_retried=%d",
        summary.indexed,
        summary.renamed,
        summary.purged,
        summary.unchanged,
        summary.errors,
        summary.embed_failed_retried,
    )
    return summary


class ReindexScheduler:
    """Runs :func:`reindex_once` on a fixed interval in a background thread.

    Mirrors the start/stop lifecycle of :class:`kb.server.watcher.InboxWatcher`.
    """

    def __init__(
        self,
        vault_path: Path,
        embed_client,
        index_client: IndexClient,
        *,
        interval_seconds: float = REINDEX_INTERVAL_SECONDS,
        reader: QdrantStateReader | None = None,
        log_lock: threading.Lock | None = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self._embed_client = embed_client
        self._index_client = index_client
        self._interval = interval_seconds
        self._reader = reader
        self._log_lock = log_lock
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                reindex_once(
                    self.vault_path,
                    self._embed_client,
                    self._index_client,
                    reader=self._reader,
                    log_lock=self._log_lock,
                )
            except Exception:  # noqa: BLE001 - keep the scheduler alive
                logger.exception("periodic reindex scan failed")
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        """Start the background scan loop (fires immediately, then every interval)."""

        self._thread = threading.Thread(
            target=self._run, name="kb-reindex-scheduler", daemon=True
        )
        self._thread.start()
        logger.info("reindex scheduler started (interval=%ss)", self._interval)

    def stop(self) -> None:
        """Stop the loop and wait for its thread to exit."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def is_alive(self) -> bool:
        """Whether the scan-loop thread is still running.

        ``False`` before ``start()``/after ``stop()``, or if an unhandled
        exception somehow escaped ``_run``'s own try/except and killed the
        thread — so callers can detect a silently-dead scheduler.
        """

        return self._thread is not None and self._thread.is_alive()
