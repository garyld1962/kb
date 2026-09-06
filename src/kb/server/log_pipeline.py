"""Log pipeline: incrementally index AI-Daily-Log entries into ``kb_logs``.

``kb log`` writes directly to ``AI-Daily-Log/<project>/<YYYY-MM-DD>.md`` (see
``kb.cli.vault``), bypassing the Inbox. Those files are append-only: new
entries are added above the nightly-summary marker, earlier entries never move.

On each file change this pipeline chunks **only the entries added since the
last index** (the delta), embeds them, and upserts them into ``kb_logs``. It
reuses the markdown chunker (Task 2) and the embed/index clients (Task 3).

Delta tracking is in-memory, keyed per document: a :class:`LogPipeline`
instance remembers which entry headings it has already indexed for each log
file. That matches the long-running watcher runtime (one process handling a
stream of change events). Reconciliation across restarts / out-of-band edits
to already-indexed entries is the hash-based re-index pass (Task 8), not this
incremental path.

Because logs are append-only, an entry's global ``chunk_index`` is stable
across re-runs, so upserting only the delta never disturbs prior points.
``chunk_of`` on already-stored chunks is intentionally left at its
index-time value rather than rewritten on every append — rewriting it would
defeat the point of delta indexing.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from pathlib import Path

import yaml
from pydantic import ValidationError

from kb.core.models import LogFileFrontmatter, QdrantPayload
from kb.server.chunker import chunk
from kb.server.embed import EmbedUnavailableError
from kb.server.index import LOG_COLLECTION, IndexClient, IndexUnavailableError

logger = logging.getLogger("kb.server.log_pipeline")

# ``## HH:MM — Title`` entry heading (em dash or hyphen), matching the format
# written by ``kb.cli.vault``. Any line matching this pattern is treated as a
# new entry boundary, so ``kb.cli.vault._render_entry`` backslash-escapes any
# logged-message line that would otherwise collide with it before writing —
# without that, caller-controlled message content could forge a bogus entry
# boundary with attacker-chosen time/title.
_ENTRY_RE = re.compile(r"^##\s+(\d{2}:\d{2})\s+[—-]\s+(.*)$")

# Nightly-summary marker; everything from here down is generated summary, not
# an indexable work-log entry.
_SUMMARY_MARKER = "<!-- === kb-server nightly summary"

# Log chunks are AI work-log entries; ``agent`` is the closest source_type.
_LOG_SOURCE_TYPE = "agent"


@dataclass(frozen=True)
class LogEntryBlock:
    """One parsed log entry: its identity and full markdown body."""

    key: str  # stable identity within the file (ordinal position + heading text)
    time: str  # "HH:MM"
    title: str
    markdown: str  # heading + body, trailing ``---`` rule/blank lines removed


@dataclass(frozen=True)
class LogIndexResult:
    """Outcome of :meth:`LogPipeline.handle` for one file-change event.

    ``status`` is ``"ok"`` unless Voyage/Qdrant was unreachable, in which case
    it is ``"embed_failed"`` and nothing from this delta was indexed; the
    in-memory ``_indexed`` set is left untouched so the same delta is retried
    on the next change event (safe, since re-index (Task 8) is
    ``doc_id``-idempotent). It is ``"error"`` when the file itself could not
    be read or parsed (unreadable, unterminated frontmatter, failed
    validation); ``doc_id`` is a placeholder in that case since the real one
    couldn't be determined. It is also ``"error"`` when chunking/embedding/
    indexing the delta raised something other than the transient-outage
    errors above (e.g. a chunker bug or an embed dimension mismatch); in that
    case ``doc_id`` is the real one, but the delta is not marked indexed and
    will keep failing on retry until the underlying bug is fixed.
    """

    doc_id: str
    entries_indexed: int
    chunks_indexed: int
    status: str = "ok"


def log_doc_id(project: str, date_value) -> str:
    """Deterministic ``doc_id`` for a project's dated log file."""

    return f"log:{project}:{date_value}"


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``---`` fenced YAML frontmatter from the document body.

    Raises ``ValueError`` if a frontmatter block is opened but never closed
    (e.g. a partial write or crash mid-append left only the leading ``---``).
    """

    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise ValueError("unterminated YAML frontmatter block")
    _, raw, body = parts
    return yaml.safe_load(raw) or {}, body


def parse_entries(body: str) -> list[LogEntryBlock]:
    """Parse a log body into its work-log entries, in file order.

    Content before the first ``## HH:MM — …`` heading (the ``# Project — Date``
    H1) and everything from the nightly-summary marker onward are ignored.

    ``key`` is derived from each entry's ordinal position in the file, not
    from ``time + title`` alone: two entries in the same minute with
    identical titles would otherwise collide, and since files are
    append-only, ordinal position is a stable identity across re-parses.
    """

    entries: list[LogEntryBlock] = []
    heading: str | None = None
    entry_time = ""
    entry_title = ""
    lines: list[str] = []
    ordinal = 0

    def _flush() -> None:
        nonlocal ordinal
        if heading is None:
            return
        block = "\n".join(lines).rstrip()
        # Drop a trailing horizontal-rule separator between entries.
        while block.endswith("---"):
            block = block[: -len("---")].rstrip()
        entries.append(
            LogEntryBlock(
                key=f"{ordinal:05d}:{heading}",
                time=entry_time,
                title=entry_title,
                markdown=block.rstrip(),
            )
        )
        ordinal += 1

    for line in body.splitlines():
        if line.startswith(_SUMMARY_MARKER):
            break
        match = _ENTRY_RE.match(line)
        if match:
            _flush()
            entry_time, entry_title = match.group(1), match.group(2).strip()
            heading = f"{entry_time} — {entry_title}"
            lines = [line]
            continue
        if heading is not None:
            lines.append(line)
    _flush()
    return entries


def select_delta(
    entries: list[LogEntryBlock], indexed_keys: set[str]
) -> list[LogEntryBlock]:
    """Return the entries not yet indexed, preserving file order."""

    return [entry for entry in entries if entry.key not in indexed_keys]


def _entry_created_at(date_value, entry_time: str) -> datetime:
    """Combine a log file's date with an entry's ``HH:MM`` into a datetime."""

    return datetime.combine(date_value, dt_time.fromisoformat(entry_time)).astimezone()


class LogPipeline:
    """Incrementally index append-only AI-Daily-Log files into ``kb_logs``.

    ``embed_client`` and ``index_client`` are the Task 3 clients (or any
    duck-typed stand-ins with the same ``embed`` / ``ensure_collections`` /
    ``upsert_chunks`` surface).

    ``lock``, when shared with :class:`kb.server.reindex.ReindexScheduler`
    (see ``kb.server.app``'s wiring), serializes this pipeline's
    read-file/embed/upsert sequence against the periodic reindex pass's
    scan-and-reconcile of ``AI-Daily-Log/``. Without that, a reindex pass can
    scan the file, then a concurrent ``handle()`` call embeds/upserts a
    just-appended entry, then the reindex pass deletes-and-rebuilds the whole
    doc_id from its now-stale scan, silently dropping the new entry. Defaults
    to a private lock when not shared, which is safe but gives no
    cross-pipeline protection.
    """

    def __init__(
        self,
        vault_path: Path,
        embed_client,
        index_client: IndexClient,
        *,
        collection: str = LOG_COLLECTION,
        lock: threading.Lock | None = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self.embed = embed_client
        self.index = index_client
        self.collection = collection
        # doc_id -> set of entry keys already embedded/upserted this session.
        self._indexed: dict[str, set[str]] = {}
        self._collections_ready = False
        self._lock = lock if lock is not None else threading.Lock()

    def handle(self, path: Path | str) -> LogIndexResult:
        """Index the delta for one changed log file. Never raises past this boundary."""

        with self._lock:
            return self._handle_locked(Path(path))

    def _handle_locked(self, path: Path) -> LogIndexResult:
        try:
            text = path.read_text()
            frontmatter, body = split_frontmatter(text)
            fm = LogFileFrontmatter.model_validate(frontmatter)
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("skipping unparseable log file %s: %s", path, exc)
            return LogIndexResult(
                doc_id=f"log:unparseable:{path}",
                entries_indexed=0,
                chunks_indexed=0,
                status="error",
            )

        doc_id = log_doc_id(fm.project, fm.date.isoformat())
        doc_hash = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_title = f"{fm.project} — {fm.date.isoformat()}"
        doc_path = self._relpath(path)

        entries = parse_entries(body)
        indexed = self._indexed.setdefault(doc_id, set())

        try:
            # Chunk every entry up front so the global chunk_index / chunk_of
            # numbering spans the whole document, then embed only new entries'
            # chunks. Append-only order keeps prior chunk_index values stable.
            per_entry = [(entry, chunk(entry.markdown)) for entry in entries]
            total_chunks = sum(len(chunks) for _, chunks in per_entry)

            payloads: list[QdrantPayload] = []
            contents: list[str] = []
            newly_indexed: list[str] = []
            indexed_at = datetime.now(timezone.utc)
            running = 0
            for entry, chunks in per_entry:
                is_new = entry.key not in indexed
                created_at = _entry_created_at(fm.date, entry.time)
                for ck in chunks:
                    chunk_index = running
                    running += 1
                    if not is_new:
                        continue
                    payloads.append(
                        QdrantPayload(
                            doc_id=doc_id,
                            doc_path=doc_path,
                            doc_hash=doc_hash,
                            doc_title=doc_title,
                            chunk_index=chunk_index,
                            chunk_of=total_chunks,
                            header_path=ck.header_path,
                            content=ck.content,
                            tags=[],
                            project=fm.project,
                            source_type=_LOG_SOURCE_TYPE,
                            source_ref=None,
                            created_at=created_at,
                            indexed_at=indexed_at,
                        )
                    )
                    contents.append(ck.content)
                if is_new:
                    newly_indexed.append(entry.key)

            if payloads:
                self._ensure_ready()
                vectors = [self.embed.embed(content) for content in contents]
                self.index.upsert_chunks(self.collection, payloads, vectors)
        except (EmbedUnavailableError, IndexUnavailableError) as exc:
            # Voyage/Qdrant is unreachable even after the clients' own
            # retry/backoff exhausted. ``indexed`` is not updated, so this
            # same delta is picked up again on the next change event.
            logger.warning("embed_failed indexing delta for %s: %s", doc_id, exc)
            return LogIndexResult(
                doc_id=doc_id,
                entries_indexed=0,
                chunks_indexed=0,
                status="embed_failed",
            )
        except Exception as exc:  # noqa: BLE001 - non-transient failure (e.g. a
            # chunker bug or an embed dimension-mismatch ValueError); retrying
            # would not help, so log and report an error instead of letting it
            # propagate to LogEventHandler and kill the watcher thread.
            logger.exception("error indexing delta for %s: %s", doc_id, exc)
            return LogIndexResult(
                doc_id=doc_id,
                entries_indexed=0,
                chunks_indexed=0,
                status="error",
            )

        indexed.update(newly_indexed)
        return LogIndexResult(
            doc_id=doc_id,
            entries_indexed=len(newly_indexed),
            chunks_indexed=len(payloads),
        )

    def _ensure_ready(self) -> None:
        """Bootstrap the Qdrant collections once per pipeline instance."""

        if not self._collections_ready:
            self.index.ensure_collections()
            self._collections_ready = True

    def _relpath(self, path: Path) -> str:
        """Vault-relative POSIX path for ``doc_path``; absolute if outside."""

        try:
            return path.relative_to(self.vault_path).as_posix()
        except ValueError:
            return path.as_posix()
