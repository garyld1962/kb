"""Inbox processing pipeline: turn a captured note into indexed chunks.

On a new file in ``Inbox/`` the pipeline (per the design spec's "Inbox
pipeline" section):

1. Parse the YAML frontmatter into :class:`~kb.core.models.InboxNoteFrontmatter`.
2. Skip the file if ``status != "inbox"``.
3. Compute ``doc_hash`` = ``sha256:<hex>`` of the raw file contents.
4. Clean/normalize the body (strip HTML for ``url`` captures, fix line endings).
5. Chunk the body (:func:`kb.server.chunker.chunk`).
6. Embed each chunk (:class:`kb.server.embed.EmbedClient`).
7. Upsert the chunks into ``kb_knowledge`` (:class:`kb.server.index.IndexClient`).
8. Move ``Inbox/note.md`` -> ``<destination>/note.md``.
9. Rewrite the frontmatter (``status: processed``, ``processed_at``,
   ``chunk_count``).

Malformed frontmatter (or an unreadable/binary file) is moved to
``Inbox/_errors/`` with a sibling ``.error.log``; the handler never raises
past its own boundary so a filesystem watcher keeps running. An unknown
``project`` is auto-registered in ``_project-registry.md`` and warned about.

If Ollama or Qdrant is unreachable, the embed/index clients retry with
exponential backoff (:mod:`kb.server.embed`, :mod:`kb.server.index`); once
those retries are exhausted the note's frontmatter is rewritten in place
with ``status: embed_failed`` and the file is left where it is (not moved,
not quarantined). ``status: embed_failed`` is treated the same as
``status: inbox`` by this module — the note stays retryable rather than
stuck — so either a later watcher event or :func:`kb.server.reindex.reindex_once`'s
periodic ``Inbox/`` sweep can pick it back up. No partial index is ever
written: chunks are only upserted once every chunk in the note has embedded
successfully.

Any other failure while chunking, embedding, or indexing (a non-network bug,
e.g. an embedding-dimension mismatch or a chunker/index defect) is treated as
non-transient: the note is quarantined to ``Inbox/_errors/`` rather than left
as ``status: inbox``, since retrying it would never succeed on its own and a
stuck-but-untouched note would never surface to an operator.

A failure moving the processed file into its destination (e.g. a permission
mismatch between the process's user and the destination directory) is
likewise non-transient: the just-upserted chunks are rolled back
(``delete_by_doc_id``) and the note is quarantined, rather than left
``status: inbox`` with content already searchable at a ``doc_path`` the file
never actually reached.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from kb.core.models import InboxNoteFrontmatter, QdrantPayload
from kb.server import registry
from kb.server.chunker import chunk
from kb.server.embed import EmbedClient, EmbedUnavailableError
from kb.server.index import KNOWLEDGE_COLLECTION, IndexClient, IndexUnavailableError

logger = logging.getLogger("kb.server.inbox_pipeline")

# Subdirectory of ``Inbox/`` that quarantined files are moved to.
ERRORS_DIRNAME = "_errors"

# Fallback destination folder when a note omits (or empties) ``destination``.
DEFAULT_DESTINATION = "Knowledge"

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Serializes process_inbox_file() per-path across threads. InboxWatcher's
# observer thread and ReindexScheduler's background thread (via
# retry_embed_failed_notes) can both dispatch the same Inbox path at once
# (e.g. an embed_failed note gets a filesystem event just as the periodic
# sweep picks it up); without this, both threads embed/upsert the same
# chunks and race on the final shutil.move, with the loser hitting
# FileNotFoundError and reporting a false status="error" for a note the
# other thread actually processed successfully. One lock per resolved path,
# guarded by a meta-lock while the per-path dict itself is mutated.
_PATH_LOCKS: dict[Path, threading.Lock] = defaultdict(threading.Lock)
_PATH_LOCKS_META_LOCK = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _PATH_LOCKS_META_LOCK:
        return _PATH_LOCKS[path]


class MalformedNoteError(Exception):
    """Raised when an Inbox file cannot be parsed into a valid note."""


@dataclass
class InboxResult:
    """Outcome of :func:`process_inbox_file`.

    ``status`` is one of ``"processed"``, ``"skipped"`` (non-inbox status or a
    file that vanished), or ``"error"`` (quarantined / unexpected failure).
    """

    status: str
    path: Path
    chunk_count: int = 0
    destination: Path | None = None
    project_registered: bool = False
    reason: str | None = None


# --------------------------------------------------------------------------- #
# Pure-logic helpers (no network, unit-tested in tests/unit/test_inbox_pipeline)
# --------------------------------------------------------------------------- #


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``---`` fenced YAML frontmatter from the note body.

    Raises :class:`MalformedNoteError` if the block is missing, unterminated,
    not valid YAML, or not a mapping.
    """

    if not text.startswith("---\n"):
        raise MalformedNoteError("file has no YAML frontmatter block")
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise MalformedNoteError("unterminated YAML frontmatter block")
    _, raw_frontmatter, body = parts
    try:
        data = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as exc:
        raise MalformedNoteError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise MalformedNoteError("frontmatter is not a mapping")
    return data, body


def parse_note(text: str) -> tuple[InboxNoteFrontmatter, str]:
    """Parse ``text`` into ``(frontmatter, body)``.

    Raises :class:`MalformedNoteError` when the frontmatter block is absent or
    fails :class:`~kb.core.models.InboxNoteFrontmatter` validation.
    """

    data, body = split_frontmatter(text)
    try:
        frontmatter = InboxNoteFrontmatter.model_validate(data)
    except ValidationError as exc:
        raise MalformedNoteError(f"frontmatter failed validation: {exc}") from exc
    return frontmatter, body


def compute_doc_hash(text: str) -> str:
    """Return ``sha256:<hex>`` of ``text`` (the raw file contents)."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def normalize_content(body: str, source_type: str) -> str:
    """Clean the note body: normalize line endings; strip HTML for ``url``."""

    text = body.replace("\r\n", "\n").replace("\r", "\n")
    if source_type == "url":
        text = _HTML_TAG_RE.sub("", text)
    return text.strip()


def resolve_destination(frontmatter: InboxNoteFrontmatter, vault_path: Path) -> Path:
    """Resolve the destination folder for a note under ``vault_path``.

    The ``destination`` frontmatter field is a vault-relative folder hint
    (e.g. ``"Knowledge/research"``); an empty value falls back to
    :data:`DEFAULT_DESTINATION`. Raises :class:`MalformedNoteError` if the
    hint resolves outside ``vault_path`` (e.g. ``"../../etc"``).
    """

    hint = (frontmatter.destination or DEFAULT_DESTINATION).strip().strip("/")
    hint = hint or DEFAULT_DESTINATION
    vault_resolved = vault_path.resolve()
    dest = (vault_path / Path(hint)).resolve()
    if not dest.is_relative_to(vault_resolved):
        raise MalformedNoteError(f"destination {hint!r} escapes the vault root")
    return dest


def quarantine(path: Path, vault_path: Path, reason: str) -> Path:
    """Move ``path`` to ``Inbox/_errors/`` and write a sibling ``.error.log``.

    Returns the quarantined file's new path.
    """

    errors_dir = vault_path / "Inbox" / ERRORS_DIRNAME
    errors_dir.mkdir(parents=True, exist_ok=True)
    dest = errors_dir / path.name
    counter = 1
    while dest.exists():
        dest = errors_dir / f"{path.stem}-{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(dest))
    stamp = datetime.now().astimezone().isoformat()
    (errors_dir / f"{dest.name}.error.log").write_text(f"{stamp}\n{reason}\n")
    return dest


def _render_embed_failed_note(
    frontmatter: InboxNoteFrontmatter,
    body: str,
    *,
    reason: str,
) -> str:
    """Serialize the note in place with ``status: embed_failed`` for later retry."""

    data = frontmatter.model_dump(mode="json")
    data["status"] = "embed_failed"
    data["embed_failed_reason"] = reason
    frontmatter_yaml = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return f"---\n{frontmatter_yaml}---\n{body}"


def _render_processed_note(
    frontmatter: InboxNoteFrontmatter,
    body: str,
    *,
    chunk_count: int,
    processed_at: datetime,
) -> str:
    """Serialize the note with updated ``status``/``processed_at``/``chunk_count``."""

    data = frontmatter.model_dump(mode="json")
    data["status"] = "processed"
    data["processed_at"] = processed_at.isoformat()
    data["chunk_count"] = chunk_count
    frontmatter_yaml = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return f"---\n{frontmatter_yaml}---\n{body}"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def process_inbox_file(
    path: Path,
    *,
    vault_path: Path,
    embed_client: EmbedClient,
    index_client: IndexClient,
    registry_file: Path | None = None,
) -> InboxResult:
    """Process one Inbox file end to end. Never raises past this boundary.

    Serialized per-path (see :data:`_PATH_LOCKS`) so the watcher thread and
    the reindex scheduler's ``embed_failed`` retry sweep can never process
    the same file at once.
    """

    path = Path(path)
    with _lock_for(path.resolve()):
        return _process_inbox_file_locked(
            path,
            vault_path=vault_path,
            embed_client=embed_client,
            index_client=index_client,
            registry_file=registry_file,
        )


def _process_inbox_file_locked(
    path: Path,
    *,
    vault_path: Path,
    embed_client: EmbedClient,
    index_client: IndexClient,
    registry_file: Path | None = None,
) -> InboxResult:
    """Body of :func:`process_inbox_file`, run while its per-path lock is held."""

    vault_path = Path(vault_path)

    if not path.exists():
        # A duplicate watcher event for an already-processed (moved) file.
        return InboxResult(status="skipped", path=path, reason="file no longer exists")

    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        try:
            dest = quarantine(path, vault_path, f"unreadable file: {exc}")
        except Exception as quarantine_exc:  # noqa: BLE001 - contain so the watcher keeps running
            logger.exception(
                "failed to quarantine unreadable inbox file %s", path
            )
            return InboxResult(status="error", path=path, reason=str(quarantine_exc))
        logger.warning("quarantined unreadable inbox file %s: %s", path, exc)
        return InboxResult(status="error", path=dest, reason=str(exc))

    try:
        frontmatter, body = parse_note(raw)
    except MalformedNoteError as exc:
        try:
            dest = quarantine(path, vault_path, f"malformed frontmatter: {exc}")
        except Exception as quarantine_exc:  # noqa: BLE001 - contain so the watcher keeps running
            logger.exception(
                "failed to quarantine malformed inbox note %s", path
            )
            return InboxResult(status="error", path=path, reason=str(quarantine_exc))
        logger.warning("quarantined malformed inbox note %s: %s", path, exc)
        return InboxResult(status="error", path=dest, reason=str(exc))

    if frontmatter.status not in ("inbox", "embed_failed"):
        logger.info("skipping %s: status=%s", path, frontmatter.status)
        return InboxResult(
            status="skipped", path=path, reason=f"status={frontmatter.status}"
        )

    try:
        return _process_valid_note(
            path,
            frontmatter,
            body,
            vault_path=vault_path,
            embed_client=embed_client,
            index_client=index_client,
            registry_file=registry_file,
        )
    except Exception as exc:  # noqa: BLE001 - contain so the watcher keeps running
        # Unexpected failures land here; the file is left in place for a later
        # retry and the failure is logged, not raised. Ollama/Qdrant outages
        # are caught earlier, in ``_process_valid_note``, and marked
        # ``embed_failed`` rather than falling through to this generic branch.
        logger.exception("inbox processing failed for %s", path)
        return InboxResult(status="error", path=path, reason=str(exc))


def _process_valid_note(
    path: Path,
    frontmatter: InboxNoteFrontmatter,
    body: str,
    *,
    vault_path: Path,
    embed_client: EmbedClient,
    index_client: IndexClient,
    registry_file: Path | None,
) -> InboxResult:
    """Chunk, embed, index, move, and rewrite one validated inbox note."""

    normalized = normalize_content(body, frontmatter.source_type)

    # Auto-register an unknown project (folder = its destination hint) and warn.
    project_registered = False
    if frontmatter.project and registry_file is not None:
        if registry.lookup(frontmatter.project, registry_file) is None:
            folder = (frontmatter.destination or DEFAULT_DESTINATION).strip("/")
            registry.register(frontmatter.project, folder, registry_file)
            project_registered = True
            logger.warning(
                "unknown project %r: auto-registered folder %r",
                frontmatter.project,
                folder,
            )

    try:
        destination_dir = resolve_destination(frontmatter, vault_path)
    except MalformedNoteError as exc:
        dest = quarantine(path, vault_path, f"invalid destination: {exc}")
        logger.warning("quarantined inbox note %s: %s", path, exc)
        return InboxResult(status="error", path=dest, reason=str(exc))
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Two different Inbox notes can resolve to the same candidate destination
    # filename (e.g. both default to "Untitled capture.md"). Lock on that
    # candidate destination path too -- not just the source Inbox path -- so
    # the exists-check/move/write sequence below is atomic across the watcher
    # thread and the reindex scheduler's embed_failed retry thread; otherwise
    # both can pass the exists() check before either has moved, and the loser
    # silently overwrites the winner's persisted note.
    dest_lock_key = (destination_dir / path.name).resolve()
    with _lock_for(dest_lock_key):
        dest_path = destination_dir / path.name
        if dest_path.exists() and dest_path != path:
            dest_path = destination_dir / f"{path.stem}-{frontmatter.id.lower()}{path.suffix}"
        doc_path_rel = dest_path.relative_to(vault_path).as_posix()

        now = datetime.now().astimezone()
        payloads: list[QdrantPayload] = []
        vectors: list[list[float]] = []
        try:
            chunks = chunk(normalized)
            # Hash the note as it will actually be persisted (status: processed,
            # processed_at, chunk_count) rather than the pre-move Inbox bytes, so
            # reindex's disk-vs-index hash comparison matches on the very next
            # pass instead of always detecting a "changed" doc and re-embedding.
            final_text = _render_processed_note(
                frontmatter, body, chunk_count=len(chunks), processed_at=now
            )
            doc_hash = compute_doc_hash(final_text)
            for piece in chunks:
                payloads.append(
                    QdrantPayload(
                        doc_id=frontmatter.id,
                        doc_path=doc_path_rel,
                        doc_hash=doc_hash,
                        doc_title=frontmatter.title,
                        chunk_index=piece.chunk_index,
                        chunk_of=piece.chunk_of,
                        header_path=piece.header_path,
                        content=piece.content,
                        tags=frontmatter.tags,
                        project=frontmatter.project,
                        source_type=frontmatter.source_type,
                        source_ref=frontmatter.source_ref,
                        created_at=frontmatter.created,
                        indexed_at=now,
                    )
                )
                vectors.append(embed_client.embed(piece.content))

            index_client.upsert_chunks(KNOWLEDGE_COLLECTION, payloads, vectors)
        except (EmbedUnavailableError, IndexUnavailableError) as exc:
            # Ollama/Qdrant is unreachable even after the clients' own retry/backoff
            # exhausted. Nothing has been upserted yet (the upsert call only ever
            # happens once all chunks are embedded), so there is no partial index
            # to unwind. Mark the note for a later retry instead of moving it: the
            # hash-based re-index (Task 8) is doc_id-idempotent, so reprocessing it
            # in place is always safe.
            path.write_text(_render_embed_failed_note(frontmatter, body, reason=str(exc)))
            logger.warning("embed_failed for %s: %s", path, exc)
            return InboxResult(status="embed_failed", path=path, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - non-transient failure, quarantine rather than leave stuck
            # Anything other than the transient-outage errors above (e.g. a bug in
            # chunk()/IndexClient, or an EmbedClient ValueError on a dimension
            # mismatch) won't be fixed by retrying, so leaving the note as
            # ``status: inbox`` would strand it forever with no way for an
            # operator to notice or for reindex's embed_failed sweep to pick it
            # up. Quarantine it instead.
            dest = quarantine(path, vault_path, f"embed/index failed: {exc}")
            logger.exception("quarantined inbox note %s after embed/index failure", path)
            return InboxResult(status="error", path=dest, reason=str(exc))

        try:
            shutil.move(str(path), str(dest_path))
            dest_path.write_text(final_text)
        except OSError as exc:
            # The chunks are already upserted at this point. Leaving them in
            # place would index a doc_path the file never actually reaches
            # (e.g. a permission error moving into the destination), so roll
            # the upsert back before quarantining -- a stuck-but-indexed note
            # is worse than a stuck note, since search would silently point
            # at content that isn't where it claims to be.
            index_client.delete_by_doc_id(KNOWLEDGE_COLLECTION, frontmatter.id)
            dest = quarantine(path, vault_path, f"move to destination failed: {exc}")
            logger.exception("quarantined inbox note %s after move failure", path)
            return InboxResult(status="error", path=dest, reason=str(exc))

    logger.info("processed %s -> %s (%d chunks)", path.name, doc_path_rel, len(chunks))
    return InboxResult(
        status="processed",
        path=dest_path,
        chunk_count=len(chunks),
        destination=dest_path,
        project_registered=project_registered,
    )
