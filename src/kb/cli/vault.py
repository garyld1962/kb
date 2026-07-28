"""Offline vault operations backing ``kb add`` / ``kb log`` / ``kb readlog``.

These never touch the network: the capture path writes directly to the local
Obsidian vault and the read path parses markdown from disk. kb-server picks up
the written files asynchronously via its filesystem watcher.
"""

from __future__ import annotations

import fcntl
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from kb.core.config import Config
from kb.core.models import (
    InboxNoteFrontmatter,
    LogFileFrontmatter,
    SourceType,
)

# Nightly-summary marker; new log entries are appended above it when present.
_SUMMARY_MARKER = "<!-- === kb-server nightly summary"

# Crockford base32 alphabet used for ULIDs.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# ``## HH:MM — Title`` entry heading in a daily log file.
_ENTRY_RE = re.compile(r"^##\s+(\d{2}:\d{2})\s+[—-]\s+(.*)$")
_MACHINE_RE = re.compile(r"^\*machine:\s*(.+?)\s*\*$")


def _ulid() -> str:
    """Generate a ULID: 48-bit millisecond timestamp + 80 random bits."""
    value = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60]


def _dump_frontmatter(data: dict) -> str:
    """Serialize a frontmatter dict as a ``---`` fenced YAML block."""
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return f"---\n{body}---\n"


def _split_document(text: str) -> tuple[dict, str]:
    """Split ``---`` fenced YAML frontmatter from the body of a document.

    Raises ``ValueError`` if a frontmatter block is opened but never closed
    (e.g. a partial write or crash mid-append left only the leading ``---``).
    """
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        raise ValueError("unterminated YAML frontmatter block")
    _, fm, body = parts
    return yaml.safe_load(fm) or {}, body


# --------------------------------------------------------------------------- #
# kb add
# --------------------------------------------------------------------------- #


def _looks_like_file(candidate: str) -> bool:
    """Whether ``candidate`` is plausibly a path worth probing with ``stat``.

    Free-form captured text (multi-line, or implausibly long) is never a
    filesystem path; skip the probe for it so ``os.stat`` never sees a
    multi-line note as a path argument. A bare word with no path separator
    and no file extension (e.g. ``TODO``, ``README``) is also skipped even
    though it might happen to name a file in the current directory —
    otherwise ``kb add "TODO"`` would silently capture that file's contents
    instead of the literal word whenever such a file exists in the cwd.
    """
    if "\n" in candidate:
        return False
    try:
        max_len = os.pathconf("/", "PC_PATH_MAX")
    except (AttributeError, OSError, ValueError):
        max_len = 4096
    if len(candidate) > max_len:
        return False
    if "/" not in candidate and not Path(candidate).suffix:
        return False
    try:
        return Path(candidate).expanduser().is_file()
    except (OSError, ValueError):
        return False


def _detect_source(
    content: str, source_type: SourceType | None
) -> tuple[SourceType, str | None, str]:
    """Resolve ``(source_type, source_ref, body)`` from raw ``content``.

    An explicit ``source_type`` wins. Otherwise a leading ``http(s)://``
    becomes a ``url`` capture and an existing local path becomes a ``file``
    capture (with the file's text as the body); everything else is ``text``.
    """
    stripped = content.strip()
    if source_type == "url" or (
        source_type is None and re.match(r"^https?://\S+$", stripped)
    ):
        return "url", stripped, content
    if source_type == "file" or (
        source_type is None and stripped and _looks_like_file(stripped)
    ):
        file_path = Path(stripped).expanduser()
        return "file", str(file_path), file_path.read_text()
    return (source_type or "text"), None, content


def _derive_title(content: str, source_type: SourceType, source_ref: str | None) -> str:
    if source_type == "url" and source_ref:
        return source_ref
    for line in content.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return "Untitled capture"


def add_note(
    config: Config,
    content: str,
    *,
    title: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    destination: str | None = None,
    source_type: SourceType | None = None,
) -> Path:
    """Write a new Inbox capture note and return its path.

    The note's frontmatter conforms to :class:`InboxNoteFrontmatter`; the body
    is a ``# <title>`` heading followed by the captured content.
    """
    resolved_type, source_ref, body = _detect_source(content, source_type)
    resolved_title = title or _derive_title(content, resolved_type, source_ref)

    note = InboxNoteFrontmatter(
        id=_ulid(),
        created=datetime.now().astimezone(),
        source_machine=config.machine_name,
        source_type=resolved_type,
        source_ref=source_ref,
        title=resolved_title,
        tags=tags or [],
        project=project if project is not None else config.default_project,
        destination=destination or "Knowledge",
        status="inbox",
    )

    inbox = config.vault_path / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    slug = _slugify(resolved_title) or note.id.lower()
    path = inbox / f"{slug}.md"
    if path.exists():
        path = inbox / f"{slug}-{note.id}.md"

    document = _dump_frontmatter(note.model_dump(mode="json"))
    document += f"\n# {resolved_title}\n\n{body.rstrip()}\n"
    path.write_text(document)
    return path


# --------------------------------------------------------------------------- #
# kb log
# --------------------------------------------------------------------------- #


class InvalidProjectError(ValueError):
    """Raised when a ``project`` argument would escape ``AI-Daily-Log/``."""


def _resolve_project_dir(config: Config, project: str) -> Path:
    """Resolve ``project`` to its directory under ``AI-Daily-Log/``.

    Mirrors the containment check ``resolve_destination`` applies to the
    ``destination`` field server-side: a ``project`` is a single directory
    name, so path separators are rejected outright, and (belt-and-braces)
    the resolved path must stay inside ``AI-Daily-Log/`` — catching
    separator-free escapes like ``".."``.
    """
    if not project or "/" in project or "\\" in project:
        raise InvalidProjectError(f"invalid project name: {project!r}")
    base = (config.vault_path / "AI-Daily-Log").resolve()
    proj_dir = (base / project).resolve()
    if not proj_dir.is_relative_to(base):
        raise InvalidProjectError(f"invalid project name: {project!r}")
    return proj_dir


@dataclass
class LogAppendResult:
    """Outcome of :func:`append_log_entry`."""

    path: Path
    project_created: bool


def _escape_heading_lookalike(line: str) -> str:
    """Escape a body line that would otherwise match ``_ENTRY_RE``.

    Both this module's ``read_log_entries`` and the server's
    ``log_pipeline.parse_entries`` treat any line matching ``## HH:MM — …``
    as a new entry boundary. Without this, a logged message whose own
    content happens to contain such a line would silently fork into a bogus
    extra entry with caller-controlled time/title. A leading backslash is
    the standard Markdown escape for a literal ``#``, so it also renders
    correctly when the file is viewed as Markdown.
    """
    return "\\" + line if _ENTRY_RE.match(line) else line


def _render_entry(message: str, machine: str) -> str:
    lines = message.splitlines() or [""]
    head = lines[0].strip()
    rest = "\n".join(_escape_heading_lookalike(line) for line in lines[1:]).strip()
    now = datetime.now().astimezone()
    block = f"\n## {now.strftime('%H:%M')} — {head}\n*machine: {machine}*\n"
    if rest:
        block += f"\n{rest}\n"
    return block


@contextmanager
def _locked(lock_path: Path):
    """Hold an exclusive advisory lock on ``lock_path`` for the ``with`` body.

    Guards the read-modify-write in :func:`append_log_entry` against
    concurrent ``kb log`` invocations for the same project/day, which would
    otherwise race and silently drop whichever entry was written first.
    """
    lock_file = open(lock_path, "a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def append_log_entry(config: Config, project: str, message: str) -> LogAppendResult:
    """Append an entry to today's ``AI-Daily-Log/<project>/<date>.md``.

    Creates the project folder and/or the dated file (with frontmatter) when
    absent. ``project_created`` is ``True`` when the project folder did not
    previously exist, so the caller can warn about an unknown project.
    """
    proj_dir = _resolve_project_dir(config, project)
    project_created = not proj_dir.exists()
    proj_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    log_file = proj_dir / f"{today.isoformat()}.md"
    lock_path = proj_dir / f".{today.isoformat()}.lock"
    entry = _render_entry(message, config.machine_name)

    with _locked(lock_path):
        if not log_file.exists():
            frontmatter = LogFileFrontmatter(
                project=project, date=today, status="active", entries=1
            )
            header = f"\n# {project} — {today.isoformat()}\n"
            log_file.write_text(
                _dump_frontmatter(frontmatter.model_dump(mode="json")) + header + entry
            )
            return LogAppendResult(path=log_file, project_created=project_created)

        fm_dict, body = _split_document(log_file.read_text())
        fm_dict["entries"] = int(fm_dict.get("entries", 0)) + 1

        marker_pos = body.find(_SUMMARY_MARKER)
        if marker_pos != -1:
            body = body[:marker_pos] + entry + "\n" + body[marker_pos:]
        else:
            body = body.rstrip() + "\n" + entry

        log_file.write_text(_dump_frontmatter(fm_dict) + body)
        return LogAppendResult(path=log_file, project_created=project_created)


# --------------------------------------------------------------------------- #
# kb readlog
# --------------------------------------------------------------------------- #


@dataclass
class LogEntry:
    """A single parsed log entry."""

    date: str
    time: str
    title: str
    machine: str | None
    body: str


def _parse_log_file(day: str, text: str) -> list[LogEntry]:
    fm_dict, body = _split_document(text)
    day = str(fm_dict.get("date", day))
    entries: list[LogEntry] = []
    current: LogEntry | None = None
    body_lines: list[str] = []

    def _flush() -> None:
        if current is not None:
            current.body = "\n".join(body_lines).strip()
            entries.append(current)

    for line in body.splitlines():
        heading = _ENTRY_RE.match(line)
        if heading:
            _flush()
            current = LogEntry(
                date=day, time=heading.group(1), title=heading.group(2).strip(),
                machine=None, body="",
            )
            body_lines = []
            continue
        if current is None:
            continue
        machine = _MACHINE_RE.match(line.strip())
        if machine and current.machine is None:
            current.machine = machine.group(1)
            continue
        body_lines.append(line)
    _flush()
    return entries


def _dates_in_range(start: str, end: str) -> list[str]:
    lo = date.fromisoformat(start)
    hi = date.fromisoformat(end)
    if hi < lo:
        lo, hi = hi, lo
    days: list[str] = []
    cur = lo
    while cur <= hi:
        days.append(cur.isoformat())
        cur = date.fromordinal(cur.toordinal() + 1)
    return days


def read_log_entries(
    config: Config,
    project: str,
    *,
    date: str | None = None,
    date_range: tuple[str, str] | None = None,
) -> list[LogEntry]:
    """Read log entries for ``project`` directly from the vault.

    With ``date_range`` (an inclusive ``(start, end)`` pair) entries from every
    existing dated file in the span are concatenated in chronological order.
    With ``date`` a single day is read. With neither, today's log is read.
    """
    proj_dir = _resolve_project_dir(config, project)
    if date_range is not None:
        days = _dates_in_range(*date_range)
    elif date is not None:
        days = [date]
    else:
        from datetime import date as _date

        days = [_date.today().isoformat()]

    entries: list[LogEntry] = []
    for day in days:
        log_file = proj_dir / f"{day}.md"
        if log_file.is_file():
            entries.extend(_parse_log_file(day, log_file.read_text()))
    return entries
