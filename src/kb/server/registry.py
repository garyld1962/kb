"""Read/write helpers for the vault's ``_project-registry.md``.

The registry file has YAML frontmatter listing known projects (each a
``project -> folder`` mapping plus ``status`` and ``created`` date) followed
by a free markdown body for human notes. kb-server reads the frontmatter
only; the body is preserved verbatim across writes.

``register`` implements the auto-create half of the CLI's auto-create/warn
flow: an unknown project gets a stub entry appended; the warning itself is
the caller's responsibility.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

import yaml

from kb.core.models import ProjectRegistryEntry

# Location of the registry relative to the vault root, per the design spec's
# vault structure (``AI-Daily-Log/_project-registry.md``).
REGISTRY_RELPATH = Path("AI-Daily-Log") / "_project-registry.md"

# Default body written when ``register`` creates the file from scratch.
_DEFAULT_BODY = "# Project Registry\n\nKnown projects and their vault folders.\n"

# Serializes register()'s read-modify-write across threads. InboxWatcher's
# observer thread and ReindexScheduler's background thread (via
# retry_embed_failed_notes) both call process_inbox_file, which can invoke
# register() concurrently for the same registry file; without this lock two
# racing unknown-project registrations can each read the same entries list
# and the second write clobbers the first's stub entry.
_REGISTRY_LOCK = threading.Lock()


def registry_path(vault_path: Path) -> Path:
    """Return the registry path for a given vault root."""

    return vault_path / REGISTRY_RELPATH


def read_registry(path: Path) -> tuple[list[ProjectRegistryEntry], str]:
    """Parse ``path`` into its project entries and free markdown body.

    A missing file is treated as an empty registry with an empty body.
    """

    if not path.exists():
        return [], ""

    frontmatter, body = _split_frontmatter(path.read_text())
    raw_entries = frontmatter.get("projects") or []
    entries = [ProjectRegistryEntry.model_validate(e) for e in raw_entries]
    return entries, body


def lookup(project: str, path: Path) -> str | None:
    """Return the vault folder registered for ``project``, or ``None``."""

    entries, _ = read_registry(path)
    for entry in entries:
        if entry.project == project:
            return entry.folder
    return None


def register(project: str, folder: str, path: Path) -> ProjectRegistryEntry:
    """Ensure ``project`` is registered, appending a stub entry if unknown.

    Idempotent: if ``project`` is already present its existing entry is
    returned and the file is left untouched. Otherwise a stub entry
    (``status: active``, ``created`` today) is appended and the file is
    rewritten, preserving existing entries and the markdown body.
    """

    with _REGISTRY_LOCK:
        entries, body = read_registry(path)
        for entry in entries:
            if entry.project == project:
                return entry

        stub = ProjectRegistryEntry(
            project=project,
            folder=folder,
            status="active",
            created=date.today(),
        )
        entries.append(stub)
        _write_registry(path, entries, body or _DEFAULT_BODY)
        return stub


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``---``-delimited YAML frontmatter from the trailing body."""

    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    _, raw_frontmatter, body = parts
    data = yaml.safe_load(raw_frontmatter) or {}
    return data, body.lstrip("\n")


def _write_registry(
    path: Path, entries: list[ProjectRegistryEntry], body: str
) -> None:
    """Serialize ``entries`` as frontmatter above ``body`` and write ``path``."""

    frontmatter = yaml.safe_dump(
        {"projects": [e.model_dump(mode="json") for e in entries]},
        sort_keys=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}---\n\n{body}")
