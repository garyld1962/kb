"""Pydantic models shared across the kb CLI and server.

These mirror the file formats and Qdrant payload defined in the design spec
(docs/specs/2026-04-12-kb-design.md). They are the single-owner contract for
frontmatter parsing/serialization and index payload shape.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["url", "text", "file", "stdin", "mcp", "agent"]
InboxStatus = Literal["inbox", "processed", "embed_failed", "error"]
LogStatus = Literal["active", "closed"]


class InboxNoteFrontmatter(BaseModel):
    """YAML frontmatter of a capture note in ``Inbox/``."""

    id: str
    created: datetime
    source_machine: str
    source_type: SourceType
    source_ref: str | None = None
    title: str
    tags: list[str] = Field(default_factory=list)
    project: str = ""
    destination: str
    status: InboxStatus = "inbox"


class LogFileFrontmatter(BaseModel):
    """YAML frontmatter of a daily project log file in ``AI-Daily-Log/``."""

    project: str
    date: date
    status: LogStatus = "active"
    entries: int = 0


class QdrantPayload(BaseModel):
    """Payload stored alongside each chunk vector in Qdrant.

    Identical shape for both the ``kb_knowledge`` and ``kb_logs`` collections.
    """

    doc_id: str
    doc_path: str
    doc_hash: str
    doc_title: str
    chunk_index: int
    chunk_of: int
    header_path: str
    content: str
    tags: list[str] = Field(default_factory=list)
    project: str = ""
    source_type: SourceType
    source_ref: str | None = None
    created_at: datetime
    indexed_at: datetime


class ProjectRegistryEntry(BaseModel):
    """One project entry from ``_project-registry.md`` frontmatter."""

    project: str
    folder: str
    status: str = "active"
    created: date
