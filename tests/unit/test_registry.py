"""Unit tests for the project registry read/write helpers."""

from __future__ import annotations

from datetime import date

import yaml

from kb.core.models import ProjectRegistryEntry
from kb.server.registry import lookup, read_registry, register


def _seed(path, entries, body="# Project Registry\n\nKnown projects.\n"):
    """Write a well-formed ``_project-registry.md`` at ``path``."""
    frontmatter = yaml.safe_dump(
        {"projects": [e.model_dump(mode="json") for e in entries]},
        sort_keys=False,
    )
    path.write_text(f"---\n{frontmatter}---\n\n{body}")


def test_lookup_known_project(tmp_path):
    reg = tmp_path / "_project-registry.md"
    _seed(
        reg,
        [
            ProjectRegistryEntry(
                project="baker-street",
                folder="AI-Daily-Log/baker-street",
                status="active",
                created=date(2026, 4, 12),
            )
        ],
    )
    assert lookup("baker-street", reg) == "AI-Daily-Log/baker-street"


def test_lookup_unknown_project_returns_none(tmp_path):
    reg = tmp_path / "_project-registry.md"
    _seed(reg, [])
    assert lookup("nonexistent", reg) is None


def test_lookup_missing_file_returns_none(tmp_path):
    assert lookup("baker-street", tmp_path / "_project-registry.md") is None


def test_register_unknown_appends_well_formed_stub(tmp_path):
    reg = tmp_path / "_project-registry.md"
    _seed(reg, [])

    entry = register("buildflow", "AI-Daily-Log/buildflow", reg)

    assert entry.project == "buildflow"
    assert entry.folder == "AI-Daily-Log/buildflow"
    assert entry.status == "active"
    assert isinstance(entry.created, date)

    # The stub is discoverable via lookup and parses back cleanly.
    assert lookup("buildflow", reg) == "AI-Daily-Log/buildflow"
    entries, body = read_registry(reg)
    assert [e.project for e in entries] == ["buildflow"]
    assert "# Project Registry" in body


def test_register_creates_file_when_absent(tmp_path):
    reg = tmp_path / "_project-registry.md"
    register("ingest", "AI-Daily-Log/ingest", reg)
    assert reg.exists()
    assert lookup("ingest", reg) == "AI-Daily-Log/ingest"


def test_register_is_idempotent(tmp_path):
    reg = tmp_path / "_project-registry.md"
    _seed(reg, [])

    first = register("buildflow", "AI-Daily-Log/buildflow", reg)
    text_after_first = reg.read_text()
    second = register("buildflow", "AI-Daily-Log/buildflow", reg)
    text_after_second = reg.read_text()

    assert first == second
    # A second registration of the same project neither duplicates the entry
    # nor rewrites the file.
    assert text_after_first == text_after_second
    entries, _ = read_registry(reg)
    assert [e.project for e in entries] == ["buildflow"]


def test_register_preserves_existing_entries_and_body(tmp_path):
    reg = tmp_path / "_project-registry.md"
    _seed(
        reg,
        [
            ProjectRegistryEntry(
                project="baker-street",
                folder="AI-Daily-Log/baker-street",
                status="active",
                created=date(2026, 4, 12),
            )
        ],
        body="# Project Registry\n\nHand-written notes about projects.\n",
    )

    register("buildflow", "AI-Daily-Log/buildflow", reg)

    entries, body = read_registry(reg)
    assert [e.project for e in entries] == ["baker-street", "buildflow"]
    assert "Hand-written notes about projects." in body
    assert lookup("baker-street", reg) == "AI-Daily-Log/baker-street"
