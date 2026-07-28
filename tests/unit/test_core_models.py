"""Round-trip tests for the shared core models and config loader."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from kb.core.config import Config, load_config
from kb.core.models import (
    InboxNoteFrontmatter,
    LogFileFrontmatter,
    ProjectRegistryEntry,
    QdrantPayload,
)


def roundtrip(model):
    """Serialize a model to YAML frontmatter and parse it back."""
    dumped = yaml.safe_dump(model.model_dump(mode="json"))
    reparsed = yaml.safe_load(dumped)
    return type(model).model_validate(reparsed)


def test_inbox_note_frontmatter_roundtrip():
    note = InboxNoteFrontmatter(
        id="01HXYZABCDEF",
        created=datetime(2026, 4, 12, 14, 32, 0, tzinfo=timezone.utc),
        source_machine="workstation-1",
        source_type="url",
        source_ref="https://example.com/karpathy",
        title="Karpathy on knowledge bases",
        tags=["research", "llm", "rag"],
        project="",
        destination="Knowledge/research",
        status="inbox",
    )
    assert roundtrip(note) == note


def test_log_file_frontmatter_roundtrip():
    log = LogFileFrontmatter(
        project="baker-street",
        date=date(2026, 4, 12),
        status="active",
        entries=2,
    )
    assert roundtrip(log) == log


def test_qdrant_payload_roundtrip():
    payload = QdrantPayload(
        doc_id="01HXYZABCDEF",
        doc_path="Knowledge/research/karpathy-kb.md",
        doc_hash="sha256:abc123",
        doc_title="Karpathy on knowledge bases",
        chunk_index=3,
        chunk_of=7,
        header_path="Key Insights > Beyond RAG",
        content="some chunk text",
        tags=["research", "llm", "rag"],
        project="",
        source_type="url",
        source_ref="https://example.com/karpathy",
        created_at=datetime(2026, 4, 12, 14, 32, 0, tzinfo=timezone.utc),
        indexed_at=datetime(2026, 4, 12, 14, 32, 15, tzinfo=timezone.utc),
    )
    assert roundtrip(payload) == payload


def test_project_registry_entry_roundtrip():
    entry = ProjectRegistryEntry(
        project="baker-street",
        folder="AI-Daily-Log/baker-street",
        status="active",
        created=date(2026, 4, 12),
    )
    assert roundtrip(entry) == entry


def test_load_config_missing_file_returns_defaults(tmp_path):
    missing = tmp_path / "nope" / "config.yaml"
    config = load_config(missing)
    assert config == Config()
    assert config.server_url == "http://localhost:8090"
    assert config.default_project == ""


def test_load_config_present_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vault_path": "/data/obsidian",
                "server_url": "http://localhost:8090",
                "machine_name": "workstation-1",
                "default_project": "baker-street",
            }
        )
    )
    config = load_config(config_path)
    assert config.vault_path == Path("/data/obsidian")
    assert config.machine_name == "workstation-1"
    assert config.default_project == "baker-street"
