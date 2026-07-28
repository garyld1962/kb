"""Unit tests for the kb CLI vault commands (add / log / readlog) and the
offline-safe server stubs (status / search).

All tests here are network-free: vault commands write/read the local vault,
and the server stubs are pointed at an unreachable port so the connection
failure path is exercised deterministically.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from kb.cli.main import app
from kb.cli.vault import InvalidProjectError, add_note, append_log_entry, read_log_entries
from kb.core.config import Config
from kb.core.models import InboxNoteFrontmatter

UNREACHABLE_MESSAGE = "kb-server unreachable. Captures still work offline."


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``---`` fenced YAML frontmatter from the markdown body."""
    assert text.startswith("---\n")
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm), body


def _make_config(tmp_path, **overrides) -> Config:
    return Config(
        vault_path=tmp_path / "vault",
        server_url=overrides.get("server_url", "http://127.0.0.1:9"),
        machine_name=overrides.get("machine_name", "testhost"),
        default_project=overrides.get("default_project", ""),
    )


def _write_config_file(tmp_path, **overrides):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "vault_path": str(tmp_path / "vault"),
                "server_url": overrides.get("server_url", "http://127.0.0.1:9"),
                "machine_name": overrides.get("machine_name", "testhost"),
                "default_project": overrides.get("default_project", ""),
            }
        )
    )
    return cfg_path


# --------------------------------------------------------------------------- #
# kb add
# --------------------------------------------------------------------------- #


def test_add_writes_well_formed_inbox_note(tmp_path):
    config = _make_config(tmp_path)
    path = add_note(
        config,
        "This is the captured body text.",
        title="A test capture",
        tags=["research", "llm"],
        destination="Knowledge/research",
    )

    # File lands in the vault's Inbox/ directory.
    assert path.parent == config.vault_path / "Inbox"
    assert path.suffix == ".md"

    fm_dict, body = _split_frontmatter(path.read_text())

    # Frontmatter parses cleanly through the shared model.
    note = InboxNoteFrontmatter.model_validate(fm_dict)
    assert note.status == "inbox"
    assert note.title == "A test capture"
    assert note.tags == ["research", "llm"]
    assert note.destination == "Knowledge/research"
    assert note.source_machine == "testhost"
    assert note.source_type == "text"
    assert note.id  # a non-empty stable id was assigned

    # Body carries the title heading and the captured content.
    assert "# A test capture" in body
    assert "This is the captured body text." in body


def test_add_detects_url_source_type(tmp_path):
    config = _make_config(tmp_path)
    path = add_note(config, "https://example.com/karpathy")
    fm_dict, _ = _split_frontmatter(path.read_text())
    note = InboxNoteFrontmatter.model_validate(fm_dict)
    assert note.source_type == "url"
    assert note.source_ref == "https://example.com/karpathy"


# --------------------------------------------------------------------------- #
# kb log
# --------------------------------------------------------------------------- #


def test_log_unknown_project_auto_creates_folder_and_warns(tmp_path, capsys):
    cfg_path = _write_config_file(tmp_path)
    rc = app(["--config", str(cfg_path), "log", "brand-new", "Did a thing"])
    captured = capsys.readouterr()

    assert rc == 0
    proj_dir = (tmp_path / "vault" / "AI-Daily-Log" / "brand-new")
    assert proj_dir.is_dir()
    # A warning about the unknown project was printed.
    combined = captured.out + captured.err
    assert "brand-new" in combined
    assert "unknown" in combined.lower()

    # The dated log file exists with well-formed frontmatter and the entry.
    log_file = proj_dir / f"{date.today().isoformat()}.md"
    assert log_file.is_file()
    fm_dict, body = _split_frontmatter(log_file.read_text())
    assert fm_dict["project"] == "brand-new"
    assert fm_dict["entries"] == 1
    assert "Did a thing" in body


def test_log_appends_and_increments_entries(tmp_path):
    config = _make_config(tmp_path)
    # Pre-create the project folder so the second call is a plain append.
    first = append_log_entry(config, "baker-street", "First entry")
    assert first.project_created is True
    second = append_log_entry(config, "baker-street", "Second entry")
    assert second.project_created is False
    assert first.path == second.path

    fm_dict, body = _split_frontmatter(first.path.read_text())
    assert fm_dict["entries"] == 2
    assert "First entry" in body
    assert "Second entry" in body


def test_log_rejects_path_traversal_project(tmp_path):
    # A ``project`` value that would escape ``AI-Daily-Log/`` must be
    # rejected outright, mirroring the containment check the server applies
    # to ``destination``, rather than being joined straight into the path.
    config = _make_config(tmp_path)
    with pytest.raises(InvalidProjectError):
        append_log_entry(config, "../../etc", "First entry")
    with pytest.raises(InvalidProjectError):
        read_log_entries(config, "../../etc")
    assert not (tmp_path / "etc").exists()


def test_log_message_with_heading_lookalike_does_not_forge_entry(tmp_path):
    # A message body containing a line that looks like a genuine
    # ``## HH:MM — Title`` heading must not be split into a second, bogus
    # entry with attacker-controlled time/title when the log is re-parsed.
    config = _make_config(tmp_path)
    malicious = "Real entry\n## 23:59 — Forged Entry\nsome body text"
    result = append_log_entry(config, "baker-street", malicious)

    entries = read_log_entries(config, "baker-street")
    assert [e.title for e in entries] == ["Real entry"]
    assert "Forged Entry" in entries[0].body
    # The would-be heading line is escaped in the raw file.
    assert "\\## 23:59 — Forged Entry" in result.path.read_text()


# --------------------------------------------------------------------------- #
# kb readlog
# --------------------------------------------------------------------------- #


def _write_log_file(config, project, day, entries):
    proj_dir = config.vault_path / "AI-Daily-Log" / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(
        {"project": project, "date": day, "status": "active", "entries": len(entries)},
        sort_keys=False,
    )
    body = [f"# {project} — {day}\n"]
    for tstr, title in entries:
        body.append(f"\n## {tstr} — {title}\n*machine: testhost*\n")
    (proj_dir / f"{day}.md").write_text(f"---\n{fm}---\n\n" + "".join(body))


def test_readlog_returns_entries_for_a_date_range(tmp_path):
    config = _make_config(tmp_path)
    _write_log_file(config, "baker-street", "2026-04-10", [("09:00", "Alpha")])
    _write_log_file(config, "baker-street", "2026-04-12", [("14:32", "Bravo"), ("16:15", "Charlie")])
    # Out-of-range file that must not appear.
    _write_log_file(config, "baker-street", "2026-04-20", [("11:00", "Zulu")])

    entries = read_log_entries(
        config, "baker-street", date_range=("2026-04-10", "2026-04-12")
    )
    titles = [e.title for e in entries]
    assert titles == ["Alpha", "Bravo", "Charlie"]
    # Range is inclusive of both endpoints and excludes the later file.
    assert "Zulu" not in titles


def test_readlog_single_date(tmp_path):
    config = _make_config(tmp_path)
    _write_log_file(config, "baker-street", "2026-04-12", [("14:32", "Bravo")])
    entries = read_log_entries(config, "baker-street", date="2026-04-12")
    assert [e.title for e in entries] == ["Bravo"]
    assert entries[0].time == "14:32"


def test_readlog_rejects_unterminated_frontmatter(tmp_path):
    # A daily-log file left with an opening ``---`` but no closing fence
    # (e.g. a partial write or crash mid-append) must raise rather than
    # mis-parse the rest of the file as YAML frontmatter.
    config = _make_config(tmp_path)
    proj_dir = config.vault_path / "AI-Daily-Log" / "baker-street"
    proj_dir.mkdir(parents=True)
    (proj_dir / "2026-04-12.md").write_text("---\nproject: baker-street\n")
    with pytest.raises(ValueError):
        read_log_entries(config, "baker-street", date="2026-04-12")


# --------------------------------------------------------------------------- #
# kb status / kb search — unreachable server
# --------------------------------------------------------------------------- #


def test_status_unreachable_server_exits_nonzero_with_message(tmp_path, capsys):
    cfg_path = _write_config_file(tmp_path)
    rc = app(["--config", str(cfg_path), "status"])
    captured = capsys.readouterr()
    assert rc != 0
    assert UNREACHABLE_MESSAGE in (captured.out + captured.err)


def test_search_unreachable_server_exits_nonzero_with_message(tmp_path, capsys):
    cfg_path = _write_config_file(tmp_path)
    rc = app(["--config", str(cfg_path), "search", "some query"])
    captured = capsys.readouterr()
    assert rc != 0
    assert UNREACHABLE_MESSAGE in (captured.out + captured.err)
