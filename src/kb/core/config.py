"""Loader for the per-machine kb config (``~/.config/kb/config.yaml``).

When the file is absent, :func:`load_config` returns a :class:`Config` built
entirely from documented defaults, so capture works out of the box.
"""

from __future__ import annotations

import socket
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Default location of the config file, per the design spec.
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "kb" / "config.yaml"


class Config(BaseModel):
    """Per-machine kb configuration.

    Defaults (used when ``~/.config/kb/config.yaml`` is absent):

    - ``vault_path``: ``~/obsidian`` — local Obsidian vault root.
    - ``server_url``: ``http://localhost:8090`` — where kb-server is reachable.
    - ``machine_name``: this host's name, for log attribution.
    - ``default_project``: ``""`` — no default project shortcut.
    """

    vault_path: Path = Field(default_factory=lambda: Path.home() / "obsidian")
    server_url: str = "http://localhost:8090"
    machine_name: str = Field(default_factory=socket.gethostname)
    default_project: str = ""


def load_config(path: Path | None = None) -> Config:
    """Load config from ``path`` (default ``~/.config/kb/config.yaml``).

    Returns a :class:`Config` of documented defaults when the file is absent.
    """

    config_path = path if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Config()
    data = yaml.safe_load(config_path.read_text()) or {}
    return Config(**data)
