"""``kb`` command-line entry point.

Capture commands (``add`` / ``log`` / ``readlog``) operate entirely on the
local vault and never touch the network. Query commands (``status`` /
``search`` / ``ask``) are HTTP clients against ``server_url``, backed by the
real API client (``kb.cli.client.KbClient``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from kb.cli import vault
from kb.cli.client import KbClient, ServerUnreachable
from kb.core.config import DEFAULT_CONFIG_PATH, Config, load_config

# Printed verbatim whenever a query command cannot reach kb-server. The exact
# wording is contract (see the plan's Task 4 acceptance).
UNREACHABLE_MESSAGE = "kb-server unreachable. Captures still work offline."

_SERVER_TIMEOUT = 2.0

# `ask` triggers server-side LLM synthesis (server LLM_TIMEOUT=240s), so it
# needs a much longer client timeout than the quick status/search calls.
_ASK_TIMEOUT = 260.0


# --------------------------------------------------------------------------- #
# command handlers
# --------------------------------------------------------------------------- #


def _cmd_add(args: argparse.Namespace, config: Config) -> int:
    if args.stdin:
        content = sys.stdin.read()
    elif args.content:
        content = args.content
    else:
        print("kb add: provide content or --stdin", file=sys.stderr)
        return 2
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    path = vault.add_note(
        config,
        content,
        title=args.title,
        tags=tags,
        project=args.project,
        destination=args.destination,
        source_type="stdin" if args.stdin and not args.source_type else args.source_type,
    )
    print(f"Captured to {path}")
    return 0


def _cmd_log(args: argparse.Namespace, config: Config) -> int:
    try:
        result = vault.append_log_entry(config, args.project, args.message)
    except vault.InvalidProjectError as exc:
        print(f"kb log: {exc}", file=sys.stderr)
        return 2
    if result.project_created:
        print(
            f"Warning: unknown project '{args.project}' — "
            f"created AI-Daily-Log/{args.project}/",
            file=sys.stderr,
        )
    print(f"Logged to {result.path}")
    return 0


def _cmd_readlog(args: argparse.Namespace, config: Config) -> int:
    date_range = None
    if args.range:
        parts = args.range.replace("..", ":").split(":")
        if len(parts) != 2:
            print("kb readlog: --range must be START:END", file=sys.stderr)
            return 2
        date_range = (parts[0].strip(), parts[1].strip())
    try:
        entries = vault.read_log_entries(
            config, args.project, date=args.date, date_range=date_range
        )
    except vault.InvalidProjectError as exc:
        print(f"kb readlog: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print("No log entries found.")
        return 0
    for entry in entries:
        print(f"## {entry.date} {entry.time} — {entry.title}")
        if entry.machine:
            print(f"   machine: {entry.machine}")
        if entry.body:
            print(entry.body)
        print()
    return 0


def _cmd_status(args: argparse.Namespace, config: Config) -> int:
    client = KbClient(config.server_url, timeout=_SERVER_TIMEOUT)
    try:
        result = client.status()
    except ServerUnreachable:
        print(UNREACHABLE_MESSAGE, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _cmd_search(args: argparse.Namespace, config: Config) -> int:
    client = KbClient(config.server_url, timeout=_SERVER_TIMEOUT)
    collection = "kb_logs" if args.logs else "kb_knowledge"
    try:
        result = client.search(args.query, collection=collection)
    except ServerUnreachable:
        print(UNREACHABLE_MESSAGE, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace, config: Config) -> int:
    client = KbClient(config.server_url, timeout=_ASK_TIMEOUT)
    try:
        result = client.ask(args.question)
    except ServerUnreachable:
        print(UNREACHABLE_MESSAGE, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _cmd_config(args: argparse.Namespace, config: Config) -> int:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if args.action == "set":
        raw = {}
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}
        raw[args.key] = args.value
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        print(f"Set {args.key} = {args.value} in {config_path}")
        return 0
    if args.action == "get":
        value = config.model_dump(mode="json").get(args.key)
        if value is None:
            print(f"kb config: unknown key '{args.key}'", file=sys.stderr)
            return 2
        print(value)
        return 0
    # Default: view the resolved config.
    print(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), end="")
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="Personal knowledge base CLI")
    parser.add_argument("--config", help="path to config.yaml (overrides default)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="capture a note to the vault Inbox")
    p_add.add_argument("content", nargs="?", help="text, url, or file path")
    p_add.add_argument("--stdin", action="store_true", help="read content from stdin")
    p_add.add_argument("--title")
    p_add.add_argument("--tags", help="comma-separated tags")
    p_add.add_argument("--project")
    p_add.add_argument("--destination", help="destination folder hint")
    p_add.add_argument(
        "--source-type", dest="source_type",
        choices=["url", "text", "file", "stdin", "mcp", "agent"],
    )
    p_add.set_defaults(func=_cmd_add)

    p_log = sub.add_parser("log", help="append an entry to today's project log")
    p_log.add_argument("project")
    p_log.add_argument("message")
    p_log.set_defaults(func=_cmd_log)

    p_readlog = sub.add_parser("readlog", help="read past log entries")
    p_readlog.add_argument("project")
    p_readlog.add_argument("--date", help="YYYY-MM-DD single day")
    p_readlog.add_argument("--range", help="START:END inclusive date range")
    p_readlog.set_defaults(func=_cmd_readlog)

    p_status = sub.add_parser("status", help="vault stats and server health")
    p_status.set_defaults(func=_cmd_status)

    p_search = sub.add_parser("search", help="semantic search over the knowledge base")
    p_search.add_argument("query")
    p_search.add_argument("--logs", action="store_true", help="search kb_logs")
    p_search.set_defaults(func=_cmd_search)

    p_ask = sub.add_parser("ask", help="RAG answer with citations")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=_cmd_ask)

    p_config = sub.add_parser("config", help="view or edit local config")
    csub = p_config.add_subparsers(dest="action")
    c_get = csub.add_parser("get", help="print one config value")
    c_get.add_argument("key")
    c_set = csub.add_parser("set", help="set one config value")
    c_set.add_argument("key")
    c_set.add_argument("value")
    p_config.set_defaults(func=_cmd_config)

    return parser


def app(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config) if args.config else None)
    return args.func(args, config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
