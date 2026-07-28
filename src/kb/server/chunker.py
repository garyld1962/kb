"""Markdown-aware chunker for the kb indexing pipelines.

Splitting strategy (hardcoded constants, not configurable in v1 per the plan's
Closed Decisions):

1. Split on markdown headers ``H1``/``H2``/``H3`` as primary boundaries. Each
   resulting section carries a ``header_path`` breadcrumb of the enclosing
   headers, e.g. ``"Key Insights > Beyond RAG"``.
2. Within any section longer than ``MAX_TOKENS`` tokens, fall back to a
   ``MAX_TOKENS``-token / ``OVERLAP_TOKENS``-overlap sliding window.

Tokens are whitespace-delimited words: a dependency-free v1 approximation of
the embedding model's tokenizer, adequate for bounding chunk size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_TOKENS = 500
OVERLAP_TOKENS = 50

# Only H1-H3 are split points; H4+ (``####``) stay inside the current section.
_HEADER_RE = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t]*$")


@dataclass
class Chunk:
    """One indexable unit: text plus its position and header breadcrumb.

    Fields map directly onto the corresponding ``QdrantPayload`` fields in
    ``kb.core.models``.
    """

    content: str
    header_path: str
    chunk_index: int = 0
    chunk_of: int = 0


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Split ``content`` into ``(header_path, body)`` sections on H1-H3.

    Body text preceding the first header is emitted with an empty
    ``header_path``. Sections whose body is empty (a header immediately
    followed by another header) are omitted by the caller.
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, title) of enclosing headers
    lines: list[str] = []
    path = ""

    for line in content.splitlines():
        match = _HEADER_RE.match(line)
        if match is None:
            lines.append(line)
            continue
        if lines:
            sections.append((path, "\n".join(lines)))
        lines = []
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = " > ".join(t for _, t in stack)

    if lines:
        sections.append((path, "\n".join(lines)))
    return sections


def _sliding_windows(tokens: list[str]) -> list[list[str]]:
    """Yield overlapping ``MAX_TOKENS``-token windows over ``tokens``."""
    if len(tokens) <= MAX_TOKENS:
        return [tokens]
    step = MAX_TOKENS - OVERLAP_TOKENS
    windows: list[list[str]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + MAX_TOKENS, n)
        windows.append(tokens[start:end])
        if end == n:
            break
        start += step
    return windows


def chunk(content: str) -> list[Chunk]:
    """Split ``content`` into indexable chunks.

    Headers (H1-H3) are the primary split points; sections exceeding
    ``MAX_TOKENS`` tokens fall back to a sliding window. Each returned chunk
    carries a 0-based ``chunk_index``, the total ``chunk_of``, and the
    ``header_path`` breadcrumb of its section.
    """
    pieces: list[tuple[str, str]] = []  # (header_path, chunk_text)
    for header_path, body in _split_sections(content):
        stripped = body.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) <= MAX_TOKENS:
            pieces.append((header_path, stripped))
        else:
            for window in _sliding_windows(tokens):
                pieces.append((header_path, " ".join(window)))

    total = len(pieces)
    return [
        Chunk(
            content=text,
            header_path=header_path,
            chunk_index=index,
            chunk_of=total,
        )
        for index, (header_path, text) in enumerate(pieces)
    ]
