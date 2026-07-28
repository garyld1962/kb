"""HTTP client for kb-server's query API.

Backs the ``kb search`` / ``kb ask`` / ``kb status`` commands. Capture stays
offline, so the only failure this client distinguishes is *unreachable* — any
transport-level error (DNS, refused connection, timeout) raises
:class:`ServerUnreachable`, which callers translate into the offline message
and a non-zero exit. An HTTP error response means the server *was* reached and
is surfaced as data, not an unreachable failure.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from kb.server.index import KNOWLEDGE_COLLECTION, LOG_COLLECTION

# Printed verbatim whenever a query command cannot reach kb-server. The exact
# wording is contract (matches ``kb.cli.main.UNREACHABLE_MESSAGE``).
UNREACHABLE_MESSAGE = "kb-server unreachable. Captures still work offline."

DEFAULT_TIMEOUT = 30.0


class ServerUnreachable(Exception):
    """Raised when kb-server cannot be reached over HTTP."""


class KbClient:
    """Thin HTTP client against kb-server's ``/search`` and ``/ask`` API."""

    def __init__(self, server_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base = server_url.rstrip("/")
        self.timeout = timeout

    def search(
        self,
        query: str,
        collection: str = KNOWLEDGE_COLLECTION,
        limit: int | None = None,
    ) -> dict:
        """Semantic search; ``collection`` targets ``kb_knowledge``/``kb_logs``."""

        payload: dict = {"query": query, "collection": collection}
        if limit is not None:
            payload["limit"] = limit
        return self._request("POST", "/search", payload)

    def ask(
        self,
        question: str,
        collection: str = KNOWLEDGE_COLLECTION,
        limit: int | None = None,
    ) -> dict:
        """RAG answer with citations for ``question``."""

        payload: dict = {"question": question, "collection": collection}
        if limit is not None:
            payload["limit"] = limit
        return self._request("POST", "/ask", payload)

    def status(self) -> dict:
        """Fetch kb-server health/status."""

        return self._request("GET", "/status", None)

    def _request(self, method: str, path: str, payload: dict | None) -> dict:
        """Issue an HTTP request, raising :class:`ServerUnreachable` on failure.

        A response with an HTTP error status means the server answered, so it is
        returned as ``{"error": <code>, "detail": <detail>}`` rather than raised
        as unreachable. ``detail`` is the JSON body's ``detail`` field when the
        body parses as JSON with one, else the HTTP reason phrase.
        """

        url = self.base + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:  # server answered — not "unreachable"
            detail = exc.reason
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            else:
                if isinstance(body, dict) and "detail" in body:
                    detail = body["detail"]
            return {"error": exc.code, "detail": detail}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ServerUnreachable(str(exc)) from exc


__all__ = [
    "KbClient",
    "ServerUnreachable",
    "UNREACHABLE_MESSAGE",
    "KNOWLEDGE_COLLECTION",
    "LOG_COLLECTION",
]
