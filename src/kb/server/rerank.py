"""Client for the Voyage AI ``rerank-2.5`` reranker.

``/search`` and ``/ask`` pull a wider candidate set from Qdrant, then ask
Voyage (``POST /v1/rerank``) to score each candidate's content against the
query and keep the top ``limit``. The API key is read from ``VOYAGE_API_KEY``
unless passed explicitly. The single HTTP call is isolated in :meth:`_post`
so tests can substitute a fake without touching the network.

Transient failures are retried exactly as in :mod:`kb.server.embed`; once
retries are exhausted :class:`RerankUnavailableError` is raised.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from kb.server.embed import (
    BACKOFF_BASE_SECONDS,
    DEFAULT_VOYAGE_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    is_transient,
    voyage_api_key,
)

RERANK_MODEL = "rerank-2.5"


class RerankUnavailableError(Exception):
    """Voyage stayed unavailable after exhausting retries."""


class RerankClient:
    """Score documents against a query via Voyage ``rerank-2.5``."""

    def __init__(
        self,
        host: str = DEFAULT_VOYAGE_URL,
        model: str = RERANK_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.api_key = voyage_api_key(api_key)

    def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """Return ``(index, relevance_score)`` pairs, best first, at most ``top_k``.

        ``index`` refers to the caller's ``documents`` list. An empty
        ``documents`` returns ``[]`` without calling Voyage.
        """

        if not documents:
            return []
        response = self._post_with_retry(
            {
                "query": query,
                "documents": documents,
                "model": self.model,
                "top_k": top_k,
            }
        )
        return [(item["index"], item["relevance_score"]) for item in response["data"]]

    def _post_with_retry(self, payload: dict) -> dict:
        attempt = 0
        while True:
            try:
                return self._post(payload)
            except urllib.error.HTTPError as exc:
                if not is_transient(exc):
                    raise
                last = exc
            except OSError as exc:
                last = exc
            attempt += 1
            if attempt > MAX_RETRIES:
                raise RerankUnavailableError(
                    f"voyage rerank unavailable after {attempt} attempts: {last}"
                ) from last
            time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    def _post(self, payload: dict) -> dict:
        """POST ``payload`` to Voyage's rerank endpoint, return the JSON."""

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/v1/rerank",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
