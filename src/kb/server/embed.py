"""Client for the Voyage AI ``voyage-4`` embedding model.

Talks to Voyage's HTTP API (``POST /v1/embeddings``) and returns
1024-dimension vectors. The API key is read from ``VOYAGE_API_KEY`` unless
passed explicitly. The single HTTP call is isolated in :meth:`_post` so tests
can substitute a fake without touching the network.

Documents are embedded with ``input_type: document`` (the default, used by
the inbox/log pipelines) and search queries with ``input_type: query``;
Voyage tunes the two differently for retrieval.

Transient failures — Voyage unreachable, a 429 rate limit, or a 5xx — are
retried with exponential backoff; once retries are exhausted,
:class:`EmbedUnavailableError` is raised so callers (the inbox/log pipelines)
can mark the affected work as ``embed_failed`` instead of crashing. A 401/403
(bad or missing key) is treated the same way: it is an operator problem, not
a defect in the note, so the note must stay retryable rather than be
quarantined.
"""

from __future__ import annotations

import os

import json
import time
import urllib.error
import urllib.request
from typing import Literal

DEFAULT_VOYAGE_URL = "https://api.voyageai.com"
EMBED_MODEL = "voyage-4"
EMBED_DIM = 1024

# Seconds to wait on the embedding HTTP call before giving up.
REQUEST_TIMEOUT = 60

# Retry/backoff for a transient failure: up to MAX_RETRIES retries beyond the
# initial attempt, waiting ``BACKOFF_BASE_SECONDS * 2 ** attempt`` between each.
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5

# HTTP statuses that mean "try again later" or "operator misconfiguration",
# never "this input is bad".
_TRANSIENT_STATUSES = {401, 403, 429}

InputType = Literal["document", "query"]


class EmbedUnavailableError(Exception):
    """Voyage stayed unavailable after exhausting retries."""


def voyage_api_key(explicit: str | None) -> str:
    return explicit if explicit is not None else os.environ.get("VOYAGE_API_KEY", "")


def is_transient(exc: urllib.error.HTTPError) -> bool:
    return exc.code in _TRANSIENT_STATUSES or exc.code >= 500


class EmbedClient:
    """Embed text into 1024-dim vectors via Voyage ``voyage-4``."""

    def __init__(
        self,
        host: str = DEFAULT_VOYAGE_URL,
        model: str = EMBED_MODEL,
        dim: int = EMBED_DIM,
        api_key: str | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.dim = dim
        self.api_key = voyage_api_key(api_key)

    def embed(self, text: str, input_type: InputType = "document") -> list[float]:
        """Return the embedding vector for ``text``.

        Raises :class:`ValueError` if Voyage returns a vector whose length
        does not match the expected dimension, or :class:`EmbedUnavailableError`
        if Voyage stays unavailable after exhausting retries.
        """

        response = self._post_with_retry(
            {"input": [text], "model": self.model, "input_type": input_type}
        )
        vector = response["data"][0]["embedding"]
        if len(vector) != self.dim:
            raise ValueError(
                f"expected {self.dim}-dim embedding, got {len(vector)}"
            )
        return vector

    def ping(self) -> bool:
        """Whether an API key is configured.

        Voyage has no free health endpoint, so ``/status`` reports key
        presence rather than spending an embedding call per check.
        """

        return bool(self.api_key)

    def _post_with_retry(self, payload: dict) -> dict:
        """Call :meth:`_post`, retrying transient failures with backoff.

        A connection-level failure, a 429, a 5xx, or a 401/403 is retried up
        to :data:`MAX_RETRIES` times before raising
        :class:`EmbedUnavailableError`. Any other HTTP error (a 4xx about the
        input itself) is raised immediately.
        """

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
                raise EmbedUnavailableError(
                    f"voyage unavailable after {attempt} attempts: {last}"
                ) from last
            time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    def _post(self, payload: dict) -> dict:
        """POST ``payload`` to Voyage's embeddings endpoint, return the JSON."""

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/v1/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
