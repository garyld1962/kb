"""Client for the Ollama ``mxbai-embed-large`` embedding model.

Talks to Ollama's HTTP API over the LAN (override with KB_OLLAMA_URL) and returns
1024-dimension vectors. The single HTTP call is isolated in :meth:`_post`
so tests can substitute a fake without touching the network.

Connection-level failures (Ollama unreachable) are retried with exponential
backoff; once retries are exhausted, :class:`EmbedUnavailableError` is raised
so callers (the inbox/log pipelines) can mark the affected work as
``embed_failed`` instead of crashing.
"""

from __future__ import annotations

import os

import json
import time
import urllib.error
import urllib.request

# Ollama host default. The per-machine kb config
# (``kb.core.config.Config``) currently owns only vault/server settings,
# so the Ollama endpoint is a module default here, overridable per client.
DEFAULT_OLLAMA_URL = os.environ.get("KB_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM = 1024

# Seconds to wait on the embedding HTTP call before giving up.
REQUEST_TIMEOUT = 60

# Seconds to wait on the lightweight reachability check used by ``/status``.
PING_TIMEOUT = 5

# Retry/backoff for a connection-level failure (Ollama unreachable): up to
# MAX_RETRIES retries beyond the initial attempt, waiting
# ``BACKOFF_BASE_SECONDS * 2 ** attempt`` between each.
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5


class EmbedUnavailableError(Exception):
    """Ollama stayed unreachable after exhausting retries."""


class EmbedClient:
    """Embed text into 1024-dim vectors via Ollama ``mxbai-embed-large``."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_URL,
        model: str = EMBED_MODEL,
        dim: int = EMBED_DIM,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``.

        Raises :class:`ValueError` if Ollama returns a vector whose length
        does not match the expected dimension, or :class:`EmbedUnavailableError`
        if Ollama is unreachable after exhausting retries.
        """

        response = self._post_with_retry({"model": self.model, "prompt": text})
        vector = response["embedding"]
        if len(vector) != self.dim:
            raise ValueError(
                f"expected {self.dim}-dim embedding, got {len(vector)}"
            )
        return vector

    def ping(self) -> bool:
        """Return whether Ollama is reachable, without invoking the model.

        Single attempt, no retry/backoff — used by ``/status`` so a health
        check stays fast even when Ollama is down.
        """

        request = urllib.request.Request(f"{self.host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=PING_TIMEOUT):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _post_with_retry(self, payload: dict) -> dict:
        """Call :meth:`_post`, retrying connection failures with backoff.

        An HTTP error response (Ollama reachable but returning an error
        status) is not retried. A connection-level failure (refused,
        timed out, DNS, etc.) is retried up to :data:`MAX_RETRIES` times
        before raising :class:`EmbedUnavailableError`.
        """

        attempt = 0
        while True:
            try:
                return self._post(payload)
            except urllib.error.HTTPError:
                raise
            except OSError as exc:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise EmbedUnavailableError(
                        f"ollama unreachable after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    def _post(self, payload: dict) -> dict:
        """POST ``payload`` to Ollama's embeddings endpoint, return the JSON."""

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
