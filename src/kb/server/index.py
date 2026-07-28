"""Qdrant client wrapper for the kb vector index.

Wraps the Qdrant REST API for the two kb collections (``kb_knowledge`` and
``kb_logs``, both 1024-dim). All wire I/O goes through a small transport
object (:class:`_HttpTransport` by default) so tests can inject an
in-memory fake.

Connection-level failures (Qdrant unreachable) are retried with exponential
backoff; once retries are exhausted, :class:`IndexUnavailableError` is
raised so callers (the inbox/log pipelines) can mark the affected work as
``embed_failed`` instead of crashing.
"""

from __future__ import annotations

import os

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Protocol

from kb.core.models import QdrantPayload

# Default Qdrant REST endpoint; override with KB_QDRANT_URL.
DEFAULT_QDRANT_URL = os.environ.get("KB_QDRANT_URL", "http://localhost:6333")

KNOWLEDGE_COLLECTION = "kb_knowledge"
LOG_COLLECTION = "kb_logs"
COLLECTIONS = (KNOWLEDGE_COLLECTION, LOG_COLLECTION)

VECTOR_SIZE = 1024
DISTANCE = "Cosine"
DEFAULT_SEARCH_LIMIT = 10
REQUEST_TIMEOUT = 30

# Retry/backoff for a connection-level failure (Qdrant unreachable): up to
# MAX_RETRIES retries beyond the initial attempt, waiting
# ``BACKOFF_BASE_SECONDS * 2 ** attempt`` between each.
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5


class IndexUnavailableError(Exception):
    """Qdrant stayed unreachable after exhausting retries."""


def _with_retry(fn):
    """Call ``fn()``, retrying connection failures with backoff.

    An HTTP error response (Qdrant reachable but returning an error status)
    is not retried. A connection-level failure (refused, timed out, DNS,
    etc.) is retried up to :data:`MAX_RETRIES` times before raising
    :class:`IndexUnavailableError`.
    """

    attempt = 0
    while True:
        try:
            return fn()
        except urllib.error.HTTPError:
            raise
        except OSError as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise IndexUnavailableError(
                    f"qdrant unreachable after {attempt} attempts: {exc}"
                ) from exc
            time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

# Stable namespace so a (doc_id, chunk_index) pair always maps to the same
# Qdrant point id, keeping re-index and upsert idempotent.
_POINT_NAMESPACE = uuid.UUID("6b6f2d6b-0000-0000-0000-6b625f696478")


def point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point id for a chunk of a document."""

    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


@dataclass(frozen=True)
class SearchHit:
    """One result from a vector search."""

    id: str
    score: float
    payload: dict


class Transport(Protocol):
    """Minimal Qdrant operations the wrapper depends on."""

    def collection_exists(self, name: str) -> bool: ...

    def create_collection(self, name: str, vector_size: int) -> None: ...

    def upsert(self, collection: str, points: list[dict]) -> None: ...

    def delete_by_doc_id(self, collection: str, doc_id: str) -> None: ...

    def delete_stale_chunks(
        self, collection: str, doc_id: str, min_chunk_index: int
    ) -> None: ...

    def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[dict]: ...


class IndexClient:
    """High-level wrapper over the Qdrant collections used by kb."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        vector_size: int = VECTOR_SIZE,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.vector_size = vector_size
        self._transport = transport if transport is not None else _HttpTransport(url)

    def ensure_collections(self) -> None:
        """Create ``kb_knowledge``/``kb_logs`` if absent. Idempotent."""

        for name in COLLECTIONS:
            if not _with_retry(lambda name=name: self._transport.collection_exists(name)):
                _with_retry(
                    lambda name=name: self._transport.create_collection(
                        name, self.vector_size
                    )
                )

    def ping(self) -> bool:
        """Return whether Qdrant is reachable.

        Single attempt, no retry/backoff — used by ``/status`` so a health
        check stays fast even when Qdrant is down. An HTTP error response
        (e.g. 404) still means Qdrant answered, so it counts as reachable.
        """

        try:
            self._transport.collection_exists(KNOWLEDGE_COLLECTION)
        except OSError:
            return False
        return True

    def upsert_chunks(
        self,
        collection: str,
        payloads: list[QdrantPayload],
        vectors: list[list[float]],
    ) -> None:
        """Upsert one point per (payload, vector) pair into ``collection``."""

        if len(payloads) != len(vectors):
            raise ValueError("payloads and vectors must be the same length")
        points = [
            {
                "id": point_id(payload.doc_id, payload.chunk_index),
                "vector": vector,
                "payload": payload.model_dump(mode="json"),
            }
            for payload, vector in zip(payloads, vectors)
        ]
        if points:
            _with_retry(lambda: self._transport.upsert(collection, points))

    def delete_by_doc_id(self, collection: str, doc_id: str) -> None:
        """Remove every chunk of ``doc_id`` from ``collection``."""

        _with_retry(lambda: self._transport.delete_by_doc_id(collection, doc_id))

    def delete_stale_chunks(
        self, collection: str, doc_id: str, min_chunk_index: int
    ) -> None:
        """Remove ``doc_id`` chunks at/after ``min_chunk_index``.

        Used after re-upserting a document's new chunks (which overwrite
        matching ``chunk_index`` points in place) to drop leftover points
        from a previous, longer version of the same document.
        """

        _with_retry(
            lambda: self._transport.delete_stale_chunks(collection, doc_id, min_chunk_index)
        )

    def search(
        self,
        query_vector: list[float],
        collection: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[SearchHit]:
        """Return the ``limit`` nearest chunks to ``query_vector``."""

        raw = _with_retry(lambda: self._transport.search(collection, query_vector, limit))
        return [
            SearchHit(id=str(hit["id"]), score=hit["score"], payload=hit["payload"])
            for hit in raw
        ]


class _HttpTransport:
    """Qdrant REST transport backing :class:`IndexClient` in production."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def collection_exists(self, name: str) -> bool:
        try:
            self._request("GET", f"/collections/{name}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        return True

    def create_collection(self, name: str, vector_size: int) -> None:
        self._request(
            "PUT",
            f"/collections/{name}",
            {"vectors": {"size": vector_size, "distance": DISTANCE}},
        )

    def upsert(self, collection: str, points: list[dict]) -> None:
        self._request(
            "PUT",
            f"/collections/{collection}/points?wait=true",
            {"points": points},
        )

    def delete_by_doc_id(self, collection: str, doc_id: str) -> None:
        self._request(
            "POST",
            f"/collections/{collection}/points/delete?wait=true",
            {"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
        )

    def delete_stale_chunks(
        self, collection: str, doc_id: str, min_chunk_index: int
    ) -> None:
        self._request(
            "POST",
            f"/collections/{collection}/points/delete?wait=true",
            {
                "filter": {
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}},
                        {"key": "chunk_index", "range": {"gte": min_chunk_index}},
                    ]
                }
            },
        )

    def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[dict]:
        response = self._request(
            "POST",
            f"/collections/{collection}/points/search",
            {"vector": vector, "limit": limit, "with_payload": True},
        )
        return response.get("result", [])

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
