"""Unit tests for the embed (Voyage) and index (Qdrant) clients.

Both clients talk to their services over HTTP. These tests exercise the
wrapper logic against in-memory fakes injected in place of the real HTTP
transport, so no network access is required.
"""

from __future__ import annotations

import urllib.error
from datetime import datetime, timezone

import pytest

from kb.core.models import QdrantPayload
from kb.server.embed import EMBED_DIM, EmbedClient, EmbedUnavailableError
from kb.server.index import (
    COLLECTIONS,
    KNOWLEDGE_COLLECTION,
    IndexClient,
    SearchHit,
)


def make_payload(doc_id: str, chunk_index: int, chunk_of: int) -> QdrantPayload:
    return QdrantPayload(
        doc_id=doc_id,
        doc_path="Knowledge/research/karpathy-kb.md",
        doc_hash="sha256:abc123",
        doc_title="Karpathy on knowledge bases",
        chunk_index=chunk_index,
        chunk_of=chunk_of,
        header_path="Key Insights > Beyond RAG",
        content=f"chunk {chunk_index} text",
        tags=["research", "llm", "rag"],
        project="",
        source_type="url",
        source_ref="https://example.com/karpathy",
        created_at=datetime(2026, 4, 12, 14, 32, 0, tzinfo=timezone.utc),
        indexed_at=datetime(2026, 4, 12, 14, 32, 15, tzinfo=timezone.utc),
    )


class FakeQdrantTransport:
    """In-memory stand-in for the Qdrant HTTP transport.

    Implements the duck-typed interface :class:`IndexClient` depends on and
    records enough state to assert on upsert/delete/bootstrap behaviour.
    """

    def __init__(self) -> None:
        # collection name -> {point_id: {"vector": [...], "payload": {...}}}
        self.collections: dict[str, dict] = {}
        self.create_calls: dict[str, int] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, vector_size: int) -> None:
        # A real Qdrant PUT would 4xx on an existing collection; the wrapper
        # must never call us twice for the same name.
        assert name not in self.collections, f"double-create of {name}"
        self.collections[name] = {}
        self.create_calls[name] = self.create_calls.get(name, 0) + 1

    def upsert(self, collection: str, points: list[dict]) -> None:
        store = self.collections.setdefault(collection, {})
        for point in points:
            store[point["id"]] = {
                "vector": point["vector"],
                "payload": point["payload"],
            }

    def delete_by_doc_id(self, collection: str, doc_id: str) -> None:
        store = self.collections.get(collection, {})
        for pid in [
            pid for pid, p in store.items() if p["payload"]["doc_id"] == doc_id
        ]:
            del store[pid]

    def delete_stale_chunks(
        self, collection: str, doc_id: str, min_chunk_index: int
    ) -> None:
        store = self.collections.get(collection, {})
        for pid in [
            pid
            for pid, p in store.items()
            if p["payload"]["doc_id"] == doc_id
            and p["payload"]["chunk_index"] >= min_chunk_index
        ]:
            del store[pid]

    def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[dict]:
        store = self.collections.get(collection, {})
        hits = [
            {"id": pid, "score": 1.0, "payload": p["payload"]}
            for pid, p in store.items()
        ]
        return hits[:limit]


def payloads_in(transport: FakeQdrantTransport, collection: str) -> list[dict]:
    return [p["payload"] for p in transport.collections[collection].values()]


# --- embed client ---------------------------------------------------------


def voyage_response(vector: list[float]) -> dict:
    return {"object": "list", "data": [{"embedding": vector, "index": 0}]}


def test_embed_returns_expected_dimension(monkeypatch):
    client = EmbedClient(api_key="k")
    monkeypatch.setattr(
        client, "_post", lambda payload: voyage_response([0.1] * EMBED_DIM)
    )
    vector = client.embed("hello world")
    assert isinstance(vector, list)
    assert len(vector) == EMBED_DIM


def test_embed_rejects_wrong_dimension(monkeypatch):
    client = EmbedClient(api_key="k")
    monkeypatch.setattr(client, "_post", lambda payload: voyage_response([0.1, 0.2]))
    with pytest.raises(ValueError):
        client.embed("hello world")


def test_embed_sends_voyage_document_payload_by_default(monkeypatch):
    client = EmbedClient(api_key="k")
    sent = []
    monkeypatch.setattr(
        client, "_post", lambda payload: sent.append(payload) or voyage_response([0.0] * EMBED_DIM)
    )
    client.embed("chunk text")
    assert sent == [{"input": ["chunk text"], "model": "voyage-4", "input_type": "document"}]


def test_embed_query_input_type(monkeypatch):
    client = EmbedClient(api_key="k")
    sent = []
    monkeypatch.setattr(
        client, "_post", lambda payload: sent.append(payload) or voyage_response([0.0] * EMBED_DIM)
    )
    client.embed("what did I decide", input_type="query")
    assert sent[0]["input_type"] == "query"


def test_embed_default_host_model_and_key_from_env(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
    client = EmbedClient()
    assert client.host == "https://api.voyageai.com"
    assert client.model == "voyage-4"
    assert client.api_key == "env-key"


def test_embed_ping_reports_whether_key_is_configured(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert EmbedClient().ping() is False
    assert EmbedClient(api_key="k").ping() is True


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.voyageai.com", status, "err", {}, None)


def test_embed_retries_rate_limit_then_succeeds(monkeypatch):
    client = EmbedClient(api_key="k")
    attempts = []

    def post(payload):
        attempts.append(1)
        if len(attempts) < 3:
            raise _http_error(429)
        return voyage_response([0.1] * EMBED_DIM)

    monkeypatch.setattr(client, "_post", post)
    monkeypatch.setattr("kb.server.embed.time.sleep", lambda s: None)
    assert len(client.embed("x")) == EMBED_DIM
    assert len(attempts) == 3


def test_embed_unauthorized_is_unavailable_not_a_note_defect(monkeypatch):
    """A bad key is an operator problem: leave the note retryable, don't quarantine it."""

    client = EmbedClient(api_key="bad")
    monkeypatch.setattr(client, "_post", lambda payload: (_ for _ in ()).throw(_http_error(401)))
    monkeypatch.setattr("kb.server.embed.time.sleep", lambda s: None)
    with pytest.raises(EmbedUnavailableError):
        client.embed("x")


def test_embed_bad_request_is_not_retried(monkeypatch):
    client = EmbedClient(api_key="k")
    attempts = []

    def post(payload):
        attempts.append(1)
        raise _http_error(400)

    monkeypatch.setattr(client, "_post", post)
    with pytest.raises(urllib.error.HTTPError):
        client.embed("x")
    assert len(attempts) == 1


# --- index client: payload shape -----------------------------------------


def test_upsert_payload_matches_qdrant_model_exactly():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    payload = make_payload("A", chunk_index=0, chunk_of=1)

    index.upsert_chunks(KNOWLEDGE_COLLECTION, [payload], [[0.0] * EMBED_DIM])

    stored = payloads_in(transport, KNOWLEDGE_COLLECTION)
    assert len(stored) == 1
    # Byte-for-byte identical to the single-owner Qdrant payload contract.
    assert stored[0] == payload.model_dump(mode="json")
    assert set(stored[0].keys()) == set(QdrantPayload.model_fields.keys())


def test_upsert_point_id_is_stable_per_chunk():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    payload = make_payload("A", chunk_index=2, chunk_of=5)

    index.upsert_chunks(KNOWLEDGE_COLLECTION, [payload], [[0.0] * EMBED_DIM])
    index.upsert_chunks(KNOWLEDGE_COLLECTION, [payload], [[0.0] * EMBED_DIM])

    # Same doc_id + chunk_index must map to one deterministic point, not two.
    assert len(payloads_in(transport, KNOWLEDGE_COLLECTION)) == 1


# --- index client: delete_by_doc_id --------------------------------------


def test_delete_by_doc_id_removes_all_chunks_for_that_doc():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    payloads = [
        make_payload("A", 0, 3),
        make_payload("A", 1, 3),
        make_payload("A", 2, 3),
        make_payload("B", 0, 2),
        make_payload("B", 1, 2),
    ]
    vectors = [[0.0] * EMBED_DIM for _ in payloads]
    index.upsert_chunks(KNOWLEDGE_COLLECTION, payloads, vectors)
    assert len(payloads_in(transport, KNOWLEDGE_COLLECTION)) == 5

    index.delete_by_doc_id(KNOWLEDGE_COLLECTION, "A")

    remaining = payloads_in(transport, KNOWLEDGE_COLLECTION)
    assert len(remaining) == 2
    assert all(p["doc_id"] == "B" for p in remaining)


def test_delete_stale_chunks_removes_only_leftover_indices():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    payloads = [make_payload("A", 0, 3), make_payload("A", 1, 3), make_payload("A", 2, 3)]
    vectors = [[0.0] * EMBED_DIM for _ in payloads]
    index.upsert_chunks(KNOWLEDGE_COLLECTION, payloads, vectors)

    # A shorter re-index of "A" upserts a single chunk 0, then trims the
    # leftover chunk_index 1 and 2 points from the previous, longer version.
    index.upsert_chunks(KNOWLEDGE_COLLECTION, [make_payload("A", 0, 1)], [[1.0] * EMBED_DIM])
    index.delete_stale_chunks(KNOWLEDGE_COLLECTION, "A", 1)

    remaining = payloads_in(transport, KNOWLEDGE_COLLECTION)
    assert len(remaining) == 1
    assert remaining[0]["chunk_index"] == 0
    assert remaining[0]["chunk_of"] == 1


# --- index client: bootstrap idempotency ---------------------------------


def test_ensure_collections_creates_both():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)

    index.ensure_collections()

    assert set(transport.collections) == set(COLLECTIONS)
    for name in COLLECTIONS:
        assert transport.create_calls[name] == 1


def test_ensure_collections_is_idempotent():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)

    index.ensure_collections()
    # Second call must not error and must not re-create (which would raise
    # inside the fake, mirroring a real Qdrant 4xx on duplicate create).
    index.ensure_collections()

    for name in COLLECTIONS:
        assert transport.create_calls[name] == 1


# --- index client: search -------------------------------------------------


def test_search_maps_hits_to_searchhit():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    payload = make_payload("A", 0, 1)
    index.upsert_chunks(KNOWLEDGE_COLLECTION, [payload], [[0.0] * EMBED_DIM])

    hits = index.search([0.0] * EMBED_DIM, KNOWLEDGE_COLLECTION, limit=5)

    assert len(hits) == 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].payload == payload.model_dump(mode="json")


def test_search_respects_limit():
    transport = FakeQdrantTransport()
    index = IndexClient(transport=transport)
    payloads = [make_payload("A", i, 3) for i in range(3)]
    vectors = [[0.0] * EMBED_DIM for _ in payloads]
    index.upsert_chunks(KNOWLEDGE_COLLECTION, payloads, vectors)

    hits = index.search([0.0] * EMBED_DIM, KNOWLEDGE_COLLECTION, limit=2)
    assert len(hits) == 2
