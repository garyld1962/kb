"""Unit tests for the embed (Ollama) and index (Qdrant) clients.

Both clients talk to their services over HTTP. These tests exercise the
wrapper logic against in-memory fakes injected in place of the real HTTP
transport, so no network access is required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kb.core.models import QdrantPayload
from kb.server.embed import EMBED_DIM, EmbedClient
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


def test_embed_returns_expected_dimension(monkeypatch):
    client = EmbedClient()
    monkeypatch.setattr(
        client, "_post", lambda payload: {"embedding": [0.1] * EMBED_DIM}
    )
    vector = client.embed("hello world")
    assert isinstance(vector, list)
    assert len(vector) == EMBED_DIM


def test_embed_rejects_wrong_dimension(monkeypatch):
    client = EmbedClient()
    monkeypatch.setattr(client, "_post", lambda payload: {"embedding": [0.1, 0.2]})
    with pytest.raises(ValueError):
        client.embed("hello world")


def test_embed_default_host_and_model():
    client = EmbedClient()
    assert client.host == "http://localhost:11434"
    assert client.model == "mxbai-embed-large"


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
