"""Unit tests for the Search + Ask HTTP API.

Exercise the FastAPI app through ``TestClient`` with the embed, index,
rerank, and LLM dependencies overridden by in-memory fakes — no network access. Covers
``/search`` (happy path, collection routing, malformed 4xx) and ``/ask``
(happy path with answer + citations, malformed 4xx).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kb.server import app as app_module
from kb.server.app import (
    app,
    get_embed_client,
    get_index_client,
    get_llm_client,
    get_rerank_client,
)
from kb.server.index import SearchHit


class FakeEmbed:
    """Returns a fixed 1024-dim vector without calling Voyage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        self.calls.append((text, input_type))
        return [0.1] * 1024

    def ping(self) -> bool:
        return True


class FakeRerank:
    """Reverses the candidate order with descending scores, without calling Voyage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        self.calls.append((query, list(documents), top_k))
        order = list(reversed(range(len(documents))))[:top_k]
        return [(i, 1.0 - 0.1 * rank) for rank, i in enumerate(order)]


class FakeIndex:
    """Returns canned hits and records the collection/limit it was asked for."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, vector, collection, limit):
        self.calls.append((collection, limit))
        return self.hits

    def ping(self) -> bool:
        return True


class FakeLLM:
    """Returns a canned answer without calling the LLM server."""

    def __init__(self, answer: str = "Synthesized answer [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer

    def ping(self) -> bool:
        return True


def make_hit(
    doc_id: str = "doc-1",
    score: float = 0.92,
    content: str = "Embeddings beat keyword search for recall.",
) -> SearchHit:
    return SearchHit(
        id=f"point-{doc_id}",
        score=score,
        payload={
            "doc_id": doc_id,
            "doc_title": "Karpathy on knowledge bases",
            "doc_path": "Knowledge/research/karpathy-kb.md",
            "header_path": "Key Insights > Beyond RAG",
            "content": content,
        },
    )


@pytest.fixture
def fakes():
    embed = FakeEmbed()
    index = FakeIndex([make_hit()])
    rerank = FakeRerank()
    llm = FakeLLM()
    app.dependency_overrides[get_embed_client] = lambda: embed
    app.dependency_overrides[get_index_client] = lambda: index
    app.dependency_overrides[get_rerank_client] = lambda: rerank
    app.dependency_overrides[get_llm_client] = lambda: llm
    yield embed, index, llm
    app.dependency_overrides.clear()


@pytest.fixture
def rerank():
    return app.dependency_overrides[get_rerank_client]()


@pytest.fixture
def client(fakes):
    return TestClient(app)


def test_status_ok(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "qdrant_ok": True,
        "llm_ok": True,
        "voyage_ok": True,
        "embed_failed_backlog": 0,
        "watchers_ok": True,
    }


def test_status_degraded_when_llm_unreachable(client, fakes):
    _embed, _index, llm = fakes
    llm.ping = lambda: False
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["llm_ok"] is False
    assert body["voyage_ok"] is True


def test_status_degraded_when_voyage_key_missing(client, fakes):
    embed, _index, _llm = fakes
    embed.ping = lambda: False
    response = client.get("/status")
    body = response.json()
    assert body["status"] == "degraded"
    assert body["voyage_ok"] is False
    assert body["llm_ok"] is True


def test_status_degraded_when_watcher_thread_dead(client):
    """A watcher killed by an unhandled exception must flip /status, not go unnoticed."""

    class DeadWatcher:
        def is_alive(self) -> bool:
            return False

    app.state.log_watcher = DeadWatcher()
    try:
        response = client.get("/status")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["watchers_ok"] is False
    finally:
        app.state.log_watcher = None


def test_search_happy_path(client, fakes):
    embed, index, _llm = fakes
    response = client.post("/search", json={"query": "vector search"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    hit = body["results"][0]
    assert hit["doc_id"] == "doc-1"
    assert hit["header_path"] == "Key Insights > Beyond RAG"
    # Queries embed as queries, not documents.
    assert embed.calls == [("vector search", "query")]
    # Defaults to the knowledge collection, over-fetching candidates for the reranker.
    assert index.calls == [
        ("kb_knowledge", app_module.DEFAULT_SEARCH_LIMIT * app_module.RERANK_CANDIDATE_FACTOR)
    ]


def test_search_returns_reranked_order_and_scores(client, fakes, rerank):
    _embed, index, _llm = fakes
    index.hits = [
        make_hit("a", 0.9, "alpha"),
        make_hit("b", 0.8, "beta"),
        make_hit("c", 0.7, "gamma"),
    ]
    response = client.post("/search", json={"query": "q", "limit": 2})
    body = response.json()
    assert rerank.calls == [("q", ["alpha", "beta", "gamma"], 2)]
    assert [r["doc_id"] for r in body["results"]] == ["c", "b"]
    assert [r["score"] for r in body["results"]] == pytest.approx([1.0, 0.9])


def test_search_candidate_fetch_is_capped(client, fakes):
    _embed, index, _llm = fakes
    client.post("/search", json={"query": "q", "limit": 100})
    assert index.calls[0][1] == app_module.RERANK_CANDIDATE_CAP


def test_search_no_candidates_skips_rerank(client, fakes, rerank):
    _embed, index, _llm = fakes
    index.hits = []
    response = client.post("/search", json={"query": "q"})
    assert response.json() == {"results": []}
    assert rerank.calls == []


def test_search_targets_logs_collection(client, fakes):
    _embed, index, _llm = fakes
    response = client.post(
        "/search", json={"query": "what did I do", "collection": "kb_logs"}
    )
    assert response.status_code == 200
    assert index.calls[0][0] == "kb_logs"


def test_search_missing_query_is_4xx(client):
    response = client.post("/search", json={})
    assert 400 <= response.status_code < 500
    assert response.status_code == 422


def test_search_empty_query_is_4xx(client):
    response = client.post("/search", json={"query": ""})
    assert 400 <= response.status_code < 500


def test_search_unknown_collection_is_4xx(client):
    response = client.post(
        "/search", json={"query": "x", "collection": "kb_bogus"}
    )
    assert 400 <= response.status_code < 500


def test_ask_happy_path(client, fakes):
    embed, _index, llm = fakes
    response = client.post("/ask", json={"question": "How does recall work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Synthesized answer [1]."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["doc_title"] == "Karpathy on knowledge bases"
    assert embed.calls == [("How does recall work?", "query")]
    # The retrieved chunk content made it into the prompt.
    assert "Embeddings beat keyword search" in llm.prompts[0]


def test_ask_cites_reranked_candidates(client, fakes, rerank):
    _embed, index, llm = fakes
    index.hits = [make_hit("a", 0.9, "alpha"), make_hit("b", 0.8, "beta")]
    response = client.post("/ask", json={"question": "q", "limit": 1})
    body = response.json()
    assert rerank.calls == [("q", ["alpha", "beta"], 1)]
    assert [c["doc_id"] for c in body["citations"]] == ["b"]
    assert "beta" in llm.prompts[0] and "alpha" not in llm.prompts[0]


def test_ask_missing_question_is_4xx(client):
    response = client.post("/ask", json={})
    assert 400 <= response.status_code < 500
    assert response.status_code == 422
