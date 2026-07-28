"""Unit tests for the Search + Ask HTTP API.

Exercise the FastAPI app through ``TestClient`` with the embed, index, and
LLM dependencies overridden by in-memory fakes — no network access. Covers
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
)
from kb.server.index import SearchHit


class FakeEmbed:
    """Returns a fixed 1024-dim vector without calling Ollama."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * 1024

    def ping(self) -> bool:
        return True


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
    """Returns a canned answer without calling Ollama."""

    def __init__(self, answer: str = "Synthesized answer [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def make_hit(doc_id: str = "doc-1", score: float = 0.92) -> SearchHit:
    return SearchHit(
        id="point-1",
        score=score,
        payload={
            "doc_id": doc_id,
            "doc_title": "Karpathy on knowledge bases",
            "doc_path": "Knowledge/research/karpathy-kb.md",
            "header_path": "Key Insights > Beyond RAG",
            "content": "Embeddings beat keyword search for recall.",
        },
    )


@pytest.fixture
def fakes():
    embed = FakeEmbed()
    index = FakeIndex([make_hit()])
    llm = FakeLLM()
    app.dependency_overrides[get_embed_client] = lambda: embed
    app.dependency_overrides[get_index_client] = lambda: index
    app.dependency_overrides[get_llm_client] = lambda: llm
    yield embed, index, llm
    app.dependency_overrides.clear()


@pytest.fixture
def client(fakes):
    return TestClient(app)


def test_status_ok(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "qdrant_ok": True,
        "ollama_ok": True,
        "embed_failed_backlog": 0,
        "watchers_ok": True,
    }


def test_status_degraded_when_dependency_unreachable(client, fakes):
    embed, _index, _llm = fakes
    embed.ping = lambda: False
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama_ok"] is False


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
    _embed, index, _llm = fakes
    response = client.post("/search", json={"query": "vector search"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    hit = body["results"][0]
    assert hit["doc_id"] == "doc-1"
    assert hit["header_path"] == "Key Insights > Beyond RAG"
    assert hit["score"] == pytest.approx(0.92)
    # Defaults to the knowledge collection.
    assert index.calls == [("kb_knowledge", app_module.DEFAULT_SEARCH_LIMIT)]


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
    _embed, _index, llm = fakes
    response = client.post("/ask", json={"question": "How does recall work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Synthesized answer [1]."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["doc_title"] == "Karpathy on knowledge bases"
    # The retrieved chunk content made it into the prompt.
    assert "Embeddings beat keyword search" in llm.prompts[0]


def test_ask_missing_question_is_4xx(client):
    response = client.post("/ask", json={})
    assert 400 <= response.status_code < 500
    assert response.status_code == 422
