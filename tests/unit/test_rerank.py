"""Unit tests for the Voyage rerank client.

The single HTTP call is faked via the ``_post`` seam, so no network access.
"""

from __future__ import annotations

import urllib.error

import pytest

from kb.server.rerank import RerankClient, RerankUnavailableError


def voyage_response(scored: list[tuple[int, float]]) -> dict:
    return {
        "object": "list",
        "data": [{"index": i, "relevance_score": s} for i, s in scored],
    }


def test_rerank_returns_index_score_pairs_in_voyage_order(monkeypatch):
    client = RerankClient(api_key="k")
    monkeypatch.setattr(
        client, "_post", lambda payload: voyage_response([(2, 0.9), (0, 0.4)])
    )
    ranked = client.rerank("q", ["a", "b", "c"], top_k=2)
    assert ranked == [(2, 0.9), (0, 0.4)]


def test_rerank_sends_voyage_payload(monkeypatch):
    client = RerankClient(api_key="k")
    sent = []
    monkeypatch.setattr(
        client, "_post", lambda payload: sent.append(payload) or voyage_response([(0, 1.0)])
    )
    client.rerank("q", ["a"], top_k=1)
    assert sent == [
        {"query": "q", "documents": ["a"], "model": "rerank-2.5", "top_k": 1}
    ]


def test_rerank_empty_documents_skips_the_call(monkeypatch):
    client = RerankClient(api_key="k")
    monkeypatch.setattr(
        client, "_post", lambda payload: pytest.fail("should not call Voyage")
    )
    assert client.rerank("q", [], top_k=5) == []


def test_rerank_default_host_model_and_key_from_env(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
    client = RerankClient()
    assert client.host == "https://api.voyageai.com"
    assert client.model == "rerank-2.5"
    assert client.api_key == "env-key"


def test_rerank_unavailable_after_retries(monkeypatch):
    client = RerankClient(api_key="k")
    monkeypatch.setattr(
        client,
        "_post",
        lambda payload: (_ for _ in ()).throw(
            urllib.error.HTTPError("https://api.voyageai.com", 503, "down", {}, None)
        ),
    )
    monkeypatch.setattr("kb.server.rerank.time.sleep", lambda s: None)
    with pytest.raises(RerankUnavailableError):
        client.rerank("q", ["a"], top_k=1)
