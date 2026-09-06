"""Unit tests for the answer-synthesis client (OpenAI-compatible chat API).

The single HTTP call is faked via the ``_post`` seam, so no network access.
"""

from __future__ import annotations

from kb.server.app import LLMClient


def chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_generate_returns_assistant_content(monkeypatch):
    client = LLMClient(host="http://llm:8085/v1", model="m")
    monkeypatch.setattr(client, "_post", lambda payload: chat_response("The answer [1]."))
    assert client.generate("prompt") == "The answer [1]."


def test_generate_sends_chat_completion_with_thinking_disabled(monkeypatch):
    client = LLMClient(host="http://llm:8085/v1", model="m")
    sent = []
    monkeypatch.setattr(
        client, "_post", lambda payload: sent.append(payload) or chat_response("x")
    )
    client.generate("prompt")
    assert sent == [
        {
            "model": "m",
            "messages": [{"role": "user", "content": "prompt"}],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


def test_default_host_and_model_from_env(monkeypatch):
    monkeypatch.setenv("KB_LLM_URL", "http://irene:8085/v1/")
    monkeypatch.setenv("KB_LLM_MODEL", "some-model")
    client = LLMClient()
    assert client.host == "http://irene:8085/v1"
    assert client.model == "some-model"
