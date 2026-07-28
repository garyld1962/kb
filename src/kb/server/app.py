"""FastAPI app exposing kb-server's query surface.

No auth (LAN-only, per the design's closed decisions):

- ``GET /status`` — health check; pings Qdrant/Ollama, reports the
  embed_failed backlog size, and reports whether the Inbox/Log watchers and
  reindex scheduler threads are still alive, so it reflects the service's
  actual health rather than only that the process is up.
- ``POST /search`` — semantic search over ``kb_knowledge`` by default;
  set ``collection`` to ``kb_logs`` to search the daily-log index instead.
- ``POST /ask`` — retrieve the top-k chunks and synthesize an answer with
  Ollama ``qwen3.5:9b``, returning the answer plus its supporting citations.

The embed, index, and LLM clients are provided through FastAPI dependencies
so tests can override them with in-memory fakes (no network access).
"""

from __future__ import annotations

import json
import threading
import urllib.request
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from kb.core.config import load_config
from kb.server.embed import DEFAULT_OLLAMA_URL, EmbedClient
from kb.server.index import (
    DEFAULT_SEARCH_LIMIT,
    KNOWLEDGE_COLLECTION,
    IndexClient,
    SearchHit,
)
from kb.server.log_pipeline import LogPipeline
from kb.server.reindex import ReindexScheduler, count_embed_failed_notes
from kb.server.watcher import InboxWatcher, LogWatcher

# Port kb-server listens on (Docker deployment, per the design spec).
SERVER_PORT = 8090

# Answer-synthesis model (per the plan's Task 9 closed decision).
ANSWER_MODEL = "qwen3.5:9b"
# Default number of chunks retrieved as context for /ask.
DEFAULT_TOP_K = 5
# Seconds to wait on the answer-generation HTTP call before giving up.
# qwen3.5:9b is a "thinking" model; observed generation latency for even a
# trivial one-word prompt on the configured Ollama host was ~139s, so this
# must sit comfortably above that rather than a nominal "should be enough".
LLM_TIMEOUT = 240

# The two collections a request may target. A Literal gives automatic 422s on
# any other value; the strings mirror the constants in ``kb.server.index``.
CollectionName = Literal["kb_knowledge", "kb_logs"]


class LLMClient:
    """Answer synthesis via Ollama's generate endpoint (``qwen3.5:9b``).

    The single HTTP call is isolated in :meth:`_post` so tests substitute a
    fake without touching the network.
    """

    def __init__(self, host: str = DEFAULT_OLLAMA_URL, model: str = ANSWER_MODEL) -> None:
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""

        response = self._post(
            {"model": self.model, "prompt": prompt, "stream": False}
        )
        return response["response"]

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# dependency providers (overridden in tests)
# --------------------------------------------------------------------------- #


def get_embed_client() -> EmbedClient:
    return EmbedClient()


def get_index_client() -> IndexClient:
    return IndexClient()


def get_llm_client() -> LLMClient:
    return LLMClient()


# --------------------------------------------------------------------------- #
# startup / shutdown: wire the Inbox and Log watchers into the app lifecycle
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the Inbox and Log filesystem watchers alongside the HTTP API.

    Without this, kb-server would only ever serve ``/search`` and ``/ask``
    and no capture would ever get indexed automatically. The reindex
    scheduler is started here too, so the periodic hash-based reindex pass
    actually runs in production.

    ``ensure_collections()`` runs before the watchers start so a fresh Qdrant
    instance has ``kb_knowledge``/``kb_logs`` created before the first Inbox
    capture's upsert; it is idempotent so re-running it on every restart is
    safe.
    """

    config = load_config()
    embed_client = get_embed_client()
    index_client = get_index_client()
    index_client.ensure_collections()

    # Shared between LogPipeline and the reindex pass so a periodic scan of
    # AI-Daily-Log/ can never interleave with a live `kb log` append: see
    # LogPipeline's and reindex_once's docstrings for the race this closes.
    log_lock = threading.Lock()

    inbox_watcher = InboxWatcher(config.vault_path, embed_client, index_client)
    log_pipeline = LogPipeline(config.vault_path, embed_client, index_client, lock=log_lock)
    log_watcher = LogWatcher(config.vault_path, log_pipeline)
    reindex_scheduler = ReindexScheduler(
        config.vault_path, embed_client, index_client, log_lock=log_lock
    )

    # Exposed on app.state so /status can report is_alive() for each: a
    # watcher thread killed by an unhandled exception must be visible there
    # rather than leaving the endpoint reporting healthy indefinitely.
    app.state.inbox_watcher = inbox_watcher
    app.state.log_watcher = log_watcher
    app.state.reindex_scheduler = reindex_scheduler

    inbox_watcher.start()
    log_watcher.start()
    reindex_scheduler.start()
    try:
        yield
    finally:
        reindex_scheduler.stop()
        log_watcher.stop()
        inbox_watcher.stop()


# --------------------------------------------------------------------------- #
# request / response models
# --------------------------------------------------------------------------- #


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: CollectionName = KNOWLEDGE_COLLECTION
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=100)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collection: CollectionName = KNOWLEDGE_COLLECTION
    limit: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)


class Citation(BaseModel):
    """One supporting chunk returned with a search or ask result."""

    doc_id: str
    doc_title: str
    doc_path: str
    header_path: str
    content: str
    score: float


class StatusResponse(BaseModel):
    status: str
    qdrant_ok: bool
    ollama_ok: bool
    embed_failed_backlog: int
    watchers_ok: bool


class SearchResponse(BaseModel):
    results: list[Citation]


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


def _citation(hit: SearchHit) -> Citation:
    """Project a Qdrant search hit into an API citation."""

    payload = hit.payload
    return Citation(
        doc_id=payload.get("doc_id", ""),
        doc_title=payload.get("doc_title", ""),
        doc_path=payload.get("doc_path", ""),
        header_path=payload.get("header_path", ""),
        content=payload.get("content", ""),
        score=hit.score,
    )


def _build_prompt(question: str, citations: list[Citation]) -> str:
    """Assemble the RAG prompt from the retrieved context chunks."""

    context = "\n\n".join(
        f"[{i}] ({c.doc_title} — {c.header_path})\n{c.content}"
        for i, c in enumerate(citations, start=1)
    )
    return (
        "Answer the question using only the context below. Cite sources by "
        "their bracketed number. If the context does not contain the answer, "
        "say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


app = FastAPI(
    title="kb-server",
    description="Search + Ask over the kb vault",
    lifespan=lifespan,
)
# Populated by ``lifespan`` on startup; ``None`` until then (e.g. in tests
# that hit the app without running its lifespan), in which case /status
# treats that watcher as not applicable rather than crashing.
app.state.inbox_watcher = None
app.state.log_watcher = None
app.state.reindex_scheduler = None


def _watchers_alive(state) -> bool:
    """Whether every watcher/scheduler that has been started is still alive.

    A watcher never started (``None`` on ``app.state``, e.g. in tests that
    bypass the app's lifespan) doesn't count against health; one started and
    then killed by an unhandled exception does.
    """

    watchers = (state.inbox_watcher, state.log_watcher, state.reindex_scheduler)
    return all(w is None or w.is_alive() for w in watchers)


@app.get("/status", response_model=StatusResponse)
def status(
    request: Request,
    embed: EmbedClient = Depends(get_embed_client),
    index: IndexClient = Depends(get_index_client),
) -> StatusResponse:
    """Health check: pings Qdrant/Ollama and reports the embed_failed backlog.

    ``status`` is ``"degraded"`` rather than ``"ok"`` when either dependency
    is unreachable, when notes are stuck in ``embed_failed`` awaiting a retry
    sweep, or when the Inbox/Log watcher or reindex scheduler thread has died
    — otherwise a watcher killed by an unhandled exception left this endpoint
    reporting healthy indefinitely, with indexing silently and permanently
    stopped.
    """

    qdrant_ok = index.ping()
    ollama_ok = embed.ping()
    backlog = count_embed_failed_notes(load_config().vault_path)
    watchers_ok = _watchers_alive(request.app.state)
    healthy = qdrant_ok and ollama_ok and backlog == 0 and watchers_ok
    return StatusResponse(
        status="ok" if healthy else "degraded",
        qdrant_ok=qdrant_ok,
        ollama_ok=ollama_ok,
        embed_failed_backlog=backlog,
        watchers_ok=watchers_ok,
    )


@app.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    embed: EmbedClient = Depends(get_embed_client),
    index: IndexClient = Depends(get_index_client),
) -> SearchResponse:
    vector = embed.embed(request.query)
    hits = index.search(vector, request.collection, request.limit)
    return SearchResponse(results=[_citation(hit) for hit in hits])


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    embed: EmbedClient = Depends(get_embed_client),
    index: IndexClient = Depends(get_index_client),
    llm: LLMClient = Depends(get_llm_client),
) -> AskResponse:
    vector = embed.embed(request.question)
    hits = index.search(vector, request.collection, request.limit)
    citations = [_citation(hit) for hit in hits]
    answer = llm.generate(_build_prompt(request.question, citations))
    return AskResponse(answer=answer, citations=citations)


def main() -> None:
    """Entry point for the ``kb-server`` script: run the app under uvicorn."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
