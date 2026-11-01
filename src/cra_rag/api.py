"""HTTP API: the assistant as a web service, for other programs to call.

Run it with `uv run cra-rag serve`, then open http://127.0.0.1:8000/docs: FastAPI
generates an interactive documentation of every endpoint from the code below.

- GET  /health  is the service up, and which search methods are available?
- GET  /search  the sections that best answer a question;
- POST /ask     a cited answer written by the language model.

Settings come from environment variables, so the same image runs anywhere
(see the Dockerfile): CRA_RAG_OLLAMA_URL, CRA_RAG_CHAT_MODEL,
CRA_RAG_EMBEDDING_MODEL, CRA_RAG_RAW_DIR and CRA_RAG_INDEX_DIR.
"""

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cra_rag import __version__, answer, dense, embed, llm, retriever, search

Lang = Literal["auto", "fr", "en"]
Method = Literal["auto", "bm25", "dense", "hybrid"]


@dataclass
class Settings:
    ollama_url: str = embed.OLLAMA_URL
    chat_model: str = llm.CHAT_MODEL
    embedding_model: str = embed.EMBEDDING_MODEL
    raw_dir: Path = Path("corpus/raw")
    index_dir: Path = dense.INDEX_DIR

    @classmethod
    def from_environment(cls) -> "Settings":
        env = os.environ.get
        return cls(
            ollama_url=env("CRA_RAG_OLLAMA_URL", embed.OLLAMA_URL),
            chat_model=env("CRA_RAG_CHAT_MODEL", llm.CHAT_MODEL),
            embedding_model=env("CRA_RAG_EMBEDDING_MODEL", embed.EMBEDDING_MODEL),
            raw_dir=Path(env("CRA_RAG_RAW_DIR", "corpus/raw")),
            index_dir=Path(env("CRA_RAG_INDEX_DIR", str(dense.INDEX_DIR))),
        )


# --- What the endpoints receive and return (shown in /docs) -----------------------


class Health(BaseModel):
    status: str
    version: str
    sections: int
    methods: list[str] = Field(description="search methods available on this server")
    chat_model: str
    ollama: str = Field(description='"up" or "down": answers need it, BM25 search does not')


class Section(BaseModel):
    id: str = Field(examples=["cra:art_14"])
    name: str = Field(examples=["Article 14: Reporting obligations of manufacturers"])
    url: str


class SearchResult(Section):
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    question: str
    lang: str
    method: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["Dans quel délai signaler une vulnérabilité activement exploitée ?"],
    )
    lang: Lang = "auto"
    method: Method = "auto"
    passages: int = Field(default=8, ge=1, le=15)


class Timings(BaseModel):
    prompt_tokens: int
    prompt_seconds: float
    output_tokens: int
    output_seconds: float


class AskResponse(BaseModel):
    question: str
    lang: str
    method: str
    model: str
    answer: str
    refusal: bool
    citations: list[Section] = Field(description="sources cited in the answer")
    unsupported_citations: list[str] = Field(description="ids cited but never given: distrust")
    sources: list[Section] = Field(description="every source given to the model")
    timings: Timings


# --- The application ---------------------------------------------------------------


@dataclass
class State:
    settings: Settings
    retriever: retriever.Retriever
    model: llm.LLM
    # Searches go one at a time: the local Qdrant database is a single file.
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_app(settings: Settings | None = None, model: llm.LLM | None = None) -> FastAPI:
    """Build the application. Tests pass their own settings and a fake model."""
    settings = settings or Settings.from_environment()
    state: dict[str, State] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        has_index = (settings.index_dir / "qdrant").exists()
        embedder = embed.OllamaEmbedder(settings.embedding_model, settings.ollama_url)
        searcher = retriever.Retriever(
            settings.raw_dir, use_dense=has_index, embedder=embedder, index_dir=settings.index_dir
        )
        chat = model or llm.OllamaLLM(settings.chat_model, settings.ollama_url)
        state["app"] = State(settings, searcher, chat)
        yield
        searcher.close()

    app = FastAPI(
        title="cra-rag",
        version=__version__,
        description="Cited answers about the Cyber Resilience Act and the RED delegated act. "
        "Not legal advice.",
        lifespan=lifespan,
    )

    def current() -> State:
        return state["app"]

    def resolve(method: str, lang: str, question: str) -> tuple[str, str]:
        s = current()
        if method == "auto":
            method = "hybrid" if s.retriever.dense else "bm25"
        if method != "bm25" and not s.retriever.dense:
            raise HTTPException(
                400, f"The {method} method needs the vector index: build it or use method=bm25."
            )
        return method, (lang if lang != "auto" else search.guess_lang(question))

    def describe(section_id: str, lang: str) -> Section:
        found = next(
            s for s in current().retriever.sections if s.id == section_id and s.lang == lang
        )
        name = f"{found.label}: {found.title}" if found.title else found.label
        return Section(id=found.id, name=name, url=found.url)

    @app.get("/health")
    def health() -> Health:
        s = current()
        try:
            httpx.get(f"{s.settings.ollama_url}/api/version", timeout=2.0, trust_env=False)
            ollama = "up"
        except httpx.HTTPError:
            ollama = "down"
        return Health(
            status="ok",
            version=__version__,
            sections=len(s.retriever.sections),
            methods=list(retriever.METHODS) if s.retriever.dense else ["bm25"],
            chat_model=s.settings.chat_model,
            ollama=ollama,
        )

    @app.get("/search")
    def search_sections(
        q: str = Query(min_length=3, max_length=1000, description="the question"),
        lang: Lang = "auto",
        method: Method = "auto",
        k: int = Query(default=5, ge=1, le=20),
    ) -> SearchResponse:
        method, lang = resolve(method, lang, q)
        s = current()
        chunk_by_id = {c.id: c for c in s.retriever.chunks}
        try:
            with s.lock:
                hits = s.retriever.search(method, q, lang, k)
        except embed.EmbeddingError as exc:
            raise HTTPException(503, str(exc)) from exc
        results = []
        for hit in hits:
            excerpt = chunk_by_id[hit.chunk_id].text.split("\n", 1)[-1].replace("\n", " ")
            section = describe(hit.section_id, lang)
            results.append(SearchResult(**section.model_dump(), score=hit.score, excerpt=excerpt))
        return SearchResponse(question=q, lang=lang, method=method, results=results)

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        method, lang = resolve(request.method, request.lang, request.question)
        s = current()
        try:
            with s.lock:
                sources = answer.find_sources(
                    s.retriever, request.question, lang, method, request.passages
                )
            prompt = answer.build_prompt(request.question, sources, lang)
            completion = s.model.chat(answer.SYSTEM_PROMPT, prompt)  # minutes on a laptop
        except (embed.EmbeddingError, llm.LLMError) as exc:
            raise HTTPException(503, str(exc)) from exc
        result = answer.Answer(
            request.question, lang, answer.clean(completion.text), sources, completion
        )
        given = {x.id for x in sources}
        return AskResponse(
            question=request.question,
            lang=lang,
            method=method,
            model=s.model.model,
            answer=result.text,
            refusal=result.is_refusal,
            citations=[describe(c, lang) for c in result.citations if c in given],
            unsupported_citations=list(result.unsupported_citations),
            sources=[Section(id=x.id, name=x.name, url=x.url) for x in sources],
            timings=Timings(
                prompt_tokens=completion.prompt_tokens,
                prompt_seconds=completion.prompt_seconds,
                output_tokens=completion.output_tokens,
                output_seconds=completion.output_seconds,
            ),
        )

    return app


app = create_app()  # what `uvicorn cra_rag.api:app` serves
