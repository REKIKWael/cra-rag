"""Tests for the HTTP API, with BM25 only and a fake model (no Ollama needed)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cra_rag import api, llm

ROOT = Path(__file__).parent.parent


class FakeLLM:
    model = "fake-model"

    def __init__(
        self, reply="Au plus tard 24 heures [cra:art_14], voir aussi [cra:art_99].", error=None
    ):
        self.reply, self.error = reply, error

    def chat(self, system, user, on_token=None):
        if self.error:
            raise llm.LLMError(self.error)
        return llm.Completion(text=self.reply, prompt_tokens=1500, prompt_seconds=40.0)


def client_for(tmp_path, model):
    settings = api.Settings(
        ollama_url="http://127.0.0.1:9",  # nothing listens there: Ollama is "down"
        raw_dir=ROOT / "corpus" / "raw",
        index_dir=tmp_path / "no-index",  # no vector index: BM25 only
    )
    return TestClient(api.create_app(settings, model=model))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with client_for(tmp_path_factory.mktemp("api"), FakeLLM()) as test_client:
        yield test_client


def test_health_says_which_methods_are_available(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["sections"] == 462
    assert body["methods"] == ["bm25"]
    assert body["ollama"] == "down"


def test_search_returns_sections_with_their_links(client):
    response = client.get("/search", params={"q": "délai de 24 heures alerte précoce ENISA"})
    body = response.json()
    assert response.status_code == 200
    assert body["method"] == "bm25"  # "auto" falls back to BM25 without a vector index
    assert body["lang"] == "fr"
    assert "cra:art_14" in [r["id"] for r in body["results"]]
    assert body["results"][0]["url"].startswith("https://eur-lex.europa.eu/")


def test_semantic_search_without_an_index_is_refused_clearly(client):
    response = client.get("/search", params={"q": "délai de signalement", "method": "dense"})
    assert response.status_code == 400
    assert "vector index" in response.json()["detail"]


def test_ask_returns_the_answer_its_citations_and_its_sources(client):
    question = "Dans quel délai faut-il signaler une vulnérabilité activement exploitée ?"
    body = client.post("/ask", json={"question": question}).json()
    assert body["answer"].startswith("Au plus tard 24 heures")
    assert [c["id"] for c in body["citations"]] == ["cra:art_14"]
    assert body["unsupported_citations"] == ["cra:art_99"]
    assert "cra:art_14" in [s["id"] for s in body["sources"]]
    assert body["timings"]["prompt_tokens"] == 1500


def test_a_too_short_question_is_rejected(client):
    assert client.post("/ask", json={"question": "?"}).status_code == 422


def test_ask_reports_an_unavailable_model_as_503(tmp_path):
    with client_for(tmp_path, FakeLLM(error="Ollama does not answer")) as failing:
        response = failing.post("/ask", json={"question": "Quels délais de signalement ?"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
