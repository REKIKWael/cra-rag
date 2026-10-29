"""Tests for the Ollama embedding client. No Ollama needed: the server is simulated."""

import json
from pathlib import Path

import httpx
import pytest

from cra_rag import embed, questions, search

ROOT = Path(__file__).parent.parent


def embedder(handler, batch_size=2):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return embed.OllamaEmbedder(client=client, batch_size=batch_size)


def fake_ollama(request):
    texts = json.loads(request.content)["input"]
    return httpx.Response(200, json={"embeddings": [[float(len(t)), 1.0] for t in texts]})


def test_returns_one_vector_per_text_in_order_using_batches():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return fake_ollama(request)

    vectors = embedder(handler).embed(["a", "bb", "ccc"])

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert [len(call["input"]) for call in calls] == [2, 1]
    assert calls[0]["model"] == "bge-m3"


def test_explains_how_to_install_a_missing_model():
    def handler(request):
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(embed.EmbeddingError, match="ollama pull bge-m3"):
        embedder(handler).embed(["text"])


def test_explains_that_ollama_is_not_running():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(embed.EmbeddingError, match="Start the Ollama application"):
        embedder(handler).embed(["text"])


def test_guess_lang_recognises_every_reference_question():
    for q in questions.load(ROOT / "eval" / "questions.toml"):
        assert search.guess_lang(q.question) == q.lang, q.id
