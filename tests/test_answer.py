"""Tests for cited answers: prompt, citations, refusals and the Ollama chat client.

No model is run: a fake model returns a fixed answer, and Ollama is simulated.
"""

import json
from pathlib import Path

import httpx
import pytest

from cra_rag import answer, llm, retriever

ROOT = Path(__file__).parent.parent


class FakeLLM:
    model = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.received = None

    def chat(self, system, user, on_token=None):
        self.received = (system, user)
        if on_token:
            on_token(self.reply)
        return llm.Completion(text=self.reply)


@pytest.fixture(scope="module")
def bm25_only():
    return retriever.Retriever(ROOT / "corpus" / "raw", use_dense=False)


def test_the_prompt_gives_labelled_passages_then_the_question(bm25_only):
    model = FakeLLM("Au plus tard 24 heures [cra:art_14].")
    question = "Dans quel délai faut-il signaler une vulnérabilité activement exploitée ?"
    result = answer.answer(question, "fr", bm25_only, model, method="bm25", passages=6)

    system, user = model.received
    assert "[cra:art_14]" in system  # the citation format is explained
    assert user.startswith("Sources:")
    assert f"Question: {question}" in user
    assert user.endswith(answer.REMINDERS["fr"])  # the rules again, last thing the model reads
    assert all(f"[{s.id}]" in user for s in result.sources)
    assert len({s.id for s in result.sources}) == len(result.sources)  # one block per section


def test_the_passage_with_the_deadlines_reaches_the_model(bm25_only):
    question = "Dans quel délai faut-il signaler une vulnérabilité activement exploitée ?"
    result = answer.answer(question, "fr", bm25_only, FakeLLM("."), method="bm25")
    article_14 = next(s for s in result.sources if s.id == "cra:art_14")
    assert "au plus tard 24 heures" in article_14.text


def test_passages_are_capped_per_section(bm25_only):
    hits = bm25_only.search_chunks("bm25", "obligations des fabricants", "fr", n=8, per_section=2)
    assert len(hits) == 8
    assert max([h.section_id for h in hits].count(s) for s in {h.section_id for h in hits}) <= 2


def test_citations_are_read_in_any_layout():
    text = "Oui [cra:art_14], voir aussi cra:anx_I et [cra:art_14, red-da:art_3]."
    assert answer.cited_ids(text) == ("cra:art_14", "cra:anx_I", "red-da:art_3")


def test_plain_references_to_given_sources_count_as_citations():
    text = "Selon l'article 54 et l'Annexe V, puis Article 30(1) ; l'article 99 n'est pas fourni."
    given = ["cra:art_54", "cra:anx_V", "cra:art_30"]
    assert answer.cited_ids(text, given) == ("cra:art_54", "cra:art_30", "cra:anx_V")


def test_ids_are_read_whatever_their_case():
    assert answer.cited_ids("voir (RED-da:art_1) et [CRA:ANX_I]") == ("red-da:art_1", "cra:anx_I")


def test_markdown_emphasis_is_removed():
    assert answer.clean("## Titre\nLe **CSIRT** et __ENISA__") == "Titre\nLe CSIRT et ENISA"


def test_a_citation_absent_from_the_sources_is_flagged():
    sources = (answer.Source("cra:art_14", "Article 14", "", "..."),)
    result = answer.Answer(
        "q", "fr", "Oui [cra:art_14] et [cra:art_99].", sources, llm.Completion(text="")
    )
    assert result.unsupported_citations == ("cra:art_99",)


@pytest.mark.parametrize(
    "text",
    [
        "Les textes fournis ne permettent pas de répondre à cette question.",
        "The provided texts do not answer this question",
        "les textes fournis ne permettent pas de répondre à cette question.",
    ],
)
def test_refusals_are_recognised(text):
    assert answer.is_refusal(text)


def test_an_ordinary_answer_is_not_a_refusal():
    assert not answer.is_refusal("Au plus tard 24 heures après en avoir eu connaissance.")


def fake_ollama_stream(request):
    lines = [
        {"message": {"content": "Au plus "}, "done": False},
        {"message": {"content": "tard 24 heures."}, "done": False},
        {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 1800,
            "prompt_eval_duration": 45_000_000_000,
            "eval_count": 120,
            "eval_duration": 30_000_000_000,
        },
    ]
    body = "\n".join(json.dumps(line) for line in lines) + "\n"
    return httpx.Response(200, content=body.encode())


def test_ollama_chat_streams_the_answer_and_reports_the_timings():
    client = httpx.Client(transport=httpx.MockTransport(fake_ollama_stream))
    pieces = []
    completion = llm.OllamaLLM(client=client).chat("system", "user", on_token=pieces.append)

    assert pieces == ["Au plus ", "tard 24 heures."]
    assert completion.text == "Au plus tard 24 heures."
    assert (completion.prompt_tokens, completion.prompt_seconds) == (1800, 45.0)
    assert (completion.output_tokens, completion.output_seconds) == (120, 30.0)


def test_ollama_chat_explains_how_to_install_a_missing_model():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(llm.LLMError, match="ollama pull ministral-3:3b"):
        llm.OllamaLLM(client=client).chat("system", "user")
