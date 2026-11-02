"""Tests for the agent, with a scripted model (no Ollama needed).

The scripted model plays a fixed sequence of replies: tool requests, then an
answer. The tools run for real, on the corpus, with BM25 search.
"""

import copy
from pathlib import Path

import pytest

from cra_rag import agent, llm, retriever

ROOT = Path(__file__).parent.parent


class ScriptedLLM:
    model = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []  # (messages, tools) received at each turn

    def chat(self, system, user, on_token=None):
        raise AssertionError("the agent must use converse")

    def converse(self, messages, tools=None, on_token=None):
        self.calls.append((copy.deepcopy(messages), tools))
        return self.replies.pop(0)


def asks(*calls):
    """A reply in which the model asks for tools: asks(("search", {"query": "..."}))."""
    return llm.Completion(text="", tool_calls=tuple(llm.ToolCall(n, a) for n, a in calls))


def says(text):
    return llm.Completion(text=text, prompt_tokens=100, output_tokens=20)


@pytest.fixture(scope="module")
def bm25_only():
    return retriever.Retriever(ROOT / "corpus" / "raw", use_dense=False)


def run(model, bm25_only, question="Quels délais pour signaler une faille exploitée ?"):
    return agent.run(question, "fr", bm25_only, model, method="bm25")


def test_the_agent_searches_reads_then_answers_with_checked_citations(bm25_only):
    model = ScriptedLLM(
        asks(("search", {"query": "vulnérabilité activement exploitée notification"})),
        asks(("read_section", {"id": "cra:art_14"})),
        says("Alerte précoce sous 24 heures, notification sous 72 heures [cra:art_14]."),
    )
    result = run(model, bm25_only)

    assert [s.tool for s in result.steps] == ["search", "read_section"]
    assert "cra:art_14" in result.steps[0].found
    assert result.citations == ("cra:art_14",)
    assert result.unsupported_citations == ()
    assert "cra:art_14" in [s.id for s in result.sources]
    # The tool results went back to the model, named, before its answer.
    last_messages, tools = model.calls[-1]
    tool_messages = [m for m in last_messages if m["role"] == "tool"]
    assert [m["tool_name"] for m in tool_messages] == ["search", "read_section"]
    assert "[cra:art_14]" in tool_messages[1]["content"]
    assert tools == agent.TOOLS


def test_an_answer_given_without_searching_is_sent_back_once(bm25_only):
    model = ScriptedLLM(
        says("De mémoire : 24 heures."),
        asks(("search", {"query": "délai alerte précoce"})),
        says("Au plus tard 24 heures [cra:art_14]."),
    )
    result = run(model, bm25_only)

    assert model.calls[1][0][-1] == {"role": "user", "content": agent.SEARCH_FIRST}
    assert result.text == "Au plus tard 24 heures [cra:art_14]."
    assert len(result.steps) == 1


def test_a_model_that_keeps_searching_must_answer_at_the_last_turn(bm25_only):
    searching = [asks(("search", {"query": f"essai {n}"})) for n in range(agent.MAX_TURNS - 1)]
    model = ScriptedLLM(*searching, says("Les textes fournis ne permettent pas de répondre."))
    result = run(model, bm25_only)

    last_messages, last_tools = model.calls[-1]
    assert last_tools is None  # no tools offered: it has to answer
    assert last_messages[-1] == {"role": "user", "content": agent.ANSWER_NOW}
    assert len(model.calls) == agent.MAX_TURNS
    assert len(result.steps) == agent.MAX_TURNS - 1


def test_a_citation_of_a_section_no_tool_returned_is_flagged(bm25_only):
    model = ScriptedLLM(
        asks(("read_section", {"id": "cra:art_14"})),
        says("Voir [cra:art_14] et [cra:art_99]."),
    )
    result = run(model, bm25_only)
    assert result.unsupported_citations == ("cra:art_99",)


def test_tool_mistakes_are_explained_to_the_model(bm25_only):
    tools = agent.Tools(bm25_only, "fr", "bm25")
    unknown_section, found = tools.run(llm.ToolCall("read_section", {"id": "cra:art_999"}))
    assert unknown_section.startswith("Error: no such section") and found == ()
    unknown_tool, _ = tools.run(llm.ToolCall("browse_web", {}))
    assert "unknown tool" in unknown_tool
    empty_query, _ = tools.run(llm.ToolCall("search", {}))
    assert empty_query.startswith("Error: give a query")


def test_a_long_section_is_cut(bm25_only):
    longest = max((s for s in bm25_only.sections if s.lang == "en"), key=lambda s: len(s.text))
    assert len(longest.text) > agent.SECTION_CHARS
    tools = agent.Tools(bm25_only, "en", "bm25")
    text, found = tools.run(llm.ToolCall("read_section", {"id": longest.id}))
    assert found == (longest.id,)
    assert len(text) < agent.SECTION_CHARS + 200
    assert text.rstrip().endswith("(cut: search for the part you need)")


def test_the_timings_of_every_turn_add_up(bm25_only):
    model = ScriptedLLM(asks(("search", {"query": "amende"})), says("15 000 000 EUR [cra:art_64]."))
    result = run(model, bm25_only, "Quelle amende maximale ?")
    assert result.completion.prompt_tokens == 100  # only the answer turn reported tokens here
    assert result.completion.text == "15 000 000 EUR [cra:art_64]."
