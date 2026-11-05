"""Tests for the intent routing: questions about the assistant get a fixed presentation."""

from pathlib import Path

import pytest

from cra_rag import agent, answer, questions, router

ROOT = Path(__file__).parent.parent

ABOUT_THE_ASSISTANT = [
    "C'est quoi ton rôle, ton objectif, tes capacités ?",
    "Qui es-tu ?",
    "Que sais-tu faire ?",
    "Qu'est-ce que vous pouvez faire ?",
    "Tu sers à quoi ?",
    "Présente-toi",
    "Comment tu fonctionnes ?",
    "Bonjour !",
    "Who are you?",
    "What can you do?",
    "What are your limitations?",
    "hello",
]
ABOUT_THE_REGULATION = [
    "Quel est le rôle de l'ENISA ?",
    "Quel est votre rôle si vous importez des caméras ?",
    "Que peut faire l'autorité de surveillance du marché ?",
    "What can you do if a vulnerability is actively exploited?",
    "Bonjour, quel délai pour signaler une faille ?",
    "Who must report an incident to the CSIRT?",
]


@pytest.mark.parametrize("question", ABOUT_THE_ASSISTANT)
def test_questions_about_the_assistant_are_recognised(question):
    assert router.is_about_assistant(question)


@pytest.mark.parametrize("question", ABOUT_THE_REGULATION)
def test_questions_about_the_regulation_go_on_to_the_search(question):
    assert not router.is_about_assistant(question)


def test_no_reference_question_is_caught_by_mistake():
    caught = [q.id for q in questions.load(ROOT / "eval" / "questions.toml")
              if router.is_about_assistant(q.question)]  # fmt: skip
    assert caught == []


class NoModel:
    model = "unused"

    def chat(self, *args, **kwargs):
        raise AssertionError("a question about the assistant must not reach the model")

    converse = chat


def test_rag_and_agent_give_the_presentation_without_search_or_model():
    rag = answer.answer("Qui es-tu ?", "fr", retriever=None, llm=NoModel())
    assert rag.route == "about" and rag.text == router.ABOUT["fr"]
    assert rag.sources == () and not rag.is_refusal

    tool_user = agent.run("What can you do?", "en", retriever=None, llm=NoModel())
    assert tool_user.route == "about" and tool_user.text == router.ABOUT["en"]
    assert tool_user.steps == ()
