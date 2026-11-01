"""Tests for the evaluation of answers, with a fake model (no Ollama needed)."""

import json
from pathlib import Path

import pytest

from cra_rag import answer_eval, llm, questions, retriever

ROOT = Path(__file__).parent.parent


class FakeLLM:
    model = "fake-model"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat(self, system, user, on_token=None):
        self.calls += 1
        return llm.Completion(text=self.reply, prompt_seconds=2.0, output_seconds=1.0)


def question(qid="q001", expected=("cra:art_14",), kind="lookup", reference="24 heures."):
    return questions.Question(
        id=qid, lang="fr", type=kind, question=f"Question {qid} sur les délais de signalement ?",
        expected=expected, answer=reference,
    )  # fmt: skip


@pytest.fixture(scope="module")
def bm25_only():
    return retriever.Retriever(ROOT / "corpus" / "raw", use_dense=False)


def test_key_figures_ignore_spacing_decimal_commas_and_article_numbers():
    assert answer_eval.key_figures("15 000 000 EUR ou 2,5 % (article 64, paragraphe 2)") == {
        "15000000",
        "2.5",
    }


def test_a_correct_answer_passes_every_check():
    record = {
        "answer": "Au plus tard 24 heures [cra:art_14].",
        "citations": ["cra:art_14"],
        "unsupported_citations": [],
        "refusal": False,
        "sources": ["cra:art_14"],
    }
    verdicts = answer_eval.checks(question(), record)
    assert verdicts == {
        "supported_citations": True,
        "answered": True,
        "cites_expected": True,
        "key_figures": True,
    }


def test_an_out_of_scope_question_must_be_declined():
    q = question(qid="q040", expected=(), kind="out_of_scope")
    invented = {"answer": "Il faut écrire au CERT-FR.", "sources": ["cra:art_14"]}
    honest = {
        "answer": "Via la plateforme unique [cra:art_14]. Les textes ne précisent pas l'organisme.",
        "sources": ["cra:art_14"],
    }
    assert answer_eval.checks(q, invented)["declines"] is False
    assert answer_eval.checks(q, honest)["declines"] is True


def test_key_figures_written_in_millions_match_digits():
    assert answer_eval.key_figures("15 millions d'euros") == answer_eval.key_figures("15 000 000")


def test_thousands_separators_of_every_language_give_the_same_figure():
    written = ["EUR 5,000,000", "5.000.000 EUR", "5 000 000 EUR", "5 millions d'euros"]
    assert all(answer_eval.key_figures(text) == {"5000000"} for text in written)


def test_a_run_limited_to_some_questions_keeps_the_other_answers(tmp_path, bm25_only):
    path = tmp_path / "answers.jsonl"
    reply = FakeLLM("Au plus tard 24 heures [cra:art_14].")
    answer_eval.run([question("q001"), question("q002")], bm25_only, reply, "bm25", 4, path)
    answer_eval.run([question("q002")], bm25_only, reply, "bm25", 4, path)
    assert sorted(json.loads(line)["id"] for line in path.read_text().splitlines()) == [
        "q001",
        "q002",
    ]


def test_answers_are_saved_and_reused_on_the_next_run(tmp_path, bm25_only):
    path = tmp_path / "answers.jsonl"
    qs = [question("q001"), question("q002")]
    first = FakeLLM("Au plus tard 24 heures [cra:art_14].")
    records = answer_eval.run(qs, bm25_only, first, "bm25", 4, path)

    assert first.calls == 2
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["q001", "q002"]
    assert records[0]["seconds"] == 3.0

    again = FakeLLM("not used")
    answer_eval.run(qs, bm25_only, again, "bm25", 4, path)
    assert again.calls == 0


def test_changing_a_setting_recomputes_the_answers(tmp_path, bm25_only):
    path = tmp_path / "answers.jsonl"
    answer_eval.run([question()], bm25_only, FakeLLM("A [cra:art_14]."), "bm25", 4, path)
    other = FakeLLM("B [cra:art_14].")
    answer_eval.run([question()], bm25_only, other, "bm25", 6, path)  # 6 passages, not 4
    assert other.calls == 1


def test_the_report_shows_scores_and_every_answer(tmp_path, bm25_only):
    qs = [question("q001"), question("q008", expected=(), kind="out_of_scope")]
    reply = "Les textes fournis ne permettent pas de répondre à cette question."
    records = answer_eval.run(qs, bm25_only, FakeLLM(reply), "bm25", 4, tmp_path / "a.jsonl")
    report = tmp_path / "a.md"
    answer_eval.write_report(qs, records, report)
    text = report.read_text(encoding="utf-8")

    assert "| Says when the texts do not answer (out of scope) | 1/1 (100%) |" in text
    assert "| Answers (no false refusal) | 0/1 (0%) |" in text
    assert "## q001 · lookup · fr · TO CHECK" in text
