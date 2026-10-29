"""Tests for the reference question set.

The last test checks the real eval/questions.toml: if an edit breaks it
(typo in a field, section id that does not exist...), the CI turns red.
"""

from pathlib import Path

import pytest

from cra_rag import parse, questions

ROOT = Path(__file__).parent.parent
VALID = """
[[questions]]
id = "q001"
lang = "fr"
type = "lookup"
question = "Quels délais pour signaler une vulnérabilité ?"
expected = ["cra:art_14"]
answer = "24 heures pour l'alerte précoce."
"""


def load_text(tmp_path, text):
    path = tmp_path / "questions.toml"
    path.write_text(text, encoding="utf-8")
    return questions.load(path)


def make(**overrides):
    fields = dict(
        id="q001", lang="fr", type="lookup", question="Q?", expected=("cra:art_14",), answer="A."
    )
    return questions.Question(**(fields | overrides))


def test_loads_a_valid_file(tmp_path):
    [question] = load_text(tmp_path, VALID)
    assert question.expected == ("cra:art_14",)
    assert question.tags == ()


def test_reports_a_misspelled_field(tmp_path):
    with pytest.raises(questions.QuestionsError, match="unknown field 'expeted'"):
        load_text(tmp_path, VALID.replace("expected", "expeted"))


def test_reports_invalid_toml_with_its_line(tmp_path):
    with pytest.raises(questions.QuestionsError, match="line"):
        load_text(tmp_path, VALID.replace('lang = "fr"', "lang = fr"))


def test_explains_a_file_not_saved_as_utf8(tmp_path):
    path = tmp_path / "questions.toml"
    path.write_bytes(VALID.encode("cp1252"))  # "é" written the Windows-1252 way

    with pytest.raises(questions.QuestionsError, match="not saved as UTF-8.*line 6"):
        questions.load(path)


@pytest.mark.parametrize(
    ("question", "problem"),
    [
        (make(expected=("cra:art_999",)), "unknown section"),
        (make(type="out_of_scope"), "expects no section"),
        (make(expected=()), "list the sections"),
        (make(type="multi"), "at least two sections"),
        (make(lang="de"), "lang must be"),
    ],
)
def test_check_finds_mistakes(question, problem):
    [found] = questions.check([question], {"cra:art_14"})
    assert problem in found


def test_check_finds_duplicate_ids():
    problems = questions.check([make(), make(question="Other?")], {"cra:art_14"})
    assert problems == ["q001: id used 2 times"]


@pytest.fixture(scope="module")
def corpus():
    return parse.parse_corpus(ROOT / "corpus" / "raw")


def test_coverage_counts_questions_per_chapter(corpus):
    counts = questions.coverage([make(), make(id="q002")], corpus)
    chapter_ii = next(part for part in counts if part.startswith("Chapter II:"))
    assert counts[chapter_ii] == 2
    assert counts["Annex I: Essential cybersecurity requirements"] == 0


def test_the_committed_question_set_is_valid(corpus):
    question_set = questions.load(ROOT / "eval" / "questions.toml")
    assert question_set, "eval/questions.toml has no question"
    assert questions.check(question_set, {s.id for s in corpus}) == []
