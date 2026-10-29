"""Tests for chunking, BM25 search and the retrieval scores.

The last test is a quality gate: it measures BM25 on the reference questions
and fails if a change makes the search worse than the level reached today.
"""

from pathlib import Path

import pytest

from cra_rag import chunks, evaluate, parse, questions, search

ROOT = Path(__file__).parent.parent


def section(text, sid="cra:art_14", title="Reporting", kind="article", lang="en"):
    return parse.Section(
        id=sid, source="cra", lang=lang, kind=kind, number="14", label="Article 14",
        title=title, chapter="", text=text, url="",
    )  # fmt: skip


# --- Chunking --------------------------------------------------------------------


def test_a_short_section_is_one_chunk_with_its_header():
    [chunk] = chunks.chunk_section(section("1. A manufacturer shall notify."))
    assert chunk.id == "cra:art_14/en#1"
    assert chunk.text == "Article 14: Reporting\n1. A manufacturer shall notify."


def test_chunk_ids_are_unique_across_languages():
    english = chunks.chunk_section(section("Text.", lang="en"))
    french = chunks.chunk_section(section("Texte.", lang="fr"))
    assert english[0].id != french[0].id


def test_a_long_section_is_cut_into_chunks_that_respect_the_size():
    text = "\n".join(f"{n}. " + "word " * 40 for n in range(1, 30))
    pieces = chunks.chunk_section(section(text), max_chars=600)
    assert len(pieces) > 1
    assert all(len(c.text) <= 600 for c in pieces)
    assert all(c.text.startswith("Article 14: Reporting\n") for c in pieces)


def test_a_list_cut_in_two_repeats_its_lead_in_sentence():
    points = "\n".join(f"({letter}) " + "detail " * 20 for letter in "abcdefgh")
    text = "2. The manufacturer shall submit:\n" + points
    pieces = chunks.chunk_section(section(text), max_chars=500)
    assert len(pieces) > 1
    assert all("2. The manufacturer shall submit:" in c.text for c in pieces)


def test_a_very_long_line_is_cut_between_sentences():
    text = " ".join(f"Sentence number {n} is here." for n in range(80))
    pieces = chunks.chunk_section(section(text), max_chars=400)
    assert all(c.text.rstrip().endswith(".") for c in pieces)


# --- BM25 ------------------------------------------------------------------------


def test_tokenize_ignores_case_accents_and_plural():
    assert search.tokenize("Les Fabricants", "fr") == search.tokenize("le fabricant", "fr")
    assert search.tokenize("sécurité", "fr") == search.tokenize("securite", "fr")


def test_bm25_ranks_the_matching_section_first_and_lists_each_section_once():
    filler = "\n".join("Member States shall cooperate on vulnerabilities." for _ in range(20))
    corpus = chunks.chunk_sections(
        [
            section("Manufacturers notify exploited vulnerabilities to ENISA.", sid="cra:art_14"),
            section("Fines may reach EUR 15 000 000.", sid="cra:art_64"),
            section(filler, sid="cra:art_59"),  # several chunks, all mentioning vulnerabilities
        ],
        max_chars=300,
    )
    hits = search.BM25Index(corpus, "en").search("notify ENISA of an exploited vulnerability")
    assert hits[0].section_id == "cra:art_14"
    assert [h.section_id for h in hits].count("cra:art_59") == 1


def test_recitals_are_not_searched():
    corpus = chunks.chunk_sections([section("vulnerability", sid="cra:rct_1", kind="recital")])
    assert search.BM25Index(corpus, "en").search("vulnerability") == []


# --- Scores ----------------------------------------------------------------------


def test_recall_and_reciprocal_rank():
    question = questions.Question(
        id="q1", lang="en", type="multi", question="?", expected=("a", "b"), answer="."
    )
    result = evaluate.Result(question, ranking=("x", "a", "y", "b"))
    assert result.recall(1) == 0
    assert result.recall(2) == 0.5
    assert result.recall(4) == 1
    assert result.reciprocal_rank == 0.5


# --- Quality gate ----------------------------------------------------------------


@pytest.fixture(scope="module")
def bm25_scores():
    sections = parse.parse_corpus(ROOT / "corpus" / "raw")
    chunk_list = chunks.chunk_sections(sections)
    indexes = {lang: search.BM25Index(chunk_list, lang) for lang in questions.LANGS}
    question_set = questions.load(ROOT / "eval" / "questions.toml")
    results = evaluate.run(question_set, lambda q: indexes[q.lang].search(q.question, 10))
    return evaluate.summary(results)


def test_bm25_does_not_get_worse(bm25_scores):
    # Measured: recall@10 = 0.86, MRR = 0.50. The margin absorbs small changes
    # to the question set; a real regression in chunking or ranking goes below.
    assert bm25_scores["recall@10"] >= 0.80
    assert bm25_scores["mrr"] >= 0.44
