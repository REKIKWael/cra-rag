"""Tests for the comparison between the expert's verdicts and the judge's."""

import pytest

from cra_rag import questions, review


def question(qid):
    return questions.Question(
        id=qid, lang="fr", type="lookup", question=f"Question {qid} ?", expected=("cra:art_14",),
        answer="Réponse.",
    )  # fmt: skip


def judged(*verdicts):
    return {f"q{n:03d}": {"verdict": v, "judge": "fake", "reason": "r"} for n, v in verdicts}


def test_the_template_lists_every_question_with_an_empty_verdict(tmp_path):
    path = tmp_path / "review.toml"
    review.write_template([question("q001"), question("q002")], path)
    reviews = review.load(path, {"q001", "q002"})
    assert [r.id for r in reviews] == ["q001", "q002"]
    assert all(r.verdict == "" and r.reference_ok for r in reviews)
    assert review.compare(reviews, judged((1, "correct"))).count == 0


def test_agreement_and_kappa():
    reviews = [
        review.Review("q001", "correct"),
        review.Review("q002", "partial"),
        review.Review("q003", "incorrect"),
        review.Review("q004", "incorrect"),
        review.Review("q005", ""),  # not reviewed yet: left out
    ]
    judgments = judged((1, "correct"), (2, "partial"), (3, "correct"), (4, "incorrect"), (5, "x"))
    result = review.compare(reviews, judgments)

    assert (result.agreed, result.count) == (3, 4)
    # Chance agreement: (1x2 + 1x1 + 2x1) / 4^2 = 5/16, so kappa = (3/4 - 5/16) / (11/16).
    assert result.kappa == pytest.approx(7 / 11)
    assert result.matrix()[("incorrect", "correct")] == 1
    assert result.disagreements() == [("q003", "incorrect", "correct")]


def test_mistakes_in_the_file_are_all_reported(tmp_path):
    path = tmp_path / "review.toml"
    path.write_text(
        '[[review]]\nid = "q001"\nverdict = "maybe"\n'
        '[[review]]\nid = "q999"\nverdict = "correct"\n'
        '[[review]]\nid = "q001"\nverdict = "correct"\nreference_ok = "no"\n',
        encoding="utf-8",
    )
    with pytest.raises(review.ReviewError) as error:
        review.load(path, {"q001"})
    message = str(error.value)
    assert "'maybe'" in message
    assert "q999: no such question" in message
    assert "q001: reviewed twice" in message
    assert "reference_ok must be true or false" in message
