"""Tests for the judge, with a fake judge model (no Ollama needed)."""

import json

from cra_rag import answer_eval, judge, llm, questions


class FakeJudge:
    model = "fake-judge"

    def __init__(self, reply='{"verdict": "partial", "reason": "The 72-hour step is missing."}'):
        self.reply = reply
        self.calls = []

    def chat(self, system, user, on_token=None, schema=None):
        self.calls.append((system, user, schema))
        return llm.Completion(text=self.reply)


def question(qid="q001", kind="lookup", expected=("cra:art_14",)):
    return questions.Question(
        id=qid, lang="fr", type=kind, question="Quels délais ?", expected=expected,
        answer="24 heures, puis 72 heures.",
    )  # fmt: skip


def answer_record(qid="q001", text="Au plus tard 24 heures [cra:art_14]."):
    return {"id": qid, "answer": text, "sources": ["cra:art_14"], "citations": [],
            "refusal": False, "seconds": 60.0, "model": "m", "method": "bm25"}  # fmt: skip


def test_the_judge_sees_the_reference_and_must_follow_the_schema(tmp_path):
    fake = FakeJudge()
    judgments = judge.run([question()], [answer_record()], fake, tmp_path / "j.jsonl")

    system, user, schema = fake.calls[0]
    assert "Reference answer:\n24 heures, puis 72 heures." in user
    assert "Answer to check:\nAu plus tard 24 heures [cra:art_14]." in user
    assert schema == judge.SCHEMA
    assert judgments["q001"]["verdict"] == "partial"


def test_out_of_scope_questions_are_flagged_to_the_judge(tmp_path):
    fake = FakeJudge('{"verdict": "correct", "reason": "Says the texts do not cover it."}')
    q = question("q040", kind="out_of_scope", expected=())
    judge.run([q], [answer_record("q040")], fake, tmp_path / "j.jsonl")
    assert fake.calls[0][1].startswith("Question (out of scope):")


def test_verdicts_are_reused_until_the_answer_changes(tmp_path):
    path = tmp_path / "j.jsonl"
    judge.run([question()], [answer_record()], FakeJudge(), path)

    same = FakeJudge()
    judge.run([question()], [answer_record()], same, path)
    assert same.calls == []

    changed = FakeJudge()
    judge.run([question()], [answer_record(text="Autre réponse.")], changed, path)
    assert len(changed.calls) == 1


def test_an_invalid_reply_is_marked_unreadable():
    assert judge.parse("not json")[0] == "unreadable"
    assert judge.parse('{"verdict": "maybe"}')[0] == "unreadable"
    assert judge.parse('{"verdict": "Correct", "reason": "ok"}') == ("correct", "ok")


def test_the_report_shows_the_verdicts(tmp_path):
    qs = [question()]
    records = [answer_record()]
    judgments = judge.run(qs, records, FakeJudge(), tmp_path / "j.jsonl")
    report = tmp_path / "r.md"
    answer_eval.write_report(qs, records, report, judgments)
    text = report.read_text(encoding="utf-8")

    assert "| all (1) | 0 | 1 | 0 |" in text
    assert "**Judge:** partial. The 72-hour step is missing." in text


def test_only_judgments_matching_the_current_answers_are_kept(tmp_path):
    path = tmp_path / "j.jsonl"
    judge.run([question()], [answer_record()], FakeJudge(), path)
    assert judge.still_valid([question()], [answer_record()], path)
    assert not judge.still_valid([question()], [answer_record(text="Changed.")], path)
    assert json.loads(path.read_text().splitlines()[0])["judge"] == "fake-judge"
