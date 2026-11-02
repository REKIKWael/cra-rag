"""Judge the answers with a language model (LLM-as-a-judge).

The automatic checks of answer_eval.py see citations and refusals, not meaning.
Here a second, larger model reads each answer next to the reference answer
of the question set, and gives a verdict:

- correct:   the key facts of the reference, nothing contradicting it;
- partial:   nothing wrong, but key facts missing;
- incorrect: a contradiction, a false fact, a wrong conclusion, or a refusal
             although the reference gives an answer.

The judge answers in JSON following a schema, so its verdict can be read
reliably. Verdicts are saved as they come and reused while the answer, the
judge model and the instructions stay the same.

A judge is a model, so it can be wrong too: review.py measures how often its
verdicts agree with a human review (`uv run cra-rag agreement`).
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from cra_rag.llm import LLM
from cra_rag.questions import Question

JUDGE_MODEL = "ministral-3:8b"  # larger than the answering model (3b)
JUDGE_VERSION = 1  # change it whenever the instructions below change
VERDICTS = ("correct", "partial", "incorrect")

SYSTEM_PROMPT = """You are a strict examiner. You check answers about EU product \
cybersecurity regulation (the Cyber Resilience Act and the RED delegated act).

You receive a question, the reference answer written by an expert, and the answer to check.
Compare the answer to check with the reference answer. Do not use your own knowledge of the law.

Verdicts:
- "correct": the answer states the key facts of the reference answer (who, deadlines, amounts, \
conditions, exceptions) and nothing that contradicts it. Wording and extra details that do not \
contradict the reference do not matter.
- "partial": nothing contradicts the reference answer, but some of its key facts are missing.
- "incorrect": the answer contradicts the reference answer, states a wrong fact, reaches the \
wrong conclusion, or refuses to answer although the reference answer gives an answer.

When the question is out of scope, the reference answer says the texts do not answer it. The \
answer is then "correct" if it says so and invents nothing, otherwise "incorrect".

Give the reason in one or two sentences, in English."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def judgments_path(answers_path: Path) -> Path:
    """eval/results/answers-X.jsonl -> eval/results/judgments-X.jsonl"""
    return answers_path.with_name(answers_path.name.replace("answers-", "judgments-", 1))


def build_prompt(question: Question, model_answer: str) -> str:
    scope = " (out of scope)" if question.type == "out_of_scope" else ""
    return (
        f"Question{scope}: {question.question}\n\n"
        f"Reference answer:\n{' '.join(question.answer.split())}\n\n"
        f"Answer to check:\n{' '.join(model_answer.split())}"
    )


def parse(text: str) -> tuple[str, str]:
    """(verdict, reason) from the judge's reply; "unreadable" if it is not valid."""
    try:
        data = json.loads(text)
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict in VERDICTS:
            return verdict, str(data.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    return "unreadable", text.strip()[:300]


def fingerprint(judge_model: str, question: Question, model_answer: str) -> str:
    text = json.dumps(
        [
            judge_model,
            JUDGE_VERSION,
            SYSTEM_PROMPT,
            question.question,
            question.answer,
            model_answer,
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(
    questions: list[Question],
    answers: list[dict],
    judge: LLM,
    path: Path,
    log: Callable[[str], None] = lambda message: None,
) -> dict[str, dict]:
    """Judge every answer not judged yet. Returns the judgments by question id."""
    by_id = {q.id: q for q in questions}
    saved = {j["fingerprint"]: j for j in _load(path)}
    path.parent.mkdir(parents=True, exist_ok=True)

    judgments: dict[str, dict] = {}
    for number, record in enumerate(answers, start=1):
        q = by_id.get(record["id"])
        if q is None:
            continue
        key = fingerprint(judge.model, q, record["answer"])
        judgment = saved.get(key)
        if judgment is None:
            log(f"[{number}/{len(answers)}] {q.id} ({q.type}) ...")
            completion = judge.chat(SYSTEM_PROMPT, build_prompt(q, record["answer"]), schema=SCHEMA)
            verdict, reason = parse(completion.text)
            judgment = {
                "id": q.id,
                "fingerprint": key,
                "judge": judge.model,
                "verdict": verdict,
                "reason": reason,
            }
            with path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            log(f"      {verdict}: {reason[:150]}")
        judgments[q.id] = judgment

    with path.open("w", encoding="utf-8", newline="\n") as file:  # drop outdated verdicts
        for judgment in judgments.values():
            file.write(json.dumps(judgment, ensure_ascii=False) + "\n")
    return judgments


def still_valid(questions: list[Question], answers: list[dict], path: Path) -> dict[str, dict]:
    """The saved judgments that still match the current answers (for the report)."""
    by_id = {q.id: q for q in questions}
    answer_of = {r["id"]: r["answer"] for r in answers}
    valid = {}
    for judgment in _load(path):
        q, text = by_id.get(judgment["id"]), answer_of.get(judgment["id"])
        if (
            q
            and text is not None
            and judgment["fingerprint"] == fingerprint(judgment["judge"], q, text)
        ):
            valid[q.id] = judgment
    return valid


def tally(judgments: dict[str, dict], questions: list[Question]) -> dict[str, Counter]:
    """Verdict counts, for all questions and by question type."""
    by_id = {q.id: q for q in questions}
    counts: dict[str, Counter] = {"all": Counter()}
    for qid, judgment in judgments.items():
        counts["all"][judgment["verdict"]] += 1
        counts.setdefault(by_id[qid].type, Counter())[judgment["verdict"]] += 1
    return counts
