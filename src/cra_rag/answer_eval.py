"""Answer all the reference questions and check the answers automatically.

Checks that need no judgement (a model acting as judge comes at step 7):
- answerable questions: the model answers (no false refusal), cites one of the
  expected sections, cites only sources it was given, and repeats the key
  figures of the reference answer (deadlines, amounts, dates);
- out_of_scope questions: the model refuses.

Answers take minutes each on a laptop, so every answer is saved as soon as it is
written, with a fingerprint of the settings (model, prompt, search method...).
An interrupted run resumes; changing a setting recomputes only what it affects.
Results are written to eval/results/, as JSON lines and as a readable report.
"""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from cra_rag import answer
from cra_rag.llm import LLM
from cra_rag.questions import Question
from cra_rag.retriever import Retriever

RESULTS_DIR = Path("eval/results")
NUMBER = re.compile(r"\d+(?:[ \u00a0\u202f.,]\d+)*")
MILLIONS = re.compile(r"(\d+(?:[.,]\d+)?)\s*millions?\b", re.IGNORECASE)
# References to articles or regulations are not facts to repeat: the citation does that.
REFERENCE = re.compile(
    r"\((?:articles?|annexe?s?)\b[^)]*\)"  # "(article 13, paragraphe 20)"
    r"|\b(?:articles?|annexe?s?|paragraph(?:e|s|es)?|points?)\s+\d+(?:\(\d+\))*"
    r"(?:\s*(?:,|et|and|à|to)\s*\d+(?:\(\d+\))*)*"  # "articles 13 et 14", "Article 64(4)"
    r"|\(\d+\)"  # "(16)" left after "Article 13(15), (16)"
    r"|\b\d{4}/\d+\b",  # regulation numbers: "2024/2847"
    re.IGNORECASE,
)


def settings_fingerprint(model: str, method: str, passages: int) -> str:
    text = json.dumps([model, method, passages, answer.PROMPT_VERSION, answer.SYSTEM_PROMPT])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def results_path(results_dir: Path, model: str) -> Path:
    return results_dir / f"answers-{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.jsonl"


def key_figures(text: str) -> set[str]:
    """Numbers of two digits or more ("24", "15 000 000", "2,5"), written one way,
    leaving out references such as "(article 13, paragraph 20)" or "2024/2847"."""
    text = REFERENCE.sub(" ", text)
    # "15 millions d'euros" and "EUR 15 000 000" are the same figure.
    text = MILLIONS.sub(lambda m: str(round(float(m.group(1).replace(",", ".")) * 1e6)), text)
    found = set()
    for number in NUMBER.findall(text):
        compact = re.sub(r"[ \u00a0\u202f]", "", number).replace(",", ".").rstrip(".")
        if len(compact.replace(".", "")) >= 2:
            found.add(compact)
    return found


def checks(question: Question, record: dict) -> dict:
    """True or False for each check that applies to this question.

    Citations and refusals are read again from the saved answer text, so an
    improvement of the reader applies to past answers without asking the model again.
    """
    cited = set(answer.cited_ids(record["answer"], record["sources"]))
    unsupported = [c for c in answer.explicit_ids(record["answer"]) if c not in record["sources"]]
    refusal = answer.is_refusal(record["answer"])
    result = {"supported_citations": not unsupported}
    if not question.expected:
        result["refused"] = refusal
        return result
    result["answered"] = not refusal
    result["cites_expected"] = bool(cited & set(question.expected))
    expected_figures = key_figures(question.answer)
    if expected_figures:
        # Figures inside citations ("[cra:art_14]") do not count as stated facts.
        stated = key_figures(answer.CITATION.sub(" ", record["answer"]))
        result["key_figures"] = expected_figures <= stated
    return result


def run(
    questions: list[Question],
    retriever: Retriever,
    llm: LLM,
    method: str,
    passages: int,
    path: Path,
    log: Callable[[str], None] = lambda message: None,
) -> list[dict]:
    """Answer every question not answered yet with these settings. Returns the records."""
    fingerprint = settings_fingerprint(llm.model, method, passages)
    saved = {(r["id"], r["question"]): r for r in _load(path) if r["settings"] == fingerprint}
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for number, q in enumerate(questions, start=1):
        record = saved.get((q.id, q.question))
        if record is None:
            log(f"[{number}/{len(questions)}] {q.id} ({q.type}, {q.lang}) ...")
            result = answer.answer(q.question, q.lang, retriever, llm, method, passages)
            c = result.completion
            record = {
                "id": q.id,
                "question": q.question,
                "settings": fingerprint,
                "model": llm.model,
                "method": method,
                "answer": result.text,
                "citations": list(result.citations),
                "unsupported_citations": list(result.unsupported_citations),
                "sources": [s.id for s in result.sources],
                "refusal": result.is_refusal,
                "seconds": round(c.prompt_seconds + c.output_seconds, 1),
                "prompt_tokens": c.prompt_tokens,
                "output_tokens": c.output_tokens,
            }
            with path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            verdicts = checks(q, record)
            failed = [name for name, ok in verdicts.items() if not ok]
            outcome = f"to check: {', '.join(failed)}" if failed else "all checks passed"
            log(f"      {record['seconds']:.0f} s, {outcome}")
        records.append(record)

    # Rewrite the file with these answers, keeping those of the questions not asked in
    # this run: a run limited with --only must not erase the others.
    asked = {r["id"] for r in records}
    others = [r for r in _load(path) if r["id"] not in asked]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in others + records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summary(questions: list[Question], records: list[dict]) -> dict[str, tuple[int, int]]:
    """For each check: (passed, applicable)."""
    by_id = {q.id: q for q in questions}
    totals: dict[str, list[int]] = {}
    for record in records:
        for name, ok in checks(by_id[record["id"]], record).items():
            passed, count = totals.setdefault(name, [0, 0])
            totals[name] = [passed + ok, count + 1]
    return {name: (passed, count) for name, (passed, count) in totals.items()}


CHECK_LABELS = {
    "answered": "Answers (no false refusal)",
    "cites_expected": "Cites an expected section",
    "supported_citations": "Cites only the sources it was given",
    "key_figures": "Repeats the key figures of the reference answer",
    "refused": "Refuses out-of-scope questions",
}


def write_report(questions: list[Question], records: list[dict], path: Path) -> None:
    """A Markdown report, readable on GitHub: scores, then every answer."""
    by_id = {q.id: q for q in questions}
    totals = summary(questions, records)
    seconds = sorted(r["seconds"] for r in records)
    lines = [
        f"# Answers of {records[0]['model']} ({records[0]['method']} search)",
        "",
        "Generated by `uv run cra-rag eval-answers`. Automatic checks only; the judgement",
        "of correctness by a model comes at step 7.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, label in CHECK_LABELS.items():
        if name in totals:
            passed, count = totals[name]
            lines.append(f"| {label} | {passed}/{count} ({passed / count:.0%}) |")
    lines += [
        "",
        f"Median time per answer: {seconds[len(seconds) // 2]:.0f} s on the machine that ran it.",
        "",
    ]
    for record in records:
        q = by_id[record["id"]]
        verdicts = checks(q, record)
        status = "OK" if all(verdicts.values()) else "TO CHECK"
        cited = answer.cited_ids(record["answer"], record["sources"])
        marks = ", ".join(f"{name} {'yes' if ok else 'NO'}" for name, ok in verdicts.items())
        lines += [
            f"## {q.id} · {q.type} · {q.lang} · {status}",
            "",
            f"**Question:** {q.question}",
            "",
            f"**Reference answer:** {' '.join(q.answer.split())}",
            "",
            f"**Model answer:** {' '.join(record['answer'].split())}",
            "",
            f"Expected: {', '.join(q.expected) or 'none (out of scope)'} · "
            f"cited: {', '.join(cited) or 'none'} · "
            f"given: {', '.join(record['sources'])}",
            "",
            f"Checks: {marks} · {record['seconds']:.0f} s",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
