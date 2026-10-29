"""The reference question set (eval/questions.toml) and its checks.

The question set is the yardstick for every later step: retrieval is scored on
whether it finds the `expected` sections, answers on whether they contain the
facts in `answer`. A mistake in this file would distort every score, so it is
checked like code, in the CI.
"""

import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

QUESTIONS_FILE = Path("eval/questions.toml")
LANGS = ("fr", "en")
TYPES = ("lookup", "multi", "trap", "out_of_scope")
REQUIRED = {"id": str, "lang": str, "type": str, "question": str, "expected": list, "answer": str}
OPTIONAL = {"tags": list, "notes": str}


@dataclass(frozen=True)
class Question:
    id: str
    lang: str
    type: str
    question: str
    expected: tuple[str, ...]
    answer: str
    tags: tuple[str, ...] = ()
    notes: str = ""


class QuestionsError(ValueError):
    """The file cannot be read as a question set."""


def load(path: Path = QUESTIONS_FILE) -> list[Question]:
    """Read the file. Raises QuestionsError listing every structural problem."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise QuestionsError(f"{path} is not valid TOML: {exc}") from exc

    problems, questions = [], []
    for position, entry in enumerate(data.get("questions", []), start=1):
        where = f"question #{position} ({entry.get('id', 'no id')})"
        for key, kind in REQUIRED.items():
            if key not in entry:
                problems.append(f"{where}: missing field '{key}'")
            elif not isinstance(entry[key], kind):
                problems.append(f"{where}: '{key}' should be a {kind.__name__}")
        for key in entry.keys() - REQUIRED.keys() - OPTIONAL.keys():
            problems.append(f"{where}: unknown field '{key}' (typo?)")
        if not any(p.startswith(where) for p in problems):
            questions.append(
                Question(
                    **{k: entry[k] for k in REQUIRED if k != "expected"},
                    expected=tuple(entry["expected"]),
                    tags=tuple(entry.get("tags", ())),
                    notes=entry.get("notes", ""),
                )
            )
    if problems:
        raise QuestionsError("\n".join(problems))
    return questions


def coverage(questions: list[Question], sections: list) -> dict[str, int]:
    """Number of questions touching each part of the corpus: every chapter and
    annex of the CRA, and the RED delegated act. Shows what is not tested yet."""
    groups: dict[str, str] = {}  # section id -> part of the corpus
    for s in sections:
        if s.lang != "en" or s.kind == "recital":
            continue
        if s.source == "red-da":
            groups[s.id] = "RED delegated act"
        elif s.kind == "article":
            groups[s.id] = s.chapter
        else:
            groups[s.id] = f"{s.label}: {s.title}"

    counts = dict.fromkeys(dict.fromkeys(groups.values()), 0)  # keeps the document order
    for q in questions:
        for part in {groups[sid] for sid in q.expected if sid in groups}:
            counts[part] += 1
    return counts


def check(questions: list[Question], section_ids: set[str]) -> list[str]:
    """Return the problems found; an empty list means the set is usable."""
    problems = []
    for qid, n in Counter(q.id for q in questions).items():
        if n > 1:
            problems.append(f"{qid}: id used {n} times")
    for text, n in Counter(q.question.strip().lower() for q in questions).items():
        if n > 1:
            problems.append(f"same question asked {n} times: {text[:60]!r}")

    for q in questions:
        if q.lang not in LANGS:
            problems.append(f"{q.id}: lang must be one of {LANGS}, not {q.lang!r}")
        if q.type not in TYPES:
            problems.append(f"{q.id}: type must be one of {TYPES}, not {q.type!r}")
        if not q.question.strip() or not q.answer.strip():
            problems.append(f"{q.id}: question and answer must not be empty")
        if q.type == "out_of_scope" and q.expected:
            problems.append(f"{q.id}: an out_of_scope question expects no section")
        if q.type != "out_of_scope" and not q.expected:
            problems.append(f"{q.id}: list the sections that answer it in 'expected'")
        if q.type == "multi" and len(q.expected) < 2:
            problems.append(f"{q.id}: a multi question needs at least two sections")
        for section_id in q.expected:
            if section_id not in section_ids:
                problems.append(f"{q.id}: unknown section {section_id!r}")
    return problems
