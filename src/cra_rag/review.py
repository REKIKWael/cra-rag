"""Check the judge against the expert: how often do the two agree?

A judge model is worth its verdicts only if they match what an expert would say.
The expert gives their own verdict on each answer in eval/expert_review.toml,
before reading the judge's, and this module compares the two:

- agreement: the share of answers where both give the same verdict;
- Cohen's kappa: the same, corrected for the agreement expected by chance
  (1 = perfect, 0 = no better than chance);
- the disagreements, to read one by one: they show what to change in the
  judge's instructions.

The same pass is used to check the reference answers themselves
(`reference_ok = false` when one needs fixing).
"""

import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cra_rag.judge import VERDICTS
from cra_rag.questions import Question

REVIEW_FILE = Path("eval/expert_review.toml")

HEADER = """\
# Expert review of the answers in eval/results/answers-ministral-3_3b.md
#
# For each question, read the question, the reference answer and the model answer,
# then write your own verdict BEFORE reading the judge's line, so as not to be swayed:
#   verdict = "correct"    the key facts of the reference answer, nothing contradicting it
#             "partial"    nothing wrong, but key facts missing
#             "incorrect"  a contradiction, a false fact, a wrong conclusion,
#                          or a refusal although the reference answer gives an answer
#   (out-of-scope questions: "correct" if the answer says the texts do not cover it
#    and invents nothing, otherwise "incorrect")
#   reference_ok = false   if the reference answer itself needs fixing (say why in note)
#
# Leave verdict empty for an answer not reviewed yet. Then: uv run cra-rag agreement

reviewer = ""  # who gave these verdicts
"""


@dataclass(frozen=True)
class Review:
    id: str
    verdict: str  # "" while not reviewed yet
    reference_ok: bool = True
    note: str = ""


class ReviewError(ValueError):
    """The review file cannot be read."""


def write_template(question_set: list[Question], path: Path) -> None:
    """An entry per question, verdicts left empty."""
    lines = [HEADER]
    for q in question_set:
        words = " ".join(q.question.split())
        summary = words if len(words) <= 80 else words[:79] + "…"
        lines += [
            "[[review]]",
            f'id = "{q.id}"  # {q.type}, {q.lang}: {summary}',
            'verdict = ""',
            "reference_ok = true",
            'note = ""',
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _read(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ReviewError(f"{path} is not saved as UTF-8 ({exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ReviewError(f"{path} is not valid TOML: {exc}") from exc


def reviewer(path: Path) -> str:
    """Who gave the verdicts, as written at the top of the file."""
    return str(_read(path).get("reviewer", "")).strip()


def load(path: Path, known_ids: set[str]) -> list[Review]:
    """Read the file. Raises ReviewError listing every problem found."""
    data = _read(path)
    problems, reviews, seen = [], [], set()
    for entry in data.get("review", []):
        rid = str(entry.get("id", ""))
        verdict = str(entry.get("verdict", "")).strip().lower()
        reference_ok = entry.get("reference_ok", True)
        if rid not in known_ids:
            problems.append(f"{rid or 'an entry without id'}: no such question")
        elif rid in seen:
            problems.append(f"{rid}: reviewed twice")
        if verdict and verdict not in VERDICTS:
            problems.append(
                f"{rid}: verdict must be {', '.join(VERDICTS)} or empty, not {verdict!r}"
            )
        if not isinstance(reference_ok, bool):
            problems.append(f"{rid}: reference_ok must be true or false")
        seen.add(rid)
        reviews.append(Review(rid, verdict, reference_ok is not False, str(entry.get("note", ""))))
    if problems:
        raise ReviewError(f"{path}:\n  " + "\n  ".join(problems))
    return reviews


@dataclass(frozen=True)
class Agreement:
    pairs: tuple[tuple[str, str, str], ...]  # (question id, expert verdict, judge verdict)

    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def agreed(self) -> int:
        return sum(expert == judged for _, expert, judged in self.pairs)

    @property
    def rate(self) -> float:
        return self.agreed / self.count if self.count else 0.0

    @property
    def kappa(self) -> float:
        """Cohen's kappa: (observed - chance) / (1 - chance)."""
        if not self.count:
            return 0.0
        expert = Counter(e for _, e, _ in self.pairs)
        judged = Counter(j for _, _, j in self.pairs)
        chance = sum(expert[v] * judged[v] for v in VERDICTS) / self.count**2
        return 1.0 if chance == 1 else (self.rate - chance) / (1 - chance)

    def matrix(self) -> Counter:
        """(expert verdict, judge verdict) -> number of answers."""
        return Counter((expert, judged) for _, expert, judged in self.pairs)

    def disagreements(self) -> list[tuple[str, str, str]]:
        return [p for p in self.pairs if p[1] != p[2]]


def compare(reviews: list[Review], judgments: dict[str, dict]) -> Agreement:
    """The answers that both the expert and the judge have given a verdict on."""
    return Agreement(
        tuple(
            (r.id, r.verdict, judgments[r.id]["verdict"])
            for r in reviews
            if r.verdict and r.id in judgments
        )
    )
