"""Score a search method on the reference questions.

For each question with expected sections (out_of_scope questions have none):
- recall@k: share of the expected sections found in the first k results;
- MRR (mean reciprocal rank): 1 if the first expected section comes first,
  1/2 if it comes second... averaged over the questions.
"""

from collections.abc import Callable
from dataclasses import dataclass

from cra_rag.questions import Question
from cra_rag.search import Hit

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Result:
    question: Question
    ranking: tuple[str, ...]  # section ids, best first

    def recall(self, k: int) -> float:
        found = set(self.ranking[:k]) & set(self.question.expected)
        return len(found) / len(self.question.expected)

    @property
    def reciprocal_rank(self) -> float:
        for position, section_id in enumerate(self.ranking, start=1):
            if section_id in self.question.expected:
                return 1 / position
        return 0.0


def run(questions: list[Question], search: Callable[[Question], list[Hit]]) -> list[Result]:
    return [Result(q, tuple(hit.section_id for hit in search(q))) for q in questions if q.expected]


def summary(results: list[Result]) -> dict[str, float]:
    n = max(len(results), 1)
    scores = {f"recall@{k}": sum(r.recall(k) for r in results) / n for k in KS}
    scores["mrr"] = sum(r.reciprocal_rank for r in results) / n
    return scores
