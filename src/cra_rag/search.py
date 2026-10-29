"""Find the sections that answer a question.

The first search method is BM25, the classic keyword ranking used by search
engines. It needs no model and runs in a fraction of a second, which makes it
the baseline that semantic search (embeddings, next step) has to beat.

Search works on chunks, then each section gets the score of its best chunk:
the result is a ranked list of sections, which is what citations need.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import snowballstemmer

from cra_rag.chunks import Chunk

WORD = re.compile(r"\w+")
STEMMERS = {"en": snowballstemmer.stemmer("english"), "fr": snowballstemmer.stemmer("french")}

# Only articles and annexes are searched: they hold the obligations, the ones
# answers must cite. Recitals explain them in long, wordy paragraphs that crowd
# the articles out of the top results: with them, BM25 recall@10 drops from
# 0.86 to 0.50 on the reference questions.
SEARCHED_KINDS = ("article", "annex")


FRENCH_WORDS = set(
    "le la les des du un une est sont et que qui quoi quel quelle quels quelles pour dans "
    "sur nous notre nos doit doivent peut peuvent ce cette avec pas faut-il".split()
)
ENGLISH_WORDS = set(
    "the a an is are and what who which for in we our us must can do does to of with not "
    "how when should".split()
)


def guess_lang(text: str) -> str:
    """Guess "fr" or "en" from the most common small words of each language."""
    words = text.lower().replace("’", "'").replace("'", " ").split()
    french = sum(w in FRENCH_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    return "en" if english > french else "fr"


def _strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))


def tokenize(text: str, lang: str) -> list[str]:
    """Words in lower case, without accents, reduced to their stem, so that
    "Fabricants" and "fabricant", or "sécurité" and "securite", match.
    Accents go first: a question typed without accents gets the same stems."""
    words = WORD.findall(_strip_accents(text.lower()))
    return STEMMERS[lang].stemWords([w for w in words if len(w) > 1 and not w.isdigit()])


@dataclass(frozen=True)
class Hit:
    section_id: str
    score: float
    chunk_id: str  # the chunk that gave the section its score


class BM25Index:
    """BM25 over the chunks of one language (k1 and b are the usual defaults)."""

    def __init__(self, chunks: list[Chunk], lang: str, k1: float = 1.5, b: float = 0.75):
        self.lang, self.k1, self.b = lang, k1, b
        self.chunks = [c for c in chunks if c.lang == lang and c.kind in SEARCHED_KINDS]
        self.lengths = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            terms = Counter(tokenize(chunk.text, lang))
            self.lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self.postings[term].append((index, count))
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)

    def _idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        return math.log(1 + (len(self.chunks) - n + 0.5) / (n + 0.5))

    def score_chunks(self, question: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(question, self.lang)):
            idf = self._idf(term)
            for index, count in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * count * (self.k1 + 1) / (count + norm)
        return scores

    def search(self, question: str, k: int = 10) -> list[Hit]:
        """The k best sections for the question, best first."""
        best: dict[str, Hit] = {}
        for index, score in self.score_chunks(question).items():
            chunk = self.chunks[index]
            if chunk.section_id not in best or score > best[chunk.section_id].score:
                best[chunk.section_id] = Hit(chunk.section_id, score, chunk.id)
        return sorted(best.values(), key=lambda hit: hit.score, reverse=True)[:k]
