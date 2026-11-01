"""Answer a question from the regulation, citing the sections used.

1. Search: the 8 best passages (hybrid search by default), at most 3 per
   section, grouped by section.
2. Prompt: the passages, each group labelled with its section id, then the question.
3. Generation: the model must answer from those passages only, cite them as
   [cra:art_14], and say so when they do not contain the answer.
4. Check: every citation is compared with the passages actually given; a cited
   id that was not among them is flagged, since it cannot be backed by a source.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from cra_rag.llm import LLM, Completion
from cra_rag.retriever import Retriever

PROMPT_VERSION = 2  # change it whenever the prompt changes: answers are then recomputed

SYSTEM_PROMPT = """You answer questions from manufacturers about EU product cybersecurity \
regulation: the Cyber Resilience Act (CRA) and the RED delegated act.

Rules:
1. Use only the sources given with the question. Never use outside knowledge: do not \
mention any law, deadline, amount or figure that is not written in the sources, even if you \
know it from elsewhere (for example from NIS2 or a standard).
2. Cite the source of each statement with its id in square brackets, for example [cra:art_14].
3. Refuse only if none of the sources deals with the question. Then reply only with this \
sentence, in the language of the question:
   French: "Les textes fournis ne permettent pas de répondre à cette question."
   English: "The provided texts do not answer this question."
   If the sources answer part of the question, answer that part and say what is missing.
4. Answer in the language of the question. Start with the direct answer. Give the key facts \
found in the sources: who, deadlines, amounts, conditions, exceptions.
5. Write plain sentences: no headings, no bold, no advice beyond the sources, 120 words at most."""

# Repeated after the question: small models follow best the last thing they read.
REMINDERS = {
    "fr": "Réponds en français, uniquement à partir des sources ci-dessus, en citant chaque "
    "affirmation avec l'identifiant de sa source entre crochets, par exemple [cra:art_14].",
    "en": "Answer in English, only from the sources above, citing each statement with the id "
    "of its source in square brackets, for example [cra:art_14].",
}

REFUSALS = {
    "fr": "Les textes fournis ne permettent pas de répondre à cette question.",
    "en": "The provided texts do not answer this question.",
}
# Ways of saying that the texts do not cover something, fully or in part.
DECLINING = re.compile(
    r"ne (?:permettent|précisent|mentionnent|contiennent|traitent|disent|indiquent) pas"
    r"|do(?:es)? not (?:answer|specify|mention|contain|cover|say|state|indicate)"
    r"|not covered",
    re.IGNORECASE,
)
CITATION = re.compile(r"\b(?:cra|red-da):(?:art|anx|rct)_(?:\d+|[IVXLC]+)\b", re.IGNORECASE)
# Plain references a model also writes: "article 54", "Article 30(1)", "annexe V", "Annex I".
ARTICLE_REFERENCE = re.compile(r"\bart(?:icle|\.)?\s*(\d+)", re.IGNORECASE)
ANNEX_REFERENCE = re.compile(r"\bannexe?\s+([IVXLC]+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Source:
    id: str  # "cra:art_14"
    name: str  # "Article 14: Reporting obligations of manufacturers"
    url: str
    text: str  # the passage given to the model


@dataclass(frozen=True)
class Answer:
    question: str
    lang: str
    text: str
    sources: tuple[Source, ...]
    completion: Completion

    @property
    def citations(self) -> tuple[str, ...]:
        return cited_ids(self.text, [s.id for s in self.sources])

    @property
    def unsupported_citations(self) -> tuple[str, ...]:
        """Ids cited although they were not among the sources: they cannot be trusted."""
        given = {s.id for s in self.sources}
        return tuple(c for c in explicit_ids(self.text) if c not in given)

    @property
    def is_refusal(self) -> bool:
        return is_refusal(self.text)


def _normalise(section_id: str) -> str:
    """Write an id the way the corpus does: "RED-da:ART_1" -> "red-da:art_1"."""
    prefix, _, number = section_id.lower().partition("_")
    return f"{prefix}_{number.upper() if number.isalpha() else number}"


def explicit_ids(text: str) -> tuple[str, ...]:
    """Ids written as such, "[cra:art_14]" or "cra:anx_I", in order, without repeats."""
    return tuple(dict.fromkeys(_normalise(m) for m in CITATION.findall(text)))


def cited_ids(text: str, given: list[str] | tuple[str, ...] = ()) -> tuple[str, ...]:
    """The sections cited in an answer, in order, without repeats.

    Besides explicit ids, plain references ("article 54", "annexe V") count when
    they designate one of the `given` sources: for a reader they are citations too.
    """
    found = list(explicit_ids(text))
    for match in ARTICLE_REFERENCE.finditer(text):
        found += [g for g in given if g.endswith(f":art_{match.group(1)}")]
    for match in ANNEX_REFERENCE.finditer(text):
        found += [g for g in given if g.endswith(f":anx_{match.group(1).upper()}")]
    return tuple(dict.fromkeys(found))


def clean(text: str) -> str:
    """Remove the Markdown emphasis and headings small models add despite the rules."""
    text = re.sub(r"\*\*|__", "", text)
    return re.sub(r"(?m)^#+\s*", "", text).strip()


def is_refusal(text: str) -> bool:
    """A refusal and nothing else. An answer that says what the texts do not cover,
    then gives what they do, is a partial answer, not a refusal."""
    flat = " ".join(text.lower().split())
    for sentence in REFUSALS.values():
        sentence = sentence.lower().rstrip(".")
        if sentence in flat and len(flat) <= len(sentence) * 1.5:
            return True
    return False


def declines(text: str) -> bool:
    """True if the answer says that the texts do not cover the question, or part of it."""
    return bool(DECLINING.search(text))


def find_sources(
    retriever: Retriever, question: str, lang: str, method: str, passages: int
) -> tuple[Source, ...]:
    """The best passages, grouped by section, the best section first."""
    section_by_key = {(s.id, s.lang): s for s in retriever.sections}
    chunk_by_id = {c.id: c for c in retriever.chunks}
    grouped: dict[str, list[str]] = {}
    for hit in retriever.search_chunks(method, question, lang, n=passages):
        grouped.setdefault(hit.section_id, []).append(hit.chunk_id)

    sources = []
    for section_id, chunk_ids in grouped.items():
        section = section_by_key[(section_id, lang)]
        name = f"{section.label}: {section.title}" if section.title else section.label
        in_order = sorted(chunk_ids, key=lambda cid: int(cid.rsplit("#", 1)[1]))
        text = "\n[...]\n".join(chunk_by_id[c].text.split("\n", 1)[-1] for c in in_order)
        sources.append(Source(section.id, name, section.url, text))
    return tuple(sources)


def build_prompt(question: str, sources: tuple[Source, ...], lang: str = "en") -> str:
    blocks = [f"[{s.id}] {s.name}\n{s.text}" for s in sources]
    sources_text = "\n\n".join(blocks)
    return f"Sources:\n\n{sources_text}\n\nQuestion: {question}\n\n{REMINDERS[lang]}"


def answer(
    question: str,
    lang: str,
    retriever: Retriever,
    llm: LLM,
    method: str = "hybrid",
    passages: int = 8,
    on_token: Callable[[str], None] | None = None,
) -> Answer:
    sources = find_sources(retriever, question, lang, method, passages)
    completion = llm.chat(SYSTEM_PROMPT, build_prompt(question, sources, lang), on_token)
    return Answer(question, lang, clean(completion.text), sources, completion)
