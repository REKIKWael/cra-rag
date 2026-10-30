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

SYSTEM_PROMPT = """You answer questions from manufacturers about EU product cybersecurity \
regulation: the Cyber Resilience Act (CRA) and the RED delegated act.

Rules:
1. Use only the sources given with the question. Never use outside knowledge: do not \
mention any law, deadline, amount or figure that is not written in the sources.
2. Cite the source of each statement with its id in square brackets, for example [cra:art_14].
3. If the sources do not contain the answer, reply only with this sentence, in the language \
of the question:
   French: "Les textes fournis ne permettent pas de répondre à cette question."
   English: "The provided texts do not answer this question."
4. Answer in the language of the question. Start with the direct answer. Give the key facts \
found in the sources: who, deadlines, amounts, conditions, exceptions.
5. Write plain sentences: no headings, no bold, no advice beyond the sources, 120 words at most."""

REFUSALS = {
    "fr": "Les textes fournis ne permettent pas de répondre à cette question.",
    "en": "The provided texts do not answer this question.",
}
CITATION = re.compile(r"\b(?:cra|red-da):(?:art|anx|rct)_(?:\d+|[IVXLC]+)\b")


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
        return cited_ids(self.text)

    @property
    def unsupported_citations(self) -> tuple[str, ...]:
        """Ids cited although they were not among the sources: they cannot be trusted."""
        given = {s.id for s in self.sources}
        return tuple(c for c in self.citations if c not in given)

    @property
    def is_refusal(self) -> bool:
        return is_refusal(self.text)


def cited_ids(text: str) -> tuple[str, ...]:
    """The section ids cited in an answer, in order, without repeats."""
    return tuple(dict.fromkeys(CITATION.findall(text)))


def is_refusal(text: str) -> bool:
    flat = " ".join(text.lower().split())
    return any(sentence.lower().rstrip(".") in flat for sentence in REFUSALS.values())


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


def build_prompt(question: str, sources: tuple[Source, ...]) -> str:
    blocks = [f"[{s.id}] {s.name}\n{s.text}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


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
    completion = llm.chat(SYSTEM_PROMPT, build_prompt(question, sources), on_token)
    return Answer(question, lang, completion.text.strip(), sources, completion)
