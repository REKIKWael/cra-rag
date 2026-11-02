"""An agent that searches the regulation itself before answering (agentic RAG).

The RAG pipeline (answer.py) always runs the same steps: one search with the
question, then the answer. Here the model decides. It may search several times,
rephrase its query in the words of the regulation, read a whole article, and it
answers when it has what it needs. It has two tools:

- search(query): the hybrid search; returns the best passages with their section ids;
- read_section(id): the full text of one section (an article, an annex).

The model only asks for a tool; this program runs it and sends the result back.
The answer is checked as in the RAG pipeline: a citation counts only if a tool
returned that section, so the agent cannot cite what it has not read. A limit on
the number of turns stops a model that would search forever.
"""

from collections.abc import Callable
from dataclasses import dataclass

from cra_rag import answer, router
from cra_rag.answer import Answer, Source
from cra_rag.llm import Completion, ToolCall, ToolLLM
from cra_rag.retriever import Retriever

AGENT_VERSION = 1  # change it whenever the prompt, the tools or the limits change
MAX_TURNS = 5  # model replies; the last one gets no tools, so it must answer
SEARCH_PASSAGES = 6  # passages returned by one search
SECTION_CHARS = 6000  # a whole article can be long: cut to keep the context small
CONTEXT_TOKENS = 16384  # system prompt + several tool results + the answer

SYSTEM_PROMPT = f"""You answer questions from manufacturers about EU product cybersecurity \
regulation: the Cyber Resilience Act (CRA) and the RED delegated act.

You have two tools:
- search: finds the passages of these texts that best match a query;
- read_section: gives the full text of one section from its id, for example cra:art_14.

Method:
1. Always search before answering. Use the words of the regulation in your queries.
2. If the passages do not answer, search again with other words, or read the full section.
3. Use only what the tools returned. Never use outside knowledge: no law, deadline, amount \
or figure that the tools did not return.
4. Cite the source of each statement with its id in square brackets, for example [cra:art_14].
5. If the texts do not deal with the question, reply only with this sentence, in the \
language of the question:
   French: "{answer.REFUSALS["fr"]}"
   English: "{answer.REFUSALS["en"]}"
   If they answer part of it, answer that part and say what is missing.
6. Answer in the language of the question, in plain sentences, 120 words at most."""

REMINDERS = {
    "fr": "Cherche dans les textes avec l'outil search avant de répondre. Réponds en français, "
    "uniquement à partir des résultats des outils, en citant chaque affirmation avec "
    "l'identifiant de sa source entre crochets, par exemple [cra:art_14].",
    "en": "Search the texts with the search tool before answering. Answer in English, only "
    "from what the tools returned, citing each statement with the id of its source in "
    "square brackets, for example [cra:art_14].",
}
SEARCH_FIRST = "Search the texts with the search tool before answering."
ANSWER_NOW = (
    "No more tools. Answer now from what the tools returned, citing the ids, "
    "or reply with the refusal sentence."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the CRA and the RED delegated act. Returns the best "
            "passages, each labelled with the id of its section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, ideally in the words of the regulation",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "The full text of one section of the regulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "A section id returned by search, e.g. cra:art_14",
                    }
                },
                "required": ["id"],
            },
        },
    },
]


@dataclass(frozen=True)
class Step:
    """One tool the agent ran: what it asked, and the sections it got back."""

    tool: str
    arguments: dict
    found: tuple[str, ...]


@dataclass(frozen=True)
class AgentAnswer(Answer):
    """An Answer (same checks of citations and refusals) plus the steps taken."""

    steps: tuple[Step, ...] = ()


class Tools:
    """Runs the tools for one question, and remembers every section returned."""

    def __init__(self, retriever: Retriever, lang: str, method: str):
        self.retriever, self.lang, self.method = retriever, lang, method
        self.sections = {(s.id, s.lang): s for s in retriever.sections}
        self.seen: dict[str, Source] = {}  # section id -> what the model was shown

    def run(self, call: ToolCall) -> tuple[str, tuple[str, ...]]:
        """The text sent back to the model, and the ids of the sections in it."""
        if call.name == "search":
            query = str(call.arguments.get("query", "")).strip()
            if not query:
                return "Error: give a query, for example search(query='reporting deadline').", ()
            found = answer.find_sources(
                self.retriever, query, self.lang, self.method, SEARCH_PASSAGES
            )
            if not found:
                return "No passage found. Try other words.", ()
            return self._show(found), tuple(s.id for s in found)
        if call.name == "read_section":
            ids = answer.explicit_ids(str(call.arguments.get("id", "")))
            section = self.sections.get((ids[0], self.lang)) if ids else None
            if section is None:
                return "Error: no such section. Use an id returned by search, e.g. cra:art_14.", ()
            text = section.text
            if len(text) > SECTION_CHARS:
                text = text[:SECTION_CHARS] + "\n[...] (cut: search for the part you need)"
            name = f"{section.label}: {section.title}" if section.title else section.label
            return self._show([Source(section.id, name, section.url, text)]), (section.id,)
        return f"Error: unknown tool {call.name!r}. The tools are search and read_section.", ()

    def _show(self, sources: list[Source] | tuple[Source, ...]) -> str:
        for source in sources:
            known = self.seen.get(source.id)
            if known is None or len(source.text) > len(known.text):
                self.seen[source.id] = source  # keep the fullest text shown
        return "\n\n".join(f"[{s.id}] {s.name}\n{s.text}" for s in sources)


def _total(completions: list[Completion], text: str) -> Completion:
    return Completion(
        text=text,
        prompt_tokens=sum(c.prompt_tokens for c in completions),
        prompt_seconds=sum(c.prompt_seconds for c in completions),
        output_tokens=sum(c.output_tokens for c in completions),
        output_seconds=sum(c.output_seconds for c in completions),
    )


def run(
    question: str,
    lang: str,
    retriever: Retriever,
    llm: ToolLLM,
    method: str = "hybrid",
    on_token: Callable[[str], None] | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> AgentAnswer:
    if router.is_about_assistant(question):  # same presentation as the RAG pipeline
        text = router.ABOUT[lang]
        if on_token:
            on_token(text)
        return AgentAnswer(question, lang, text, (), Completion(text=text), route="about")
    tools = Tools(retriever, lang, method)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n{REMINDERS[lang]}"},
    ]
    completions: list[Completion] = []
    steps: list[Step] = []
    text = ""
    for turn in range(1, MAX_TURNS + 1):
        last = turn == MAX_TURNS
        if last:
            messages.append({"role": "user", "content": ANSWER_NOW})
        completion = llm.converse(messages, tools=None if last else TOOLS, on_token=on_token)
        completions.append(completion)
        if last or not completion.tool_calls:
            if not steps and not last:  # answered from memory: send it to search first
                messages += [
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": SEARCH_FIRST},
                ]
                continue
            text = completion.text
            break
        messages.append(
            {
                "role": "assistant",
                "content": completion.text,
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in completion.tool_calls
                ],
            }
        )
        for call in completion.tool_calls:
            result, found = tools.run(call)
            step = Step(call.name, dict(call.arguments), found)
            steps.append(step)
            if on_step:
                on_step(step)
            messages.append({"role": "tool", "tool_name": call.name, "content": result})

    return AgentAnswer(
        question=question,
        lang=lang,
        text=answer.clean(text),
        sources=tuple(tools.seen.values()),
        completion=_total(completions, text),
        steps=tuple(steps),
    )
