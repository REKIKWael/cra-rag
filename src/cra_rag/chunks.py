"""Cut sections into chunks small enough to be searched precisely.

Some sections are long (Article 13 is about 15 000 characters): searched as a
whole, a precise passage would drown in the rest. Each chunk:
- starts with a header naming its section ("Article 14: Reporting obligations
  of manufacturers"), so it still makes sense on its own;
- keeps whole lines (paragraphs and points) together when it can;
- repeats the lead-in sentence of a list ("the manufacturer shall submit:")
  when the list continues in a new chunk.
"""

import re
from dataclasses import dataclass

from cra_rag.parse import Section

MAX_CHARS = 1200
# A line that is a point of a list: "(a) ...", "a) ...", "(iv) ...", "  (i) ..."
POINT = re.compile(r"^\s*(\(?[a-z]{1,4}\)|\(?\d{1,2}\)|\s)")
SENTENCE_END = re.compile(r"(?<=[.;])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str  # "cra:art_14#2"
    section_id: str  # "cra:art_14"
    lang: str
    kind: str  # kind of the section: "article", "annex" or "recital"
    text: str  # header + body


def _header(section: Section) -> str:
    return f"{section.label}: {section.title}" if section.title else section.label


def _split_long_line(line: str, limit: int) -> list[str]:
    """Split a line longer than `limit` at sentence ends, as evenly as possible."""
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(line):
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_section(section: Section, max_chars: int = MAX_CHARS) -> list[Chunk]:
    header = _header(section)
    budget = max(max_chars - len(header) - 1, 200)

    lines: list[str] = []
    for line in section.text.splitlines():
        lines.extend(_split_long_line(line, budget) if len(line) > budget else [line])

    bodies: list[list[str]] = []
    current: list[str] = []
    size = 0
    lead_in = ""  # last line announcing a list, e.g. "... the manufacturer shall submit:"
    for line in lines:
        if current and size + len(line) + 1 > budget:
            bodies.append(current)
            current, size = [], 0
            if POINT.match(line) and lead_in:
                current, size = [lead_in], len(lead_in) + 1
        current.append(line)
        size += len(line) + 1
        if line.rstrip().endswith(":") and len(line) <= budget // 2:
            lead_in = line
    if current:
        bodies.append(current)

    return [
        Chunk(
            id=f"{section.id}#{n}",
            section_id=section.id,
            lang=section.lang,
            kind=section.kind,
            text=header + "\n" + "\n".join(body),
        )
        for n, body in enumerate(bodies, start=1)
    ]


def chunk_sections(sections: list[Section], max_chars: int = MAX_CHARS) -> list[Chunk]:
    return [chunk for section in sections for chunk in chunk_section(section, max_chars)]
