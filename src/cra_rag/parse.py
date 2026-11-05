"""Turn the raw Official Journal HTML into sections: articles, annexes and recitals.

The files are XHTML (valid XML). They are parsed with defusedxml, which refuses
the entity tricks a malicious XML file could use (XML bombs, external entities):
the texts come from the internet, so they are treated as untrusted input.
Every section carries a stable identifier from the European Legislation
Identifier (ELI) markup, e.g. ``art_14`` or ``anx_I``. We prefix it with the
source, ``cra:art_14``, and the same id is used in every language.
"""

import json
import re

# Parsing goes through defusedxml, which refuses entity tricks (XML bombs, external
# entities). The standard module is imported for the Element type only.
import xml.etree.ElementTree as ET  # nosemgrep
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import defusedxml.ElementTree as SafeET

SECTION_ID = re.compile(r"^(art|anx|rct)_([0-9]+|[IVXLC]+)$")
KINDS = {"art": "article", "anx": "annex", "rct": "recital"}
RECITAL_LABEL = {"en": "Recital", "fr": "Considérant"}
# Words kept in capitals when an all-caps heading is turned into sentence case.
ACRONYMS = {"EU", "UE", "CE"}
# A table row whose first cell is this short is a numbered point: "(a)", "(1)", "—"...
MAX_LABEL_LENGTH = 10


@dataclass(frozen=True)
class Section:
    id: str  # "cra:art_14", identical in every language
    source: str  # "cra"
    lang: str  # "en"
    kind: str  # "article", "annex" or "recital"
    number: str  # "14", "I", "5"
    label: str  # "Article 14", "Annexe I", "Considérant 5"
    title: str  # article or annex title, "" for recitals
    chapter: str  # "Chapter II: Obligations of economic operators...", "" if none
    text: str  # plain text, one line per paragraph or point
    url: str  # EUR-Lex link straight to the section


# --- Turning XHTML elements into text ------------------------------------------


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _classes(element: ET.Element) -> set[str]:
    return set(element.get("class", "").split())


def _is_footnote(element: ET.Element) -> bool:
    """Footnotes and the "(1)" links pointing to them, which we leave out of the text."""
    if _tag(element) == "a" and element.get("href", "").startswith("#ntr"):
        return True
    return "oj-note" in _classes(element)


def _is_inline(element: ET.Element) -> bool:
    """Some paragraphs are displayed on one line: "1." then the text, side by side."""
    return "display: inline" in element.get("style", "")


def _clean(text: str) -> str:
    """Collapse all whitespace (non-breaking spaces included) and remove the space
    a dropped footnote leaves before a comma or full stop: "Council (38), the"."""
    return re.sub(r" ([,.])(?=\s|$)", r"\1", " ".join(text.split()))


def _inline(element: ET.Element) -> str:
    parts = [element.text or ""]
    for child in element:
        if not _is_footnote(child):
            parts.append(_inline(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _lines(element: ET.Element) -> list[str]:
    """Render a block of XHTML as lines of text: one per paragraph or point."""
    lines: list[str] = []
    buffer = [element.text or ""]

    def flush() -> None:
        text = _clean("".join(buffer))
        buffer.clear()
        if text:
            lines.append(text)

    for child in element:
        tag = _tag(child)
        if _is_footnote(child) or tag in ("col", "hr"):
            pass
        elif tag == "table":
            flush()
            lines.extend(_table_lines(child))
        elif tag in ("p", "div") and not _is_inline(child):
            flush()
            lines.extend(_lines(child))
        else:
            buffer.append(_inline(child))
        buffer.append(child.tail or "")
    flush()
    return lines


def _table_lines(table: ET.Element) -> list[str]:
    """The Official Journal lays out numbered points as two-column tables:
    "(a)" in the first cell, the text in the second. We write "(a) text"
    and indent the sub-points found inside."""
    rows = []
    for child in table:
        if _tag(child) == "tr":
            rows.append(child)
        elif _tag(child) in ("thead", "tbody"):
            rows.extend(row for row in child if _tag(row) == "tr")
    data_table = "oj-table" in _classes(table)
    lines: list[str] = []
    for row in rows:
        cells = [_lines(cell) for cell in row if _tag(cell) in ("td", "th")]
        cells = [cell for cell in cells if cell]
        is_point = len(cells) == 2 and len(cells[0]) == 1 and len(cells[0][0]) <= MAX_LABEL_LENGTH
        if is_point and not data_table:
            label, body = cells[0][0], cells[1]
            lines.append(f"{label} {body[0]}")
            lines.extend("  " + line for line in body[1:])
        elif len(cells) >= 2:
            # A real data table: one line per row, cells separated by " | ".
            lines.append(" | ".join(" ".join(cell) for cell in cells))
        else:
            lines.extend(line for cell in cells for line in cell)
    return lines


def _render(element: ET.Element) -> list[str]:
    if _is_footnote(element):
        return []
    return _table_lines(element) if _tag(element) == "table" else _lines(element)


def _sentence_case(heading: str) -> str:
    """'OBLIGATIONS OF ECONOMIC OPERATORS' -> 'Obligations of economic operators'."""
    if heading != heading.upper():
        return heading
    words = [w if w.strip("(),.:’'") in ACRONYMS else w.lower() for w in heading.split()]
    sentence = " ".join(words)
    return sentence[:1].upper() + sentence[1:]


def _numbered_label(heading: str) -> str:
    """'ANNEX VIII' -> 'Annex VIII', 'CHAPTER II' -> 'Chapter II' (numeral kept)."""
    word, _, number = heading.partition(" ")
    return f"{word.capitalize()} {number}".strip()


# --- Building sections ----------------------------------------------------------


def _chapter_name(chapter: ET.Element) -> str:
    heading = title = ""
    for element in chapter.iter():
        classes = _classes(element)
        if not heading and "oj-ti-section-1" in classes:
            heading = _clean(_inline(element))
        elif not title and "oj-ti-section-2" in classes:
            title = _clean(_inline(element))
    if not heading:
        return ""
    return f"{_numbered_label(heading)}: {_sentence_case(title)}" if title else heading


def _article(element: ET.Element) -> tuple[str, str, list[str]]:
    label, title, body = "", "", []
    for child in element:
        classes = _classes(child)
        if "oj-ti-art" in classes:
            label = _clean(_inline(child))
        elif "eli-title" in classes:
            title = _clean(" ".join(_lines(child)))
        else:
            body.extend(_render(child))
    return label, title, body


def _annex(element: ET.Element) -> tuple[str, str, list[str]]:
    headings, body = [], []
    for child in element:
        if "oj-doc-ti" in _classes(child) and not body:
            headings.append(_clean(_inline(child)))
        else:
            body.extend(_render(child))
    label = _numbered_label(headings[0]) if headings else ""
    title = " ".join(_sentence_case(h) for h in headings[1:])
    return label, title, body


def _recital(element: ET.Element) -> list[str]:
    lines = _lines(element)
    if lines:
        lines[0] = re.sub(r"^\(\d+\)\s*", "", lines[0])  # the number is already in the label
    return lines


def _walk(element: ET.Element, chapter: str = "") -> Iterator[tuple[ET.Element, str]]:
    """Yield every article, annex and recital with the chapter it belongs to."""
    for child in element:
        section_id = child.get("id", "")
        if section_id.startswith("cpt_"):
            yield from _walk(child, _chapter_name(child))
        elif SECTION_ID.match(section_id):
            yield child, chapter
        else:
            yield from _walk(child, chapter)


def parse_file(path: Path, *, source: str, lang: str, url: str) -> list[Section]:
    root = SafeET.parse(path).getroot()
    sections = []
    for element, chapter in _walk(root):
        eli_id = element.get("id")
        prefix, number = SECTION_ID.match(eli_id).groups()
        kind = KINDS[prefix]
        if kind == "article":
            label, title, lines = _article(element)
        elif kind == "annex":
            label, title, lines = _annex(element)
        else:
            label, title, lines = f"{RECITAL_LABEL[lang]} {number}", "", _recital(element)
        sections.append(
            Section(
                id=f"{source}:{eli_id}",
                source=source,
                lang=lang,
                kind=kind,
                number=number,
                label=label,
                title=title,
                chapter=chapter if kind == "article" else "",
                text="\n".join(lines),
                url=f"{url}#{eli_id}",
            )
        )
    return sections


def parse_corpus(raw_dir: Path) -> list[Section]:
    """Parse every file listed in raw_dir/manifest.json."""
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = []
    for filename, entry in manifest.items():
        source = filename.rsplit("_", 1)[0]  # "red-da_en.html" -> "red-da"
        sections += parse_file(
            raw_dir / filename, source=source, lang=entry["lang"].lower(), url=entry["url"]
        )
    return sections


# --- Checks and output ------------------------------------------------------------


def check(sections: list[Section]) -> list[str]:
    """Return the problems found; an empty list means the corpus looks sound."""
    problems = []
    keys = Counter((s.id, s.lang) for s in sections)
    problems += [f"duplicate section {sid} ({lang})" for (sid, lang), n in keys.items() if n > 1]
    problems += [f"empty text in {s.id} ({s.lang})" for s in sections if not s.text]
    problems += [f"no label for {s.id} ({s.lang})" for s in sections if not s.label]

    # Every language must have exactly the same sections.
    ids_by_lang: dict[tuple[str, str], set[str]] = {}
    for s in sections:
        ids_by_lang.setdefault((s.source, s.lang), set()).add(s.id)
    for source in {s.source for s in sections}:
        versions = {lang: ids for (src, lang), ids in ids_by_lang.items() if src == source}
        reference = set.union(*versions.values())
        for lang, ids in sorted(versions.items()):
            for missing in sorted(reference - ids):
                problems.append(f"{missing} missing in {lang}")
    return problems


def write_jsonl(sections: list[Section], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for section in sections:
            file.write(json.dumps(asdict(section), ensure_ascii=False) + "\n")
