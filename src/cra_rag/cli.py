"""Command-line entry point: `uv run cra-rag <command>`."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from cra_rag import chunks, download, evaluate, parse, questions, search

RAW_DIR = Path("corpus/raw")
SECTIONS_FILE = Path("data/sections.jsonl")
QUESTIONS_TARGET = 40


def _download(args: argparse.Namespace) -> int:
    try:
        manifest = download.download_all(args.raw, log=print)
    except download.DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for filename, entry in manifest.items():
        print(f"  {filename}: {entry['bytes']:,} bytes, sha256 {entry['sha256'][:12]}...")
    print(f"Done: {len(manifest)} files in {args.raw}")
    return 0


def _parse(args: argparse.Namespace) -> int:
    sections = parse.parse_corpus(args.raw)
    problems = parse.check(sections)
    for problem in problems:
        print(f"Problem: {problem}", file=sys.stderr)

    counts = Counter((s.source, s.lang, s.kind) for s in sections)
    print(f"{'source':<8} {'lang':<5} {'articles':>8} {'annexes':>8} {'recitals':>9}")
    for source, lang in sorted({(s.source, s.lang) for s in sections}):
        row = [counts[(source, lang, kind)] for kind in ("article", "annex", "recital")]
        print(f"{source:<8} {lang:<5} {row[0]:>8} {row[1]:>8} {row[2]:>9}")

    parse.write_jsonl(sections, args.out)
    print(f"Wrote {len(sections)} sections to {args.out}")
    return 1 if problems else 0


def _show(args: argparse.Namespace) -> int:
    sections = parse.parse_corpus(args.raw)
    for section in sections:
        if section.id == args.id and section.lang == args.lang:
            print(f"{section.label}: {section.title}" if section.title else section.label)
            if section.chapter:
                print(section.chapter)
            print(section.url)
            print()
            print(section.text)
            return 0
    known = sorted({s.id for s in sections})
    print(f"Unknown section {args.id!r}. Examples: {', '.join(known[:4])}...", file=sys.stderr)
    return 1


def _questions(args: argparse.Namespace) -> int:
    sections = parse.parse_corpus(args.raw)
    try:
        question_set = questions.load(args.file)
    except questions.QuestionsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    problems = questions.check(question_set, {s.id for s in sections})
    for problem in problems:
        print(f"Problem: {problem}", file=sys.stderr)

    langs = Counter(q.lang for q in question_set)
    types = Counter(q.type for q in question_set)
    print(f"{len(question_set)} questions (target: {QUESTIONS_TARGET})")
    print("  by language: " + ", ".join(f"{lang} {langs[lang]}" for lang in questions.LANGS))
    print("  by type:     " + ", ".join(f"{kind} {types[kind]}" for kind in questions.TYPES))
    print("Questions per part of the corpus:")
    for part, count in questions.coverage(question_set, sections).items():
        print(f"  {count:>3}  {part}{'   <- not covered yet' if count == 0 else ''}")
    if problems:
        print(f"{len(problems)} problem(s) to fix, listed above.", file=sys.stderr)
    return 1 if problems else 0


def _indexes(raw: Path) -> tuple[list[parse.Section], list[chunks.Chunk], dict]:
    sections = parse.parse_corpus(raw)
    chunk_list = chunks.chunk_sections(sections)
    indexes = {lang: search.BM25Index(chunk_list, lang) for lang in questions.LANGS}
    return sections, chunk_list, indexes


def _search(args: argparse.Namespace) -> int:
    sections, chunk_list, indexes = _indexes(args.raw)
    section_by_key = {(s.id, s.lang): s for s in sections}
    chunk_by_id = {c.id: c for c in chunk_list}
    for rank, hit in enumerate(indexes[args.lang].search(args.question, args.k), start=1):
        section = section_by_key[(hit.section_id, args.lang)]
        name = f"{section.label}: {section.title}" if section.title else section.label
        print(f"{rank:>2}. {hit.section_id:<14} score {hit.score:5.1f}  {name}")
        excerpt = chunk_by_id[hit.chunk_id].text.split("\n", 1)[-1].replace("\n", " ")
        print(f"    {excerpt[:150]}...")
    return 0


def _eval_search(args: argparse.Namespace) -> int:
    _, _, indexes = _indexes(args.raw)
    question_set = questions.load(args.file)
    results = evaluate.run(question_set, lambda q: indexes[q.lang].search(q.question, 10))

    columns = [f"recall@{k}" for k in evaluate.KS] + ["mrr"]
    groups = {"all": results}
    for lang in questions.LANGS:
        groups[lang] = [r for r in results if r.question.lang == lang]
    for kind in ("lookup", "multi", "trap"):
        groups[kind] = [r for r in results if r.question.type == kind]

    print(f"BM25 on {len(results)} questions (out_of_scope questions have nothing to find)")
    print(f"{'':<8}{'n':>4}" + "".join(f"{c:>11}" for c in columns))
    for name, group in groups.items():
        scores = evaluate.summary(group)
        print(f"{name:<8}{len(group):>4}" + "".join(f"{scores[c]:>11.2f}" for c in columns))

    misses = [r for r in results if r.recall(5) < 1]
    print(f"\nNot fully found in the top 5 ({len(misses)} questions):")
    for r in misses:
        print(f"  {r.question.id}  expected {', '.join(r.question.expected)}")
        print(f"        got      {', '.join(r.ranking[:5])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cra-rag",
        description="Cited question-answering over EU product cybersecurity regulation.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cmd = commands.add_parser(
        "download", help="download the regulatory texts (EU Publications Office) into corpus/raw/"
    )
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_download)

    cmd = commands.add_parser("parse", help="split the texts into articles, annexes and recitals")
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.add_argument("--out", type=Path, default=SECTIONS_FILE)
    cmd.set_defaults(handler=_parse)

    cmd = commands.add_parser("show", help="print one section, e.g. `show cra:art_14 --lang fr`")
    cmd.add_argument("id", help="section id, e.g. cra:art_14, cra:anx_I, red-da:rct_3")
    cmd.add_argument("--lang", choices=["en", "fr"], default="en")
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_show)

    cmd = commands.add_parser("questions", help="check the reference questions and their coverage")
    cmd.add_argument("--file", type=Path, default=questions.QUESTIONS_FILE)
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_questions)

    cmd = commands.add_parser("search", help='find the sections answering a question: search "..."')
    cmd.add_argument("question")
    cmd.add_argument("--lang", choices=list(questions.LANGS), default="fr")
    cmd.add_argument("-k", type=int, default=5, help="number of sections to show")
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_search)

    cmd = commands.add_parser("eval-search", help="score the search on the reference questions")
    cmd.add_argument("--file", type=Path, default=questions.QUESTIONS_FILE)
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_eval_search)

    args = parser.parse_args(argv)
    return args.handler(args)
