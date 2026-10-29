"""Command-line entry point: `uv run cra-rag <command>`."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from cra_rag import download, parse

RAW_DIR = Path("corpus/raw")
SECTIONS_FILE = Path("data/sections.jsonl")


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

    args = parser.parse_args(argv)
    return args.handler(args)
