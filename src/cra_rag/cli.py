"""Command-line entry point: `uv run cra-rag <command>`."""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from cra_rag import (
    answer,
    answer_eval,
    chunks,
    dense,
    download,
    embed,
    evaluate,
    llm,
    parse,
    questions,
    retriever,
    search,
)

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


def _open_retriever(raw: Path, methods: list[str]) -> retriever.Retriever | None:
    """A retriever able to run `methods`; None (after a message) if the index is missing."""
    try:
        return retriever.Retriever(raw, use_dense=any(m != "bm25" for m in methods))
    except dense.IndexMissingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None


def _index(args: argparse.Namespace) -> int:
    sections = parse.parse_corpus(args.raw)
    embedder = embed.OllamaEmbedder(model=args.model)
    print(f"Building the vector index with {args.model} (interrupt any time: it resumes)")
    try:
        count = dense.build_index(chunks.chunk_sections(sections), embedder, log=print)
    except embed.EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Index ready: {count} chunks in {dense.INDEX_DIR}")
    return 0


def _search(args: argparse.Namespace) -> int:
    method = args.method
    if method == "auto":
        method = "hybrid" if (dense.INDEX_DIR / "qdrant").exists() else "bm25"
    searcher = _open_retriever(args.raw, [method])
    if searcher is None:
        return 1
    lang = args.lang if args.lang != "auto" else search.guess_lang(args.question)
    language = "French" if lang == "fr" else "English"
    print(f"Searching the {language} text with {method} (--lang and --method to change)")
    section_by_key = {(s.id, s.lang): s for s in searcher.sections}
    chunk_by_id = {c.id: c for c in searcher.chunks}
    try:
        hits = searcher.search(method, args.question, lang, args.k)
    except embed.EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        searcher.close()
    for rank, hit in enumerate(hits, start=1):
        section = section_by_key[(hit.section_id, lang)]
        name = f"{section.label}: {section.title}" if section.title else section.label
        print(f"{rank:>2}. {hit.section_id:<14} score {hit.score:6.3f}  {name}")
        excerpt = chunk_by_id[hit.chunk_id].text.split("\n", 1)[-1].replace("\n", " ")
        print(f"    {excerpt[:150]}...")
    return 0


def _ask(args: argparse.Namespace) -> int:
    method = args.method
    if method == "auto":
        method = "hybrid" if (dense.INDEX_DIR / "qdrant").exists() else "bm25"
    searcher = _open_retriever(args.raw, [method])
    if searcher is None:
        return 1
    lang = args.lang if args.lang != "auto" else search.guess_lang(args.question)
    model = llm.OllamaLLM(model=args.model)
    print(f"Searching with {method}, then writing the answer with {args.model}...\n")
    try:
        result = answer.answer(
            args.question,
            lang,
            searcher,
            model,
            method=method,
            passages=args.passages,
            on_token=lambda piece: print(piece, end="", flush=True),
        )
    except (embed.EmbeddingError, llm.LLMError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    finally:
        searcher.close()

    print("\n\nSources given to the model:")
    for source in result.sources:
        mark = "*" if source.id in result.citations else " "
        print(f" {mark} [{source.id}] {source.name}\n     {source.url}")
    print("   (* = cited in the answer)")
    for cited in result.unsupported_citations:
        print(f"Warning: the answer cites {cited}, which was not among the sources.")
    c = result.completion
    print(
        f"\nRead {c.prompt_tokens} tokens in {c.prompt_seconds:.0f} s, "
        f"wrote {c.output_tokens} tokens in {c.output_seconds:.0f} s."
    )
    return 0


def _eval_answers(args: argparse.Namespace) -> int:
    method = args.method
    if method == "auto":
        method = "hybrid" if (dense.INDEX_DIR / "qdrant").exists() else "bm25"
    question_set = questions.load(args.file)
    if args.only:
        wanted = set(args.only.split(","))
        question_set = [q for q in question_set if q.id in wanted]
    searcher = _open_retriever(args.raw, [method])
    if searcher is None:
        return 1
    model = llm.OllamaLLM(model=args.model)
    path = answer_eval.results_path(answer_eval.RESULTS_DIR, args.model)
    print(f"{len(question_set)} questions, {method} search, {args.model}.")
    print("Answers already saved with the same settings are reused: interrupt any time.")
    try:
        records = answer_eval.run(
            question_set, searcher, model, method, args.passages, path, log=print
        )
    except (embed.EmbeddingError, llm.LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        searcher.close()

    report = path.with_suffix(".md")
    if args.only:  # a partial run shows its answers but leaves the full report alone
        for record in records:
            q = next(q for q in question_set if q.id == record["id"])
            verdicts = answer_eval.checks(q, record)
            marks = ", ".join(f"{n} {'yes' if ok else 'NO'}" for n, ok in verdicts.items())
            print(f"\n{q.id}: {record['answer']}\n  -> {marks}")
    else:
        answer_eval.write_report(question_set, records, report)
    print("\nAutomatic checks:")
    totals = answer_eval.summary(question_set, records)
    for name, label in answer_eval.CHECK_LABELS.items():
        if name in totals:
            passed, count = totals[name]
            print(f"  {label:<50} {passed:>2}/{count:<2} ({passed / count:.0%})")
    if not args.only:
        print(f"\nEvery answer, with its checks: {report}")
    return 0


COLUMNS = [f"recall@{k}" for k in evaluate.KS] + ["mrr"]


def _score_table(groups: dict[str, list[evaluate.Result]]) -> None:
    print(f"{'':<8}{'n':>4}" + "".join(f"{c:>11}" for c in COLUMNS))
    for name, group in groups.items():
        scores = evaluate.summary(group)
        print(f"{name:<8}{len(group):>4}" + "".join(f"{scores[c]:>11.2f}" for c in COLUMNS))


def _eval_search(args: argparse.Namespace) -> int:
    methods = list(retriever.METHODS) if args.method == "all" else [args.method]
    searcher = _open_retriever(args.raw, methods)
    if searcher is None:
        return 1
    question_set = questions.load(args.file)
    try:
        results = {
            m: evaluate.run(question_set, lambda q, m=m: searcher.search(m, q.question, q.lang))
            for m in methods
        }
    except embed.EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        searcher.close()

    count = len(next(iter(results.values())))
    print(f"{count} questions (out_of_scope questions have nothing to find)")
    if len(methods) > 1:
        _score_table(results)
        print("\nFor the details of one method: uv run cra-rag eval-search --method dense")
        return 0

    [method_results] = results.values()
    groups = {"all": method_results}
    for lang in questions.LANGS:
        groups[lang] = [r for r in method_results if r.question.lang == lang]
    for kind in ("lookup", "multi", "trap"):
        groups[kind] = [r for r in method_results if r.question.type == kind]
    print(f"Method: {methods[0]}")
    _score_table(groups)

    misses = [r for r in method_results if r.recall(5) < 1]
    print(f"\nNot fully found in the top 5 ({len(misses)} questions):")
    for r in misses:
        print(f"  {r.question.id}  expected {', '.join(r.question.expected)}")
        print(f"        got      {', '.join(r.ranking[:5])}")
    return 0


def _embed_speed(args: argparse.Namespace) -> int:
    """Time the embedding model on real chunks, to estimate how long indexing takes."""
    sections = parse.parse_corpus(args.raw)
    searched = [c for c in chunks.chunk_sections(sections) if c.kind in search.SEARCHED_KINDS]
    step = max(len(searched) // args.n, 1)
    sample = [c.text for c in searched[::step][: args.n]]  # spread over the whole corpus
    embedder = embed.OllamaEmbedder(model=args.model)

    try:
        print(f"Loading {args.model} (the first call loads the model into memory)...")
        started = time.perf_counter()
        [vector] = embedder.embed(["warm-up"])
        load_time = time.perf_counter() - started

        print(f"Embedding {len(sample)} chunks of the corpus...")
        started = time.perf_counter()
        embedder.embed(sample)
        elapsed = time.perf_counter() - started
    except embed.EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    per_chunk = elapsed / len(sample)
    total = per_chunk * len(searched)
    print(f"  model loaded in {load_time:.1f} s, vectors of {len(vector)} dimensions")
    print(f"  {per_chunk:.2f} s per chunk ({len(sample)} chunks in {elapsed:.1f} s)")
    duration = f"{total:.0f} s" if total < 120 else f"{total / 60:.0f} min"
    print(f"  indexing all {len(searched)} chunks (fr + en) would take about {duration}")
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
    cmd.add_argument("--lang", choices=["auto", *questions.LANGS], default="auto")
    cmd.add_argument("--method", choices=["auto", *retriever.METHODS], default="auto")
    cmd.add_argument("-k", type=int, default=5, help="number of sections to show")
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_search)

    cmd = commands.add_parser("eval-search", help="score the search on the reference questions")
    cmd.add_argument("--method", choices=["all", *retriever.METHODS], default="bm25")
    cmd.add_argument("--file", type=Path, default=questions.QUESTIONS_FILE)
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_eval_search)

    cmd = commands.add_parser("ask", help='answer a question with citations: ask "..."')
    cmd.add_argument("question")
    cmd.add_argument("--lang", choices=["auto", *questions.LANGS], default="auto")
    cmd.add_argument("--method", choices=["auto", *retriever.METHODS], default="auto")
    cmd.add_argument("--passages", type=int, default=8, help="passages given to the model")
    cmd.add_argument("--model", default=llm.CHAT_MODEL)
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_ask)

    cmd = commands.add_parser(
        "eval-answers", help="answer the reference questions and check the answers"
    )
    cmd.add_argument("--only", help="comma-separated question ids, e.g. q001,q008")
    cmd.add_argument("--method", choices=["auto", *retriever.METHODS], default="auto")
    cmd.add_argument("--passages", type=int, default=8)
    cmd.add_argument("--model", default=llm.CHAT_MODEL)
    cmd.add_argument("--file", type=Path, default=questions.QUESTIONS_FILE)
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_eval_answers)

    cmd = commands.add_parser("index", help="build the vector index (needs Ollama, ~12 min)")
    cmd.add_argument("--model", default=embed.EMBEDDING_MODEL)
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_index)

    cmd = commands.add_parser("embed-speed", help="time the embedding model on this computer")
    cmd.add_argument("--model", default=embed.EMBEDDING_MODEL)
    cmd.add_argument("-n", type=int, default=16, help="number of chunks to embed")
    cmd.add_argument("--raw", type=Path, default=RAW_DIR)
    cmd.set_defaults(handler=_embed_speed)

    args = parser.parse_args(argv)
    return args.handler(args)
