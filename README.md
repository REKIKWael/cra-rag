# cra-rag

[![CI](https://github.com/REKIKWael/cra-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/REKIKWael/cra-rag/actions/workflows/ci.yml)

A question-answering assistant for EU product cybersecurity regulation: the
**Cyber Resilience Act** (Regulation (EU) 2024/2847) and the **RED delegated act**
(Regulation (EU) 2022/30).

Ask *"What must a manufacturer report, and within what deadline, when an actively
exploited vulnerability is found?"* and get a short answer that **cites the
article it relies on**, so the answer can be checked against the source.

It is built as a small, production-style LLM system rather than a notebook:
reproducible ingestion, measured retrieval quality, cited answers, a containerised
API and evaluation running in CI.

> **Status:** work in progress, step 1 of 7 (project skeleton and CI).
> This tool helps navigate the regulatory texts. It is not legal advice.

## Design principles

- **Every answer cites its sources** (article, annex). No source, no answer.
- **Quality is measured, not assumed.** A reference question set, written by a
  domain practitioner, is run automatically in CI.
- **Reproducible.** The corpus is downloaded and processed by scripts, not
  committed by hand.
- **Provider-agnostic.** The language model sits behind an interface and can be
  swapped (hosted API or local model).

## Corpus and licensing

| Source | In this repository |
|---|---|
| Cyber Resilience Act, Regulation (EU) 2024/2847 | Downloaded from EUR-Lex by script (step 2). Reuse of EU legal texts is authorised under Commission Decision 2011/833/EU. |
| RED delegated act, Regulation (EU) 2022/30 | Same as above. |
| EN 18031-1/-2/-3 harmonised standards | **Not included.** The standards are copyrighted by CEN-CENELEC and are not redistributed here. |

## Roadmap

- [x] 1. Project skeleton: packaging (uv), lint (ruff), tests (pytest), CI
- [ ] 2. Corpus ingestion: download the texts from EUR-Lex, split them by article and annex
- [ ] 3. Reference question set: questions with the articles that answer them
- [ ] 4. Retrieval: embeddings and a vector store, measured with recall@k
- [ ] 5. Generation: cited answers from a language model behind a provider interface
- [ ] 6. FastAPI service and Docker image
- [ ] 7. End-to-end evaluation (LLM-as-a-judge) in CI

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create the virtual environment and install dependencies
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
