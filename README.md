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

> **Status:** work in progress, step 2 of 7 done (corpus ingestion).
> This tool helps navigate the regulatory texts. It is not legal advice.

## Design principles

- **Every answer cites its sources** (article, annex). No source, no answer.
- **Quality is measured, not assumed.** A reference question set, written by a
  domain practitioner, is run automatically in CI.
- **Reproducible.** The texts are downloaded by script and committed with their
  SHA-256 checksums, so every result can be traced to an exact version.
- **Provider-agnostic.** The language model sits behind an interface and can be
  swapped (hosted API or local model).

## Corpus and licensing

| Source | In this repository |
|---|---|
| Cyber Resilience Act, Regulation (EU) 2024/2847 | English and French, in [`corpus/raw/`](corpus/). Reuse of EU legal texts is authorised under Commission Decision 2011/833/EU. |
| RED delegated act, Regulation (EU) 2022/30 | Same as above. |
| EN 18031-1/-2/-3 harmonised standards | **Not included.** The standards are copyrighted by CEN-CENELEC and are not redistributed here. |

Once split, the corpus holds 462 sections: for each language, the 71 articles,
8 annexes and 130 recitals of the CRA, and the 3 articles and 19 recitals of the
RED delegated act.

**Known limitation:** the RED delegated act is in its original version. Its date
of application was postponed to 1 August 2025 by Delegated Regulation (EU)
2023/2444, which is not in the corpus yet.

## Roadmap

- [x] 1. Project skeleton: packaging (uv), lint (ruff), tests (pytest), CI
- [x] 2. Corpus ingestion: download the texts, split them into articles, annexes and recitals
- [ ] 3. Reference question set: questions with the articles that answer them
- [ ] 4. Retrieval: embeddings and a vector store, measured with recall@k
- [ ] 5. Generation: cited answers from a language model behind a provider interface
- [ ] 6. FastAPI service and Docker image
- [ ] 7. End-to-end evaluation (LLM-as-a-judge) in CI

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv run cra-rag parse                    # split the texts into sections -> data/sections.jsonl
uv run cra-rag show cra:art_14          # print one section (add --lang fr for French)
uv run cra-rag questions                # check the reference questions in eval/questions.toml
uv run cra-rag download                 # refresh the texts from the EU Publications Office
```

Section ids follow the European Legislation Identifier markup of the Official
Journal: `cra:art_14` (Article 14), `cra:anx_I` (Annex I), `cra:rct_5` (Recital 5).

## Development

```bash
uv sync                      # create the virtual environment and install dependencies
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
