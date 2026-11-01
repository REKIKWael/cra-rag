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

> **Status:** work in progress, step 4 of 7 done (retrieval).
> This tool helps navigate the regulatory texts. It is not legal advice.

## Design principles

- **Every answer cites its sources** (article, annex). No source, no answer.
- **Quality is measured, not assumed.** A set of 40 reference questions, reviewed
  by a practitioner of product cybersecurity regulation, is checked in CI and will
  score retrieval and answers. See [`eval/questions.toml`](eval/questions.toml).
- **Reproducible.** The texts are downloaded by script and committed with their
  SHA-256 checksums, so every result can be traced to an exact version.
- **Local and free.** The models run on the machine with [Ollama](https://ollama.com):
  no API key, no cost, and no document leaves the computer.
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

**Language versions differ in places.** For example, Article 69(3) of the CRA
covers products placed on the market "before 11 December 2027" in English, but
"le 11 décembre 2027" (on that date) in the published French text.

## Roadmap

- [x] 1. Project skeleton: packaging (uv), lint (ruff), tests (pytest), CI
- [x] 2. Corpus ingestion: download the texts, split them into articles, annexes and recitals
- [x] 3. Reference question set: 40 questions in French and English, with the sections that answer them
- [x] 4. Retrieval: BM25, semantic search (bge-m3 + Qdrant) and hybrid fusion, measured with recall@k
- [ ] 5. Generation: cited answers from a language model behind a provider interface (in progress)
- [ ] 6. FastAPI service and Docker image
- [ ] 7. End-to-end evaluation (LLM-as-a-judge) in CI

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv run cra-rag parse                    # split the texts into sections -> data/sections.jsonl
uv run cra-rag show cra:art_14          # print one section (add --lang fr for French)
uv run cra-rag questions                # check the reference questions in eval/questions.toml
uv run cra-rag search "Qui doit-on prévenir quand une faille est exploitée ?"
uv run cra-rag eval-search              # score the search on the reference questions
uv run cra-rag embed-speed              # time the embedding model (needs Ollama and bge-m3)
uv run cra-rag index                    # build the vector index (Ollama + bge-m3, once)
uv run cra-rag eval-search --method all # compare BM25, semantic and hybrid search
uv run cra-rag ask "Dans quel délai signaler une faille activement exploitée ?"
uv run cra-rag eval-answers             # answer the 40 questions, check them (resumable)
uv run cra-rag download                 # refresh the texts from the EU Publications Office
```

Section ids follow the European Legislation Identifier markup of the Official
Journal: `cra:art_14` (Article 14), `cra:anx_I` (Annex I), `cra:rct_5` (Recital 5).

## Retrieval results

Measured on the 36 reference questions that have an answer in the corpus
(the 4 `out_of_scope` ones have nothing to find). Recall@k is the share of the
expected sections found in the first k results; MRR rewards finding the first
one early (1 if it is ranked first, 1/2 if second...).

| Method | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| BM25 (keywords, stemmed) | 0.25 | 0.44 | 0.57 | 0.86 | 0.50 |
| Semantic (bge-m3 + Qdrant) | 0.37 | **0.75** | 0.81 | 0.87 | **0.65** |
| Hybrid (reciprocal rank fusion) | 0.34 | 0.66 | **0.82** | **0.91** | 0.59 |

Recall@5 by question type:

| Type | BM25 | Semantic | Hybrid |
|---|---|---|---|
| lookup (24) | 0.62 | 0.96 | 0.96 |
| multi (8) | 0.60 | 0.67 | 0.71 |
| trap (4) | 0.21 | 0.25 | 0.25 |

What this shows:

- **Semantic search closes the vocabulary gap.** The questions are worded the way
  manufacturers ask them ("protections désactivées"), not with the words of the
  regulation ("configuration de sécurité par défaut"). Recall@5 on single-section
  questions goes from 0.62 to 0.96.
- **Hybrid is the default.** The answer step will read the first 5 sections, and
  at 5 the semantic and hybrid methods are level; hybrid keeps the exact-term
  matches of BM25 (product names, "module H") and finds more at 10. The
  differences between the two are one or two questions out of 36: not enough to
  justify tuning fusion weights, which would only fit these 36 questions.
- **Weak spots:** questions whose answer spans several sections, and traps, where
  the decisive provision (a date in Article 71, a definition in Article 3) sits in
  a long article about something else.

Searching both languages at once (cross-lingual) changed nothing, so each
question is searched in its own language. A test in CI fails if BM25 falls
below recall@10 = 0.80; the semantic and hybrid scores need Ollama and will be
checked in CI with the end-to-end evaluation (step 7).

## Development

```bash
uv sync                      # create the virtual environment and install dependencies
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
