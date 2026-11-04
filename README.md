# cra-rag

[![CI](https://github.com/REKIKWael/cra-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/REKIKWael/cra-rag/actions/workflows/ci.yml)
[![Evaluate](https://github.com/REKIKWael/cra-rag/actions/workflows/evaluate.yml/badge.svg)](https://github.com/REKIKWael/cra-rag/actions/workflows/evaluate.yml)

A question-answering assistant for EU product cybersecurity regulation: the
**Cyber Resilience Act** (Regulation (EU) 2024/2847) and the **RED delegated act**
(Regulation (EU) 2022/30).

Ask *"What must a manufacturer report, and within what deadline, when an actively
exploited vulnerability is found?"* and get a short answer that **cites the
article it relies on**, so the answer can be checked against the source.

It is built as a small, production-style LLM system rather than a notebook:
reproducible ingestion, measured retrieval quality, cited answers, a containerised
API and evaluation running in CI.

> **Status:** the 8 steps are done. The judge's verdicts are being confirmed by the
> domain expert (see [Evaluation in CI](#evaluation-in-ci)).
> This tool helps navigate the regulatory texts. It is not legal advice.

## Design principles

- **Every answer cites its sources** (article, annex). No source, no answer.
- **Questions about the assistant itself** ("who are you?", "what can you do?") are
  recognised before any search and get a fixed presentation (intent routing,
  [`router.py`](src/cra_rag/router.py)). A test checks that none of the reference
  questions is caught by mistake.
- **Quality is measured, not assumed.** 40 reference questions in French and
  English, each with a reference answer and the sections that answer it, score the
  retrieval and the answers. See [`eval/questions.toml`](eval/questions.toml).
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
- [x] 5. Generation: cited answers from a local model behind a provider interface, checked on the 40 questions
- [x] 6. FastAPI service and Docker image, built and smoke-tested in CI
- [x] 7. End-to-end evaluation in CI, with a judge model checked against a human review
- [x] 8. An agent that searches with tools, measured against the RAG pipeline (see [Agent mode](#agent-mode))

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
uv run cra-rag ask --mode agent "..."   # the model searches the texts itself, with tools
uv run cra-rag eval-answers             # answer the 40 questions, check them (resumable)
uv run cra-rag judge                    # judge the answers with ministral-3:8b (resumable)
uv run cra-rag eval-answers --mode agent # the same for the agent (then judge --mode agent)
uv run cra-rag agreement                # compare the judge with eval/expert_review.toml
uv run cra-rag download                 # refresh the texts from the EU Publications Office
```

Section ids follow the European Legislation Identifier markup of the Official
Journal: `cra:art_14` (Article 14), `cra:anx_I` (Annex I), `cra:rct_5` (Recital 5).

## Run as a service

The same features as an HTTP API (FastAPI), documented at `/docs`:

```bash
uv run cra-rag serve                    # then open http://127.0.0.1:8000/docs
curl "http://127.0.0.1:8000/search?q=reporting+deadline+exploited+vulnerability"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What must the EU declaration of conformity contain?"}'
```

| Endpoint | Does |
|---|---|
| `GET /health` | Service status, search methods available, whether Ollama answers |
| `GET /search?q=...` | The sections that best answer a question, with links to EUR-Lex |
| `POST /ask` | A cited answer, its sources, and any citation not backed by a source |

With Docker, the API and Ollama run together ([`compose.yaml`](compose.yaml)):

```bash
docker compose up --detach
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull ministral-3:3b
docker compose run --rm api cra-rag index    # once: the vector index
docker compose restart api                   # then http://localhost:8000/docs
```

The CI builds the image on every push, starts it and checks `/health` and `/search`.

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
below recall@10 = 0.80; the semantic and hybrid scores need Ollama and are held
to the same floor by the [evaluation workflow](#evaluation-in-ci).

## Answer results

Answers of `ministral-3:3b` (3 billion parameters, run with Ollama on a 2016
laptop CPU: about 2 minutes per answer) to the 40 reference questions, with the
automatic checks of `uv run cra-rag eval-answers`. Every question, reference
answer and model answer is in
[`eval/results/answers-ministral-3_3b.md`](eval/results/answers-ministral-3_3b.md).

| Check | Prompt v1 | Prompt v2 |
|---|---|---|
| Answers, no false refusal (36 answerable questions) | 31 | **34** |
| Cites an expected section (36) | 24 | **29** |
| Cites only the sources it was given (40) | 38 | **39** |
| Repeats the key figures of the reference answer (8) | 2 | 3 |
| Says when the texts do not answer (4 out-of-scope) | 2 | **4** |

Prompt v2 asks to refuse only when no source deals with the question, allows
partial answers, forbids other laws by name (NIS2), and repeats the rules after
the question, in its language: small models follow best what they read last.

What reading the answers shows, beyond the checks:

- **Retrieval** missed the decisive section three times (q007, q022, q037), and
  the model then answered wrongly or refused.
- **The model** makes its own errors: incomplete answers (q001 gives the 24-hour
  early warning but not the 72-hour notification nor the 14-day final report),
  a wrong source (q023 attributes an Annex I rule to Article 13), misreadings
  (q013, q021).
- **The corpus** matters: q036 repeats the French wording of Article 69(3)
  ("le 11 décembre 2027" instead of "before"), so the divergence between
  language versions reaches the answers.

The checks above are indicators. Judging whether an answer is right needs a
judge model comparing it with the reference answer (step 7). The next lever on
quality is a larger model behind the same interface, to be measured the same way.

## Evaluation in CI

`uv run cra-rag judge` asks a larger model, `ministral-3:8b`, to compare every
answer with the reference answer of the question set, and to return a verdict
in JSON that follows a schema (Ollama structured outputs), so it can always be read:

| Verdict | Means |
|---|---|
| correct | The key facts of the reference answer, nothing contradicting it |
| partial | Nothing wrong, but key facts missing |
| incorrect | A contradiction, a false fact, a wrong conclusion, or a refusal although the texts answer |

Verdicts are saved in `eval/results/judgments-*.jsonl` with a fingerprint of the
answer, the judge model and its instructions, and reused while none of them changes.

The [Evaluate workflow](.github/workflows/evaluate.yml) runs the whole chain on
GitHub's free runners, with the real models, when started by hand from the
Actions tab: it installs Ollama, builds the vector index, scores the three search
methods (and fails below recall@10 = 0.80), recomputes the answers whose
settings changed, judges them, and publishes the scores on the run page with the
result files as an artifact. A choice of mode runs it for the RAG pipeline or the agent.

### What the judge says, and how far to trust it

Verdicts of `ministral-3:8b` on the 40 answers of `ministral-3:3b`, from the
Evaluate workflow (every verdict and its reason are in
[the report](eval/results/answers-ministral-3_3b.md)):

| Questions | Correct | Partial | Incorrect |
|---|---|---|---|
| all (40) | 17 | 8 | 15 |
| lookup (24) | 9 | 5 | 10 |
| multi (8) | 2 | 3 | 3 |
| trap (4) | 3 | 0 | 1 |
| out of scope (4) | 3 | 0 | 1 |

A judge is a model too. To know how far its verdicts can be trusted, every
answer was also reviewed by hand in [`eval/expert_review.toml`](eval/expert_review.toml),
and `uv run cra-rag agreement` compares the two. This first review was made by
Claude (an AI assistant), checking each answer against the text of the
regulation; the domain expert's confirmation is pending.

| Judge against the review | 40 answers |
|---|---|
| Same verdict | 28 (70 %) |
| Cohen's kappa (agreement corrected for chance) | 0.55 |

What the 12 disagreements show:

- **Totals hide the errors.** The review counts 14 correct, 14 partial and 12
  incorrect, close to the judge's totals, yet the two disagree on 12 answers,
  6 in each direction.
- **Too harsh when the judge trusts its own memory.** It rejected true
  statements, claiming that Article 13(20) does not exist (q020), that the CRA
  requires no coordinated vulnerability disclosure policy (q025), or that ENISA
  receives no notifications (q040), although its instructions forbid using its
  own knowledge of the law.
- **Too lenient with long or hedged answers.** It saw items that are not there
  (q015, q017), accepted a refusal although the texts answer (q037), and missed
  a wrong conclusion (q036).

The judge's totals are therefore an indicator, not yet a score to compare models
with. The next lever: give the judge the passages each answer cites, so that it
checks facts against the text rather than its memory, then measure the agreement
again with the same command.

## Agent mode

`uv run cra-rag ask --mode agent "..."` lets the model lead the search
([`agent.py`](src/cra_rag/agent.py)). Through tool calling (Ollama), it can run
`search(query)` again with other words or `read_section(id)` to read a whole
article, and answers when it has enough. The program runs the tools, the model
only asks for them: a citation counts only if a tool returned that section, and
after 5 turns the model gets no more tools and must answer. Since version 2, the
first search uses the user's whole question, as the RAG pipeline does.

Measured on the same 40 questions with the same judge (Evaluate workflow, mode
agent; every run is kept in [`eval/results/runs/`](eval/results/runs/)):

| | RAG pipeline | Agent v1, run 1 | Agent v1, run 2 | Agent v2 |
|---|---|---|---|---|
| Judge: correct | 17 | 8 | 14 | 12 |
| Judge: partial | 8 | 15 | 9 | 13 |
| Judge: incorrect | 15 | 17 | 17 | 15 |
| Cites an expected section (36) | 29 | 27 | 26 | **30** |
| Answers, no false refusal (36) | 34 | 36 | 36 | 36 |

The RAG answers come from one run on a laptop, the agent's from GitHub runners.

What this shows:

- **The same code does not give the same answers twice.** Agent v1 ran twice,
  with temperature 0 and a fixed seed: 38 of its 40 answers differed, and the
  judge counted 8 correct answers, then 14. From one machine to another the
  floating-point computations differ slightly, one different word changes the
  rest of the answer. A gap of a few answers between two single runs is noise.
- **Reading the answers of v1 showed its faults.** It searched with its own
  shorter queries, added "CRA RED" to them, which pulled the search off target,
  never read a whole section, and after refusing an out-of-scope question still
  added what the CRA says. Version 2 starts from a search with the whole question
  and states a stricter refusal rule.
- **With a 3-billion-parameter model, the autonomy is barely used.** Agent v2
  answered right after its first search for 39 questions out of 40, and read a
  whole section twice. It cites an expected section most often (30 of 36), but
  its judged results are within the noise of v1's, and it still adds content to
  out-of-scope refusals (q038, q039, q040).

The RAG pipeline therefore stays the default. Next: several runs per
configuration, compared on their mean and spread, a scope check that refuses
other laws, countries and standards before any answer, and a larger model for
the agent.

## Development

```bash
uv sync                      # create the virtual environment and install dependencies
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
