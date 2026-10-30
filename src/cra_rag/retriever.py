"""One entry point for the three search methods, so they can be compared fairly.

- "bm25": keywords (search.py), no model needed;
- "dense": meaning, with embeddings and Qdrant (dense.py), needs Ollama;
- "hybrid": both rankings merged by reciprocal rank fusion.
"""

from pathlib import Path

from cra_rag import chunks, dense, embed, parse, questions, search
from cra_rag.search import Hit

METHODS = ("bm25", "dense", "hybrid")
FUSION_DEPTH = 30  # how many results of each method go into the fusion


class Retriever:
    def __init__(
        self,
        raw_dir: Path,
        use_dense: bool,
        embedder: embed.OllamaEmbedder | None = None,
        index_dir: Path = dense.INDEX_DIR,
    ):
        self.sections = parse.parse_corpus(raw_dir)
        self.chunks = chunks.chunk_sections(self.sections)
        self.bm25 = {lang: search.BM25Index(self.chunks, lang) for lang in questions.LANGS}
        self.dense = None
        if use_dense:
            self.dense = dense.DenseIndex(embedder or embed.OllamaEmbedder(), index_dir)

    def close(self) -> None:
        if self.dense:
            self.dense.close()

    def search(self, method: str, question: str, lang: str, k: int = 10) -> list[Hit]:
        if method == "bm25":
            return self.bm25[lang].search(question, k)
        if self.dense is None:
            raise dense.IndexMissingError(f"The {method} method needs the vector index.")
        if method == "dense":
            return self.dense.search(question, lang, k)
        rankings = [
            self.bm25[lang].search(question, FUSION_DEPTH),
            self.dense.search(question, lang, FUSION_DEPTH),
        ]
        return dense.reciprocal_rank_fusion(rankings, k)

    def search_chunks(
        self, method: str, question: str, lang: str, n: int = 8, per_section: int = 3
    ) -> list[Hit]:
        """The n best passages, at most `per_section` from the same section.

        Used to build the answer: the passage holding the answer is not always the
        one that ranked its section first (Article 14 opens with the deadlines,
        but a question can match its paragraph 8 better).
        """
        if method == "bm25":
            ranked = self.bm25[lang].search_chunks(question)
        elif self.dense is None:
            raise dense.IndexMissingError(f"The {method} method needs the vector index.")
        elif method == "dense":
            ranked = self.dense.search_chunks(question, lang)
        else:
            ranked = dense.fuse_chunks(
                [self.bm25[lang].search_chunks(question), self.dense.search_chunks(question, lang)]
            )
        kept: list[Hit] = []
        per: dict[str, int] = {}
        for hit in ranked:
            if per.get(hit.section_id, 0) < per_section:
                kept.append(hit)
                per[hit.section_id] = per.get(hit.section_id, 0) + 1
            if len(kept) == n:
                break
        return kept
