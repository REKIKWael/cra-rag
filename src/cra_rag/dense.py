"""Semantic search: every chunk becomes a vector, stored in a Qdrant vector database.

To answer a question, its vector is compared with those of the chunks: the
closest chunks (cosine similarity) give the closest sections.

Computing the vectors is the slow part (about 1.4 s per chunk on a laptop CPU),
so they are kept in a cache on disk, keyed by the exact text and model. The
cache is written as it goes: an interrupted indexing resumes where it stopped,
and after a change only the new or modified chunks are computed again.
"""

import base64
import hashlib
import json
import re
import time
from array import array
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient, models

from cra_rag.chunks import Chunk
from cra_rag.embed import OllamaEmbedder
from cra_rag.search import SEARCHED_KINDS, Hit, best_per_section

INDEX_DIR = Path("data/index")


def _safe(model: str) -> str:
    """A model name usable in a file name, even on Windows: "bge-m3:latest" -> "bge-m3_latest"."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def cache_path(index_dir: Path, model: str) -> Path:
    return index_dir / f"{_safe(model)}.cache.jsonl"


def collection_name(model: str) -> str:
    """One collection per model: vectors of different models cannot be compared."""
    return f"chunks-{_safe(model)}"


class IndexMissingError(RuntimeError):
    """The vector index has not been built yet."""


class EmbeddingCache:
    """Vectors already computed, one JSON line each, in data/index/<model>.cache.jsonl."""

    def __init__(self, path: Path, model: str):
        self.path, self.model = path, model
        self.vectors: dict[str, list[float]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.vectors[entry["key"]] = _decode(entry["vector"])

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self.vectors.get(self.key(text))

    def add(self, texts: list[str], vectors: list[list[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            for text, vector in zip(texts, vectors, strict=True):
                key = self.key(text)
                self.vectors[key] = vector
                file.write(json.dumps({"key": key, "vector": _encode(vector)}) + "\n")


def _encode(vector: list[float]) -> str:
    """Store a vector compactly: 32-bit floats, in base64."""
    return base64.b64encode(array("f", vector).tobytes()).decode("ascii")


def _decode(data: str) -> list[float]:
    return array("f", base64.b64decode(data)).tolist()


def embed_cached(
    texts: list[str],
    embedder: OllamaEmbedder,
    cache: EmbeddingCache,
    log: Callable[[str], None] = lambda message: None,
) -> list[list[float]]:
    """Vectors for `texts`, computing only those missing from the cache."""
    todo = list(dict.fromkeys(t for t in texts if cache.get(t) is None))
    if todo:
        started = time.perf_counter()
        for done in range(0, len(todo), embedder.batch_size):
            batch = todo[done : done + embedder.batch_size]
            cache.add(batch, embedder.embed(batch))
            count = done + len(batch)
            elapsed = time.perf_counter() - started
            left = elapsed / count * (len(todo) - count)
            log(f"  {count}/{len(todo)} computed, {elapsed / 60:.1f} min elapsed, "
                f"about {left / 60:.1f} min left")  # fmt: skip
    return [cache.get(t) for t in texts]


def build_index(
    chunks: list[Chunk],
    embedder: OllamaEmbedder,
    index_dir: Path = INDEX_DIR,
    log: Callable[[str], None] = lambda message: None,
) -> int:
    """Embed the searched chunks (cached) and load them into Qdrant. Returns the count."""
    searched = [c for c in chunks if c.kind in SEARCHED_KINDS]
    cache = EmbeddingCache(cache_path(index_dir, embedder.model), embedder.model)
    missing = sum(1 for c in searched if cache.get(c.text) is None)
    log(f"{len(searched)} chunks, {len(searched) - missing} already in the cache")
    vectors = embed_cached([c.text for c in searched], embedder, cache, log)

    collection = collection_name(embedder.model)
    client = QdrantClient(path=str(index_dir / "qdrant"))
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=len(vectors[0]), distance=models.Distance.COSINE
            ),
        )
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=number,
                    vector=vector,
                    payload={"chunk_id": c.id, "section_id": c.section_id, "lang": c.lang},
                )
                for number, (c, vector) in enumerate(zip(searched, vectors, strict=True))
            ],
        )
    finally:
        client.close()
    return len(searched)


class DenseIndex:
    """Search the Qdrant index built by `build_index`."""

    def __init__(self, embedder: OllamaEmbedder, index_dir: Path = INDEX_DIR):
        advice = f"Build it with: uv run cra-rag index --model {embedder.model}"
        if not (index_dir / "qdrant").exists():
            raise IndexMissingError(f"No vector index yet. {advice}")
        self.embedder = embedder
        self.collection = collection_name(embedder.model)
        self.cache = EmbeddingCache(cache_path(index_dir, embedder.model), embedder.model)
        self.client = QdrantClient(path=str(index_dir / "qdrant"))
        if not self.client.collection_exists(self.collection):
            self.client.close()
            raise IndexMissingError(f"No vector index for {embedder.model} yet. {advice}")

    def close(self) -> None:
        self.client.close()

    def search_chunks(self, question: str, lang: str | None, n: int = 50) -> list[Hit]:
        """The n closest chunks, best first; `lang` limits the search to one language."""
        [vector] = embed_cached([question], self.embedder, self.cache)
        condition = None
        if lang:
            match = models.FieldCondition(key="lang", match=models.MatchValue(value=lang))
            condition = models.Filter(must=[match])
        points = self.client.query_points(
            self.collection, query=vector, limit=n, query_filter=condition, with_payload=True
        ).points
        return [Hit(p.payload["section_id"], p.score, p.payload["chunk_id"]) for p in points]

    def search(self, question: str, lang: str | None, k: int = 10) -> list[Hit]:
        """The k closest sections, each scored by its closest chunk."""
        return best_per_section(self.search_chunks(question, lang))[:k]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], k: int = 10, smoothing: int = 60
) -> list[Hit]:
    """Merge several rankings into one (hybrid search).

    Each section earns 1 / (60 + rank) from every ranking it appears in. A section
    ranked well by both methods comes first, without having to compare BM25
    scores with cosine similarities, which are on different scales.
    """
    scores: dict[str, float] = {}
    chunk_of: dict[str, str] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.section_id] = scores.get(hit.section_id, 0.0) + 1 / (smoothing + rank)
            chunk_of.setdefault(hit.section_id, hit.chunk_id)
    ordered = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(section_id, scores[section_id], chunk_of[section_id]) for section_id in ordered]


def fuse_chunks(rankings: list[list[Hit]], n: int = 50, smoothing: int = 60) -> list[Hit]:
    """The same fusion, passage by passage instead of section by section."""
    scores: dict[str, float] = {}
    hit_of: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (smoothing + rank)
            hit_of.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=scores.get, reverse=True)[:n]
    return [Hit(hit_of[c].section_id, scores[c], c) for c in ordered]
