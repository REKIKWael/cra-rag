"""Tests for the vector index, with a fake embedding model (no Ollama needed).

The fake model turns a text into a "bag of words" vector: texts sharing words
get close vectors. That is enough to test the plumbing: cache, Qdrant, filters.
"""

import hashlib
import math

import pytest

from cra_rag import chunks, dense, parse
from cra_rag.search import Hit


class FakeEmbedder:
    model = "fake"
    batch_size = 2

    def __init__(self, fail_after_batches=None):
        self.calls = 0
        self.fail_after_batches = fail_after_batches

    def embed(self, texts):
        if self.fail_after_batches is not None and self.calls >= self.fail_after_batches:
            raise RuntimeError("interrupted")
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * 32
            for word in text.lower().split():
                vector[int(hashlib.sha1(word.encode()).hexdigest(), 16) % 32] += 1
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors


def make_chunks():
    def section(sid, lang, text, kind="article"):
        return parse.Section(
            id=sid, source="cra", lang=lang, kind=kind, number="1", label=sid, title="",
            chapter="", text=text, url="",
        )  # fmt: skip

    return chunks.chunk_sections(
        [
            section("cra:art_14", "en", "manufacturers notify exploited vulnerabilities to enisa"),
            section("cra:art_64", "en", "fines up to fifteen million euros"),
            section("cra:art_14", "fr", "le fabricant notifie les vulnérabilités exploitées"),
            section("cra:rct_1", "en", "notify exploited vulnerabilities recital", kind="recital"),
        ]
    )


def test_index_then_search_finds_the_closest_section_in_the_right_language(tmp_path):
    embedder = FakeEmbedder()
    assert dense.build_index(make_chunks(), embedder, tmp_path) == 3  # recital left out

    index = dense.DenseIndex(embedder, tmp_path)
    try:
        hits = index.search("notify exploited vulnerabilities", lang="en")
    finally:
        index.close()

    assert hits[0].section_id == "cra:art_14"
    assert "cra:rct_1" not in [h.section_id for h in hits]
    assert all("/en#" in h.chunk_id for h in hits)


def test_rebuilding_the_index_reuses_the_cached_vectors(tmp_path):
    dense.build_index(make_chunks(), FakeEmbedder(), tmp_path)
    again = FakeEmbedder()
    dense.build_index(make_chunks(), again, tmp_path)
    assert again.calls == 0


def test_an_interrupted_indexing_resumes_where_it_stopped(tmp_path):
    with pytest.raises(RuntimeError):
        dense.build_index(make_chunks(), FakeEmbedder(fail_after_batches=1), tmp_path)
    resumed = FakeEmbedder()
    dense.build_index(make_chunks(), resumed, tmp_path)
    assert resumed.calls == 1  # 3 chunks, 2 already computed, batches of 2: one call left


def test_search_without_an_index_explains_how_to_build_it(tmp_path):
    with pytest.raises(dense.IndexMissingError, match="cra-rag index"):
        dense.DenseIndex(FakeEmbedder(), tmp_path)


def test_fusion_puts_first_the_section_both_methods_rank_well():
    bm25 = [Hit("a", 9.0, "a#1"), Hit("b", 8.0, "b#1"), Hit("c", 7.0, "c#1")]
    semantic = [Hit("c", 0.9, "c#1"), Hit("b", 0.8, "b#1"), Hit("d", 0.7, "d#1")]
    fused = dense.reciprocal_rank_fusion([bm25, semantic], k=4)
    assert {h.section_id for h in fused[:2]} == {"b", "c"}  # found by both methods
    assert {h.section_id for h in fused} == {"a", "b", "c", "d"}


def test_model_names_become_safe_file_names(tmp_path):
    assert dense.cache_path(tmp_path, "bge-m3:latest").name == "bge-m3_latest.cache.jsonl"


def test_an_index_built_with_another_model_is_not_used(tmp_path):
    dense.build_index(make_chunks(), FakeEmbedder(), tmp_path)
    other = FakeEmbedder()
    other.model = "other-model"
    with pytest.raises(dense.IndexMissingError, match="--model other-model"):
        dense.DenseIndex(other, tmp_path)
