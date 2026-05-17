"""
tests/test_pipeline.py - Pytest suite for the RAG pipeline.

Covers:
- MockTextEmbeddingModel produces correct-shape L2-normalised vectors
- MockGenerativeModel expand path runs without GCP credentials
- VectorStore add / search round-trip
- RAGPipeline.ingest + retrieve_raw + retrieve_expanded end-to-end
- Benchmark output structure
"""

import json
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedder import MockTextEmbeddingModel
from src.vector_store import VectorStore
from src.query_expander import MockGenerativeModel, QueryExpander
from src.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestMockTextEmbeddingModel:
    def setup_method(self):
        self.model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    def test_returns_list_of_arrays(self):
        results = self.model.get_embeddings(["hello world"])
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)

    def test_embedding_dimension_is_384(self):
        results = self.model.get_embeddings(["test"])
        assert results[0].shape == (384,)

    def test_embeddings_are_l2_normalised(self):
        results = self.model.get_embeddings(["normalised vector check"])
        norm = float(np.linalg.norm(results[0]))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_batch_embeddings(self):
        texts = ["doc one", "doc two", "doc three"]
        results = self.model.get_embeddings(texts)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def _make_unit_vec(self, dim=384, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.random(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_add_and_search_returns_top_k(self):
        store = VectorStore(dim=4)
        docs = [{"id": f"d{i}", "text": f"doc {i}"} for i in range(5)]
        # Create 5 unit vecs in 4-d space
        vecs = [np.array([1, 0, 0, 0], dtype=np.float32)]  # perfect match for query
        for i in range(1, 5):
            v = np.zeros(4, dtype=np.float32)
            v[i % 4] = 1.0
            vecs.append(v)
        store.add(docs, vecs)

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=3)

        assert len(results) == 3
        # d0 and d4 both map to [1,0,0,0] (4%4==0), so either can rank first
        assert results[0]["id"] in ("d0", "d4")
        assert results[0]["score"] > 0.99

    def test_result_keys(self):
        store = VectorStore(dim=4)
        docs = [{"id": "x", "text": "hello"}]
        vecs = [np.array([1, 0, 0, 0], dtype=np.float32)]
        store.add(docs, vecs)
        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
        assert set(results[0].keys()) == {"rank", "score", "id", "text"}

    def test_rank_starts_at_one(self):
        store = VectorStore(dim=4)
        docs = [{"id": f"d{i}", "text": f"t{i}"} for i in range(3)]
        vecs = [np.eye(4, dtype=np.float32)[i % 4] for i in range(3)]
        store.add(docs, vecs)
        results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3)
        assert results[0]["rank"] == 1


# ---------------------------------------------------------------------------
# QueryExpander / MockGenerativeModel tests
# ---------------------------------------------------------------------------

class TestQueryExpander:
    def test_mock_generative_model_returns_string(self):
        model = MockGenerativeModel("gemini-pro")
        response = model.generate_content("expand: peak load")
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_expander_enriches_known_term(self):
        expander = QueryExpander()
        result = expander.expand("How does the system handle peak load?")
        # Expanded query should contain at least one synonym
        assert any(
            term in result
            for term in ["traffic spike", "auto-scaling", "concurrency", "queue"]
        )

    def test_expander_fallback_for_unknown_query(self):
        expander = QueryExpander()
        result = expander.expand("What is the colour of the sky?")
        assert "What is the colour of the sky?" in result or len(result) > 20


# ---------------------------------------------------------------------------
# RAGPipeline end-to-end tests
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"id": "a", "text": "Auto-scaling handles traffic spikes by adding nodes."},
    {"id": "b", "text": "Cosine similarity measures the angle between vectors."},
    {"id": "c", "text": "Redis cache reduces database load significantly."},
]


class TestRAGPipeline:
    def setup_method(self):
        self.pipeline = RAGPipeline()
        self.pipeline.ingest(SAMPLE_DOCS)

    def test_ingest_populates_store(self):
        assert self.pipeline.store.index.ntotal == len(SAMPLE_DOCS)

    def test_retrieve_raw_returns_results(self):
        out = self.pipeline.retrieve_raw("traffic handling", top_k=2)
        assert out["strategy"].startswith("A")
        assert len(out["results"]) == 2

    def test_retrieve_expanded_returns_results(self):
        out = self.pipeline.retrieve_expanded("peak load", top_k=2)
        assert out["strategy"].startswith("B")
        assert len(out["results"]) == 2

    def test_expanded_query_differs_from_raw(self):
        raw_out = self.pipeline.retrieve_raw("peak load", top_k=3)
        exp_out = self.pipeline.retrieve_expanded("peak load", top_k=3)
        # The query used for embedding should differ between strategies
        assert raw_out["query_used"] != exp_out["query_used"]

    def test_benchmark_structure(self):
        report = self.pipeline.benchmark(["peak load"], top_k=2)
        assert len(report) == 1
        entry = report[0]
        assert "query" in entry
        assert "strategy_a" in entry
        assert "strategy_b" in entry

    def test_benchmark_json_is_valid_json(self):
        json_str = self.pipeline.benchmark_json(["embeddings", "cache"], top_k=2)
        parsed = json.loads(json_str)
        assert len(parsed) == 2

    def test_retrieve_before_ingest_raises(self):
        fresh = RAGPipeline()
        with pytest.raises(RuntimeError, match="ingest"):
            fresh.retrieve_raw("anything")
