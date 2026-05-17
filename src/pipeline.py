"""
pipeline.py - Orchestrates ingestion, embedding, and retrieval.

This is the central class that wires together:
    Corpus  ->  Embedder  ->  VectorStore
and exposes two retrieval strategies:
    retrieve_raw()      Strategy A - embed the query as-is
    retrieve_expanded() Strategy B - expand query first, then embed
"""

from __future__ import annotations
import json

import numpy as np

from .corpus import DOCUMENTS
from .embedder import get_embedder
from .vector_store import VectorStore
from .query_expander import QueryExpander


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage
    -----
        pipeline = RAGPipeline()
        pipeline.ingest()
        results_a = pipeline.retrieve_raw("peak load", top_k=3)
        results_b = pipeline.retrieve_expanded("peak load", top_k=3)
    """

    def __init__(self):
        self.embedder = get_embedder()
        self.store = VectorStore(dim=384)
        self.expander = QueryExpander()
        self._ingested = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[dict] | None = None) -> None:
        """
        Embed and store documents.  Defaults to the built-in DOCUMENTS corpus.
        Calling ingest() multiple times is safe - it resets the store first.
        """
        docs = documents or DOCUMENTS
        texts = [d["text"] for d in docs]
        embeddings = self.embedder.get_embeddings(texts)

        # Reset store so re-ingestion doesn't duplicate entries
        self.store = VectorStore(dim=len(embeddings[0]))
        self.store.add(docs, embeddings)
        self._ingested = True
        print(f"[Pipeline] Ingested {len(docs)} documents.")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        vecs = self.embedder.get_embeddings([query])
        return vecs[0]

    def retrieve_raw(self, query: str, top_k: int = 3) -> dict:
        """Strategy A - embed the raw query and search."""
        self._check_ingested()
        vec = self._embed_query(query)
        hits = self.store.search(vec, top_k=top_k)
        return {
            "strategy": "A - Raw Vector Search",
            "query_used": query,
            "results": hits,
        }

    def retrieve_expanded(self, query: str, top_k: int = 3) -> dict:
        """Strategy B - expand the query, then embed and search."""
        self._check_ingested()
        expanded = self.expander.expand(query)
        vec = self._embed_query(expanded)
        hits = self.store.search(vec, top_k=top_k)
        return {
            "strategy": "B - AI-Enhanced Retrieval (Query Expansion)",
            "query_used": expanded,
            "results": hits,
        }

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(self, queries: list[str], top_k: int = 3) -> list[dict]:
        """
        Run Strategy A and B for each query and return a structured report.

        Returns
        -------
        list of dicts, one per query, each containing:
            {
                "query": <original query>,
                "strategy_a": { ... },
                "strategy_b": { ... },
            }
        """
        self._check_ingested()
        report = []
        for q in queries:
            report.append(
                {
                    "query": q,
                    "strategy_a": self.retrieve_raw(q, top_k=top_k),
                    "strategy_b": self.retrieve_expanded(q, top_k=top_k),
                }
            )
        return report

    def benchmark_json(self, queries: list[str], top_k: int = 3) -> str:
        """Return the benchmark report as a formatted JSON string."""
        return json.dumps(self.benchmark(queries, top_k=top_k), indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_ingested(self) -> None:
        if not self._ingested:
            raise RuntimeError("Call pipeline.ingest() before retrieving.")
