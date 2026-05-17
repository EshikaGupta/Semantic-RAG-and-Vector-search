"""
query_expander.py - Mocks Vertex AI GenerativeModel for query expansion.

Strategy B works by rewriting the user's terse query into a richer,
embedding-friendly sentence before we embed and search.  In production this
would call GenerativeModel("gemini-pro").generate_content(...).  Here we mock
the SDK class but keep the expansion logic real (rule-based + synonym map)
so the benchmark shows a genuine difference vs Strategy A.

"""

from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Mock of vertexai.generative_models.GenerativeModel
# ---------------------------------------------------------------------------
class MockGenerativeModel:
    """
    Mirrors the Vertex AI SDK surface:
        model = GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        text = response.text
    """

    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name

    def generate_content(self, prompt: str) -> "_MockResponse":
        expanded = _expand(prompt)
        return _MockResponse(expanded)


class _MockResponse:
    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Expansion logic
# ---------------------------------------------------------------------------

# Domain synonym map - each key expands to additional terms when found in query
_SYNONYMS: dict[str, list[str]] = {
    "peak load": [
        "traffic spike", "high concurrency", "auto-scaling threshold",
        "request queue saturation", "load surge",
    ],
    "embedding": [
        "vector representation", "dense vector", "semantic encoding",
        "sentence-transformers", "dimensionality",
    ],
    "retrieval": [
        "similarity search", "nearest neighbour lookup", "top-k search",
        "semantic retrieval", "RAG pipeline",
    ],
    "similarity": [
        "cosine distance", "inner product", "vector proximity",
        "semantic closeness",
    ],
    "cache": [
        "in-memory store", "Redis", "TTL", "cache invalidation",
        "write-through policy",
    ],
    "database": [
        "connection pool", "query latency", "read replica",
        "transaction throughput",
    ],
    "latency": [
        "response time", "p99", "tail latency", "millisecond delay",
    ],
    "scale": [
        "horizontal scaling", "auto-scaling", "load balancer",
        "distributed system",
    ],
    "vertex ai": [
        "Matching Engine", "Vector Search", "GCP managed service",
        "cloud deployment",
    ],
    "observability": [
        "monitoring", "logging", "tracing", "OpenTelemetry", "alerting",
    ],
}


def _expand(raw_query: str) -> str:
    """
    Produce an enriched query string by:
    1. Detecting known domain terms in the raw query (case-insensitive).
    2. Appending their synonyms as a comma-separated phrase.
    3. Wrapping everything in a full sentence for better embedding coverage.
    """
    lower = raw_query.lower()
    extra_terms: list[str] = []

    for key, synonyms in _SYNONYMS.items():
        if key in lower:
            extra_terms.extend(synonyms)

    if extra_terms:
        unique_terms = list(dict.fromkeys(extra_terms))  # preserve order, dedupe
        synonym_clause = ", ".join(unique_terms)
        expanded = (
            f"{raw_query.rstrip('?')} — including concepts such as "
            f"{synonym_clause}?"
        )
    else:
        # Fallback: wrap in a generic elaboration sentence
        expanded = (
            f"Provide a detailed technical explanation of the following: {raw_query}"
        )

    return expanded


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

class QueryExpander:
    """
    Thin wrapper that exposes expand(query) -> str.
    Uses MockGenerativeModel internally; swap for real SDK in production.
    """

    def __init__(self):
        self._model = MockGenerativeModel("gemini-pro")

    def expand(self, query: str) -> str:
        """Return an expanded version of *query* suitable for embedding."""
        prompt = (

            f"Query: {query}\n\n"
            f"Expanded query:"
        )
        response = self._model.generate_content(prompt)
        return response.text
