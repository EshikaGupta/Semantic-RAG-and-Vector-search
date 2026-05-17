"""
embedder.py - Wraps a local embedding model to simulate Vertex AI textembedding-gecko.
"""

from __future__ import annotations
import re
import hashlib
import numpy as np

DIM = 384  # matches textembedding-gecko@003 and all-MiniLM-L6-v2


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash_token(token: str, dim: int = DIM) -> int:
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return h % dim


def _embed(text: str, dim: int = DIM) -> np.ndarray:
    """Hashing-trick TF vector, L2-normalised to unit length."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        vec[_hash_token(tok, dim)] += 1.0
    # Sublinear TF scaling
    vec = np.where(vec > 0, 1.0 + np.log(vec), 0.0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class MockTextEmbeddingModel:
    """
    Drop-in mock for vertexai.language_models.TextEmbeddingModel.
    Produces L2-normalised 384-dim vectors locally — no GCP credentials needed.
    """

    @classmethod
    def from_pretrained(cls, vertex_model_name: str = "textembedding-gecko@003"):
        return cls()

    def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        return [_embed(t) for t in texts]


def get_embedder() -> MockTextEmbeddingModel:
    return MockTextEmbeddingModel.from_pretrained()
