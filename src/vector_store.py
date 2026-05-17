"""
vector_store.py - Lightweight FAISS-backed vector store.

Uses IndexFlatIP (inner product) on L2-normalised vectors, which is
mathematically equivalent to cosine similarity.
"""

import numpy as np
import faiss


class VectorStore:
    """
    Stores document embeddings in a FAISS index and supports top-k retrieval.

    Attributes
    ----------
    dim : int
        Dimensionality of the embedding vectors (384 for all-MiniLM-L6-v2).
    index : faiss.IndexFlatIP
        The FAISS index - exact inner product search (≡ cosine on unit vecs).
    documents : list[dict]
        The original document dicts {id, text} stored in insertion order.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        # IndexFlatIP = brute-force inner-product search (exact, no quantisation)
        self.index = faiss.IndexFlatIP(dim)
        self.documents: list[dict] = []

    def add(self, docs: list[dict], embeddings: list[np.ndarray]) -> None:
        """
        Add documents and their precomputed embeddings to the store.

        Parameters
        ----------
        docs : list[dict]   Each dict must have 'id' and 'text' keys.
        embeddings : list[np.ndarray]  One L2-normalised vector per doc.
        """
        matrix = np.array(embeddings, dtype=np.float32)
        self.index.add(matrix)
        self.documents.extend(docs)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Return the top-k most similar documents for a query embedding.

        Parameters
        ----------
        query_vec : np.ndarray  Shape (dim,) - must be L2-normalised.
        top_k : int             Number of results to return.

        Returns
        -------
        list of dicts with keys: rank, score, id, text
        """
        q = np.array([query_vec], dtype=np.float32)          # (1, dim)
        scores, indices = self.index.search(q, top_k)        # each shape (1, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:          # FAISS returns -1 when fewer docs than k
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "id": doc["id"],
                    "text": doc["text"],
                }
            )
        return results
