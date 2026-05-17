
    # Retrieval Benchmark

    ## Query
    How does the system handle peak load?

    ## JSON Output

    ```json
    {
  "timestamp": "2026-05-16T20:54:56.925942",
  "query": "How does the system handle peak load?",
  "strategy_a": {
    "strategy": "A - Raw Vector Search",
    "query_used": "How does the system handle peak load?",
    "results": [
      {
        "rank": 1,
        "score": 0.34371381998062134,
        "id": "doc_1",
        "text": "The system uses a horizontal auto-scaling strategy to handle peak load. When CPU utilization exceeds 70% or request queue depth surpasses 500, the orchestration layer spins up additional compute nodes within 90 seconds. Load balancers distribute traffic using a weighted round-robin algorithm, ensuring no single node becomes a bottleneck during traffic spikes."
      },
      {
        "rank": 2,
        "score": 0.2563765048980713,
        "id": "doc_7",
        "text": "Query expansion improves retrieval recall by rewriting the user's raw query into a richer, embedding-friendly form before search. A generative model identifies implicit intent, adds synonyms, and injects domain terminology. For example, 'peak load' may be expanded to include 'traffic spike'."
      },
      {
        "rank": 3,
        "score": 0.21855108439922333,
        "id": "doc_6",
        "text": "FAISS (Facebook AI Similarity Search) provides an efficient in-memory index for approximate nearest-neighbour lookups. The IndexFlatIP index performs exact inner-product search, which is equivalent to cosine similarity when vectors are L2-normalised. For production scale, the IVF index reduces search time from O(n) to O(sqrt(n)) by clustering vectors into Voronoi cells."
      }
    ]
  },
  "strategy_b": {
    "strategy": "B - AI-Enhanced Retrieval (Query Expansion)",
    "query_used": "Query: How does the system handle peak load?\n\nExpanded query: \u2014 including concepts such as traffic spike, high concurrency, auto-scaling threshold, request queue saturation, load surge?",
    "results": [
      {
        "rank": 1,
        "score": 0.4215165376663208,
        "id": "doc_1",
        "text": "The system uses a horizontal auto-scaling strategy to handle peak load. When CPU utilization exceeds 70% or request queue depth surpasses 500, the orchestration layer spins up additional compute nodes within 90 seconds. Load balancers distribute traffic using a weighted round-robin algorithm, ensuring no single node becomes a bottleneck during traffic spikes."
      },
      {
        "rank": 2,
        "score": 0.35942739248275757,
        "id": "doc_7",
        "text": "Query expansion improves retrieval recall by rewriting the user's raw query into a richer, embedding-friendly form before search. A generative model identifies implicit intent, adds synonyms, and injects domain terminology. For example, 'peak load' may be expanded to include 'traffic spike'."
      },
      {
        "rank": 3,
        "score": 0.29081618785858154,
        "id": "doc_6",
        "text": "FAISS (Facebook AI Similarity Search) provides an efficient in-memory index for approximate nearest-neighbour lookups. The IndexFlatIP index performs exact inner-product search, which is equivalent to cosine similarity when vectors are L2-normalised. For production scale, the IVF index reduces search time from O(n) to O(sqrt(n)) by clustering vectors into Voronoi cells."
      }
    ]
  }
}
    # Retrieval Benchmark

    ## Query
    What similarity metric is used for vector search?

    ## JSON Output

    ```json
    {
  "timestamp": "2026-05-16T20:55:27.151722",
  "query": "What similarity metric is used for vector search?",
  "strategy_a": {
    "strategy": "A - Raw Vector Search",
    "query_used": "What similarity metric is used for vector search?",
    "results": [
      {
        "rank": 1,
        "score": 0.2824857234954834,
        "id": "doc_6",
        "text": "FAISS (Facebook AI Similarity Search) provides an efficient in-memory index for approximate nearest-neighbour lookups. The IndexFlatIP index performs exact inner-product search, which is equivalent to cosine similarity when vectors are L2-normalised. For production scale, the IVF index reduces search time from O(n) to O(sqrt(n)) by clustering vectors into Voronoi cells."
      },
      {
        "rank": 2,
        "score": 0.1999911516904831,
        "id": "doc_4",
        "text": "Cosine similarity is preferred over Euclidean distance for semantic retrieval because it measures the angle between two vectors rather than their absolute distance. This makes it invariant to vector magnitude, which matters when comparing documents of different lengths that have been embedded into the same space."
      },
      {
        "rank": 3,
        "score": 0.1583380401134491,
        "id": "doc_7",
        "text": "Query expansion improves retrieval recall by rewriting the user's raw query into a richer, embedding-friendly form before search. A generative model identifies implicit intent, adds synonyms, and injects domain terminology. For example, 'peak load' may be expanded to include 'traffic spike'."
      }
    ]
  },
  "strategy_b": {
    "strategy": "B - AI-Enhanced Retrieval (Query Expansion)",
    "query_used": "Query: What similarity metric is used for vector search?\n\nExpanded query: \u2014 including concepts such as cosine distance, inner product, vector proximity, semantic closeness?",
    "results": [
      {
        "rank": 1,
        "score": 0.3258560597896576,
        "id": "doc_4",
        "text": "Cosine similarity is preferred over Euclidean distance for semantic retrieval because it measures the angle between two vectors rather than their absolute distance. This makes it invariant to vector magnitude, which matters when comparing documents of different lengths that have been embedded into the same space."
      },
      {
        "rank": 2,
        "score": 0.3195536434650421,
        "id": "doc_6",
        "text": "FAISS (Facebook AI Similarity Search) provides an efficient in-memory index for approximate nearest-neighbour lookups. The IndexFlatIP index performs exact inner-product search, which is equivalent to cosine similarity when vectors are L2-normalised. For production scale, the IVF index reduces search time from O(n) to O(sqrt(n)) by clustering vectors into Voronoi cells."
      },
      {
        "rank": 3,
        "score": 0.23280581831932068,
        "id": "doc_7",
        "text": "Query expansion improves retrieval recall by rewriting the user's raw query into a richer, embedding-friendly form before search. A generative model identifies implicit intent, adds synonyms, and injects domain terminology. For example, 'peak load' may be expanded to include 'traffic spike'."
      }
    ]
  }
}