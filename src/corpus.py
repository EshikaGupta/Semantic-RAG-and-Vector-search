"""
corpus.py - The raw text dataset used for ingestion.
10 technical paragraphs covering system design topics.
"""

DOCUMENTS = [
    {
        "id": "doc_1",
        "text": (
            "The system uses a horizontal auto-scaling strategy to handle peak load. "
            "When CPU utilization exceeds 70% or request queue depth surpasses 500, "
            "the orchestration layer spins up additional compute nodes within 90 seconds. "
            "Load balancers distribute traffic using a weighted round-robin algorithm, "
            "ensuring no single node becomes a bottleneck during traffic spikes."
        ),
    },
    {
        "id": "doc_2",
        "text": (
            "Cache invalidation follows a write-through policy combined with a TTL of 300 seconds. "
            "Frequently accessed data is stored in an in-memory Redis cluster, "
            "reducing database read pressure by up to 80% under high concurrency. "
            "Cache warm-up routines execute at deployment time to pre-populate keys "
            "before live traffic is admitted."
        ),
    },
    {
        "id": "doc_3",
        "text": (
            "Vector embeddings are generated using the sentence-transformers library, "
            "specifically the all-MiniLM-L6-v2 model, which produces 384-dimensional dense vectors. "
            "These vectors capture semantic meaning rather than lexical overlap, "
            "enabling similarity searches that surface conceptually related documents "
            "even when they share no common keywords."
        ),
    },
    {
        "id": "doc_4",
        "text": (
            "Cosine similarity is preferred over Euclidean distance for semantic retrieval "
            "because it measures the angle between two vectors rather than their absolute distance. "
            "This makes it invariant to vector magnitude, which matters when comparing "
            "documents of different lengths that have been embedded into the same space."
        ),
    },
    {
        "id": "doc_5",
        "text": (
            "The ingestion pipeline normalises raw text through tokenisation, "
            "stop-word removal, and lower-casing before embedding. "
            "Chunking splits documents into overlapping windows of 256 tokens with a 32-token stride, "
            "preserving context at chunk boundaries and preventing information loss "
            "for long-form technical documents."
        ),
    },
    {
        "id": "doc_6",
        "text": (
            "FAISS (Facebook AI Similarity Search) provides an efficient in-memory index "
            "for approximate nearest-neighbour lookups. The IndexFlatIP index performs "
            "exact inner-product search, which is equivalent to cosine similarity when vectors "
            "are L2-normalised. For production scale, the IVF index reduces search time "
            "from O(n) to O(sqrt(n)) by clustering vectors into Voronoi cells."
        ),
    },
    {
        "id": "doc_7",
        "text": (
            "Query expansion improves retrieval recall by rewriting the user's raw query "
            "into a richer, embedding-friendly form before search. A generative model "
            "identifies implicit intent, adds synonyms, and injects domain terminology. "
            "For example, 'peak load' may be expanded to include 'traffic spike'."
        ),
    },
    {
        "id": "doc_8",
        "text": (
            "Database connection pooling limits the number of simultaneous connections "
            "to prevent resource exhaustion during load surges. The pool is configured "
            "with a minimum of 10 and a maximum of 100 connections, with a 30-second "
            "idle timeout. Queries that exceed 5 seconds are automatically killed "
            "and retried with exponential back-off to protect overall throughput."
        ),
    },
    {
        "id": "doc_9",
        "text": (
            "Migrating a local FAISS index to Vertex AI Vector Search (Matching Engine) "
            "requires exporting embeddings as JSON Lines to Cloud Storage, then creating "
            "an Index resource via the Vertex AI API. The managed service handles sharding, "
            "replication, and online serving, offering single-digit millisecond latency "
            "at billion-vector scale without operational overhead."
        ),
    },
    {
        "id": "doc_10",
        "text": (
            "Observability is implemented through structured JSON logging shipped to "
            "Cloud Logging, distributed tracing via OpenTelemetry, and custom metrics "
            "published to Cloud Monitoring. Dashboards track p50/p95/p99 latency, "
            "error rates, and retrieval hit-rates. Alerts fire when p99 latency "
            "exceeds 2 seconds or error rate climbs above 1% over a 5-minute window."
        ),
    },
]
