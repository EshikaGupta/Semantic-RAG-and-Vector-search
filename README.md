# RAG Benchmark — Strategy A vs Strategy B

A local Retrieval-Augmented Generation pipeline that compares two retrieval strategies:

- **Strategy A** — embed the user's raw query and search
- **Strategy B** — expand the query with synonyms first, then embed and search

Strategy B consistently outperforms A because the expanded query spans a broader semantic region of the vector space, reducing the vocabulary gap between user language and document language.

---

## Folder Structure

```
rag_project/
├── src/
│   ├── __init__.py
│   ├── corpus.py          # 10 technical paragraphs (the dataset)
│   ├── embedder.py        # MockTextEmbeddingModel (mocks gecko locally)
│   ├── vector_store.py    # FAISS IndexFlatIP (cosine similarity)
│   ├── query_expander.py  # MockGenerativeModel + synonym expansion
│   └── pipeline.py        # RAGPipeline orchestrator
├── tests/
│   └── test_pipeline.py   # 17 pytest tests
├── app.py                 # Streamlit visualisation UI
├── retrieval_benchmark.md # Strategy A vs B comparison report
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run Streamlit UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Select a query, click **Run Benchmark**, and see Strategy A vs B side-by-side with scores and analysis.

---

## Run Tests

```bash
pytest tests/ -v
```

All 17 tests should pass. Tests mock GCP SDK calls — no credentials required.
---

## Key Design Decisions
# Retrieval Benchmark: Strategy A vs Strategy B

## Similarity Metric Choice: Cosine vs Euclidean

**We use Cosine Similarity (via FAISS `IndexFlatIP` on L2-normalised vectors).**

| Property | Cosine Similarity | Euclidean Distance |
|----------|-------------------|--------------------|
| What it measures | Angle between vectors | Absolute distance |
| Magnitude-invariant | ✅ Yes | ❌ No |
| Good for varied-length docs | ✅ Yes | ❌ No |
| FAISS implementation | `IndexFlatIP` + L2-norm | `IndexFlatL2` |

When two vectors are L2-normalised (unit length), inner product equals cosine similarity. This is why we normalise all embeddings at ingestion time. A short and a long document about the same topic will have the same cosine similarity to a query, which is the desired behaviour for semantic search.

---

## How Query Expansion Works (Strategy B)

The raw user query is passed to a `MockGenerativeModel` (mocking `vertexai.generative_models.GenerativeModel`). The model rewrites the query by injecting synonyms and related domain terms from a curated synonym map. The expanded text is then embedded and used for search.

**Why does it help?** The embedding of a short query like `"peak load"` only occupies a narrow region of the vector space. The embedding of the expanded query spans a broader semantic region, pulling in documents that are conceptually related but use different vocabulary.

---

## Migration to Vertex AI Vector Search (Matching Engine)

1. **Export embeddings** — serialise all vectors as JSON Lines to Cloud Storage:
   ```
   {"id": "doc_1", "embedding": [0.12, -0.34, ...]}
   ```
2. **Create Index** — call `aiplatform.MatchingEngineIndex.create_tree_ah_index(...)` pointing to the GCS path.
3. **Deploy to Endpoint** — `MatchingEngineIndexEndpoint.deploy_index(...)` for online serving.
4. **Query** — replace `VectorStore.search()` with `endpoint.find_neighbors(query_embedding, num_neighbors=k)`.
5. **Swap embedder** — replace `MockTextEmbeddingModel` with the real `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`.

All three are behind clean interfaces — the orchestration layer (`pipeline.py`) doesn't change.
