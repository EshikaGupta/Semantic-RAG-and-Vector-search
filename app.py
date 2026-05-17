"""
app.py - Streamlit UI for the RAG Benchmark.

Run with:
    streamlit run app.py

The app lets you:
1. Pick any of the pre-defined benchmark queries (or type your own).
2. Click "Run Benchmark" to fire both Strategy A and Strategy B.
3. See the retrieved chunks, similarity scores, and the expanded query side-by-side.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from src.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Benchmark: Strategy A vs B",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS - dark, minimal, clear
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .stButton > button {
        background: #0f62fe;
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background: #0353e9; }

    .score-badge {
        display: inline-block;
        background: #161616;
        color: #42be65;
        border: 1px solid #42be65;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        margin-bottom: 4px;
    }
    .rank-badge {
        display: inline-block;
        background: #0f62fe;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        line-height: 24px;
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .expanded-box {
        background: #1c1c1c;
        border-left: 3px solid #0f62fe;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #c6c6c6;
        margin-bottom: 1rem;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .doc-card {
        border: 1px solid #393939;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        background: #111;
    }
    .stExpander { border: 1px solid #393939 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached pipeline (load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model & ingesting corpus…")
def load_pipeline() -> RAGPipeline:
    p = RAGPipeline()
    p.ingest()
    return p

pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Sidebar - query selector
# ---------------------------------------------------------------------------
PRESET_QUERIES = [
    "How does the system handle peak load?",
    "What similarity metric is used for vector search?",
    "How does query expansion improve retrieval?",
    "How do embeddings capture semantic meaning?",
    "How would you migrate to Vertex AI in production?",
]

st.sidebar.title("🔍 RAG Benchmark")
st.sidebar.markdown("**Corpus**: 10 technical paragraphs")
st.sidebar.markdown("**Model**: all-MiniLM-L6-v2 (mock gecko)")
st.sidebar.markdown("---")

query_choice = st.sidebar.radio("Choose a preset query:", PRESET_QUERIES)
custom = st.sidebar.text_input("…or type your own query:")
query = custom.strip() if custom.strip() else query_choice

top_k = st.sidebar.slider("Top-K results", min_value=1, max_value=5, value=3)

run = st.sidebar.button("▶ Run Benchmark")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("RAG Retrieval Benchmark")
st.markdown("**Strategy A** — raw embedding search &nbsp;·&nbsp; **Strategy B** — query expansion + embedding search")
st.markdown("---")

if run:
    with st.spinner("Running both strategies…"):
        result_a = pipeline.retrieve_raw(query, top_k=top_k)
        result_b = pipeline.retrieve_expanded(query, top_k=top_k)
    # ---- Save benchmark output to markdown -------------------------------
    import json
    from datetime import datetime

    benchmark_output = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "strategy_a": result_a,
        "strategy_b": result_b,
    }
    md_content = f"""
    # Retrieval Benchmark

    ## Query
    {query}

    ## JSON Output

    ```json
    {json.dumps(benchmark_output, indent=2)}"""
    with open("retrival_benchmark.md", "a", encoding="utf-8") as f:
        f.write(md_content)
    # ---- Query row --------------------------------------------------------
    st.subheader(f"Query: *{query}*")
    st.markdown("---")

    col_a, col_b = st.columns(2, gap="large")

    # ---- Strategy A -------------------------------------------------------
    with col_a:
        st.markdown("### 🔵 Strategy A — Raw Vector Search")
        st.markdown(f"**Query used for embedding:**")
        st.markdown(f'<div class="expanded-box">{result_a["query_used"]}</div>', unsafe_allow_html=True)

        for hit in result_a["results"]:
            st.markdown(
                f'<div class="doc-card">'
                f'<span class="rank-badge">{hit["rank"]}</span>'
                f'<span class="score-badge">score: {hit["score"]:.4f}</span>&nbsp;&nbsp;'
                f'<small style="color:#8d8d8d">{hit["id"]}</small><br><br>'
                f'<p style="color:#c6c6c6">{hit["text"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---- Strategy B -------------------------------------------------------
    with col_b:
        st.markdown("### 🟢 Strategy B — Query Expansion + Search")
        st.markdown(f"**Expanded query used for embedding:**")
        st.markdown(f'<div class="expanded-box">{result_b["query_used"]}</div>', unsafe_allow_html=True)

        for hit in result_b["results"]:
            st.markdown(
                f'<div class="doc-card">'
                f'<span class="rank-badge">{hit["rank"]}</span>'
                f'<span class="score-badge">score: {hit["score"]:.4f}</span>&nbsp;&nbsp;'
                f'<small style="color:#8d8d8d">{hit["id"]}</small><br><br>'
                f'<p style="color:#c6c6c6">{hit["text"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---- Score comparison chart -------------------------------------------
    st.markdown("---")
    st.subheader("Score Comparison")

    df_a = pd.DataFrame(result_a["results"]).rename(columns={"score": "Score A"})
    df_b = pd.DataFrame(result_b["results"]).rename(columns={"score": "Score B"})

    # Build a unified comparison table keyed on doc id
    merged = pd.merge(
        df_a[["id", "Score A"]],
        df_b[["id", "Score B"]],
        on="id",
        how="outer",
    ).fillna(0)
    merged = merged.set_index("id")

    st.bar_chart(merged, color=["#0f62fe", "#42be65"])

    # ---- Δ analysis -------------------------------------------------------
    st.markdown("---")
    st.subheader("Analysis")

    ids_a = {r["id"] for r in result_a["results"]}
    ids_b = {r["id"] for r in result_b["results"]}
    new_in_b = ids_b - ids_a
    dropped_in_b = ids_a - ids_b

    avg_a = sum(r["score"] for r in result_a["results"]) / len(result_a["results"])
    avg_b = sum(r["score"] for r in result_b["results"]) / len(result_b["results"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Score A", f"{avg_a:.4f}")
    m2.metric("Avg Score B", f"{avg_b:.4f}", delta=f"{avg_b - avg_a:+.4f}")
    m3.metric("New docs surfaced by B", len(new_in_b))

    if new_in_b:
        st.info(f"📄 Strategy B surfaced **{', '.join(sorted(new_in_b))}** which Strategy A missed.")
    if dropped_in_b:
        st.warning(f"Strategy B did not retrieve **{', '.join(sorted(dropped_in_b))}** (present in A).")
    if not new_in_b and not dropped_in_b:
        st.success("Both strategies retrieved the same document set for this query.")

    # ---- Raw JSON -----------------------------------------------------------
    with st.expander("View raw JSON output"):
        import json
        st.json({"strategy_a": result_a, "strategy_b": result_b})

else:
    st.info("👈 Select a query in the sidebar and click **▶ Run Benchmark** to start.")

    # Show corpus at rest
    with st.expander("📚 View corpus (10 documents)"):
        from src.corpus import DOCUMENTS
        for doc in DOCUMENTS:
            st.markdown(f"**{doc['id']}** — {doc['text']}")
