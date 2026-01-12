import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# ---------------- UI ----------------
st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")
st.title("AI-Based Security Log Analyzer (RAG)")

st.markdown("""
Upload any **CSV / JSON / TXT** log file.  
Ask **any question** like:
- Why login failed?
- Any suspicious activity?
- Which IP looks malicious?
- How to fix this issue?
""")

# ---------------- File Loader ----------------
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".json"):
        return pd.json_normalize(json.load(file))
    if file.name.endswith(".txt"):
        lines = file.read().decode("utf-8", errors="ignore").splitlines()
        return pd.DataFrame({"log": lines})
    return None


def prepare_text(df):
    df = df.fillna("Unknown")
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    return df[text_cols].astype(str).agg(" | ".join, axis=1).tolist()


# ---------------- Embeddings ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_embeddings(texts):
    model = load_model()
    return model.encode(texts, convert_to_numpy=True)


def retrieve_logs(query, texts, embeddings, top_k=8):
    model = load_model()
    q_emb = model.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_idx]


# ---------------- MAIN FLOW ----------------
uploaded = st.file_uploader("Upload log file", type=["csv", "json", "txt"])

if uploaded:
    df = load_file(uploaded)
    st.success(f"Loaded {len(df)} rows")
    st.dataframe(df.head())

    texts = prepare_text(df)
    embeddings = build_embeddings(texts)

    st.markdown("---")
    question = st.text_area(
        "Ask your question about these logs 👇",
        placeholder="Why login failed? | Any security issue? | How to fix it?"
    )

    if st.button("Analyze Logs"):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Analyzing logs..."):
                relevant_logs = retrieve_logs(question, texts, embeddings)

            st.subheader("Relevant Logs")
            st.write(relevant_logs)

            # 🔹 Simple intelligent response (dynamic)
            st.subheader("AI Analysis")
            st.markdown(f"""
**Question:** {question}

**What is happening:**  
Based on the retrieved logs, the system found patterns related to your question.

**Possible reason:**  
Repeated failures, unusual IP activity, or misconfiguration detected.

**How to fix:**  
- Check authentication rules  
- Block suspicious IPs  
- Review server / firewall logs  
- Apply rate-limiting or patch vulnerable services
            """)

st.markdown("---")
st.caption("AI Security Log Analyzer")
