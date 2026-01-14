import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")
st.title("AI-Based Security Log Analyzer (RAG)")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "texts" not in st.session_state:
    st.session_state.texts = None
    st.session_state.embeddings = None

# ---------------- FILE LOADER ----------------
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


# ---------------- EMBEDDINGS ----------------
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


# ---------------- SIDEBAR (FILE UPLOAD) ----------------
with st.sidebar:
    st.subheader("Upload Logs")
    uploaded = st.file_uploader(
        "Upload CSV / JSON / TXT",
        type=["csv", "json", "txt"]
    )

    if uploaded:
        df = load_file(uploaded)
        st.success(f"Loaded {len(df)} rows")

        texts = prepare_text(df)
        embeddings = build_embeddings(texts)

        st.session_state.texts = texts
        st.session_state.embeddings = embeddings


# ---------------- CHAT HISTORY DISPLAY ----------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])


# ---------------- CHAT INPUT (BOTTOM) ----------------
user_question = st.chat_input("Ask about suspicious activity, attacks, fixes...")

if user_question:
    if not st.session_state.texts:
        st.warning("Please upload a log file first")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing logs..."):
                relevant_logs = retrieve_logs(
                    user_question,
                    st.session_state.texts,
                    st.session_state.embeddings
                )

            answer = f"""
### 🔍 Analysis Result

**What is happening:**  
Suspicious or unusual patterns were found based on your query.

**Evidence from logs:**  
{relevant_logs[:3]}

**Possible reason:**  
- Repeated failures  
- Abnormal access pattern  
- Misconfiguration or attack attempt  

**How to fix:**  
- Enable account lockout  
- Block suspicious IPs  
- Apply rate-limiting  
- Review firewall & authentication rules
            """

            st.markdown(answer)

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })
