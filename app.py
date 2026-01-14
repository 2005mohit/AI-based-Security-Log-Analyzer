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


# ---------------- EMBEDDINGS (UNCHANGED) ----------------
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


# ---------------- ISSUE ANALYSIS HELPERS ----------------
def detect_issue_type(logs):
    text = " ".join(logs).lower()

    if "network_scan" in text:
        return "Network scanning activity was detected."
    if "file_access" in text:
        return "Someone tried to access files without permission."
    if "account locked" in text:
        return "An account was locked after multiple failed login attempts."
    if "login | failed" in text:
        return "Repeated failed login attempts were detected."
    return "Unusual system activity was detected."


def get_reason(issue):
    issue = issue.lower()
    if "network" in issue:
        return "A system was scanning the network to find open ports or weaknesses."
    if "file" in issue:
        return "The user did not have permission to access the requested files."
    if "login" in issue:
        return "Wrong passwords were entered multiple times in a short period."
    return "The activity does not match normal system behavior."


def get_fix(issue):
    issue = issue.lower()
    if "login" in issue:
        return [
            "Enable account lockout and multi-factor authentication",
            "Add CAPTCHA to the login page"
        ]
    if "file" in issue:
        return [
            "Review file permissions",
            "Apply role-based access control"
        ]
    if "network" in issue:
        return [
            "Block the IP address using firewall rules",
            "Enable IDS/IPS monitoring"
        ]
    return ["Monitor the system and review logs regularly"]


# ---------------- IP ANALYSIS (NEW – FOR QUESTION 2) ----------------
def extract_malicious_ips(logs):
    ip_reasons = {}

    for log in logs:
        parts = log.split("|")
        if len(parts) < 6:
            continue

        ip = parts[2].strip()
        event = parts[3].strip().lower()
        details = parts[5].lower()

        if "network_scan" in event:
            ip_reasons[ip] = "Detected network scanning activity"
        elif "suspicious location" in details or "unknown location" in details:
            ip_reasons[ip] = "Login attempt from a suspicious location"
        elif "account locked" in details:
            ip_reasons[ip] = "Account locked after repeated failed logins"

    return ip_reasons


# ---------------- SIDEBAR (UPLOAD) ----------------
with st.sidebar:
    st.subheader("Upload Log File")
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


# ---------------- CHAT HISTORY ----------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])


# ---------------- CHAT INPUT ----------------
user_question = st.chat_input("Ask about suspicious activity, IPs, or fixes...")

if user_question:
    if not st.session_state.texts:
        st.warning("Please upload a log file first.")
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

            # -------- IP INTENT HANDLING --------
            if "ip" in user_question.lower():
                ip_map = extract_malicious_ips(relevant_logs)

                if not ip_map:
                    answer = "No clearly malicious IP addresses were identified."
                else:
                    lines = [
                        f"- **{ip}**: {reason}"
                        for ip, reason in ip_map.items()
                    ]

                    answer = f"""
### 🔍 Malicious IP Analysis

The following IP addresses appear suspicious:

{chr(10).join(lines)}
                    """
            else:
                issue_summary = detect_issue_type(relevant_logs)
                reason = get_reason(issue_summary)
                fixes = get_fix(issue_summary)

                answer = f"""
### 🔍 Analysis Result

**What happened (Simple):**  
{issue_summary}

**Why it happened:**  
{reason}

**Evidence from logs:**  
{relevant_logs[:3]}

**How to fix:**  
- {fixes[0]}
- {fixes[1] if len(fixes) > 1 else ""}
                """

            st.markdown(answer)

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })
