import streamlit as st
import requests
import os
import json
import datetime

# --------------------------
# Session Configuration
# --------------------------
st.set_page_config(page_title="Clinical Trial Chatbot", layout="wide")
st.title("💬 Clinical Trials Assistant (via Flask API)")

# Allow user to set a session ID (can be tied to user or timestamp)
session_id = st.text_input("Session ID", value="default")

# Load or initialize conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------
# Conversation Logging
# --------------------------
def log_conversation(query, answer, session_id="default"):
    log_folder = "chat_logs"
    os.makedirs(log_folder, exist_ok=True)
    filename = os.path.join(log_folder, f"session_{session_id}.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# --------------------------
# Chat Input
# --------------------------
user_input = st.chat_input("Ask something about clinical trials...")

if user_input:
    try:
        with st.spinner("Querying the backend model..."):
            res = requests.post(
                "http://localhost:5000/v1/predict_trial",
                json={"query": user_input, "session_id": session_id}
            )
            data = res.json()
            if res.status_code != 200:
                answer = f"❌ Error: {data.get('error', 'Unknown error')}"
                st.session_state.chat_history.append((user_input, answer, []))
                log_conversation(user_input, answer, session_id)
            else:
                answer = data.get("prediction_explanation", "")
                sources = data.get("source_documents_retrieved", [])
                st.session_state.chat_history.append((user_input, answer, sources))
                log_conversation(user_input, answer, session_id)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Flask API at http://localhost:5000")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# --------------------------
# Display Conversation
# --------------------------
for entry in st.session_state.chat_history:
    q, a = entry[0], entry[1]
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        if len(entry) > 2 and entry[2]:
            with st.expander("📄 Retrieved Context"):
                for i, doc in enumerate(entry[2]):
                    st.markdown(f"**Source {i+1}** — Metadata: `{doc.get('metadata', {})}`")
                    st.code(doc.get("page_content", "")[:1500])  # truncate if needed
