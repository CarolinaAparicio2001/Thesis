import streamlit as st
import requests

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Clinical Trial RAG Chat", layout="wide")
st.title("💬 Clinical Trial Chatbot (Streamlit Frontend)")

# --------------------------
# Session state for history
# --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------
# Input box
# --------------------------
user_input = st.chat_input("Ask a question about the clinical trials...")

if user_input:
    try:
        with st.spinner("Querying the backend model..."):
            res = requests.post(
                "http://localhost:5000/v1/predict_trial",
                json={"query": user_input}
            )
            if res.status_code != 200:
                st.error(f"Error from backend: {res.status_code}")
                st.stop()

            data = res.json()
            answer = data.get("prediction_explanation", "No answer returned.")
            sources = data.get("source_documents_retrieved", [])

            st.session_state.chat_history.append((user_input, answer, sources))

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Flask API. Is it running on http://localhost:5000?")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# --------------------------
# Display chat history
# --------------------------
for q, a, sources in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        with st.expander("📄 Retrieved Context"):
            for doc in sources:
                st.markdown(f"**Metadata**: `{doc.get('metadata', {})}`")
                st.code(doc.get("page_content", "")[:1500])  # truncated for clarity
