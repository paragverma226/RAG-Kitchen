import streamlit as st
import yaml
from src.pipeline import CorrectiveRAG
from dotenv import load_dotenv

load_dotenv()  # load .env file

st.set_page_config(page_title="Corrective RAG", layout="wide")
st.title("🔧 Corrective RAG — Streamlit Demo")
st.markdown("Ask questions against ingested documents. The pipeline will validate retrieved docs and re-retrieve if necessary.")

cfg = yaml.safe_load(open("config.yaml"))
rag = CorrectiveRAG("config.yaml")

# Sidebar
st.sidebar.header("Settings")
llm = st.sidebar.selectbox("Answer LLM (for QA)", [cfg["models"]["final_llm"]], index=0)
k = st.sidebar.slider("Top-K", 1, 10, cfg["retrieval"]["k"])
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []

# Chat input
query = st.text_input("Ask a question about the documents:")
if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running Corrective RAG..."):
            ans = rag.answer(query, k=k)
        st.session_state.chat_history.append({"query": query, "answer": ans})

# Display history
for item in st.session_state.chat_history:
    st.markdown(f"**User:** {item['query']}")
    st.markdown(f"**Assistant:** {item['answer']}")

st.sidebar.info("Pipeline: HyDE → Retrieval → Evaluation → Corrective Retrieval → QA")