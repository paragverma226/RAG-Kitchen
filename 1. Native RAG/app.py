# streamlit_app.py
import os
import yaml
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# Load config + index
cfg = load_config()
faiss_path = cfg["faiss"]["index_path"]

st.set_page_config(page_title="Naive RAG", layout="wide")
st.title("📚 Naive RAG with Streamlit")

# Sidebar configs
with st.sidebar:
    st.header("⚙️ Settings")
    llm_model = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    k = st.slider("Top K Docs", 1, 10, 4)

# Embeddings + FAISS
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # 🔹 ensures no GPU meta-tensor issue
)

st.info("🔄 Loading FAISS index...")
vectorstore = FAISS.load_local(
    faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": k})

llm = ChatOpenAI(model=llm_model, temperature=0.0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# =========================
# 🔹 Chat History Handling
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"query":..., "answer":..., "sources":...}

# User query input
query = st.text_area("💬 Ask a question:", placeholder="e.g., Explain quantum entanglement in simple terms.")

if st.button("Run Query") and query.strip():
    with st.spinner("Fetching answer..."):
        result = qa_chain({"query": query})

    # Save to history
    st.session_state.chat_history.append({
        "query": query,
        "answer": result["result"],
        "sources": result.get("source_documents", [])
    })

# =========================
# 🔹 Show Chat History
# =========================
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for i, chat in enumerate(st.session_state.chat_history[::-1], start=1):  # latest on top
        with st.expander(f"Q{i}: {chat['query']}"):
            st.markdown(f"**📝 Answer:** {chat['answer']}")
            st.markdown("**📂 Sources:**")
            for doc in chat["sources"]:
                with st.expander(doc.metadata.get("source", "Unknown Source")):
                    st.write(doc.page_content[:1000])

