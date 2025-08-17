"""
Streamlit App: HyDE RAG
1) Generate hypothetical answer for query (HyDE).
2) Embed hypothetical answer (free HuggingFace embeddings).
3) Retrieve top-k chunks by FAISS similarity.
4) Synthesize final answer with LLM and show sources.
"""

import os
import yaml
import streamlit as st
from dotenv import load_dotenv
from typing import List

# LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# -------------------
# CONFIG LOADING
# -------------------
load_dotenv()
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# -------------------
# HELPERS
# -------------------
def get_llm(spec: str):
    """spec format: 'openai:gpt-4o' or 'ollama:llama3'"""
    prov, model = spec.split(":", 1)
    if prov == "openai":
        return ChatOpenAI(model=model, temperature=0.0)
    elif prov == "ollama":
        return ChatOllama(model=model, temperature=0.0)
    else:
        raise ValueError(f"Unknown LLM provider spec: {spec}")

def make_final_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """You are a precise assistant. Use the provided context to answer the question.
If the answer isn't fully supported by the context, say you don't know.

Question:
{question}

Context:
{context}

Answer in a concise, well-structured paragraph.
Cite source filenames in-line like [source].
"""
    )

def hyde_generate(llm, query: str) -> str:
    return llm.predict(
        f"Write a detailed, factual answer to the following question. "
        f"Keep it strictly on-topic and grounded in general knowledge.\n\nQuestion: {query}"
    )

def embed_and_retrieve(embeddings, vs: FAISS, hypothetical_doc: str, k: int) -> List[Document]:
    vec = embeddings.embed_query(hypothetical_doc)
    return vs.similarity_search_by_vector(vec, k=k)

def synthesize_answer(llm, question: str, docs: List[Document]) -> str:
    context = "\n\n".join([
        f"{i+1}. ({os.path.basename(d.metadata.get('source',''))}) {d.page_content}"
        for i, d in enumerate(docs)
    ])
    prompt = make_final_prompt().format(
        question=question, 
        context=context if context else "(no context)"
    )
    return llm.predict(prompt)

# -------------------
# PIPELINE
# -------------------
@st.cache_resource(show_spinner=False)
def build_pipeline():
    cfg = load_config()
    embeddings = HuggingFaceEmbeddings(model_name=cfg["models"]["text_embedding"])
    vs = FAISS.load_local(
        cfg["faiss"]["index_path"],
        embeddings,
        allow_dangerous_deserialization=True
    )
    return cfg, embeddings, vs

# -------------------
# STREAMLIT APP
# -------------------
def main():
    st.set_page_config(page_title="HyDE RAG", layout="wide")

    st.title("🔎 HyDE RAG with Streamlit")
    st.markdown("**HyDE RAG Pipeline:** Hypothetical Document → Embedding → Retrieval → Final Answer")

    # Load pipeline
    cfg, embeddings, vs = build_pipeline()

    # Sidebar Settings
    st.sidebar.header("⚙️ Settings")

    final_llm_spec = st.sidebar.selectbox(
        "Answer LLM",
        ["openai:gpt-4o", "openai:gpt-4o-mini", "ollama:llama3"],
        index=0 if cfg["final_llm"].startswith("openai") else 2
    )

    k = st.sidebar.slider("Top-K Documents", 1, 10, cfg["retrieval"]["k"])

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Input
    query = st.chat_input("💬 Ask a question about your documents...")

    if query:
        with st.spinner("Generating answer..."):
            hyde_llm = get_llm(cfg["hyde_llm"])
            final_llm = get_llm(final_llm_spec)

            # HyDE generation
            hypo = hyde_generate(hyde_llm, query)

            # Retrieval
            docs = embed_and_retrieve(embeddings, vs, hypo, k=k)

            # Final synthesis
            answer = synthesize_answer(final_llm, query, docs)

        # Store in chat history
        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "sources": [d.metadata.get("source") for d in docs]
        })

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            with st.expander("📂 Sources"):
                for s in chat["sources"]:
                    st.write("-", os.path.basename(s) if s else "Unknown")

    st.sidebar.markdown("---")
    st.sidebar.info("✅ OpenAI & Ollama models\n✅ HuggingFace embeddings\n✅ FAISS-powered retrieval")

if __name__ == "__main__":
    main()
