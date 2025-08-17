# streamlit_app_mm.py
import os, json, yaml
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import faiss
from PIL import Image

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain.chat_models import ChatOpenAI

from sentence_transformers import SentenceTransformer

load_dotenv()
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("🖼️📚 Multimodal RAG (Text + Images) — Free Embeddings")

cfg = load_config()

# sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    k_text = st.slider("Top-K Text", 1, 10, 4)
    k_image = st.slider("Top-K Images", 1, 10, 4)
    # llm_model = st.selectbox("LLM", [cfg["openai"]["model"], "gpt-4o", "gpt-4o-mini"], index=0)
    llm_model = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini"])
    st.caption("Embeddings: Text=all-MiniLM-L6-v2, Image=clip-ViT-B-32")

# load text index
st.info("Loading text index…")
txt_embed = HuggingFaceEmbeddings(model_name=cfg["models"]["text_embedding"])
text_vs = LCFAISS.load_local(
    cfg["faiss"]["text_index_path"], 
    txt_embed, 
    allow_dangerous_deserialization=True
)

# load image index
st.info("Loading image index…")
img_index_file = os.path.join(cfg["faiss"]["image_index_path"], "index.faiss")
img_meta_file = os.path.join(cfg["faiss"]["image_index_path"], "meta.json")
if os.path.exists(img_index_file) and os.path.exists(img_meta_file):
    img_index = faiss.read_index(img_index_file)
    img_meta = json.load(open(img_meta_file, "r", encoding="utf-8"))
else:
    img_index, img_meta = None, []

clip = SentenceTransformer(cfg["models"]["image_embedding"])
llm = ChatOpenAI(model=llm_model, temperature=0.0)

def search_images(query, k):
    if img_index is None or not img_meta:
        return []
    qv = clip.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = img_index.search(qv, k)
    out = []
    for i, score in zip(I[0], D[0]):
        if i < 0: continue
        m = img_meta[i]
        out.append({"source": m["source"], "score": float(score), "ocr_text": (m.get("ocr_text") or "").strip()})
    return out

query = st.text_area("💬 Ask a question", placeholder="e.g., Show me Einstein’s handwritten notes and explain them.")

if st.button("Run"):
    with st.spinner("Retrieving…"):
        text_docs = text_vs.as_retriever(search_kwargs={"k": k_text}).get_relevant_documents(query)
        img_hits = search_images(query, k_image)

    text_ctx = "\n\n".join([f"[TEXT:{i+1}] {d.page_content}" for i, d in enumerate(text_docs)])
    ocr_ctx = "\n\n".join([f"[IMG:{i+1}] {h['ocr_text']}" for i, h in enumerate(img_hits) if h.get("ocr_text")])

    prompt = f"""You are a helpful assistant.
Use both the TEXT and IMAGE OCR context to answer the user's question.
Cite sources by filename when relevant.

Question: {query}

--- TEXT CONTEXT ---
{text_ctx if text_ctx else "(none)"}

--- IMAGE OCR CONTEXT ---
{ocr_ctx if ocr_ctx else "(none)"}
"""
    with st.spinner("Synthesizing…"):
        answer = llm.predict(prompt)

    st.subheader("📝 Answer")
    st.write(answer)

    st.subheader("📂 Sources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Text Chunks**")
        for d in text_docs:
            with st.expander(d.metadata.get("source","(unknown)")):
                st.write(d.page_content[:1200])
    with col2:
        st.markdown("**Images**")
        for h in img_hits:
            st.caption(f"{h['source']} (score={h['score']:.3f})")
            try:
                st.image(Image.open(h["source"]), use_column_width=True)
            except Exception:
                st.write("Preview unavailable")
            if h.get("ocr_text"):
                with st.expander("OCR text"):
                    st.write(h["ocr_text"][:1200])


