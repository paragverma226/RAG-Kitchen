# app_mm.py
import os, json, yaml
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import faiss

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from sentence_transformers import SentenceTransformer

load_dotenv()
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

# ---- TEXT retriever (LangChain+FAISS)
text_embeddings = HuggingFaceEmbeddings(model_name=cfg["models"]["text_embedding"])
text_vs = LCFAISS.load_local(
    cfg["faiss"]["text_index_path"],
    text_embeddings,
    allow_dangerous_deserialization=True
)
text_retriever = text_vs.as_retriever(search_kwargs={"k": 4})

# ---- IMAGE retriever (CLIP + raw FAISS)
clip = SentenceTransformer(cfg["models"]["image_embedding"])
img_index_path = os.path.join(cfg["faiss"]["image_index_path"], "index.faiss")
img_meta_path = os.path.join(cfg["faiss"]["image_index_path"], "meta.json")
img_index = faiss.read_index(img_index_path) if os.path.exists(img_index_path) else None
img_meta = json.load(open(img_meta_path, "r", encoding="utf-8")) if os.path.exists(img_meta_path) else []

# ---- LLM
llm = ChatOpenAI(model=cfg["openai"]["model"], temperature=0.0)

class QueryReq(BaseModel):
    query: str
    k_text: int = 4
    k_image: int = 4

app = FastAPI(title="Multimodal RAG API")

def search_images(query: str, k: int = 4) -> List[Dict]:
    if img_index is None or not img_meta:
        return []
    # CLIP supports text→image by encoding text using the same model
    qv = clip.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = img_index.search(qv, k)
    results = []
    for i, score in zip(I[0], D[0]):
        if i < 0: 
            continue
        m = img_meta[i]
        results.append({"source": m["source"], "score": float(score), "ocr_text": m.get("ocr_text","")})
    return results

def synthesize_answer(query: str, text_docs, image_hits):
    # Build a compact context with top text chunks + OCR snippets
    text_context = "\n\n".join([f"[TEXT:{i}] {d.page_content}" for i, d in enumerate(text_docs, 1)])
    ocr_context = "\n\n".join([f"[IMG:{i}] {ih['ocr_text']}".strip() for i, ih in enumerate(image_hits, 1) if ih.get("ocr_text")])
    src_list = [d.metadata.get("source") for d in text_docs] + [ih["source"] for ih in image_hits]

    prompt = f"""You are a helpful assistant.
Use the provided multimodal context (text chunks + OCR from images) to answer the question.
Cite filenames inline like [source].

Question: {query}

--- TEXT CONTEXT ---
{text_context if text_context else "(none)"}

--- IMAGE OCR CONTEXT ---
{ocr_context if ocr_context else "(none)"}

Provide a precise, concise answer. If unsure, say you don't know.
"""
    # Use a simple LLM call via RetrievalQA (we'll inject text via a fake Document)
    # Easiest: directly call llm with the prompt.
    ans = llm.predict(prompt)
    return ans, src_list

@app.post("/query")
async def query(req: QueryReq):
    # text retrieval
    tr = text_vs.as_retriever(search_kwargs={"k": req.k_text})
    text_docs = tr.get_relevant_documents(req.query)

    # image retrieval
    image_hits = search_images(req.query, k=req.k_image)

    # synthesize
    answer, sources = synthesize_answer(req.query, text_docs, image_hits)
    return {
        "answer": answer,
        "text_sources": [
            {"source": d.metadata.get("source"), "preview": d.page_content[:400]} for d in text_docs
        ],
        "image_sources": image_hits
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    elif args.query:
        # quick CLI
        text_docs = text_retriever.get_relevant_documents(args.query)
        image_hits = search_images(args.query, k=4)
        ans, srcs = synthesize_answer(args.query, text_docs, image_hits)
        print("ANSWER:\n", ans, "\n")
        print("TEXT SOURCES:")
        for d in text_docs: print("-", d.metadata.get("source"))
        print("\nIMAGE SOURCES:")
        for ih in image_hits: print("-", ih["source"], f"(score={ih['score']:.3f})")
    else:
        print("Use --serve or --query.")
