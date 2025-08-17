# scripts/build_index.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils import load_documents_from_folder, chunk_documents


CONFIG = "config.yaml"

if __name__ == "__main__":
    cfg = yaml.safe_load(open(CONFIG))
    chunk_size = cfg["ingest"]["chunk_size"]
    overlap = cfg["ingest"]["chunk_overlap"]

    docs = load_documents_from_folder("./data")
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)

    texts = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name=cfg["models"]["text_embedding"])
    faiss_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    idx_path = cfg["faiss"]["index_path"]
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    faiss_index.save_local(idx_path)
    print("Index built and saved to", idx_path)