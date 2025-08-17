# indexer.py
import os
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import load_all_files_from_dir

CONFIG_PATH = "config.yaml"
load_dotenv()

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    data_path = cfg.get("data_path", "data")
    index_path = cfg["faiss"]["index_path"]
    chunk_cfg = cfg["chunk"]

    print(f"[Indexer] Loading docs from: {data_path}")
    file_docs = load_all_files_from_dir(data_path)
    if not file_docs:
        print("[Indexer] No documents found in data/.")
        return

    docs = [Document(page_content=d["text"], metadata={"source": d["path"]}) for d in file_docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["size"],
        chunk_overlap=chunk_cfg["overlap"]
    )
    print("[Indexer] Splitting into chunks…")
    chunks = splitter.split_documents(docs)
    print(f"[Indexer] Total chunks: {len(chunks)}")

    emb_model = cfg["models"]["text_embedding"]
    print(f"[Indexer] Building embeddings with: {emb_model} (free)")
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)

    os.makedirs(index_path, exist_ok=True)
    print(f"[Indexer] Creating FAISS at {index_path} …")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(index_path)
    print("[Indexer] Done.")

if __name__ == "__main__":
    main()
