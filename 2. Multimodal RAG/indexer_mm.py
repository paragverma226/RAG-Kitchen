# indexer_mm.py
import os, json, yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

import numpy as np
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain.schema import Document

from sentence_transformers import SentenceTransformer
from PIL import Image

from utils import load_text_files, load_image_files

load_dotenv()
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def build_text_index(cfg):
    text_docs = load_text_files(cfg["data_path"])
    if not text_docs:
        print("No text docs found.")
        return
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk"]["size"],
        chunk_overlap=cfg["chunk"]["overlap"]
    )
    # Build LangChain Documents
    docs = [Document(page_content=d["text"], metadata={"source": d["path"], "modality":"text"})
            for d in text_docs]
    chunks = splitter.split_documents(docs)
    print(f"Text chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=cfg["models"]["text_embedding"])
    os.makedirs(cfg["faiss"]["text_index_path"], exist_ok=True)
    vs = LCFAISS.from_documents(chunks, embeddings)
    vs.save_local(cfg["faiss"]["text_index_path"])
    print("Saved text FAISS:", cfg["faiss"]["text_index_path"])

def build_image_index(cfg):
    image_items = load_image_files(cfg["data_path"])
    if not image_items:
        print("No images found.")
        return

    model_name = cfg["models"]["image_embedding"]
    clip = SentenceTransformer(model_name)

    vectors = []
    meta = []

    for it in tqdm(image_items, desc="Embedding images (CLIP)"):
        try:
            img = Image.open(it["path"]).convert("RGB")
            # SentenceTransformers CLIP can encode images directly
            vec = clip.encode(img, convert_to_numpy=True, normalize_embeddings=True)
            vectors.append(vec)
            meta.append({
                "source": it["path"],
                "ocr_text": it["ocr_text"]
            })
        except Exception as e:
            print("Image failed:", it["path"], e)

    if not vectors:
        print("No image vectors produced.")
        return

    mat = np.vstack(vectors).astype("float32")
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors (dot product)
    index.add(mat)

    out_dir = cfg["faiss"]["image_index_path"]
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved image FAISS + meta:", out_dir)

if __name__ == "__main__":
    cfg = load_config()
    build_text_index(cfg)
    build_image_index(cfg)
