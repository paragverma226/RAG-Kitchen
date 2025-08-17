# indexer.py
import os
import yaml
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from utils import load_all_files_from_dir
from tqdm.auto import tqdm

# indexer.py
from langchain_community.embeddings import HuggingFaceEmbeddings  # instead of OpenAIEmbeddings




load_dotenv()
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def create_or_update_index(config):
    data_path = config.get("data_path", "data")
    faiss_path = config["faiss"]["index_path"]
    chunk_cfg = config["chunk"]

    print("Loading documents from", data_path)
    raw_docs = load_all_files_from_dir(data_path)
    docs = []
    for item in raw_docs:
        # meta includes source file path
        docs.append(Document(page_content=item["text"], metadata={"source": item["path"]}))

    # chunker
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["size"],
        chunk_overlap=chunk_cfg["overlap"]
    )
    print("Splitting documents into chunks...")
    splitted = text_splitter.split_documents(docs)

    # # embeddings
    # print("Creating embeddings (OpenAI)...")
    # embeddings = OpenAIEmbeddings()

    # embeddings
    print("Creating embeddings (HuggingFace Sentence-Transformers)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
    # model_name can be changed, e.g., "all-mpnet-base-v2"

    # Build FAISS index (overwrites if exists)
    if not os.path.exists(faiss_path):
        os.makedirs(faiss_path, exist_ok=True)

    print(f"Creating FAISS index at {faiss_path} (this may take a while)...")
    vectorstore = FAISS.from_documents(splitted, embeddings)
    vectorstore.save_local(faiss_path)
    print("Index saved.")
    return vectorstore

if __name__ == "__main__":
    cfg = load_config()
    create_or_update_index(cfg)
