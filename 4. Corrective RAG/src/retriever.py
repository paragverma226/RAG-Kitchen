# src/retriever.py
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class FaissRetriever:
    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(open(config_path))
        self.cfg = cfg
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg["models"]["text_embedding"])
        self.vs = FAISS.load_local(cfg["faiss"]["index_path"], self.embeddings, allow_dangerous_deserialization=True)

    def get_relevant_documents(self, query: str, k: int = None):
        if k is None:
            k = self.cfg["retrieval"]["k"]
        vec = self.embeddings.embed_query(query)
        return self.vs.similarity_search_by_vector(vec, k=k)