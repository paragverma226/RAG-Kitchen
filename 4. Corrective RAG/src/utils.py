import os
from typing import List, Tuple
from pypdf import PdfReader
from scripts.chunker import chunk_text

SUPPORTED = (".txt", ".md")

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)

def load_documents_from_folder(folder: str) -> List[Tuple[str,str]]:
    """Return list of tuples (source_path, content)."""
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            try:
                if ext in SUPPORTED:
                    text = read_text_file(path)
                elif ext == ".pdf":
                    text = read_pdf(path)
                else:
                    continue
                docs.append((path, text))
            except Exception as e:
                print(f"Failed to read {path}: {e}")
    return docs

def chunk_documents(docs: List[Tuple[str,str]], chunk_size: int, overlap: int):
    """Return list of dicts with 'page_content' and 'metadata'."""
    out = []
    for path, text in docs:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            out.append({
                "page_content": c,
                "metadata": {"source": path, "chunk": i}
            })
    return out