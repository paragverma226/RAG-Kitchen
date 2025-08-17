# utils.py
import os
from typing import List, Dict
from PyPDF2 import PdfReader
import docx

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            text.append(t)
    return "\n".join(text)

def read_docx(path: str) -> str:
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)

def load_all_files_from_dir(directory: str, allowed_ext=("txt","pdf","docx")) -> List[Dict]:
    docs: List[Dict] = []
    for root, _, files in os.walk(directory):
        for fn in files:
            ext = fn.split(".")[-1].lower()
            if ext not in allowed_ext:
                continue
            path = os.path.join(root, fn)
            if ext == "txt":
                text = read_txt(path)
            elif ext == "pdf":
                text = read_pdf(path)
            elif ext == "docx":
                text = read_docx(path)
            else:
                text = ""
            if text and text.strip():
                docs.append({"path": path, "text": text})
    return docs
