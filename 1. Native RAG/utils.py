# utils.py
import os
from typing import List
from PyPDF2 import PdfReader
import docx

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def read_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def load_all_files_from_dir(directory: str, allowed_ext=("txt","pdf","docx")) -> List[dict]:
    docs = []
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
            docs.append({"path": path, "text": text})
    return docs
