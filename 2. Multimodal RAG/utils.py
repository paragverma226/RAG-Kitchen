# utils.py
import os
from typing import List, Dict
from PyPDF2 import PdfReader
import docx
from PIL import Image
import pytesseract

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
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs if p.text])

def read_image_ocr(path: str) -> str:
    # Ensure Tesseract is installed system-wide.
    img = Image.open(path).convert("RGB")
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def load_text_files(directory: str, exts=("txt","pdf","docx")) -> List[Dict]:
    docs = []
    for root, _, files in os.walk(directory):
        for fn in files:
            ext = fn.split(".")[-1].lower()
            if ext not in exts: 
                continue
            p = os.path.join(root, fn)
            if ext == "txt":
                text = read_txt(p)
            elif ext == "pdf":
                text = read_pdf(p)
            else:
                text = read_docx(p)
            if text.strip():
                docs.append({"type":"text","path":p,"text":text})
    return docs

def load_image_files(directory: str, exts=("png","jpg","jpeg","webp")) -> List[Dict]:
    items = []
    for root, _, files in os.walk(directory):
        for fn in files:
            ext = fn.split(".")[-1].lower()
            if ext not in exts:
                continue
            p = os.path.join(root, fn)
            ocr = read_image_ocr(p)
            items.append({"type":"image","path":p,"ocr_text":ocr})
    return items
