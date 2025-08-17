from typing import List

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Simple whitespace-based chunker with overlap."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks