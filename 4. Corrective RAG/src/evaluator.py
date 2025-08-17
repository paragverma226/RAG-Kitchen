# src/evaluator.py
from typing import List
from langchain.schema import Document

class SimpleEvaluator:
    """A naive evaluator that looks for keywords like 'official' or compares expected numeric tokens.
    This is pluggable — replace with a stronger, external evaluator if needed.
    """
    def rank_by_relevance(self, docs: List[Document], query: str) -> List[Document]:
        # naive: prefer docs whose filename or content mention 'official' or 'gov' or the entity
        def score(d):
            txt = (d.metadata.get("source","") + " " + d.page_content).lower()
            s = 0
            if "official" in txt or "gov" in txt or "government" in txt:
                s += 2
            if any(tok in txt for tok in ["gdp", "gross domestic product", "gross domestic"]):
                s += 1
            return s
        return sorted(docs, key=score, reverse=True)

    def is_sufficient(self, graded_docs: List[Document]) -> bool:
        # If the top doc has a non-zero heuristic score or mentions a number, treat as sufficient.
        if not graded_docs:
            return False
        top = graded_docs[0]
        text = (top.metadata.get("source","") + " " + top.page_content).lower()
        # simple numeric-check
        import re
        m = re.search(r"\b\d{3,}\b", text)
        return bool(m)