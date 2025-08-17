# src/qa_chain.py
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List
import os

PROMPT = PromptTemplate.from_template(
    """You are a precise assistant. Use the provided context to answer the question.
If the answer isn't fully supported by the context, say you don't know.

Question:
{question}

Context:
{context}

Answer in a concise, well-structured paragraph.
Cite source filenames in-line like [source].
"""
)

class QAChain:
    def __init__(self, llm_spec: str = "openai:gpt-4o-mini"):
        prov, model = llm_spec.split(":",1)
        if prov == "openai":
            self.llm = ChatOpenAI(model=model, temperature=0.0)
        else:
            raise ValueError("Only openai supported in this simple QA chain")

    def run(self, question: str, context_docs: List[dict]) -> str:
        context = "\n\n".join([
            f"({os.path.basename(d.metadata.get('source',''))}) {d.page_content[:1000]}"
            for d in context_docs
        ])
        prompt = PROMPT.format(question=question, context=context if context else "(no context)")
        return self.llm.predict(prompt)