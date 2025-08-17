# src/pipeline.py
from src.retriever import FaissRetriever
from src.evaluator import SimpleEvaluator
from src.qa_chain import QAChain
from langchain_openai import ChatOpenAI

class CorrectiveRAG:
    def __init__(self, config_path: str = "config.yaml"):
        self.retriever = FaissRetriever(config_path)
        self.evaluator = SimpleEvaluator()
        cfg = __import__("yaml").safe_load(open(config_path))
        self.hyde_spec = cfg["models"]["hyde_llm"]
        self.final_spec = cfg["models"]["final_llm"]
        # instantiate only final QA chain when needed
        self.qa_chain = QAChain(self.final_spec)

    def hyde_generate(self, query: str) -> str:
        # HyDE: create hypothetical answer using the HyDE LLM
        prov, model = self.hyde_spec.split(":",1)
        if prov == "openai":
            hyde_llm = ChatOpenAI(model=model, temperature=0.0)
        else:
            raise ValueError("HyDE only supports openai in this simple demo")
        prompt = f"Write a detailed, factual answer to the following question:\n\nQuestion: {query}"
        return hyde_llm.predict(prompt)

    def retrieve_and_correct(self, query: str, k: int = None):
        docs = self.retriever.get_relevant_documents(query, k=k)
        graded = self.evaluator.rank_by_relevance(docs, query)
        if not self.evaluator.is_sufficient(graded):
            # corrective retrieval using an augmented query
            corrected = self.retriever.get_relevant_documents(query + " official statistics", k=k)
            return corrected
        return graded

    def answer(self, query: str, k: int = None) -> str:
        hypo = self.hyde_generate(query)
        # embed hypothetical doc and use to retrieve (simple demo: use hypo as query too)
        docs = self.retrieve_and_correct(query, k=k)
        return self.qa_chain.run(question=query, context_docs=docs)