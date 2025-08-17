# HyDE RAG (Retrieval-Augmented Generation)

Production-ready **HyDE RAG** pipeline:
- Ingest external files (**PDF, TXT, DOCX**)
- **Chunk** and embed with **free HuggingFace embeddings**
- Persist vectors in **FAISS**
- **HyDE**: Generate a hypothetical answer via LLM → embed → retrieve top-K
- Synthesize final answer with sources

## Features
- ✅ Free embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- ✅ FAISS vector store (local)
- ✅ Works with **OpenAI** or **Ollama** for HyDE + final LLM
- ✅ CLI and **FastAPI** server
- ✅ Config-driven (`config.yaml`)

## Project Structure

rag-hyde/
├─ data/ # Put your PDFs / TXTs / DOCX here
├─ faiss_index/ # Created by indexer
├─ config.yaml
├─ requirements.txt
├─ .env
├─ utils.py
├─ indexer.py
├─ app_hyde.py
└─ README.md


## Tech Stack
- **LangChain** (chains, prompts)
- **FAISS** (vector search)
- **HuggingFace sentence-transformers** (free embeddings)
- **OpenAI / Ollama** (LLMs for HyDE + synthesis)
- **PyPDF2 / python-docx** (document parsing)
- **FastAPI + Uvicorn** (optional API)
- **Python 3.10+**

## Setup

```bash
# 1) Create venv
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# - For OpenAI: add OPENAI_API_KEY
# - For Ollama: ensure `ollama serve` is running and model pulled (e.g., `ollama pull llama3`)


Configure models (config.yaml)
models:
  text_embedding: "all-MiniLM-L6-v2"
hyde_llm: "openai:gpt-4o"     # or "ollama:llama3"
final_llm: "openai:gpt-4o"    # or "ollama:llama3"
retrieval:
  k: 5

Index your data
Add files to data/ then run:

python indexer.py


Run HyDE RAG (CLI)

python app_hyde.py --query "Impact of AI in climate change"

Run API
python app_hyde.py --serve
# POST to http://localhost:8002/query
# body: {"query":"Impact of AI in climate change", "k":5}

Notes
We set allow_dangerous_deserialization=True when loading FAISS (safe only for your own indexes).

For higher retrieval quality, try all-mpnet-base-v2 (heavier) as the embeddings model.

HyDE prompts are designed to stay grounded; you can harden the prompt if your domain needs stricter factuality.

License
MIT

---

## Sample data

Create minimal files in `data/`:

- `sample.txt`  

Artificial intelligence can optimize energy consumption in data centers and smart grids. Techniques like predictive maintenance and load forecasting help reduce emissions.

- `sample.pdf`  
A single-page PDF describing AI for climate modeling, mitigation, and adaptation.

- `sample.docx`  
A short paragraph on AI-assisted renewable integration and demand response.

---

## How to run (quick)

```bash
# install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY or run Ollama

# index
python indexer.py

# query
python app_hyde.py --query "Impact of AI in climate change"

# or API
python app_hyde.py --serve


