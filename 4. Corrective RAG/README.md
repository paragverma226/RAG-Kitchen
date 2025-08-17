# Corrective RAG — Quickstart

1. Install requirements:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Place PDFs / TXT files into ./data/

Build the FAISS index:

python scripts/build_index.py

Run the Streamlit app:

streamlit run app_streamlit.py

Notes:

Export your OpenAI API key: export OPENAI_API_KEY=sk-...

The evaluator is intentionally simple — replace with stronger grader.