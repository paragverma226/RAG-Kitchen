# 📚 Multimodal RAG (Retrieval-Augmented Generation)

A **production-ready Multimodal RAG system** that combines **text + image understanding** to provide context-aware answers using external data sources such as **PDFs, TXT files, and images**.

This project demonstrates how to:

* Ingest multimodal documents (text + images)
* Chunk & embed content using **free HuggingFace embeddings**
* Store embeddings in **FAISS** (vector database)
* Query via **LangChain Multimodal Retrieval Chains**
* Use **GPT-4o** for natural language + multimodal reasoning

---

## 🚀 Features

✅ **Multimodal Retrieval** (text + images)
✅ **Free Embeddings** via HuggingFace (`all-MiniLM-L6-v2` for text, `clip-ViT-B-32` for images)
✅ **FAISS Vector Database** for fast similarity search
✅ **PDF, TXT, and Image Support** for knowledge ingestion
✅ **Naive RAG & Multimodal RAG pipelines**
✅ **Configurable project layout with `config.yaml`**
✅ **Production-ready modular codebase**

---

## 🏗️ Project Structure

```
multimodal-rag/
│── data/                   # External raw files (PDFs, TXTs, Images)
│── faiss_index/            # Serialized FAISS vector store
│── src/
│   ├── indexer.py          # Build multimodal index (text + images)
│   ├── rag_chain.py        # Multimodal RAG pipeline
│   ├── utils.py            # Helper functions (chunking, preprocessing, embeddings)
│   ├── config.yaml         # Configurations
│── app.py                  # Streamlit UI for querying
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
```

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **LangChain** → RAG pipeline & multimodal chain
* **FAISS** → Vector database for retrieval
* **HuggingFace Transformers** → Free embeddings (text + image)

  * `sentence-transformers/all-MiniLM-L6-v2` (text)
  * `openai/clip-vit-base-patch32` (images)
* **PyPDF2** → PDF parsing
* **Pillow (PIL)** → Image processing
* **Streamlit** → Web UI for interaction
* **dotenv / YAML** → Config management

---

## 🔧 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv rag_env
source rag_env/bin/activate   # Mac/Linux
rag_env\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Settings

Edit `src/config.yaml` to specify:

```yaml
faiss:
  index_path: "faiss_index"
chunk_size: 500
chunk_overlap: 50
```

### 5️⃣ Add Data

Place your files in `data/` folder:

* `.pdf` → research papers, docs
* `.txt` → raw text files
* `.jpg/.png` → images

### 6️⃣ Build Index

```bash
python src/indexer.py
```

### 7️⃣ Run Multimodal RAG

```bash
python src/rag_chain.py
```

### 8️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Example Queries

**Text Query**

```text
"Summarize the PDF report on climate change."
```

**Image Query**

```text
"Show me Einstein’s handwritten notes."
```

**Mixed Query**

```text
"Find the diagram from the physics textbook and explain it."
```

---

## 🛠️ Future Enhancements

* [ ] Add **Whisper** for audio-to-text RAG ingestion
* [ ] Integrate **Milvus / Weaviate** for scalable vector search
* [ ] Add **Fine-tuning adapters** for domain-specific multimodal embeddings
* [ ] Deploy with **Docker + FastAPI** backend

---

## 👨‍💻 Author

**Parag Verma** – Data Scientist | Generative AI | Multimodal RAG Enthusiast