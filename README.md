# Pro-Level RAG Assistant using Mistral & Streamlit

A **production-grade Retrieval-Augmented Generation (RAG) application** built using **Mistral LLMs**, **hybrid retrieval (FAISS + BM25)**, and **Streamlit**.
This project follows real-world AI engineering practices with a strong focus on **reliability, explainability, and hallucination prevention**.

---

## 🚀 Features

- 📄 Upload and query **PDF / TXT documents**
- 🔍 **Hybrid Retrieval**
  - Semantic search using **FAISS**
  - Keyword search using **BM25**
- 🧠 **Mistral LLM (API-based)** for grounded answers
- 💬 Interactive **Streamlit chat UI**
- 📚 Source-aware responses (shows retrieved document chunks)
- 🛡️ Hallucination-safe design (responds with *"I don’t know"* when context is insufficient)
- 🏗️ Explicit RAG orchestration (no fragile LangChain chains)

---

## 🧠 Architecture

```
User Query
   ↓
BM25 Keyword Search
   ↓
FAISS Vector Search
   ↓
Hybrid Merge & Deduplication
   ↓
Context Construction
   ↓
Mistral LLM
   ↓
Answer + Sources
```

---

## ⚙️ Tech Stack

- Python 3.10+
- Mistral AI (LLM & Embeddings via API)
- FAISS (CPU-based vector search)
- BM25 (rank-bm25)
- LangChain (core & community modules)
- Streamlit
- PyPDF

---

## 📂 Project Structure

```
rag-pro-app/
│
├── app.py              # Streamlit UI
├── rag_engine.py       # Core RAG logic
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🛠️ Installation

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Set Mistral API Key

**Linux / macOS**
```bash
export MISTRAL_API_KEY=your_api_key_here
```

**Windows**
```bash
set MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Upload a PDF or TXT file and start asking questions.

---

## ❓ Why Does the App Say "I Don’t Know" Sometimes?

This behavior is **intentional**.

The system is designed to:
- Answer **only using retrieved document context**
- Avoid hallucinations or guessing
- Respond with *"I don’t know"* if the document does not contain enough information

This is a **best practice in enterprise RAG systems**.



