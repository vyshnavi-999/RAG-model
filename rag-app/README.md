# Pro RAG Assistant (Mistral)

This project is a production-grade Retrieval-Augmented Generation (RAG) system
using Mistral LLMs, FAISS, BM25 hybrid retrieval, and Streamlit.

## Features
- PDF/TXT document upload
- Hybrid retrieval (BM25 + FAISS)
- Conversational chat UI
- Source citations
- Streamlit deployment ready

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment
Set your MISTRAL_API_KEY as an environment variable or Streamlit secret.
