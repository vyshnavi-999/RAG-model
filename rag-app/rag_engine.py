import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


# ---------------------------
# Build index from uploaded files
# ---------------------------
def build_index(files):
    docs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in files:
            file_path = os.path.join(tmpdir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue

            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(docs)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    corpus = [c.page_content for c in chunks]
    bm25 = BM25Okapi([c.split() for c in corpus])

    return chunks, retriever, bm25


# ---------------------------
# PRO-LEVEL RAG QUERY
# ---------------------------
def ask_pro(question, chunks, retriever, bm25, llm, chat_history):
    # --- BM25 ---
    bm25_scores = bm25.get_scores(question.split())
    bm25_docs = sorted(
        zip(chunks, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:6]

    # --- Vector ---
    vector_docs = retriever.invoke(question)

    # --- Merge ---
    docs = []
    seen = set()

    for d, _ in bm25_docs:
        if d.page_content not in seen:
            docs.append(d)
            seen.add(d.page_content)

    for d in vector_docs:
        if d.page_content not in seen:
            docs.append(d)
            seen.add(d.page_content)

    # --- Context ---
    context = "\n\n".join(d.page_content for d in docs[:4])

    prompt = f"""
You are an AI assistant.
Answer primarily from the context.
If the context partially answers the question, explain using it.
Only say "I don't know" if the context has no relevant information at all.


Chat history:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt).content

    return answer, docs
