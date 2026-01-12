import streamlit as st
import os
from rag_engine import build_index, ask_pro
from langchain_mistralai.chat_models import ChatMistralAI

st.set_page_config(page_title="Pro RAG Assistant", layout="wide")

st.title(" Pro RAG Assistant")
st.caption("Hybrid Retrieval • Verified Answers • AI Engineer Grade")

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = st.text_input("Enter your Mistral API Key", type="password")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

uploaded_files = st.file_uploader("Upload PDF / TXT files", accept_multiple_files=True)

if uploaded_files and not st.session_state.rag_ready:
    with st.spinner("Indexing documents..."):
        chunks, retriever, bm25 = build_index(uploaded_files)
        st.session_state.chunks = chunks
        st.session_state.retriever = retriever
        st.session_state.bm25 = bm25
        st.session_state.llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
        st.session_state.rag_ready = True
    st.success("Documents indexed successfully")

if st.session_state.rag_ready:
    question = st.text_input("Ask a question")
    if st.button("Ask"):
        answer, sources = ask_pro(
            question,
            st.session_state.chunks,
            st.session_state.retriever,
            st.session_state.bm25,
            st.session_state.llm,
            st.session_state.chat_history
        )
        st.session_state.chat_history.append(f"Q: {question}\nA: {answer}")
        st.markdown("###  Answer")
        st.write(answer)
        