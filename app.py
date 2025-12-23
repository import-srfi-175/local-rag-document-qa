# chromadb persists embeddings. don't want to re-embed on every question
# embed once per uploaded pdf, then store vectors, reuse those same vectors for semantic search and retrieval task

import streamlit as st
import os
import shutil

from src.ingest import load_pdf_text
from src.chunking import split_text_into_chunks
from src.vectorstore import VectorStore
from src.rag import RAGPipeline

# App title
st.set_page_config(page_title="Local Document Q&A", layout="centered")
st.title("Local Document Q&A Assistant")

# Session state to track indexing
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully")

    # Index document (only once)
    if not st.session_state.indexed:
        with st.spinner("Processing and indexing document..."):
            # Optional: clear old DB
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            text = load_pdf_text(pdf_path)
            chunks = split_text_into_chunks(text)

            store = VectorStore()
            store.add_chunks(chunks)

            st.session_state.indexed = True

        st.success("Document indexed successfully!")

# Question answering
if st.session_state.indexed:
    question = st.text_input("Ask a question about the document")

    if question:
        rag = RAGPipeline()

        with st.spinner("Generating answer..."):
            answer = rag.answer(question)

        st.subheader("Answer")
        st.write(answer)
