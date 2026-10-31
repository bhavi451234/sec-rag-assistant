import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.answer import (
    CHUNKS_PATH,
    MODEL_NAME,
    RERANKER_NAME,
    VECTORS_PATH,
    answer,
    build_index,
    load_chunks,
)

load_dotenv()


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_pipeline():
    embedder = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))
    return embedder, reranker, chunks, index


st.title("SEC 10-K assistant")
embedder, reranker, chunks, index = load_pipeline()

question = st.text_input("Ask a question about the filings")
if st.button("Ask") and question.strip():
    with st.spinner("Searching the filings..."):
        result = answer(question, embedder, index, chunks, reranker=reranker)
    st.write(result["answer"])
    st.subheader("Sources")
    for number, source in enumerate(result["sources"], start=1):
        st.write(f"[{number}] {source['ticker']} {source['section']}")