import re

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.answer import (
    CHUNKS_PATH,
    COMPANY_ALIASES,
    MODEL_NAME,
    RERANKER_NAME,
    VECTORS_PATH,
    answer,
    build_index,
    load_chunks,
)

load_dotenv()

EXAMPLES = {
    "Apple net sales": "What was Apple's total net sales in fiscal 2025?",
    "Apple vs Exxon": "Which had higher total revenue in its latest fiscal year, Apple or ExxonMobil, and what were the two figures in millions of dollars?",
    "Snap legal": "What legal proceedings is Snap involved in?",
}

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "SNAP": "Snap",
    "JPM": "JPMorgan Chase",
    "XOM": "ExxonMobil",
    "JNJ": "Johnson & Johnson",
}


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_pipeline():
    embedder = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))
    return embedder, reranker, chunks, index


def use_example(text):
    st.session_state["question"] = text
    st.session_state["run_now"] = True


st.title("SEC 10-K assistant")
embedder, reranker, chunks, index = load_pipeline()

st.write("Try an example:")
for label, text in EXAMPLES.items():
    st.button(label, on_click=use_example, args=(text,))

selected = st.sidebar.multiselect(
    "Companies to search",
    list(COMPANY_ALIASES),
    format_func=lambda ticker: f"{COMPANY_NAMES[ticker]} ({ticker})",
    help="Leave empty to detect the companies from your question.",
)

question = st.text_input("Ask a question about the filings", key="question")
if (st.button("Ask") or st.session_state.pop("run_now", False)) and question.strip():
    try:
        with st.spinner("Searching the filings..."):
            result = answer(
                question, embedder, index, chunks,
                reranker=reranker, tickers=selected or None,
            )
    except RuntimeError:
        st.error(
            "The language model is not available right now. "
            "The free daily limit may be used up, so please try again later."
        )
        st.stop()
    except Exception as error:
        print("answer() failed:", repr(error))
        st.error("Something went wrong while answering. Please try again.")
        st.stop()
    st.write(result["answer"].replace("$", "\\$"))
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    if cited:
        st.subheader("Sources")
    else:
        st.subheader("Passages searched")
        st.caption("The answer cites none of these.")
    for number, source in enumerate(result["sources"], start=1):
        mark = "✓ cited · " if number in cited else ""
        label = f"{mark}[{number}] {source['ticker']} · {source['section']} · period {source['period']}"
        with st.expander(label, expanded=number in cited):
            st.text(source["text"])
            if source.get("source_url"):
                st.markdown(f"[Open the filing on SEC.gov]({source['source_url']})")