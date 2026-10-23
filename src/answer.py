"""Answer a question from the filings: search, build the prompt, ask the LLM."""

import re
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks, search

load_dotenv()

LLM_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
RETRIES = 3
RETRY_CODES = (429, 500, 503)

COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "TSLA": ["tesla", "tsla"],
    "SNAP": ["snap", "snapchat"],
    "JPM": ["jpmorgan chase", "jpmorgan", "jp morgan", "jpm"],
    "XOM": ["exxonmobil", "exxon", "xom"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
}

INSTRUCTIONS = """You are a careful financial research assistant.
Answer the question using ONLY the numbered sources below, which are excerpts from SEC 10-K filings.
Cite the sources you use like [1] or [2][3] right after each claim.
If the sources do not contain the answer, say "I can't find this in the provided filings." and do not guess.
Give exact figures as written in the sources. Write plain text without markdown formatting."""


def detect_tickers(question: str) -> list[str]:
    """Return the tickers of the companies named in the question, in the order they appear."""
    text = question.lower()
    found = []
    for ticker, aliases in COMPANY_ALIASES.items():
        positions = []
        for alias in aliases:
            for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
                positions.append(match.start())
        if positions:
            found.append((min(positions), ticker))
    return [ticker for _, ticker in sorted(found)]


def format_sources(results: list[dict]) -> str:
    """Turn search results into numbered source blocks the LLM can cite."""
    blocks = []
    for number, result in enumerate(results, start=1):
        header = f"[{number}] {result['ticker']} 10-K, period {result['period']}, {result['section']}"
        blocks.append(header + "\n" + result["text"])
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[dict]) -> str:
    """Instructions + numbered sources + the question, as one prompt string."""
    return f"{INSTRUCTIONS}\n\nSOURCES:\n{format_sources(results)}\n\nQUESTION: {question}\n\nANSWER:"


def ask_llm(prompt: str) -> str:
    """Send the prompt to Gemini. If it is busy, wait and retry, then try the next model."""
    client = genai.Client()
    for llm_model in LLM_MODELS:
        for attempt in range(RETRIES):
            try:
                response = client.models.generate_content(model=llm_model, contents=prompt)
                return response.text
            except errors.APIError as error:
                if error.code not in RETRY_CODES:
                    raise
                time.sleep(5 * (attempt + 1))
    raise RuntimeError("All Gemini models are busy right now. Please try again in a minute.")


def retrieve(
    question: str, embedder: SentenceTransformer, index, chunks: list[dict], k: int = 8
) -> list[dict]:
    """Find the best chunks. Search the companies named in the question, or all of them if none is named."""
    tickers = detect_tickers(question)
    if not tickers:
        tickers = list(COMPANY_ALIASES)

    per_company = max(2, k // len(tickers))
    results = []
    for ticker in tickers:
        results += search(question, embedder, index, chunks, k=per_company, ticker=ticker)
    return results


def answer(
    question: str,
    embedder: SentenceTransformer,
    index,
    chunks: list[dict],
    k: int = 8,
    llm=ask_llm,
) -> dict:
    """Return {"answer": text, "sources": the chunks used}."""
    results = retrieve(question, embedder, index, chunks, k=k)
    if not results:
        return {"answer": "I can't find this in the provided filings.", "sources": []}
    prompt = build_prompt(question, results)
    return {"answer": llm(prompt), "sources": results}


if __name__ == "__main__":
    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    for question in [
        "What legal proceedings is Snap involved in?",
        "How do Apple and Microsoft describe their competition?",
        "Which companies warn about supply chain disruptions?",
    ]:
        result = answer(question, embedder, index, chunks)
        print("\nQ:", question)
        print(result["answer"])
        for number, source in enumerate(result["sources"], start=1):
            print(f"  [{number}] {source['ticker']} {source['section']} (score {source['score']:.2f})")