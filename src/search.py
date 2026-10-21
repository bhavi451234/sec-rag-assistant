"""Search the chunks: embed a question and find the closest chunk vectors."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embed_chunks import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, load_chunks

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUESTIONS = [
    ("What are Tesla's main risk factors?", "TSLA"),
    ("What legal proceedings is Snap involved in?", "SNAP"),
    ("What did Microsoft say about artificial intelligence risks?", "MSFT"),
    ("How much did Exxon spend on capital expenditures?", "XOM"),
]


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Put all chunk vectors into a FAISS index that can find the closest ones."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(
    question: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    k: int = 5,
    ticker: str | None = None,
    min_chars: int = 150,
) -> list[dict]:
    """Return the k chunks closest in meaning to the question, each with a score.

    ticker: only keep chunks of this company. min_chars: skip tiny chunks (table-of-contents entries).
    """
    query = model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
    scores, ids = index.search(query, index.ntotal)

    results = []
    for score, chunk_number in zip(scores[0], ids[0]):
        chunk = chunks[chunk_number]
        if ticker and chunk["ticker"] != ticker:
            continue
        if len(chunk["text"]) < min_chars:
            continue
        results.append(dict(chunk, score=float(score)))
        if len(results) == k:
            break
    return results


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    for question, ticker in QUESTIONS:
        print("\nQ:", question)
        for result in search(question, model, index, chunks, k=3, ticker=ticker):
            preview = result["text"][:150].replace("\n", " ")
            print(f"  {result['score']:.2f} {result['ticker']} {result['section']}: {preview}")