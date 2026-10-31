"""Experiment: does a cross-encoder reranker move the evidence chunk up the ranking?"""

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.check_eval_set import evidence_pairs, load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks, search

RERANKER_NAME = "BAAI/bge-reranker-base"
CANDIDATES = 60


def rank_of(results, phrase):
    """1-based position of the first result containing the phrase, or None."""
    return next(
        (position for position, r in enumerate(results, start=1) if phrase in r["text"]),
        None,
    )


def main():
    embedder = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    for entry in load_eval():
        for ticker, phrase in evidence_pairs(entry):
            candidates = search(entry["question"], embedder, index, chunks, k=CANDIDATES, ticker=ticker)
            scores = reranker.predict([(entry["question"], c["text"]) for c in candidates])
            reranked = [candidates[i] for i in np.argsort(-scores)]
            before = rank_of(candidates, phrase)
            after = rank_of(reranked, phrase)
            print(f"{entry['id']:28} {ticker:5} embedding rank: {str(before):5} after rerank: {after}")


if __name__ == "__main__":
    main()