"""For every eval question, find how deep the evidence chunk sits in the full ranking."""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.check_eval_set import evidence_pairs, load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks, search


def main():
    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    for entry in load_eval():
        for ticker, phrase in evidence_pairs(entry):
            ranked = search(entry["question"], embedder, index, chunks, k=1000, ticker=ticker)
            rank = next(
                (position for position, r in enumerate(ranked, start=1) if phrase in r["text"]),
                None,
            )
            print(f"{entry['id']:28} {ticker:5} rank among {ticker} chunks: {rank}")


if __name__ == "__main__":
    main()