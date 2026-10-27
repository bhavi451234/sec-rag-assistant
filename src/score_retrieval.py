"""Retrieval score: how often does a retrieved chunk contain the evidence phrase?"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer import retrieve
from src.check_eval_set import load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks

def hit_rank(entry, results):
    """Return the 1-based position of the first chunk containing the evidence, or None."""
    for position, result in enumerate(results, start=1):
        if entry["evidence"] in result["text"]:
            return position
    return None

def main():
    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    entries = [entry for entry in load_eval() if entry["evidence"] is not None]
    hits5 = 0
    hits8 = 0
    for entry in entries:
        results = retrieve(entry["question"], embedder, index, chunks, k=8)
        rank = hit_rank(entry, results)
        print(f"{entry['id']:20} evidence found at position: {rank}")
        if rank is not None:
            hits8 += 1
            if rank <= 5:
                hits5 += 1

    total = len(entries)
    print(f"\nhit@5: {hits5}/{total} | hit@8: {hits8}/{total}")

if __name__ == "__main__":
    main()