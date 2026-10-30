"""Retrieval score: how often does a retrieved chunk contain the evidence phrase?"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer import retrieve
from src.check_eval_set import load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks

def hit_rank(entry, results):
    """Return the worst per-company position of the evidence, or None if any company's evidence is missing."""
    evidence = entry["evidence"]
    if isinstance(evidence, str):
        evidence = {entry["ticker"]: evidence}

    worst = 0
    for ticker, phrase in evidence.items():
        company_results = [r for r in results if r["ticker"] == ticker]
        rank = next(
            (position for position, r in enumerate(company_results, start=1) if phrase in r["text"]),
            None,
        )
        if rank is None:
            return None
        worst = max(worst, rank)
    return worst

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