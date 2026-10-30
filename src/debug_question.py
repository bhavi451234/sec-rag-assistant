"""Debug one eval question: show which chunks retrieval returns for it."""

import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer import retrieve
from src.check_eval_set import load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks


def main():
    wanted_id = sys.argv[1]
    question_override = sys.argv[2] if len(sys.argv) > 2 else None
    entry = next(e for e in load_eval() if e["id"] == wanted_id)
    question = question_override or entry["question"]

    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    results = retrieve(question, embedder, index, chunks, k=8)
    print(f"\nQUESTION: {question}")
    print(f"EVIDENCE: {entry['evidence']!r}\n")
    for position, result in enumerate(results, start=1):
        has_evidence = entry["evidence"] is not None and entry["evidence"] in result["text"]
        print(f"[{position}] {result.get('chunk_id')} | {result.get('section')} | "
              f"{len(result['text'])} chars | has evidence: {has_evidence}")
        print("    " + result["text"][:200].replace("\n", " "))


if __name__ == "__main__":
    main()