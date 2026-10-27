"""Why does the Exxon capex question miss? Find the right chunk and measure it."""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks, search

QUESTION = "How much did Exxon spend on capital expenditures in 2025?"
NEEDLE = "Capex in 2025 was"

def main():
    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    ranked = search(QUESTION, embedder, index, chunks, k=len(chunks), ticker="XOM")
    print("best score overall:", round(ranked[0]["score"], 3))

    found = 0
    for rank, result in enumerate(ranked, start=1):
        text = result["text"]
        if NEEDLE not in text:
            continue
        found += 1
        position = text.index(NEEDLE)
        total_tokens = len(embedder.tokenizer.encode(text))
        tokens_before = len(embedder.tokenizer.encode(text[:position]))
        print(f"\nchunk {result['chunk_id']} | {result['section']}")
        print(f"  rank among XOM chunks: {rank} of {len(ranked)} | score {result['score']:.3f}")
        print(f"  chars: {len(text)} | tokens: {total_tokens} | model limit: {embedder.max_seq_length}")
        print(f"  sentence starts at char {position}, token about {tokens_before}")

    if found == 0:
        print("no chunk contains the needle")

if __name__ == "__main__":
    main()