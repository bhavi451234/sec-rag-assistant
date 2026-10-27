"""Look at what the model saw for the Exxon capex question."""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer import retrieve
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks

QUESTION = "How much did Exxon spend on capital expenditures in 2025?"
OUT_PATH = Path("data/eval/capex_debug.txt")

def show_around(text, needle, width=250):
    """Return the text around the first place needle appears."""
    position = text.index(needle)
    start = max(0, position - width)
    return text[start : position + width]

def main():
    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    results = retrieve(QUESTION, embedder, index, chunks, k=8)
    lines = []
    for number, result in enumerate(results, start=1):
        text = result["text"]
        lines.append(f"[{number}] {result['chunk_id']} score {result['score']:.2f}")
        for needle in ["3,053", "Capex in 2025 was"]:
            if needle in text:
                lines.append(f"   contains {needle}:")
                lines.append(show_around(text, needle))
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT_PATH)

if __name__ == "__main__":
    main()