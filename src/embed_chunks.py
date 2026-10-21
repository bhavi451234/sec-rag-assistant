"""Turn every chunk into a vector and save the vectors to disk."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
VECTORS_PATH = Path("data/index/vectors.npy")


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl: one JSON record per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    model = SentenceTransformer(MODEL_NAME)
    counts = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    too_long = sum(1 for n in counts if n > model.max_seq_length)
    print(f"{len(texts)} chunks | model reads {model.max_seq_length} tokens | too long: {too_long}")

    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    print("saved", vectors.shape, "to", VECTORS_PATH)


if __name__ == "__main__":
    main()