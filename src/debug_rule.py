"""Ask the LLM one eval question twice: with the current instructions, and without the last two rule lines."""

import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer import INSTRUCTIONS, ask_llm, build_prompt, retrieve
from src.check_eval_set import load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks


def main():
    wanted_id = sys.argv[1]
    entry = next(e for e in load_eval() if e["id"] == wanted_id)

    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    results = retrieve(entry["question"], embedder, index, chunks, k=8)
    prompt = build_prompt(entry["question"], results)

    short_instructions = "\n".join(INSTRUCTIONS.split("\n")[:-2])
    prompt_without_rules = prompt.replace(INSTRUCTIONS, short_instructions, 1)
    assert prompt_without_rules != prompt, "the replacement did not change the prompt"

    print("\nQUESTION:", entry["question"])
    print("WITH RULES:   ", ask_llm(prompt))
    print("WITHOUT RULES:", ask_llm(prompt_without_rules))


if __name__ == "__main__":
    main()