"""Run the questions in a text file through the assistant and save every answer for review."""

import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.answer import answer
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks

QUESTIONS_PATH = Path("data/eval/questions.txt")
LOG_PATH = Path("data/eval/answers_log.txt")


def main() -> None:
    lines = QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    questions = [line.strip() for line in lines if line.strip()]

    embedder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    with LOG_PATH.open("w", encoding="utf-8") as log:
        for number, question in enumerate(questions, start=1):
            result = answer(question, embedder, index, chunks)
            log.write(f"Q{number}: {question}\n{result['answer']}\n")
            for position, source in enumerate(result["sources"], start=1):
                log.write(f"  [{position}] {source['ticker']} {source['section']} (score {source['score']:.2f})\n")
            log.write("\n")
            print("done", number, "of", len(questions))
            time.sleep(4)


if __name__ == "__main__":
    main()