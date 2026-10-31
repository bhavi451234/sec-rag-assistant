"""Answer score: is the final answer correct according to the answer key?"""

import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.answer import RERANKER_NAME, answer
from src.check_eval_set import load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks

CANT_FIND = "find this in the provided filings"
LOG_PATH = Path("data/eval/score_answers_log.txt")
RUNS = 3
USE_RERANKER = True


def judge(entry, answer_text):
    text = answer_text.lower()
    if entry["type"] == "unanswerable":
        if CANT_FIND in text:
            return True, "ok"
        return False, "should have said it can't find this"

    expected_values = entry["expected"]
    if isinstance(expected_values, str):
        expected_values = [expected_values]

    positions = []
    for expected in expected_values:
        position = text.find(expected.lower())
        if position == -1:
            return False, f"missing expected {expected!r}"
        positions.append(position)

    first_position = min(positions)
    for bad in entry["must_not_contain"]:
        bad_position = text.find(bad.lower())
        if bad_position != -1 and bad_position < first_position:
            return False, f"wrong value {bad!r} comes before the expected one"
    return True, "ok"


def main():
    embedder = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME) if USE_RERANKER else None
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    entries = load_eval()
    total_passes = 0
    log = []
    started = time.time()
    for entry in entries:
        passes = 0
        for run in range(1, RUNS + 1):
            result = answer(entry["question"], embedder, index, chunks, reranker=reranker)
            ok, reason = judge(entry, result["answer"])
            if ok:
                passes += 1
            verdict = "PASS" if ok else "FAIL"
            log.append(f"{entry['id']} run {run}  {verdict}  ({reason})\n{result['answer']}\n")
            time.sleep(4)
        total_passes += passes
        print(f"{entry['id']:28} {passes}/{RUNS} runs passed")

    LOG_PATH.write_text("\n".join(log), encoding="utf-8")
    total = len(entries) * RUNS
    minutes = (time.time() - started) / 60
    print(f"\nanswer score: {total_passes}/{total}  (reranker: {USE_RERANKER}, {minutes:.1f} min)")
    print(f"all answers saved to {LOG_PATH}")


if __name__ == "__main__":
    main()