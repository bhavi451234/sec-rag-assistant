"""Answer score: is the final answer correct according to the answer key?"""

import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from src.answer import COMPANY_ALIASES

from src.answer import RERANKER_NAME, answer
from src.check_eval_set import load_eval
from src.search import CHUNKS_PATH, MODEL_NAME, VECTORS_PATH, build_index, load_chunks

CANT_FIND = "find this in the provided filings"
LOG_PATH = Path("data/eval/score_answers_log.txt")
PARTIAL_LOG_PATH = Path("data/eval/score_answers_partial_log.txt")
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

    winner = entry.get("winner")
    if winner is not None:
        if "higher:" in text:
            claim_text = text.rsplit("higher:", 1)[1]
        else:
            claim_text = text
        earliest = len(claim_text)
        first_company = None
        for ticker in entry["ticker"]:
            for alias in COMPANY_ALIASES[ticker]:
                alias_position = claim_text.find(alias)
                if alias_position != -1 and alias_position < earliest:
                    earliest = alias_position
                    first_company = ticker
        if first_company != winner:
            return False, f"names {first_company} as the winner, but the higher one is {winner}"
    return True, "ok"


def main():
    wanted_ids = sys.argv[1:]
    log_path = PARTIAL_LOG_PATH if wanted_ids else LOG_PATH

    embedder = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME) if USE_RERANKER else None
    chunks = load_chunks(CHUNKS_PATH)
    index = build_index(np.load(VECTORS_PATH))

    entries = load_eval()
    if wanted_ids:
        entries = [entry for entry in entries if entry["id"] in wanted_ids]

    total_passes = 0
    done = 0
    log = []
    started = time.time()
    try:
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
            done += 1
            print(f"{entry['id']:28} {passes}/{RUNS} runs passed")
            log_path.write_text("\n".join(log), encoding="utf-8")
    except RuntimeError as error:
        print(f"\nSTOPPED EARLY: {error}")

    total = done * RUNS
    minutes = (time.time() - started) / 60
    print(f"\nanswer score: {total_passes}/{total} over {done} of {len(entries)} questions "
          f"(reranker: {USE_RERANKER}, {minutes:.1f} min)")
    print(f"all answers saved to {log_path}")


if __name__ == "__main__":
    main()