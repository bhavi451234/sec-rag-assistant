"""Check that every evaluation entry is well formed and its evidence really exists."""

import json
from pathlib import Path

from src.search import CHUNKS_PATH, load_chunks

EVAL_PATH = Path("data/eval/eval_set.jsonl")
REQUIRED = {"id", "question", "ticker", "type", "expected", "must_not_contain", "evidence"}


def load_eval(path=EVAL_PATH):
    """Read the evaluation set: one dict per non-blank line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evidence_pairs(entry):
    """Return (ticker, phrase) pairs: one for a single-company entry, one per company otherwise."""
    evidence = entry["evidence"]
    if evidence is None:
        return []
    if isinstance(evidence, str):
        return [(entry["ticker"], evidence)]
    return list(evidence.items())


def main():
    chunks = load_chunks(CHUNKS_PATH)
    entries = load_eval()
    problems = 0
    for entry in entries:
        missing = REQUIRED - entry.keys()
        if missing:
            print(entry.get("id"), "missing fields:", sorted(missing))
            problems += 1
            continue
        if isinstance(entry["evidence"], dict) and set(entry["ticker"]) != set(entry["evidence"]):
            print(entry["id"], "tickers and evidence keys do not match")
            problems += 1
        for ticker, phrase in evidence_pairs(entry):
            count = sum(
                1
                for chunk in chunks
                if (ticker is None or chunk["ticker"] == ticker) and phrase in chunk["text"]
            )
            if count == 0:
                print(entry["id"], f"evidence not found for {ticker}:", repr(phrase))
                problems += 1
    print(f"{len(entries)} entries, {problems} problems")


if __name__ == "__main__":
    main()