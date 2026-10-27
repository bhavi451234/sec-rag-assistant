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
        evidence = entry["evidence"]
        if evidence is None:
            continue
        count = sum(
            1
            for chunk in chunks
            if (entry["ticker"] is None or chunk["ticker"] == entry["ticker"])
            and evidence in chunk["text"]
        )
        if count == 0:
            print(entry["id"], "evidence not found in any chunk:", repr(evidence))
            problems += 1
    print(f"{len(entries)} entries, {problems} problems")

if __name__ == "__main__":
    main()