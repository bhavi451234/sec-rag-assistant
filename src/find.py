"""Find chunks containing a phrase, to help write evaluation questions."""

import sys

from src.search import CHUNKS_PATH, load_chunks

def find(phrase, ticker=None, width=150):
    """Return (chunk_id, section, snippet) for every chunk containing the phrase."""
    matches = []
    for chunk in load_chunks(CHUNKS_PATH):
        if ticker and chunk["ticker"] != ticker:
            continue
        text = chunk["text"]
        position = text.lower().find(phrase.lower())
        if position == -1:
            continue
        start = max(0, position - width)
        snippet = text[start : position + len(phrase) + width].replace("\n", " ")
        matches.append((chunk["chunk_id"], chunk["section"], snippet))
    return matches

if __name__ == "__main__":
    phrase = sys.argv[1]
    ticker = sys.argv[2] if len(sys.argv) > 2 else None
    for chunk_id, section, snippet in find(phrase, ticker):
        print(f"{chunk_id} | {section}\n   {snippet}\n")