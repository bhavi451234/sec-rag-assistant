"""Turn every downloaded filing into chunks and save them as JSONL, one chunk per line."""

import json
from pathlib import Path

from src.extract_text import chunk_lines, html_to_text, label_lines, remove_furniture

FILINGS_DIR = Path("data/filings")
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")


def build_filing_chunks(html_path: Path) -> list[dict]:
    """Return one record (a dict) per chunk of a single filing."""
    meta = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    labelled = label_lines(html_to_text(html_path))
    kept, _removed = remove_furniture(labelled)
    chunks = chunk_lines(kept)

    records = []
    for number, (section, text) in enumerate(chunks):
        records.append(
            {
                "ticker": meta["ticker"],
                "period": meta["period"],
                "section": section,
                "chunk_id": f"{meta['ticker']}_{meta['period']}_{number:04d}",
                "text": text,
                "source_url": meta["url"],
            }
        )
    return records


def main() -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for html_path in sorted(FILINGS_DIR.glob("*.html")):
            records = build_filing_chunks(html_path)
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            chars = sum(len(r["text"]) for r in records)
            other = sum(len(r["text"]) for r in records if r["section"] == "Other")
            long = sum(1 for r in records if len(r["text"]) > 1500)
            print(
                f"{html_path.stem:24} {len(records):>5} chunks | {chars:>9,} chars"
                f" | Other {other / chars:>4.0%} | over 1500: {long}"
            )
            total += len(records)
    print(total, "chunks written to", CHUNKS_PATH)


if __name__ == "__main__":
    main()