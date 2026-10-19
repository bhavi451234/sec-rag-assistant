"""Turn a downloaded 10-K HTML file into labelled lines of plain text."""

import re
from pathlib import Path

from bs4 import BeautifulSoup

ITEM_LINE = re.compile(r"item\s+(\d+[a-c]?)\b", re.IGNORECASE)

CAPPED = {"Item 15", "Item 16"}
MAX_CAPPED_LINES = 300

BARE_NUMBER = re.compile(r"\d{1,3}")
FOOTER = re.compile(r"form #-k", re.IGNORECASE)
ANNUAL_REPORT_FOOTER = re.compile(r"\d{4} annual report", re.IGNORECASE)
FURNITURE_LINES = {"table of contents", "financial table of contents"}


def html_to_text(path: Path) -> str:
    """Read one filing and return its readable text, one block per line."""
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["ix:header", "title", "script", "style"]):
        tag.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(["p", "div", "tr", "li", "table"]):
        tag.append("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")

    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def label_lines(text: str) -> list[tuple[str, str]]:
    """Return (label, line) for every line. A line takes the label of the last Item heading above it.

    Items 15 and 16 are short by nature, so after MAX_CAPPED_LINES lines they revert to Other.
    """
    labelled = []
    current = "Other"
    since = 0
    for line in text.splitlines():
        match = ITEM_LINE.match(line)
        if match:
            current = "Item " + match.group(1).upper()
            since = 0
        else:
            since += 1
        label = current
        if current in CAPPED and since > MAX_CAPPED_LINES:
            label = "Other"
        labelled.append((label, line))
    return labelled


def repeated_short_lines(lines: list[str], min_count: int = 5, max_len: int = 80) -> dict[str, int]:
    """Count short lines that repeat, treating every number as the same, e.g. page numbers."""
    counts = {}
    for line in lines:
        if len(line) > max_len:
            continue
        key = re.sub(r"\d+", "#", line)
        counts[key] = counts.get(key, 0) + 1
    return {key: n for key, n in counts.items() if n >= min_count}


def is_furniture(line: str, footer_keys: set[str]) -> bool:
    """True for page numbers, repeated 'Form 10-K' footers, 'YYYY Annual Report' and 'Table of Contents' lines."""
    key = re.sub(r"\d+", "#", line)
    return bool(
        BARE_NUMBER.fullmatch(line)
        or key in footer_keys
        or ANNUAL_REPORT_FOOTER.fullmatch(line)
        or line.lower() in FURNITURE_LINES
    )


def remove_furniture(
    labelled: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Drop furniture lines. Return (kept, removed_lines)."""
    lines = [line for _, line in labelled]
    footer_keys = {key for key in repeated_short_lines(lines) if FOOTER.search(key)}

    kept = []
    removed = []
    for label, line in labelled:
        if is_furniture(line, footer_keys):
            removed.append(line)
        else:
            kept.append((label, line))
    return kept, removed


def chunk_lines(
    labelled: list[tuple[str, str]], max_chars: int = 1500, overlap_chars: int = 300
) -> list[tuple[str, str]]:
    """Group labelled lines into (label, text) chunks of whole lines, never mixing labels.

    When a chunk fills up, the next one starts with a copy of its last line (if short enough).
    """
    chunks = []
    current_label = None
    current_lines = []
    size = 0
    for label, line in labelled:
        too_big = size + len(line) + 1 > max_chars
        if current_lines and (label != current_label or too_big):
            chunks.append((current_label, "\n".join(current_lines)))
            last = current_lines[-1]
            if label == current_label and len(last) <= overlap_chars:
                current_lines = [last]
                size = len(last) + 1
            else:
                current_lines = []
                size = 0
        current_label = label
        current_lines.append(line)
        size += len(line) + 1
    if current_lines:
        chunks.append((current_label, "\n".join(current_lines)))
    return chunks


if __name__ == "__main__":
    for path in sorted(Path("data/filings").glob("*.html")):
        labelled = label_lines(html_to_text(path))
        kept, removed = remove_furniture(labelled)
        numbers = [line for line in removed if line.isdigit()]
        others = sorted(set(removed) - set(numbers))
        print(path.name)
        print(f"   lines {len(labelled):,} | removed {len(removed):,} ({len(numbers):,} bare numbers)")
        for line in others[:5]:
            print("   removed:", repr(line))