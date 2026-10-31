"""Download the most recent 10-K annual report for each company from SEC EDGAR."""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# The SEC requires every request to say who is asking.
# Put your name and email in .env as: SEC_USER_AGENT="Your Name your.email@example.com"
HEADERS = {"User-Agent": os.environ.get("SEC_USER_AGENT", "")}

TICKERS = ["AAPL", "MSFT", "TSLA", "SNAP", "JPM", "XOM", "JNJ"]
OUT_DIR = Path("data/filings")

# The SEC's ticker file now maps XOM to a new holding company with no 10-K history.
# Its annual reports were filed under the original Exxon Mobil Corp CIK.
CIK_OVERRIDES = {"XOM": "0000034088"}


def get(url: str) -> requests.Response:
    """One polite request: the SEC allows at most 10 per second."""
    time.sleep(0.2)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response


def ticker_to_cik() -> dict:
    """Map each ticker to the 10-digit company ID (CIK) EDGAR uses."""
    data = get("https://www.sec.gov/files/company_tickers.json").json()
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in data.values()}


def latest_10k(cik: str) -> dict:
    """Find the company's most recent 10-K in its filing history."""
    recent = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "document": recent["primaryDocument"][i],
                "filed": recent["filingDate"][i],
                "period": recent["reportDate"][i],
            }
    raise ValueError(
        f"no 10-K in recent filings for CIK {cik}. "
        "Possible reasons: foreign company (files 20-F), recent IPO, "
        "or the ticker moved to a new holding company. Check the ticker, "
        "or add the right CIK to CIK_OVERRIDES."
    )


def download(ticker: str, cik: str) -> None:
    filing = latest_10k(cik)
    path = OUT_DIR / f"{ticker}_10K_{filing['period']}.html"
    if path.exists():
        print(f"{ticker}: already have {path.name}")
        return
    folder = filing["accession"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{filing['document']}"
    path.write_bytes(get(url).content)
    path.with_suffix(".json").write_text(
        json.dumps({**filing, "ticker": ticker, "url": url}, indent=2)
    )
    print(f"{ticker}: saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ciks = ticker_to_cik()
    ciks.update(CIK_OVERRIDES)

    failed = {}
    for ticker in TICKERS:
        try:
            if ticker not in ciks:
                raise ValueError("unknown ticker, check the spelling")
            download(ticker, ciks[ticker])
        except (ValueError, requests.RequestException) as error:
            print(f"{ticker}: SKIPPED - {error}")
            failed[ticker] = str(error)

    print(f"\nDone: {len(TICKERS) - len(failed)} of {len(TICKERS)} downloaded.")
    if failed:
        print("Needs attention:", ", ".join(failed))


if __name__ == "__main__":
    main()