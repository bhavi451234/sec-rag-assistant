# SEC 10-K assistant

Ask questions about the latest annual reports (10-K filings) of seven companies and get answers with citations. Every answer shows the exact passages it came from, with a link to the filing on SEC.gov.

**Companies:** Apple, Microsoft, Tesla, Snap, JPMorgan Chase, ExxonMobil, Johnson & Johnson.

This is a learning project. It is not investment advice, and answers can be wrong. Check the cited passage.

## How it works

1. **Download** the latest 10-K of each company from SEC EDGAR (`src/fetch_filings.py`).
2. **Clean and split** the HTML into labelled text chunks of up to 1,500 characters, about 3,100 in total (`src/extract_text.py`, `src/build_chunks.py`).
3. **Embed** every chunk with `BAAI/bge-small-en-v1.5` and store the vectors in a FAISS index (`src/embed_chunks.py`, `src/search.py`).
4. **Retrieve:** find the companies named in the question, take the 60 closest chunks per company from the index, and re-rank them with the cross-encoder `BAAI/bge-reranker-base`. Keep the best ones (`src/answer.py`).
5. **Answer:** send the question and the numbered passages to Gemini (`gemini-3.5-flash-lite`, temperature 0), which must cite them as [1], [2] and say "I can't find this in the provided filings" when the passages don't contain the answer.
6. **Show:** a Streamlit app (`app.py`) with example questions, a company filter, the answer, and the passages with SEC links. Cited passages open automatically.

## Run it

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your Gemini key and your name/email for the SEC
python -m src.fetch_filings
python -m src.build_chunks
python -m src.embed_chunks
python -m streamlit run app.py
```

The downloaded filings, chunks and vectors are not in git (`data/filings`, `data/chunks`, `data/index`); the three build commands recreate them. The first question after starting the app takes about 30-40 seconds while the models load; later ones take about 15-20 seconds.

Tests: `python -m pytest -q`

## How well does it work?

I wrote an answer key of 33 questions (single figures, text lookups, two-company comparisons, and questions that cannot be answered from the filings). Only the question goes into the pipeline. Each question is asked 3 times, and an answer passes if it contains all the expected figures, no wrong "lure" figure before them, and, for "which is higher" questions, names the right winner.

| Set | Result |
|---|---|
| Development set, 22 questions x 3 runs | about 65/66 (tuned on this set, assembled from runs on different days) |
| Held-out batch 1, 8 questions x 3 runs (never tuned on) | 21/24 |
| Held-out batch 2, 3 "which is higher" questions x 3 runs | 9/9 |
| Retrieval: right passage in the top 8 | 27 of 30 questions |

How the development score improved: 48-49/66 (first version) -> 54 (narrower prompt rule) -> 58 (cross-encoder reranker) -> 65 (more passages per company for two-company questions, and a format rule for "which is higher" questions).

Things I tried that did not help: adding neighbouring chunks to each hit, and a smaller reranker (MiniLM), which made numeric table questions worse.

## What I got wrong along the way

- My first scorer only checked that the right numbers appeared in the answer. In the app I noticed an answer that quoted both figures correctly but named the wrong company as higher. Reading the saved answers showed 6 of 12 comparison answers had this error, so the development score was overstated (about 59/66 with the stricter check). I added a winner check to the scorer and changed the prompt so the model writes both figures first and ends with a `Higher:` line.
- The one held-out failure is a question I wrote badly: Tesla reports two net income figures (3,855 and 3,794 attributable to common stockholders) and my question did not say which.

## Limitations

- Small evaluation: 33 questions. The development score is optimistic because I tuned settings while looking at those questions; the held-out sets are small.
- The model sometimes picks the wrong row of a table, or refuses, even when the right passage is in its context.
- Wide tables are flattened into text, and some chunks glue two statements together.
- "Right passage in the top 8" is a strict test (one specific chunk), so it understates how often the answer is reachable.
- The scorer checks headline figures and the winner, not whether the whole answer is correct, and not units.
- Only seven companies and one year each. No live data (for example, stock prices).
- Free-tier Gemini limits (about 20 requests/day for the main model) restrict how many evaluation runs I can do and how many questions the app can answer per day. Reranking adds latency and about 1 GB of memory.