"""Tests for answer.py. No real LLM or embedding model is used."""

from src.answer import answer, build_prompt, detect_tickers, format_sources

RESULTS = [
    {"ticker": "AAPL", "period": "2025-09-27", "section": "Item 7", "text": "Net sales were $416,161 million."},
    {"ticker": "MSFT", "period": "2026-06-30", "section": "Item 1A", "text": "We face intense competition."},
]

def test_detect_one_company():
    assert detect_tickers("What was Apple's total net sales?") == ["AAPL"]

def test_detect_keeps_order_of_appearance():
    assert detect_tickers("Compare Microsoft and Apple.") == ["MSFT", "AAPL"]

def test_detect_none():
    assert detect_tickers("Which companies mention climate change?") == []

def test_detect_needs_whole_word():
    assert detect_tickers("Show me a snapshot of the results.") == []
    assert detect_tickers("What is Snapchat's revenue?") == ["SNAP"]

def test_detect_alias_with_symbols():
    assert detect_tickers("What does Johnson & Johnson say about talc?") == ["JNJ"]

def test_format_sources_numbers_each_block():
    text = format_sources(RESULTS)
    assert text.startswith("[1] AAPL 10-K, period 2025-09-27, Item 7\nNet sales were")
    assert "[2] MSFT 10-K, period 2026-06-30, Item 1A" in text

def test_build_prompt_order():
    prompt = build_prompt("What were sales?", RESULTS)
    assert prompt.index("SOURCES:") < prompt.index("QUESTION: What were sales?")
    assert prompt.endswith("ANSWER:")

def test_answer_sends_prompt_to_llm(monkeypatch):
    monkeypatch.setattr("src.answer.retrieve", lambda *args, **kwargs: RESULTS)
    seen = []

    def fake_llm(prompt):
        seen.append(prompt)
        return "fake answer [1]"

    result = answer("What were sales?", None, None, [], llm=fake_llm)

    assert result["answer"] == "fake answer [1]"
    assert result["sources"] == RESULTS
    assert len(seen) == 1
    assert "Net sales were $416,161 million." in seen[0]

def test_answer_with_no_results_skips_llm(monkeypatch):
    monkeypatch.setattr("src.answer.retrieve", lambda *args, **kwargs: [])
    calls = []

    result = answer("anything", None, None, [], llm=lambda prompt: calls.append(prompt))

    assert result["answer"] == "I can't find this in the provided filings."
    assert result["sources"] == []
    assert calls == []