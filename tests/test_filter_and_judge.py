from src import answer as answer_module
from src.score_answers import judge

PAIR = {
    "type": "compare",
    "ticker": ["AAPL", "XOM"],
    "expected": ["416,161", "332,238"],
    "must_not_contain": [],
    "winner": "AAPL",
}


def run_retrieve(monkeypatch, question, tickers=None):
    searched = []

    def fake_search(question, embedder, index, chunks, k, ticker):
        searched.append(ticker)
        return []

    monkeypatch.setattr(answer_module, "search", fake_search)
    answer_module.retrieve(question, None, None, [], tickers=tickers)
    return searched


def test_filter_overrides_the_question(monkeypatch):
    assert run_retrieve(monkeypatch, "What did Apple earn?", tickers=["XOM"]) == ["XOM"]


def test_without_filter_the_question_decides(monkeypatch):
    assert run_retrieve(monkeypatch, "What did Apple earn?") == ["AAPL"]


def test_judge_accepts_the_right_winner():
    text = "Apple: $416,161 million [9]\nExxonMobil: $332,238 million [17]\nHigher: Apple"
    assert judge(PAIR, text)[0] is True


def test_judge_rejects_the_wrong_winner():
    text = "Apple: $416,161 million [9]\nExxonMobil: $332,238 million [17]\nHigher: ExxonMobil"
    assert judge(PAIR, text)[0] is False


def test_judge_without_higher_line_uses_first_company_named():
    text = "ExxonMobil had higher revenue. Apple 416,161 and Exxon 332,238"
    assert judge(PAIR, text)[0] is False