import numpy as np

from src.search import build_index, search


class FakeModel:
    """Stands in for the embedding model: always returns the same query vector."""

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype="float32")


CHUNKS = [
    {"ticker": "AAA", "text": "a" * 200},
    {"ticker": "BBB", "text": "b" * 200},
    {"ticker": "AAA", "text": "tiny"},
    {"ticker": "AAA", "text": "c" * 200},
]
VECTORS = np.array([[0.6, 0.8], [1.0, 0.0], [0.99, 0.141], [0.8, 0.6]], dtype="float32")


def run_search(**settings):
    index = build_index(VECTORS)
    return search("any question", FakeModel(), index, CHUNKS, **settings)


def test_best_match_comes_first_and_tiny_chunks_are_skipped():
    results = run_search(k=2)
    assert [r["text"][0] for r in results] == ["b", "c"]


def test_ticker_filter_keeps_only_that_company():
    results = run_search(k=5, ticker="AAA")
    assert [r["text"][0] for r in results] == ["c", "a"]
    assert all(r["ticker"] == "AAA" for r in results)


def test_tiny_chunks_come_back_when_min_chars_is_zero():
    results = run_search(k=1, ticker="AAA", min_chars=0)
    assert results[0]["text"] == "tiny"


def test_never_more_than_k_results():
    assert len(run_search(k=1)) == 1


def test_fewer_results_when_the_company_has_few_chunks():
    assert len(run_search(k=5, ticker="BBB")) == 1