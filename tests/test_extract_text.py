from src.extract_text import chunk_lines, is_furniture, label_lines


def test_lines_take_the_label_of_the_last_item_heading():
    text = (
        "Cover page\n"
        "Item 1A. Risk Factors\n"
        "Supply can fail.\n"
        "Item 1B. Unresolved Staff Comments\n"
        "None."
    )
    assert label_lines(text) == [
        ("Other", "Cover page"),
        ("Item 1A", "Item 1A. Risk Factors"),
        ("Item 1A", "Supply can fail."),
        ("Item 1B", "Item 1B. Unresolved Staff Comments"),
        ("Item 1B", "None."),
    ]


def test_item_heading_is_case_insensitive():
    assert label_lines("ITEM 7. MD&A") == [("Item 7", "ITEM 7. MD&A")]


def test_item_in_the_middle_of_a_sentence_is_not_a_heading():
    assert label_lines("See Item 1A for details.") == [("Other", "See Item 1A for details.")]


def test_items_15_and_16_revert_to_other_after_300_lines():
    text = "Item 15.\n" + "\n".join(["line"] * 301)
    labelled = label_lines(text)
    assert labelled[300][0] == "Item 15"
    assert labelled[301][0] == "Other"


def test_page_numbers_are_furniture():
    assert is_furniture("40", set())
    assert is_furniture("192", set())


def test_years_sentences_and_table_rows_are_kept():
    assert not is_furniture("2025", set())
    assert not is_furniture("Revenue was $391 million.", set())
    assert not is_furniture("Net sales 391,035 383,285", set())


def test_table_of_contents_is_furniture_in_any_case():
    assert is_furniture("TABLE OF CONTENTS", set())
    assert is_furniture("Financial Table of Contents", set())


def test_annual_report_footer_is_furniture_but_a_sentence_is_not():
    assert is_furniture("2025 Annual Report", set())
    assert not is_furniture("See the 2025 Annual Report for more.", set())


def test_known_footer_pattern_is_furniture():
    footer_keys = {"Apple Inc. | # Form #-K | #"}
    assert is_furniture("Apple Inc. | 2025 Form 10-K | 17", footer_keys)


def test_chunks_never_mix_labels():
    labelled = [("Item 1", "a"), ("Item 2", "b")]
    assert chunk_lines(labelled, max_chars=100) == [("Item 1", "a"), ("Item 2", "b")]


def test_full_chunk_closes_and_next_one_starts_with_its_last_line():
    labelled = [("Item 1", "aaaa"), ("Item 1", "bbbb"), ("Item 1", "cccc")]
    assert chunk_lines(labelled, max_chars=10, overlap_chars=300) == [
        ("Item 1", "aaaa\nbbbb"),
        ("Item 1", "bbbb\ncccc"),
    ]


def test_oversize_line_stays_whole_and_is_not_carried_forward():
    long_line = "x" * 50
    labelled = [("Item 1", "a"), ("Item 1", long_line), ("Item 1", "b")]
    chunks = chunk_lines(labelled, max_chars=20, overlap_chars=10)
    assert chunks[1][1].endswith(long_line)
    assert chunks[-1] == ("Item 1", "b")