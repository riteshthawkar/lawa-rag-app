from modules.citations import process_citations, validate_citation_numbers


def test_validate_citation_numbers_filters_out_of_range_values():
    assert validate_citation_numbers([0, 1, 2, 99], max_docs=2) == [1, 2]


def test_process_citations_encodes_urls_and_deduplicates_adjacent_references():
    ranked_docs = [
        {"page_source": "https://example.com/visa path/overview", "chunk": "Doc 1"},
        {"page_source": "https://example.com/visa path/overview", "chunk": "Doc 2"},
        {"page_source": "https://example.com/policy", "chunk": "Doc 3"},
    ]

    updated_answer, citations = process_citations(
        "Golden Visa benefits [2][2] and policy details [3].",
        ranked_docs,
    )

    assert (
        updated_answer
        == "Golden Visa benefits [1](https://example.com/visa%20path/overview) and policy details [2](https://example.com/policy)."
    )
    assert citations == [
        {"url": "https://example.com/visa%20path/overview", "cite_num": "1"},
        {"url": "https://example.com/policy", "cite_num": "2"},
    ]


def test_process_citations_preserves_invalid_citation_numbers_without_linking():
    ranked_docs = [
        {"page_source": "https://example.com/valid", "chunk": "Doc 1"},
    ]

    updated_answer, citations = process_citations(
        "Valid source [1] and unknown source [9].",
        ranked_docs,
    )

    assert updated_answer == "Valid source [1](https://example.com/valid) and unknown source [9]."
    assert citations == [{"url": "https://example.com/valid", "cite_num": "1"}]
