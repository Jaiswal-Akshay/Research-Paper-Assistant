import pytest
from langchain_core.documents import Document

from arxiv_client import (
    ArxivPaper,
    build_search_query,
    extract_arxiv_id,
    normalize_whitespace,
    paper_to_document,
)


def test_normalize_whitespace():
    value = "Retrieval   augmented\n generation"

    assert normalize_whitespace(value) == (
        "Retrieval augmented generation"
    )


def test_build_search_query():
    query = build_search_query(
        "retrieval augmented generation"
    )

    assert query == (
        'all:"retrieval augmented generation"'
    )


def test_empty_topic_is_rejected():
    with pytest.raises(ValueError):
        build_search_query("   ")


def test_extract_arxiv_id_removes_version():
    identifier = extract_arxiv_id(
        "https://arxiv.org/abs/2608.12345v2"
    )

    assert identifier == "2608.12345"


def test_paper_to_document():
    paper = ArxivPaper(
        arxiv_id="2608.12345",
        title="Example RAG Study",
        abstract="This paper evaluates RAG.",
        authors=("Author One", "Author Two"),
        published="2026-08-01T00:00:00Z",
        updated="2026-08-02T00:00:00Z",
        categories=("cs.AI", "cs.IR"),
        abstract_url=(
            "https://arxiv.org/abs/2608.12345"
        ),
        pdf_url=(
            "https://arxiv.org/pdf/2608.12345"
        ),
    )

    document = paper_to_document(paper)

    assert isinstance(document, Document)
    assert "Example RAG Study" in document.page_content
    assert "Author One" in document.page_content
    assert document.metadata["arxiv_id"] == "2608.12345"
    assert (
        document.metadata["content_type"]
        == "arxiv_metadata"
    )