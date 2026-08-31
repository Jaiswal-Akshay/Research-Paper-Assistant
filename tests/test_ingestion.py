from pathlib import Path

import pymupdf
import pytest
from langchain_core.documents import Document

from ingestion import (
    DocumentProcessingError,
    clean_text,
    extract_pdf_documents,
    process_pdf,
    table_to_documents,
)


def create_test_pdf(pdf_path: Path) -> None:
    document = pymupdf.open()

    page_one = document.new_page()
    page_one.insert_text(
        (72, 72),
        "This study evaluates retrieval augmented generation.",
    )

    page_two = document.new_page()
    page_two.insert_text(
        (72, 72),
        "The researchers report the experimental results.",
    )

    document.save(pdf_path)
    document.close()


def test_clean_text():
    cleaned = clean_text(
        "retrieval-\naugmented   generation"
    )

    assert cleaned == "retrievalaugmented generation"


def test_table_is_converted_to_markdown():
    raw_table = [
        ["Model", "Accuracy", "Precision"],
        ["Llama 3", "91.4%", "89.2%"],
        ["BERT", "88.6%", "87.1%"],
    ]

    documents = table_to_documents(
        raw_table,
        metadata={
            "document_id": "test123",
            "source": "sample.pdf",
            "title": "Sample Study",
            "page": 4,
            "table_number": 1,
        },
    )

    assert len(documents) == 1
    assert "| Model | Accuracy | Precision |" in (
        documents[0].page_content
    )
    assert "| Llama 3 | 91.4% | 89.2% |" in (
        documents[0].page_content
    )
    assert (
        documents[0].metadata["content_type"]
        == "table"
    )
    assert documents[0].metadata["page"] == 4


def test_missing_table_cells_are_preserved():
    raw_table = [
        ["Method", "Score", None],
        ["RAG", None, "High"],
    ]

    documents = table_to_documents(
        raw_table,
        metadata={
            "document_id": "test123",
            "source": "sample.pdf",
            "title": "Sample Study",
            "page": 2,
            "table_number": 1,
        },
    )

    assert documents
    assert "RAG" in documents[0].page_content
    assert "High" in documents[0].page_content


def test_extract_pdf_returns_langchain_documents(
    tmp_path,
):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    documents = extract_pdf_documents(
        pdf_path,
        title="Sample Study",
    )

    assert len(documents) >= 2
    assert all(
        isinstance(document, Document)
        for document in documents
    )
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["title"] == "Sample Study"


def test_process_pdf_adds_chunk_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)

    chunks = process_pdf(
        pdf_path,
        title="Sample Study",
    )

    assert chunks

    for chunk in chunks:
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["chunk_number"] > 0
        assert chunk.metadata["content_type"] in {
            "text",
            "table",
        }


def test_missing_pdf_raises_error(tmp_path):
    with pytest.raises(DocumentProcessingError):
        extract_pdf_documents(
            tmp_path / "missing.pdf"
        )