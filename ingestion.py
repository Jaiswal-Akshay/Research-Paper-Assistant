import hashlib
import re
from pathlib import Path

import pymupdf
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TABLE_ROWS_PER_CHUNK,
)


class DocumentProcessingError(Exception):
    """Raised when a research paper cannot be processed."""


def clean_text(text: str) -> str:
    """
    Clean narrative PDF text while retaining readable paragraph
    boundaries for LangChain's text splitter.
    """
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_table_cell(cell: object) -> str:
    """Normalize a cell without destroying its meaning."""
    if cell is None:
        return ""

    value = str(cell)
    value = re.sub(r"\s+", " ", value).strip()

    # Escape Markdown table separators.
    return value.replace("|", r"\|")


def create_document_id(pdf_path: str | Path) -> str:
    """Generate a stable ID from the file contents."""
    pdf_bytes = Path(pdf_path).read_bytes()
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def normalize_table(
    raw_table: list[list[object]],
) -> list[list[str]]:
    """
    Clean a table and make every row the same width.
    """
    rows = []

    for raw_row in raw_table:
        if not raw_row:
            continue

        row = [clean_table_cell(cell) for cell in raw_row]

        if any(row):
            rows.append(row)

    if not rows:
        return []

    column_count = max(len(row) for row in rows)

    return [
        row + [""] * (column_count - len(row))
        for row in rows
    ]


def create_unique_headers(first_row: list[str]) -> list[str]:
    """
    Create valid Markdown headers.

    Missing or duplicated headers receive generated names.
    """
    headers = []
    used_headers: dict[str, int] = {}

    for index, value in enumerate(first_row, start=1):
        base_header = value or f"Column {index}"
        occurrence = used_headers.get(base_header, 0) + 1
        used_headers[base_header] = occurrence

        if occurrence > 1:
            header = f"{base_header} {occurrence}"
        else:
            header = base_header

        headers.append(header)

    return headers


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    """Convert table rows into Markdown."""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(
        ["---"] * len(headers)
    ) + " |"

    data_lines = [
        "| " + " | ".join(row) + " |"
        for row in rows
    ]

    return "\n".join(
        [header_line, separator_line, *data_lines]
    )


def table_to_documents(
    raw_table: list[list[object]],
    *,
    metadata: dict,
) -> list[Document]:
    """
    Convert one extracted table into one or more LangChain
    Documents.

    Large tables are divided by rows while repeating the headers.
    """
    normalized_rows = normalize_table(raw_table)

    if not normalized_rows:
        return []

    headers = create_unique_headers(normalized_rows[0])
    data_rows = normalized_rows[1:]

    # Retain a table even if it contains only one extracted row.
    if not data_rows:
        data_rows = [[""] * len(headers)]

    documents = []

    for start in range(
        0,
        len(data_rows),
        TABLE_ROWS_PER_CHUNK,
    ):
        batch = data_rows[
            start : start + TABLE_ROWS_PER_CHUNK
        ]

        table_part = (
            start // TABLE_ROWS_PER_CHUNK
        ) + 1

        table_label = (
            f"Table {metadata['table_number']} "
            f"from {metadata['title']}, "
            f"page {metadata['page']}, "
            f"part {table_part}"
        )

        content = (
            f"{table_label}\n\n"
            f"{markdown_table(headers, batch)}"
        )

        part_metadata = {
            **metadata,
            "table_part": table_part,
            "content_type": "table",
        }

        documents.append(
            Document(
                page_content=content,
                metadata=part_metadata,
            )
        )

    return documents


def extract_pdf_documents(
    pdf_path: str | Path,
    title: str | None = None,
) -> list[Document]:
    """
    Extract narrative text and tables from a research paper.

    Text and tables are returned as separate LangChain Documents.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise DocumentProcessingError(
            f"PDF does not exist: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise DocumentProcessingError(
            f"Expected a PDF file: {pdf_path.name}"
        )

    document_id = create_document_id(pdf_path)
    paper_title = title or pdf_path.stem
    documents: list[Document] = []

    try:
        pymupdf_document = pymupdf.open(pdf_path)
        plumber_document = pdfplumber.open(pdf_path)
    except Exception as exc:
        raise DocumentProcessingError(
            f"Could not open PDF: {pdf_path.name}"
        ) from exc

    try:
        if pymupdf_document.needs_pass:
            raise DocumentProcessingError(
                "Password-protected PDFs are not supported."
            )

        for page_index, pymupdf_page in enumerate(
            pymupdf_document
        ):
            page_number = page_index + 1
            base_metadata = {
                "document_id": document_id,
                "source": pdf_path.name,
                "title": paper_title,
                "page": page_number,
            }

            # Extract narrative text.
            raw_text = pymupdf_page.get_text("text")
            cleaned_text = clean_text(raw_text)

            if cleaned_text:
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            **base_metadata,
                            "content_type": "text",
                        },
                    )
                )

            # Extract tables from the same page.
            plumber_page = plumber_document.pages[page_index]
            extracted_tables = (
                plumber_page.extract_tables() or []
            )

            for table_index, raw_table in enumerate(
                extracted_tables,
                start=1,
            ):
                table_documents = table_to_documents(
                    raw_table,
                    metadata={
                        **base_metadata,
                        "table_number": table_index,
                    },
                )

                documents.extend(table_documents)

    finally:
        pymupdf_document.close()
        plumber_document.close()

    if not documents:
        raise DocumentProcessingError(
            "No readable text or tables were found. The PDF "
            "may be scanned and require OCR."
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split narrative text while keeping prepared table batches intact.
    """
    text_documents = [
        document
        for document in documents
        if document.metadata["content_type"] == "text"
    ]

    table_documents = [
        document
        for document in documents
        if document.metadata["content_type"] == "table"
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    text_chunks = splitter.split_documents(text_documents)

    return text_chunks + table_documents


def process_pdf(
    pdf_path: str | Path,
    title: str | None = None,
) -> list[Document]:
    """
    Run the complete table-aware LangChain ingestion pipeline.
    """
    extracted_documents = extract_pdf_documents(
        pdf_path,
        title=title,
    )

    chunks = split_documents(extracted_documents)
    document_id = create_document_id(pdf_path)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_number"] = index
        chunk.metadata["chunk_id"] = (
            f"{document_id}-chunk-{index}"
        )

    return chunks