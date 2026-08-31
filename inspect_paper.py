from collections import Counter
from pathlib import Path

from ingestion import process_pdf


PDF_PATH = Path("data/uploads/Sample_Research.pdf")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Research paper not found: {PDF_PATH}"
        )

    documents = process_pdf(PDF_PATH)

    counts = Counter(
        document.metadata["content_type"]
        for document in documents
    )

    print(f"Paper: {PDF_PATH.name}")
    print(f"Total chunks: {len(documents)}")
    print(f"Text chunks: {counts['text']}")
    print(f"Table chunks: {counts['table']}")
    print()

    table_documents = [
        document
        for document in documents
        if document.metadata["content_type"] == "table"
    ]

    if not table_documents:
        print("No tables were detected.")
        return

    for document in table_documents:
        metadata = document.metadata

        print(
            f"Page {metadata['page']} | "
            f"Table {metadata['table_number']} | "
            f"Part {metadata['table_part']}"
        )
        print(document.page_content)
        print("=" * 80)


if __name__ == "__main__":
    main()