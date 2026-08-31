from pathlib import Path

from ingestion import process_pdf
from vector_store import index_documents


PDF_PATH = Path("data/uploads/Sample_Research.pdf")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Research paper not found: {PDF_PATH}"
        )

    print("Processing paper...")
    documents = process_pdf(PDF_PATH)

    print(f"Created {len(documents)} chunks.")
    print("Generating embeddings and indexing...")

    indexed_count = index_documents(documents)

    print(f"Successfully indexed {indexed_count} chunks.")


if __name__ == "__main__":
    main()