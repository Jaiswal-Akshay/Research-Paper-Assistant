from vector_store import search_documents


QUESTION = "What results are reported in the table 1    ?"


def main() -> None:
    documents = search_documents(
        question=QUESTION,
        limit=6,
    )

    print(f"Question: {QUESTION}")
    print(f"Retrieved documents: {len(documents)}")
    print()

    for position, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        print(f"RESULT {position}")
        print(f"Title: {metadata['title']}")
        print(f"Page: {metadata['page']}")
        print(
            f"Content type: "
            f"{metadata['content_type']}"
        )

        if metadata["content_type"] == "table":
            print(
                f"Table: "
                f"{metadata['table_number']}"
            )

        print(document.page_content[:1000])
        print("=" * 80)


if __name__ == "__main__":
    main()