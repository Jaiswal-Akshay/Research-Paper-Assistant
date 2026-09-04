from arxiv_rag import answer_arxiv_question


def display_sources(documents) -> None:
    """Display metadata for the retrieved arXiv papers."""
    if not documents:
        print("No supporting papers were retrieved.")
        return

    print()
    print("RETRIEVED PAPERS")

    for position, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        title = metadata.get(
            "title",
            "Untitled paper",
        )

        arxiv_id = metadata.get(
            "arxiv_id",
            "Unknown ID",
        )

        authors = metadata.get(
            "authors",
            "Authors not available",
        )

        published = metadata.get(
            "published",
            "Date not available",
        )

        abstract_url = metadata.get(
            "abstract_url",
            "",
        )

        print()
        print(
            f"{position}. {title}"
        )
        print(
            f"   arXiv ID: {arxiv_id}"
        )
        print(
            f"   Authors: {authors}"
        )
        print(
            f"   Published: {published}"
        )

        if abstract_url:
            print(
                f"   URL: {abstract_url}"
            )


def main() -> None:
    """Run the command-line arXiv research assistant."""
    print("=" * 80)
    print("arXiv Metadata Research Assistant")
    print("=" * 80)
    print(
        "Ask questions about the indexed papers "
        "and their abstracts."
    )
    print(
        "Enter 'exit' or 'quit' to stop."
    )
    print()

    while True:
        try:
            question = input(
                "Question: "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye.")
            break

        if question.lower() in {
            "exit",
            "quit",
        }:
            print("Goodbye.")
            break

        if not question:
            print(
                "Please enter a question."
            )
            print()
            continue

        try:
            print()
            print(
                "Retrieving papers and "
                "generating an answer..."
            )
            print()

            result = answer_arxiv_question(
                question
            )

            print("ANSWER")
            print(result["answer"])

            display_sources(
                result["sources"]
            )

        except Exception as exc:
            print("ERROR")
            print(
                "The question could not be processed."
            )
            print(
                f"Details: {exc}"
            )

        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()