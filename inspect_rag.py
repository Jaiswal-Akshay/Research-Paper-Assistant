from rag import answer_question, create_citation


QUESTION = "What are the challenges?"


def main() -> None:
    print(f"Question: {QUESTION}")
    print("Retrieving evidence and generating answer...")
    print()

    result = answer_question(QUESTION)

    print("ANSWER")
    print(result["answer"])
    print()
    print("=" * 80)
    print("RETRIEVED SOURCES")

    for position, document in enumerate(
        result["sources"],
        start=1,
    ):
        print()
        print(
            f"{position}. {create_citation(document)}"
        )
        print(
            f"Content type: "
            f"{document.metadata['content_type']}"
        )
        print(document.page_content[:500])


if __name__ == "__main__":
    main()