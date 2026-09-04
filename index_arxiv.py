import argparse

from arxiv_ingestion import index_arxiv_topic
from config import ARXIV_MAX_RESULTS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve and index research-paper metadata "
            "from arXiv."
        )
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Research topic to search for.",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=ARXIV_MAX_RESULTS,
        help="Number of arXiv results to retrieve.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    print(
        f'Searching arXiv for: "{arguments.topic}"'
    )

    result = index_arxiv_topic(
        topic=arguments.topic,
        max_results=arguments.max_results,
    )

    print(
        f"Retrieved {result['retrieved_count']} papers."
    )
    print(
        f"Indexed {result['indexed_count']} papers."
    )
    print()

    for paper in result["papers"][:10]:
        print(f"{paper.arxiv_id}: {paper.title}")


if __name__ == "__main__":
    main()