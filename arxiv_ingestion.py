from arxiv_client import (
    papers_to_documents,
    search_arxiv,
)
from vector_store import index_arxiv_documents


def index_arxiv_topic(
    topic: str,
    max_results: int = 100,
) -> dict:
    """
    Retrieve and index arXiv metadata for a topic.
    """
    papers = search_arxiv(
        topic=topic,
        max_results=max_results,
    )

    documents = papers_to_documents(papers)
    indexed_count = index_arxiv_documents(documents)

    return {
        "topic": topic,
        "retrieved_count": len(papers),
        "indexed_count": indexed_count,
        "papers": papers,
    }