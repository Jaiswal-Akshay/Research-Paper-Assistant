import re
from dataclasses import dataclass

import feedparser
import requests
from langchain_core.documents import Document

from config import (
    ARXIV_API_URL,
    ARXIV_REQUEST_TIMEOUT,
    ARXIV_USER_AGENT,
)


class ArxivAPIError(Exception):
    """Raised when arXiv metadata cannot be retrieved."""


@dataclass(frozen=True)
class ArxivPaper:
    """Structured metadata for one arXiv paper."""

    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str
    updated: str
    categories: tuple[str, ...]
    abstract_url: str
    pdf_url: str


def normalize_whitespace(value: str) -> str:
    """Remove unnecessary whitespace from API text."""
    return re.sub(r"\s+", " ", value).strip()


def build_search_query(topic: str) -> str:
    """
    Build an arXiv full-text metadata query.

    Quotation marks are removed to prevent malformed queries.
    """
    cleaned_topic = normalize_whitespace(topic)
    cleaned_topic = cleaned_topic.replace('"', "")

    if not cleaned_topic:
        raise ValueError("The arXiv topic cannot be empty.")

    return f'all:"{cleaned_topic}"'


def extract_arxiv_id(entry_id: str) -> str:
    """Extract the arXiv identifier from its abstract URL."""
    identifier = entry_id.rstrip("/").split("/")[-1]

    # Remove a version suffix such as v1 or v2.
    return re.sub(r"v\d+$", "", identifier)


def find_pdf_url(entry) -> str:
    """Find the PDF link in an arXiv Atom entry."""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            return link.get("href", "")

        if link.get("title") == "pdf":
            return link.get("href", "")

    arxiv_id = extract_arxiv_id(entry.get("id", ""))

    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"

    return ""


def parse_entry(entry) -> ArxivPaper:
    """Convert an Atom entry into an ArxivPaper."""
    abstract_url = entry.get("id", "")
    arxiv_id = extract_arxiv_id(abstract_url)

    authors = tuple(
        normalize_whitespace(author.get("name", ""))
        for author in entry.get("authors", [])
        if author.get("name")
    )

    categories = tuple(
        tag.get("term", "")
        for tag in entry.get("tags", [])
        if tag.get("term")
    )

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=normalize_whitespace(
            entry.get("title", "Untitled paper")
        ),
        abstract=normalize_whitespace(
            entry.get("summary", "")
        ),
        authors=authors,
        published=entry.get("published", ""),
        updated=entry.get("updated", ""),
        categories=categories,
        abstract_url=abstract_url,
        pdf_url=find_pdf_url(entry),
    )


def search_arxiv(
    topic: str,
    max_results: int = 100,
) -> list[ArxivPaper]:
    """
    Search arXiv and return metadata ordered by relevance.
    """
    if not 1 <= max_results <= 200:
        raise ValueError(
            "max_results must be between 1 and 200."
        )

    parameters = {
        "search_query": build_search_query(topic),
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": ARXIV_USER_AGENT,
    }

    try:
        response = requests.get(
            ARXIV_API_URL,
            params=parameters,
            headers=headers,
            timeout=ARXIV_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ArxivAPIError(
            f"arXiv request failed: {exc}"
        ) from exc

    feed = feedparser.parse(response.content)

    if feed.bozo and not feed.entries:
        raise ArxivAPIError(
            "arXiv returned an invalid Atom response."
        )

    papers = [
        parse_entry(entry)
        for entry in feed.entries
    ]

    # Prevent duplicate identifiers.
    unique_papers = {
        paper.arxiv_id: paper
        for paper in papers
        if paper.arxiv_id
    }

    return list(unique_papers.values())


def paper_to_document(paper: ArxivPaper) -> Document:
    """
    Convert arXiv metadata into a searchable LangChain Document.
    """
    authors = "; ".join(paper.authors) or "Not provided"
    categories = ", ".join(
        paper.categories
    ) or "Not provided"

    content = f"""
Title: {paper.title}
Authors: {authors}
Published: {paper.published}
Updated: {paper.updated}
Categories: {categories}
arXiv ID: {paper.arxiv_id}

Abstract:
{paper.abstract}
""".strip()

    return Document(
        page_content=content,
        metadata={
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": authors,
            "published": paper.published,
            "updated": paper.updated,
            "categories": categories,
            "abstract_url": paper.abstract_url,
            "pdf_url": paper.pdf_url,
            "content_type": "arxiv_metadata",
            "chunk_id": f"arxiv-{paper.arxiv_id}",
        },
    )


def papers_to_documents(
    papers: list[ArxivPaper],
) -> list[Document]:
    """Convert multiple papers into LangChain documents."""
    return [
        paper_to_document(paper)
        for paper in papers
    ]