from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    ARXIV_COLLECTION_NAME,
    CHROMA_DIR,
    COLLECTION_NAME,
)
from embeddings import get_embedding_model


def get_vector_store() -> Chroma:
    """
    Connect to the persistent Chroma collection used for
    full research-paper text and table chunks.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )


def get_arxiv_vector_store() -> Chroma:
    """
    Connect to the persistent Chroma collection used for
    arXiv metadata and abstracts.
    """
    return Chroma(
        collection_name=ARXIV_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )


def index_documents(
    documents: list[Document],
) -> int:
    """
    Embed and index full-paper text and table documents.

    Stable chunk IDs allow an existing record to be updated when
    the same paper is indexed again.
    """
    if not documents:
        return 0

    vector_store = get_vector_store()

    ids = [
        document.metadata["chunk_id"]
        for document in documents
    ]

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return len(documents)


def index_arxiv_documents(
    documents: list[Document],
) -> int:
    """
    Embed and index arXiv metadata and abstract documents.

    Each arXiv paper uses its arXiv ID as part of a stable
    document ID to help prevent duplicate records.
    """
    if not documents:
        return 0

    vector_store = get_arxiv_vector_store()

    ids = [
        document.metadata["chunk_id"]
        for document in documents
    ]

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return len(documents)


def search_documents(
    question: str,
    limit: int = 6,
) -> list[Document]:
    """
    Search full research-paper text and table chunks.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Search question cannot be empty.")

    if limit <= 0:
        raise ValueError(
            "Search limit must be greater than zero."
        )

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=cleaned_question,
        k=limit,
    )


def search_arxiv_documents(
    question: str,
    limit: int = 8,
) -> list[Document]:
    """
    Search arXiv metadata and abstracts.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Search question cannot be empty.")

    if limit <= 0:
        raise ValueError(
            "Search limit must be greater than zero."
        )

    vector_store = get_arxiv_vector_store()

    return vector_store.similarity_search(
        query=cleaned_question,
        k=limit,
    )


def get_full_paper_retriever(
    top_k: int = 6,
    fetch_k: int = 15,
):
    """
    Create an MMR retriever for full-paper text and tables.

    MMR balances relevance with diversity so that the retrieved
    chunks are not all nearly identical.
    """
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if fetch_k < top_k:
        raise ValueError(
            "fetch_k must be greater than or equal to top_k."
        )

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.7,
        },
    )


def get_arxiv_retriever(
    top_k: int = 8,
    fetch_k: int = 24,
):
    """
    Create an MMR retriever for arXiv abstracts and metadata.
    """
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if fetch_k < top_k:
        raise ValueError(
            "fetch_k must be greater than or equal to top_k."
        )

    vector_store = get_arxiv_vector_store()

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.7,
        },
    )


def collection_count(
    collection: str,
) -> int:
    """
    Return the number of stored records in a collection.

    Supported values:
    - "papers"
    - "arxiv"
    """
    if collection == "papers":
        vector_store = get_vector_store()
    elif collection == "arxiv":
        vector_store = get_arxiv_vector_store()
    else:
        raise ValueError(
            "Collection must be either 'papers' or 'arxiv'."
        )

    return vector_store._collection.count()