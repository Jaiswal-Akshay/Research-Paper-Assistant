from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import ARXIV_TOP_K, LLM_MODEL
from vector_store import get_arxiv_vector_store


SYSTEM_PROMPT = """
You are an academic literature-discovery assistant.

Answer using only the supplied arXiv metadata and abstracts.

Rules:
1. Cite papers using [Title, arXiv:ID].
2. Do not claim that an abstract proves a result.
3. Clearly use language such as "the abstract states,"
   "the authors propose," or "the paper reports."
4. Do not invent authors, methods, results or citations.
5. If the retrieved metadata does not provide enough evidence,
   say:
   "The indexed arXiv metadata does not contain enough
   information to answer this question."
6. When recommending papers, explain their relevance based on
   the abstract.
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Retrieved arXiv metadata:

{context}

Question:
{question}

Answer concisely and include citations.
""",
        ),
    ]
)


def format_arxiv_document(
    document: Document,
    position: int,
) -> str:
    """Format one metadata record for Llama 3."""
    metadata = document.metadata

    citation = (
        f"[{metadata['title']}, "
        f"arXiv:{metadata['arxiv_id']}]"
    )

    return (
        f"SOURCE {position}\n"
        f"REQUIRED CITATION: {citation}\n"
        f"ABSTRACT URL: {metadata['abstract_url']}\n"
        f"{document.page_content}"
    )


def retrieve_arxiv_metadata(
    question: str,
) -> list[Document]:
    """Retrieve relevant arXiv papers."""
    vector_store = get_arxiv_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": ARXIV_TOP_K,
            "fetch_k": min(ARXIV_TOP_K * 3, 30),
            "lambda_mult": 0.7,
        },
    )

    return retriever.invoke(question)


@lru_cache(maxsize=1)
def get_arxiv_llm() -> ChatOllama:
    """Create and cache the local Llama 3 connection."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


def answer_arxiv_question(question: str) -> dict:
    """Answer a question using indexed arXiv metadata."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    documents = retrieve_arxiv_metadata(
        cleaned_question
    )

    if not documents:
        return {
            "answer": (
                "The indexed arXiv metadata does not contain "
                "enough information to answer this question."
            ),
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        format_arxiv_document(document, position)
        for position, document in enumerate(
            documents,
            start=1,
        )
    )

    chain = (
        PROMPT
        | get_arxiv_llm()
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "question": cleaned_question,
            "context": context,
        }
    )

    return {
        "answer": answer,
        "sources": documents,
    }