from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import FETCH_K, LLM_MODEL, TOP_K
from vector_store import get_vector_store


SYSTEM_PROMPT = """
You are an academic research assistant.

Answer the user's question using only the retrieved evidence.

Rules:
1. Every factual claim must include the exact citation supplied
   with its source.
2. Preserve numerical values, units and percentages exactly.
3. When using a table, cite the table number.
4. Distinguish findings, methods and limitations.
5. If papers disagree, explain the disagreement.
6. Do not invent facts, authors, results or citations.
7. Do not cite a source unless it supports the claim.
8. If the evidence is insufficient, respond:
   "The indexed papers do not contain enough evidence to answer
   this question."
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Retrieved research evidence:

{context}

Research question:
{question}

Provide a concise, evidence-based answer with citations.
""",
        ),
    ]
)


def create_citation(document: Document) -> str:
    """Create a citation from document metadata."""
    metadata = document.metadata

    citation = (
        f"[{metadata['title']}, "
        f"p. {metadata['page']}"
    )

    if metadata["content_type"] == "table":
        citation += (
            f", Table {metadata['table_number']}"
        )

    citation += "]"
    return citation


def format_document(
    document: Document,
    source_number: int,
) -> str:
    """Format one retrieved document for the LLM."""
    citation = create_citation(document)

    return (
        f"SOURCE {source_number}\n"
        f"REQUIRED CITATION: {citation}\n"
        f"CONTENT TYPE: "
        f"{document.metadata['content_type']}\n"
        f"EVIDENCE:\n{document.page_content}"
    )


def retrieve_documents(
    question: str,
) -> list[Document]:
    """
    Retrieve relevant and diverse evidence.

    MMR reduces the chance that all retrieved chunks contain
    nearly identical information.
    """
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": 0.7,
        },
    )

    return retriever.invoke(question)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Load and reuse the local Ollama chat model."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """Generate an answer from retrieved documents."""
    if not documents:
        return (
            "The indexed papers do not contain enough evidence "
            "to answer this question."
        )

    context = "\n\n---\n\n".join(
        format_document(document, source_number)
        for source_number, document in enumerate(
            documents,
            start=1,
        )
    )

    chain = PROMPT | get_llm() | StrOutputParser()

    return chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )


def answer_question(question: str) -> dict:
    """Run retrieval and answer generation."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    documents = retrieve_documents(cleaned_question)

    answer = generate_answer(
        cleaned_question,
        documents,
    )

    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": documents,
    }