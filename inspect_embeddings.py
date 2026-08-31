from embeddings import get_embedding_model


def main() -> None:
    model = get_embedding_model()

    documents = [
        "The RAG model achieved 91.4 percent accuracy.",
        "The retrieval system improved answer quality.",
    ]

    vectors = model.embed_documents(documents)
    query_vector = model.embed_query(
        "What accuracy did the RAG model achieve?"
    )

    print(f"Document vectors: {len(vectors)}")
    print(f"Embedding dimensions: {len(vectors[0])}")
    print(f"Query dimensions: {len(query_vector)}")
    print(f"First five values: {query_vector[:5]}")


if __name__ == "__main__":
    main()