from config import (
    CHROMA_DIR,
    UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    LLM_MODEL,
)


def test_directories_exist():
    assert UPLOAD_DIR.exists()
    assert CHROMA_DIR.exists()


def test_chunk_configuration():
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_retrieval_configuration():
    assert TOP_K > 0


def test_llm_configuration():
    assert LLM_MODEL == "llama3:latest"