from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Local models
LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Document processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Vector database
COLLECTION_NAME = "research_papers"

# Create required directories automatically
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)