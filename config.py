from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Local models
LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LangChain text splitting
# RecursiveCharacterTextSplitter measures characters by default.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Number of table rows stored in each table chunk
TABLE_ROWS_PER_CHUNK = 20

# Retrieval
TOP_K = 4
FETCH_K = 10

# Vector database
COLLECTION_NAME = "research_papers"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# arXiv metadata retrieval
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_COLLECTION_NAME = "arxiv_metadata"
ARXIV_MAX_RESULTS = 100
ARXIV_REQUEST_TIMEOUT = 60

# Number of metadata records retrieved for a question
ARXIV_TOP_K = 8

# Identify our application to arXiv.
ARXIV_USER_AGENT = "ResearchPaperAssistant/0.1"