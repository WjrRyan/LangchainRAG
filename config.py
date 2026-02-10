"""
Global configuration for the Agentic RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ─── API Keys ────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ─── LLM Configuration ──────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.0  # Deterministic for grading/routing; override per-node if needed

# ─── Embedding Configuration ────────────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ─── ChromaDB Configuration ─────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "langchain_rag"

# ─── Text Splitting ─────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ─── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVER_TOP_K = 5               # Number of documents to retrieve
MULTI_QUERY_COUNT = 4             # Number of sub-queries for Multi-Query retrieval

# ─── Agent ───────────────────────────────────────────────────────────────────
MAX_QUERY_REWRITE_COUNT = 3       # Max rewrite attempts to avoid infinite loops
MAX_DECOMPOSITION_STEPS = 4       # Max sub-questions for query decomposition
