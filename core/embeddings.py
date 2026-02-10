"""
Embedding model wrapper for Google Generative AI Embeddings.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embeddings(model: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Return a configured GoogleGenerativeAIEmbeddings instance.

    Parameters
    ----------
    model : str, optional
        Override the default embedding model name from config.
    """
    return GoogleGenerativeAIEmbeddings(
        model=model or EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
