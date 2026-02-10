"""
External tools available to the Agent.
"""

import os
from langchain_core.documents import Document

from config import TAVILY_API_KEY


def web_search(query: str, max_results: int = 3) -> list[Document]:
    """Search the web using Tavily and return results as Documents.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int
        Maximum number of results.

    Returns
    -------
    list[Document]
        Web search results as LangChain Documents.
    """
    if not TAVILY_API_KEY:
        return [
            Document(
                page_content="Web search is not available. Please set TAVILY_API_KEY in your .env file.",
                metadata={"source": "web_search_error"},
            )
        ]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=max_results)

        docs = []
        for r in results.get("results", []):
            docs.append(
                Document(
                    page_content=r.get("content", ""),
                    metadata={
                        "source": r.get("url", "web"),
                        "title": r.get("title", ""),
                        "type": "web_search",
                    },
                )
            )
        return docs

    except Exception as e:
        return [
            Document(
                page_content=f"Web search failed: {str(e)}",
                metadata={"source": "web_search_error"},
            )
        ]
