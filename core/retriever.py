"""
Retriever wrapper with configurable search strategies.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import RETRIEVER_TOP_K
from core.vectorstore import VectorStoreManager


def get_retriever(
    vs_manager: VectorStoreManager | None = None,
    top_k: int | None = None,
) -> BaseRetriever:
    """Return a LangChain retriever backed by the vector store.

    Parameters
    ----------
    vs_manager : VectorStoreManager, optional
        Pre-existing vector store manager. Creates a default one if None.
    top_k : int, optional
        Number of documents to retrieve. Defaults to config value.

    Returns
    -------
    BaseRetriever
        A LangChain-compatible retriever.
    """
    manager = vs_manager or VectorStoreManager()
    k = top_k or RETRIEVER_TOP_K
    return manager.as_retriever(search_kwargs={"k": k})


def retrieve_documents(
    query: str,
    retriever: BaseRetriever | None = None,
    vs_manager: VectorStoreManager | None = None,
    top_k: int | None = None,
) -> list[Document]:
    """Retrieve documents relevant to a query.

    Parameters
    ----------
    query : str
        The search query.
    retriever : BaseRetriever, optional
        Pre-configured retriever. If None, one is created from vs_manager.
    vs_manager : VectorStoreManager, optional
        Vector store manager to create a retriever from.
    top_k : int, optional
        Number of results.

    Returns
    -------
    list[Document]
        Retrieved documents.
    """
    if retriever is None:
        retriever = get_retriever(vs_manager, top_k)
    return retriever.invoke(query)
