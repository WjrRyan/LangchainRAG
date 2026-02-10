"""
ChromaDB vector store management.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import VECTORSTORE_DIR, CHROMA_COLLECTION_NAME
from core.embeddings import get_embeddings


class VectorStoreManager:
    """Manage a ChromaDB-backed vector store with persistence.

    Parameters
    ----------
    persist_directory : str or Path, optional
        Directory for ChromaDB persistence. Defaults to config value.
    collection_name : str, optional
        ChromaDB collection name. Defaults to config value.
    embedding_function : Embeddings, optional
        LangChain embedding function. Defaults to Google Generative AI.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        embedding_function: Embeddings | None = None,
    ):
        self._persist_dir = str(persist_directory or VECTORSTORE_DIR)
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME
        self._embedding_fn = embedding_function or get_embeddings()
        self._store: Chroma | None = None

    @property
    def store(self) -> Chroma:
        """Lazy-initialised Chroma vector store."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_fn,
                persist_directory=self._persist_dir,
            )
        return self._store

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Parameters
        ----------
        documents : list[Document]
            Documents with page_content and metadata.

        Returns
        -------
        list[str]
            Document IDs assigned by ChromaDB.
        """
        if not documents:
            return []
        return self.store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """Perform similarity search.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.

        Returns
        -------
        list[Document]
            Most similar documents.
        """
        return self.store.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs: dict | None = None):
        """Return a LangChain Retriever wrapping the vector store.

        Parameters
        ----------
        search_kwargs : dict, optional
            Keyword arguments for the retriever (e.g. {"k": 5}).
        """
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": 5},
        )

    def get_document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.store._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        collection = self.store._collection
        if collection.count() > 0:
            # Get all IDs and delete them
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
