"""
Text splitting strategies for different document types.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a configured RecursiveCharacterTextSplitter.

    Parameters
    ----------
    chunk_size : int, optional
        Override default chunk size from config.
    chunk_overlap : int, optional
        Override default chunk overlap from config.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split a list of documents into chunks.

    CSV documents (detected by 'row' in metadata) are returned as-is since
    each row is already a self-contained unit.

    Parameters
    ----------
    documents : list[Document]
        Documents to split.
    chunk_size : int, optional
        Override default chunk size.
    chunk_overlap : int, optional
        Override default chunk overlap.

    Returns
    -------
    list[Document]
        Chunked documents.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)

    # Separate CSV docs (no splitting needed) from others
    csv_docs = [d for d in documents if "row" in d.metadata]
    other_docs = [d for d in documents if "row" not in d.metadata]

    chunks = splitter.split_documents(other_docs) if other_docs else []
    return chunks + csv_docs
