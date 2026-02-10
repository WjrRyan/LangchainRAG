"""
Multi-format document loader supporting PDF, Markdown, and CSV files.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)


# Map file extensions to loader classes
_LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())


def load_document(file_path: str | Path) -> list[Document]:
    """Load a single document file and return a list of LangChain Document objects.

    Supported formats: .pdf, .md, .csv

    Each Document's metadata will include the original ``source`` file path.
    PDF documents additionally have a ``page`` field (0-indexed).
    CSV documents additionally have a ``row`` field.

    Parameters
    ----------
    file_path : str or Path
        Path to the document file.

    Returns
    -------
    list[Document]
        Loaded documents.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in _LOADER_MAP:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    loader_cls = _LOADER_MAP[ext]
    loader = loader_cls(str(path))
    docs = loader.load()

    # Ensure consistent metadata
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("source", str(path))
        if ext == ".csv":
            doc.metadata.setdefault("row", i)

    return docs


def load_documents(file_paths: list[str | Path]) -> list[Document]:
    """Load multiple document files.

    Parameters
    ----------
    file_paths : list
        List of file paths to load.

    Returns
    -------
    list[Document]
        All loaded documents concatenated.
    """
    all_docs: list[Document] = []
    for fp in file_paths:
        all_docs.extend(load_document(fp))
    return all_docs


def load_directory(directory: str | Path) -> list[Document]:
    """Load all supported documents from a directory (non-recursive).

    Parameters
    ----------
    directory : str or Path
        Directory to scan for documents.

    Returns
    -------
    list[Document]
        All loaded documents from the directory.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    files = [
        f for f in sorted(dir_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return load_documents(files)
