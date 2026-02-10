"""
Document ingestion CLI script.

Usage:
    # Ingest a single file
    python ingest.py path/to/document.pdf

    # Ingest multiple files
    python ingest.py doc1.pdf doc2.md data.csv

    # Ingest all supported files in a directory
    python ingest.py --dir path/to/documents/

    # Clear the vector store before ingesting
    python ingest.py --clear --dir path/to/documents/

    # Show vector store stats
    python ingest.py --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DATA_DIR
from core.document_loader import load_document, load_directory, SUPPORTED_EXTENSIONS
from core.text_splitter import split_documents
from core.vectorstore import VectorStoreManager


def ingest_files(
    file_paths: list[str | Path],
    vs_manager: VectorStoreManager | None = None,
    verbose: bool = True,
) -> int:
    """Ingest files into the vector store.

    Parameters
    ----------
    file_paths : list
        Paths to document files.
    vs_manager : VectorStoreManager, optional
        Pre-existing vector store manager.
    verbose : bool
        Print progress info.

    Returns
    -------
    int
        Number of chunks added to the vector store.
    """
    manager = vs_manager or VectorStoreManager()
    total_chunks = 0

    for fp in file_paths:
        path = Path(fp)
        if verbose:
            print(f"  Loading: {path.name} ... ", end="", flush=True)

        try:
            docs = load_document(path)
            chunks = split_documents(docs)
            manager.add_documents(chunks)
            total_chunks += len(chunks)

            if verbose:
                print(f"OK ({len(docs)} docs → {len(chunks)} chunks)")
        except Exception as e:
            if verbose:
                print(f"FAILED ({e})")

    return total_chunks


def ingest_directory(
    directory: str | Path,
    vs_manager: VectorStoreManager | None = None,
    verbose: bool = True,
) -> int:
    """Ingest all supported files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to scan.
    vs_manager : VectorStoreManager, optional
        Pre-existing vector store manager.
    verbose : bool
        Print progress info.

    Returns
    -------
    int
        Number of chunks added.
    """
    dir_path = Path(directory)
    files = [
        f for f in sorted(dir_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if verbose:
        print(f"Found {len(files)} supported files in {dir_path}")

    return ingest_files(files, vs_manager, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Agentic RAG vector store."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Document files to ingest (PDF, Markdown, CSV).",
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory to ingest all supported documents from.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector store before ingesting.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show vector store statistics and exit.",
    )

    args = parser.parse_args()

    manager = VectorStoreManager()

    # Show stats
    if args.stats:
        count = manager.get_document_count()
        print(f"Vector store contains {count} document chunks.")
        return

    # Clear if requested
    if args.clear:
        print("Clearing vector store...")
        manager.clear()
        print("Done.")

    # Ingest
    total = 0
    if args.dir:
        print(f"\nIngesting directory: {args.dir}")
        total += ingest_directory(args.dir, manager)

    if args.files:
        print(f"\nIngesting {len(args.files)} file(s):")
        total += ingest_files(args.files, manager)

    if not args.dir and not args.files and not args.clear and not args.stats:
        parser.print_help()
        return

    print(f"\nTotal: {total} chunks added to vector store.")
    print(f"Vector store now contains {manager.get_document_count()} chunks.")


if __name__ == "__main__":
    main()
