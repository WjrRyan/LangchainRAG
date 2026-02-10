"""
Multi-Query retrieval: generate multiple sub-queries from different angles
and merge deduplicated results to improve recall.
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import MULTI_QUERY_COUNT, RETRIEVER_TOP_K
from core.llm import get_llm
from core.vectorstore import VectorStoreManager


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that generates multiple search queries "
        "based on a single input question. Generate {count} different versions "
        "of the given question to retrieve relevant documents from a vector "
        "database. Each version should approach the question from a different "
        "angle or use different keywords.\n\n"
        "Output ONLY the queries, one per line, without numbering or bullet points.",
    ),
    ("human", "{question}"),
])


def generate_multi_queries(
    question: str,
    count: int | None = None,
) -> list[str]:
    """Generate multiple search queries from different angles.

    Parameters
    ----------
    question : str
        Original user question.
    count : int, optional
        Number of sub-queries to generate. Defaults to config.

    Returns
    -------
    list[str]
        Generated sub-queries.
    """
    llm = get_llm(temperature=0.7)  # Slightly creative for diversity
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()

    result = chain.invoke({
        "question": question,
        "count": count or MULTI_QUERY_COUNT,
    })

    # Parse the multi-line output
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return queries


def multi_query_retrieve(
    question: str,
    vs_manager: VectorStoreManager | None = None,
    query_count: int | None = None,
    top_k: int | None = None,
) -> list[Document]:
    """Perform multi-query retrieval: generate sub-queries, retrieve for each,
    then merge and deduplicate results.

    Parameters
    ----------
    question : str
        Original user question.
    vs_manager : VectorStoreManager, optional
        Vector store manager. Creates default if None.
    query_count : int, optional
        Number of sub-queries.
    top_k : int, optional
        Documents per sub-query.

    Returns
    -------
    list[Document]
        Merged and deduplicated documents.
    """
    manager = vs_manager or VectorStoreManager()
    k = top_k or RETRIEVER_TOP_K

    # Generate sub-queries
    sub_queries = generate_multi_queries(question, query_count)

    # Retrieve for each sub-query
    seen_contents: set[str] = set()
    unique_docs: list[Document] = []

    for query in [question] + sub_queries:  # Include original query
        docs = manager.similarity_search(query, k=k)
        for doc in docs:
            # Deduplicate by content hash
            content_key = doc.page_content[:200]  # Use first 200 chars as key
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_docs.append(doc)

    return unique_docs
