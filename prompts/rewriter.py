"""
Query rewriting prompt: transforms a vague or failed query into a more
precise search query for better retrieval results.
"""

from langchain_core.prompts import ChatPromptTemplate


REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query rewriter that converts a user question into a better version optimized for vector store retrieval.

Look at the original question and try to reason about the underlying semantic intent. Then rewrite it to be:
1. More specific and focused
2. Using different keywords that might match document content
3. Removing ambiguity

Output ONLY the rewritten query, nothing else.""",
    ),
    ("human", "Original question: {question}"),
])
