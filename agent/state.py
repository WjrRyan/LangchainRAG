"""
LangGraph Agent State definition.

The state flows through the graph and is updated by each node.
"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the Agentic RAG graph.

    Attributes
    ----------
    question : str
        The current user question (may be rewritten).
    original_question : str
        The original user question (preserved for reference).
    chat_history : list
        Conversation history as LangChain messages, accumulated via add_messages.
    documents : list[Document]
        Retrieved documents from the current retrieval step.
    generation : str
        The generated answer text.
    route : str
        The chosen routing destination (vectorstore/multi_query/decompose/web_search/direct).
    route_reasoning : str
        Brief explanation from the router for debugging.
    query_rewrite_count : int
        Number of times the query has been rewritten (to prevent infinite loops).
    web_search_needed : bool
        Whether web search should be performed.
    citations : list[dict]
        Extracted citation metadata from the generation.
    sub_questions : list[str]
        Decomposed sub-questions for complex queries.
    sub_answers : list[dict]
        Answers to each sub-question with their sources.
    steps : list[dict]
        Agent reasoning trace for UI display.
    """
    question: str
    original_question: str
    chat_history: Annotated[list, add_messages]
    documents: list[Document]
    generation: str
    route: str
    route_reasoning: str
    query_rewrite_count: int
    web_search_needed: bool
    citations: list[dict]
    sub_questions: list[str]
    sub_answers: list[dict]
    steps: list[dict]
