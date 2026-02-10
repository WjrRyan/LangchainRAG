"""
LangGraph Agentic RAG graph construction.

Builds the complete state machine that orchestrates:
- Query routing (Adaptive RAG)
- Standard / Multi-Query / Decomposition retrieval
- Document grading (Corrective RAG)
- Query rewriting
- Web search fallback
- Answer generation with citations
- Hallucination & relevance grading (Self-RAG)
- Conversation memory
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    route_query,
    retrieve,
    multi_query_retrieve,
    decompose_and_answer,
    grade_documents,
    rewrite_query,
    generate,
    grade_generation,
    web_search_node,
    # Conditional edge functions
    route_after_query_analysis,
    route_after_grading,
    route_after_generation_grade,
)


def build_graph() -> StateGraph:
    """Construct the Agentic RAG state graph.

    Returns
    -------
    StateGraph
        The compiled LangGraph graph (not yet compiled — call .compile()).
    """
    workflow = StateGraph(AgentState)

    # ─── Add Nodes ───────────────────────────────────────────────────────
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("multi_query_retrieve", multi_query_retrieve)
    workflow.add_node("decompose_and_answer", decompose_and_answer)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_node("web_search_node", web_search_node)

    # ─── Entry Point ─────────────────────────────────────────────────────
    workflow.set_entry_point("route_query")

    # ─── Conditional Edge: After routing ─────────────────────────────────
    workflow.add_conditional_edges(
        "route_query",
        route_after_query_analysis,
        {
            "retrieve": "retrieve",
            "multi_query_retrieve": "multi_query_retrieve",
            "decompose_and_answer": "decompose_and_answer",
            "web_search_node": "web_search_node",
            "generate": "generate",  # direct route
        },
    )

    # ─── After retrieval → grade documents ───────────────────────────────
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("multi_query_retrieve", "grade_documents")

    # ─── After decompose → grade generation (skip doc grading) ───────────
    workflow.add_edge("decompose_and_answer", "grade_generation")

    # ─── Conditional Edge: After document grading ────────────────────────
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search_node": "web_search_node",
        },
    )

    # ─── After rewrite → re-retrieve ────────────────────────────────────
    workflow.add_edge("rewrite_query", "retrieve")

    # ─── After web search → generate ────────────────────────────────────
    workflow.add_edge("web_search_node", "generate")

    # ─── After generate → grade generation ──────────────────────────────
    workflow.add_edge("generate", "grade_generation")

    # ─── Conditional Edge: After generation grading ──────────────────────
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_generation_grade,
        {
            "finish": END,
            "rewrite_query": "rewrite_query",
        },
    )

    return workflow


def create_app(thread_id: str | None = None):
    """Build and compile the graph with memory checkpointing.

    Parameters
    ----------
    thread_id : str, optional
        Conversation thread ID for memory persistence.

    Returns
    -------
    tuple
        (compiled_graph, config_dict)
    """
    workflow = build_graph()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id or "default"}}
    return app, config


def invoke_agent(
    question: str,
    thread_id: str = "default",
    app=None,
    config: dict | None = None,
) -> AgentState:
    """Invoke the Agentic RAG pipeline with a user question.

    Parameters
    ----------
    question : str
        User question.
    thread_id : str
        Conversation thread ID for memory.
    app : optional
        Pre-compiled graph. If None, creates a new one.
    config : dict, optional
        Configuration dict. If None, creates from thread_id.

    Returns
    -------
    AgentState
        The final state after graph execution.
    """
    if app is None or config is None:
        app, config = create_app(thread_id)

    initial_state = {
        "question": question,
        "original_question": question,
        "chat_history": [],
        "documents": [],
        "generation": "",
        "route": "",
        "route_reasoning": "",
        "query_rewrite_count": 0,
        "web_search_needed": False,
        "citations": [],
        "sub_questions": [],
        "sub_answers": [],
        "steps": [],
    }

    result = app.invoke(initial_state, config=config)
    return result
