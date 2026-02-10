"""
LangGraph node implementations for the Agentic RAG pipeline.

Each function takes and returns an AgentState dict, updating the
relevant fields. Nodes:

1. route_query          — Classify the question and choose a retrieval strategy
2. retrieve             — Standard vector similarity search
3. multi_query_retrieve — Multi-angle retrieval with sub-query generation
4. decompose_and_answer — Break complex question into sub-questions
5. grade_documents      — Evaluate retrieved documents for relevance
6. rewrite_query        — Rewrite query for better retrieval
7. generate             — Generate an answer with citations
8. grade_generation     — Check for hallucinations and answer relevance
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config import MAX_QUERY_REWRITE_COUNT, MAX_DECOMPOSITION_STEPS, RETRIEVER_TOP_K
from core.llm import get_llm
from core.vectorstore import VectorStoreManager
from core.multi_query import multi_query_retrieve as _multi_query_retrieve
from agent.tools import web_search
from agent.state import AgentState

from prompts.router import ROUTER_PROMPT, RouteDecision
from prompts.grader import GRADER_PROMPT, GradeDecision
from prompts.generator import (
    GENERATOR_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
)
from prompts.rewriter import REWRITER_PROMPT
from prompts.decomposer import (
    DECOMPOSER_PROMPT,
    SUB_ANSWER_SYNTHESIS_PROMPT,
    DecomposedQuestions,
)


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _get_vs_manager() -> VectorStoreManager:
    """Return a shared VectorStoreManager instance."""
    return VectorStoreManager()


def _format_docs(docs: list[Document]) -> str:
    """Format documents into a single string with source metadata."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        row = doc.metadata.get("row")
        location = ""
        if page is not None:
            location = f", page {page + 1}"  # 0-indexed to 1-indexed
        elif row is not None:
            location = f", row {row + 1}"
        parts.append(
            f"[Document {i}] (Source: {source}{location})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _format_chat_history(messages: list) -> str:
    """Format chat history messages into a readable string."""
    if not messages:
        return "(No prior conversation)"
    lines = []
    for msg in messages[-10:]:  # Keep last 10 messages for context window
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _add_step(state: dict, step_name: str, detail: str) -> list[dict]:
    """Append a reasoning step to the steps trace."""
    steps = list(state.get("steps", []))
    steps.append({"step": step_name, "detail": detail})
    return steps


# ─── Node 1: Route Query ────────────────────────────────────────────────────

def route_query(state: AgentState) -> dict:
    """Analyse the question and decide the retrieval strategy.

    Updates: route, route_reasoning, steps
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    chain = ROUTER_PROMPT | structured_llm

    question = state["question"]
    chat_history = _format_chat_history(state.get("chat_history", []))

    decision: RouteDecision = chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    steps = _add_step(
        state,
        "Query Routing",
        f"Route: {decision.route} | Reason: {decision.reasoning}",
    )

    return {
        "route": decision.route,
        "route_reasoning": decision.reasoning,
        "steps": steps,
    }


# ─── Node 2: Retrieve ───────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    """Perform standard vector similarity search.

    Updates: documents, steps
    """
    question = state["question"]
    manager = _get_vs_manager()
    docs = manager.similarity_search(question, k=RETRIEVER_TOP_K)

    steps = _add_step(
        state,
        "Vector Retrieval",
        f"Retrieved {len(docs)} documents for: '{question}'",
    )

    return {"documents": docs, "steps": steps}


# ─── Node 3: Multi-Query Retrieve ───────────────────────────────────────────

def multi_query_retrieve(state: AgentState) -> dict:
    """Generate multiple sub-queries and retrieve with deduplication.

    Updates: documents, steps
    """
    question = state["question"]
    manager = _get_vs_manager()
    docs = _multi_query_retrieve(question, vs_manager=manager)

    steps = _add_step(
        state,
        "Multi-Query Retrieval",
        f"Generated multiple sub-queries and retrieved {len(docs)} unique documents",
    )

    return {"documents": docs, "steps": steps}


# ─── Node 4: Decompose and Answer ───────────────────────────────────────────

def decompose_and_answer(state: AgentState) -> dict:
    """Break complex question into sub-questions, answer each, then synthesize.

    Updates: sub_questions, sub_answers, documents, generation, steps
    """
    llm = get_llm()
    question = state["question"]
    chat_history = _format_chat_history(state.get("chat_history", []))
    manager = _get_vs_manager()

    # Step 1: Decompose
    decompose_chain = DECOMPOSER_PROMPT | llm.with_structured_output(DecomposedQuestions)
    decomposed: DecomposedQuestions = decompose_chain.invoke({"question": question})
    sub_questions = decomposed.sub_questions[:MAX_DECOMPOSITION_STEPS]

    steps = _add_step(
        state,
        "Query Decomposition",
        f"Decomposed into {len(sub_questions)} sub-questions: {sub_questions}",
    )

    # Step 2: Answer each sub-question
    gen_chain = GENERATOR_PROMPT | llm | StrOutputParser()
    all_docs: list[Document] = []
    sub_answers: list[dict] = []

    for i, sub_q in enumerate(sub_questions, 1):
        # Retrieve for sub-question
        sub_docs = manager.similarity_search(sub_q, k=RETRIEVER_TOP_K)
        all_docs.extend(sub_docs)

        # Generate answer for sub-question
        context = _format_docs(sub_docs)
        sub_answer = gen_chain.invoke({
            "context": context,
            "question": sub_q,
            "chat_history": chat_history,
        })

        sub_answers.append({
            "question": sub_q,
            "answer": sub_answer,
            "sources": [d.metadata for d in sub_docs],
        })

        steps = _add_step(
            {"steps": steps},
            f"Sub-question {i}",
            f"Q: {sub_q}\nA: {sub_answer[:200]}...",
        )

    # Step 3: Synthesize
    sub_answers_text = "\n\n".join(
        f"**Sub-question {i}**: {sa['question']}\n**Answer**: {sa['answer']}"
        for i, sa in enumerate(sub_answers, 1)
    )

    synthesis_chain = SUB_ANSWER_SYNTHESIS_PROMPT | llm | StrOutputParser()
    final_answer = synthesis_chain.invoke({
        "question": question,
        "sub_answers": sub_answers_text,
    })

    steps = _add_step(
        {"steps": steps},
        "Synthesis",
        "Synthesized sub-answers into final answer",
    )

    return {
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "documents": all_docs,
        "generation": final_answer,
        "steps": steps,
    }


# ─── Node 5: Grade Documents ────────────────────────────────────────────────

def grade_documents(state: AgentState) -> dict:
    """Evaluate each retrieved document for relevance, filter out irrelevant ones.

    Updates: documents, web_search_needed, steps
    """
    llm = get_llm()
    grade_chain = GRADER_PROMPT | llm.with_structured_output(GradeDecision)

    question = state["question"]
    documents = state.get("documents", [])

    relevant_docs: list[Document] = []
    irrelevant_count = 0

    for doc in documents:
        grade: GradeDecision = grade_chain.invoke({
            "document": doc.page_content,
            "question": question,
        })
        if grade.relevant.lower() == "yes":
            relevant_docs.append(doc)
        else:
            irrelevant_count += 1

    # If no relevant docs found, flag for potential web search or rewrite
    web_search_needed = len(relevant_docs) == 0

    steps = _add_step(
        state,
        "Document Grading",
        f"Relevant: {len(relevant_docs)}, Irrelevant: {irrelevant_count}. "
        f"Web search needed: {web_search_needed}",
    )

    return {
        "documents": relevant_docs,
        "web_search_needed": web_search_needed,
        "steps": steps,
    }


# ─── Node 6: Rewrite Query ──────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:
    """Rewrite the query for better retrieval results.

    Updates: question, query_rewrite_count, steps
    """
    llm = get_llm()
    rewrite_chain = REWRITER_PROMPT | llm | StrOutputParser()

    question = state["question"]
    rewrite_count = state.get("query_rewrite_count", 0) + 1

    new_question = rewrite_chain.invoke({"question": question})

    steps = _add_step(
        state,
        f"Query Rewrite (attempt {rewrite_count})",
        f"'{question}' → '{new_question}'",
    )

    return {
        "question": new_question,
        "query_rewrite_count": rewrite_count,
        "steps": steps,
    }


# ─── Node 7: Generate ───────────────────────────────────────────────────────

def generate(state: AgentState) -> dict:
    """Generate an answer with citations based on retrieved documents.

    Updates: generation, citations, steps
    """
    llm = get_llm(temperature=0.3)  # Slightly creative for natural answers
    gen_chain = GENERATOR_PROMPT | llm | StrOutputParser()

    question = state["question"]
    documents = state.get("documents", [])
    chat_history = _format_chat_history(state.get("chat_history", []))
    route = state.get("route", "")

    # For direct route, generate without context
    if route == "direct":
        context = "(No documents retrieved — this is a direct response.)"
    else:
        context = _format_docs(documents) if documents else "(No relevant documents found.)"

    generation = gen_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })

    # Extract citation metadata from documents
    citations = []
    for doc in documents:
        citation = {
            "source": doc.metadata.get("source", "unknown"),
        }
        if "page" in doc.metadata:
            citation["page"] = doc.metadata["page"] + 1  # 1-indexed
        if "row" in doc.metadata:
            citation["row"] = doc.metadata["row"] + 1
        if "title" in doc.metadata:
            citation["title"] = doc.metadata["title"]
        if "type" in doc.metadata:
            citation["type"] = doc.metadata["type"]
        citations.append(citation)

    # Deduplicate citations
    seen = set()
    unique_citations = []
    for c in citations:
        key = (c["source"], c.get("page"), c.get("row"))
        if key not in seen:
            seen.add(key)
            unique_citations.append(c)

    steps = _add_step(
        state,
        "Answer Generation",
        f"Generated answer ({len(generation)} chars) with {len(unique_citations)} source(s)",
    )

    return {
        "generation": generation,
        "citations": unique_citations,
        "steps": steps,
    }


# ─── Node 8: Grade Generation ───────────────────────────────────────────────

def grade_generation(state: AgentState) -> dict:
    """Check for hallucinations and answer relevance.

    Updates: steps
    Returns the state as-is; the graph uses conditional edges based on grading.
    """
    llm = get_llm()
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    question = state.get("original_question", state.get("question", ""))

    # Skip grading for direct responses or decomposed answers
    route = state.get("route", "")
    if route in ("direct", "decompose"):
        steps = _add_step(
            state,
            "Grade Skipped",
            f"Route '{route}' — skipping hallucination/relevance check",
        )
        return {"steps": steps}

    # Grade 1: Hallucination check
    hall_chain = HALLUCINATION_GRADER_PROMPT | llm.with_structured_output(GradeDecision)
    docs_text = _format_docs(documents) if documents else ""

    hallucination_grade: GradeDecision = hall_chain.invoke({
        "documents": docs_text,
        "generation": generation,
    })

    is_grounded = hallucination_grade.relevant.lower() == "yes"

    # Grade 2: Answer relevance check
    rel_chain = ANSWER_RELEVANCE_PROMPT | llm.with_structured_output(GradeDecision)
    relevance_grade: GradeDecision = rel_chain.invoke({
        "question": question,
        "generation": generation,
    })

    is_relevant = relevance_grade.relevant.lower() == "yes"

    steps = _add_step(
        state,
        "Generation Grading",
        f"Grounded in facts: {'Yes' if is_grounded else 'No'} | "
        f"Answers question: {'Yes' if is_relevant else 'No'}",
    )

    # Store grading results for conditional edge routing
    result = {"steps": steps}
    if not is_grounded or not is_relevant:
        result["web_search_needed"] = True

    return result


# ─── Conditional Edge Functions ──────────────────────────────────────────────

def route_after_query_analysis(state: AgentState) -> str:
    """Conditional edge after route_query: determine next node."""
    route = state.get("route", "vectorstore")
    if route == "multi_query":
        return "multi_query_retrieve"
    elif route == "decompose":
        return "decompose_and_answer"
    elif route == "web_search":
        return "web_search_node"
    elif route == "direct":
        return "generate"
    else:  # "vectorstore" or default
        return "retrieve"


def route_after_grading(state: AgentState) -> str:
    """Conditional edge after grade_documents: decide to generate or rewrite."""
    documents = state.get("documents", [])
    rewrite_count = state.get("query_rewrite_count", 0)
    web_search_needed = state.get("web_search_needed", False)

    if documents:
        return "generate"
    elif rewrite_count >= MAX_QUERY_REWRITE_COUNT:
        # Exhausted rewrites — try web search or generate with what we have
        return "web_search_node"
    else:
        return "rewrite_query"


def route_after_generation_grade(state: AgentState) -> str:
    """Conditional edge after grade_generation: accept or retry."""
    web_search_needed = state.get("web_search_needed", False)
    rewrite_count = state.get("query_rewrite_count", 0)

    if not web_search_needed:
        return "finish"
    elif rewrite_count >= MAX_QUERY_REWRITE_COUNT:
        return "finish"  # Accept imperfect answer rather than loop forever
    else:
        return "rewrite_query"


# ─── Web Search Node ────────────────────────────────────────────────────────

def web_search_node(state: AgentState) -> dict:
    """Perform web search and add results to documents.

    Updates: documents, steps
    """
    question = state["question"]
    web_docs = web_search(question)

    # Combine with any existing relevant docs
    existing_docs = state.get("documents", [])
    all_docs = existing_docs + web_docs

    steps = _add_step(
        state,
        "Web Search",
        f"Found {len(web_docs)} web results for: '{question}'",
    )

    return {"documents": all_docs, "steps": steps}
