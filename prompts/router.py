"""
Query routing prompt: determines the best retrieval strategy for a user question.

Routes to one of:
- "vectorstore"   : standard vector similarity search
- "multi_query"   : multi-angle retrieval for vague/broad questions
- "decompose"     : break complex question into sub-questions
- "web_search"    : search the web for recent/external information
- "direct"        : answer directly without retrieval
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output for query routing."""
    route: str = Field(
        description=(
            "The routing destination. Must be one of: "
            "'vectorstore', 'multi_query', 'decompose', 'web_search', 'direct'."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen."
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at routing user questions to the most appropriate retrieval strategy.

You have access to a vector store containing user-uploaded documents (PDFs, Markdown files, CSVs).

Based on the user question, choose ONE of the following routes:

1. **vectorstore** — The question is clear, specific, and likely answerable from the uploaded documents. A single focused search should find relevant content.

2. **multi_query** — The question is vague, broad, or could benefit from searching from multiple angles. Examples: "Tell me about this system", "What are the key points?", "Summarize the document".

3. **decompose** — The question is complex and involves multiple entities, comparisons, or requires multi-step reasoning. It needs to be broken into sub-questions. Examples: "What are the differences between X and Y?", "How does A affect B and what are the implications for C?".

4. **web_search** — The question is about recent events, current information, or topics clearly outside the uploaded documents. Examples: "What is today's weather?", "Latest news about...".

5. **direct** — The question is a greeting, casual chat, or can be answered without any retrieval. Examples: "Hello", "What can you do?", "Thanks".

Consider the chat history for context if available.

Respond with the route and a brief reasoning.""",
    ),
    ("human", "Chat history:\n{chat_history}\n\nQuestion: {question}"),
])
