"""
Query decomposition prompt: breaks a complex question into smaller,
sequential sub-questions that can be answered independently.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class DecomposedQuestions(BaseModel):
    """Structured output for query decomposition."""
    sub_questions: list[str] = Field(
        description=(
            "A list of 2-4 sub-questions that, when answered in order, "
            "will provide a complete answer to the original complex question."
        )
    )


DECOMPOSER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at breaking down complex questions into simpler sub-questions.

Given a complex question, decompose it into 2-4 smaller sub-questions that:
1. Can each be answered independently through document retrieval
2. Are ordered logically (later questions may depend on earlier answers)
3. Together provide all the information needed to answer the original question
4. Are specific enough for effective vector store retrieval

Return the sub-questions as a structured list.""",
    ),
    ("human", "Complex question: {question}"),
])


SUB_ANSWER_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that synthesizes answers from multiple sub-question answers into a comprehensive final answer.

**Citation rules:**
- Preserve all citations from the sub-answers in the format: [Source: filename, page/row N]
- Combine and organize the information logically.
- Eliminate redundancy while keeping all important details.
- If sub-answers contradict each other, mention the discrepancy.""",
    ),
    (
        "human",
        "Original question: {question}\n\n"
        "Sub-questions and their answers:\n{sub_answers}\n\n"
        "Please synthesize a comprehensive answer to the original question.",
    ),
])
