"""
Document relevance grading prompt: evaluates whether a retrieved document
is relevant to the user question.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDecision(BaseModel):
    """Binary relevance grade for a retrieved document."""
    relevant: str = Field(
        description="Whether the document is relevant to the question. Must be 'yes' or 'no'."
    )


GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader assessing the relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.

Give a binary score: 'yes' or 'no' to indicate whether the document is relevant to the question.""",
    ),
    (
        "human",
        "Retrieved document:\n\n{document}\n\nUser question: {question}",
    ),
])
