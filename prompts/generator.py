"""
Answer generation prompt with citation support.
Generates answers based on retrieved documents and marks source references.
"""

from langchain_core.prompts import ChatPromptTemplate


GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant for question-answering tasks.

Use the following retrieved context to answer the question. If the context does not contain enough information to answer the question, say so clearly — do not make up information.

**Citation rules:**
- After each key claim or piece of information, add a citation in the format: [Source: filename, page/row N]
- Use the metadata from each document chunk to identify the source file and location.
- If multiple sources support the same point, cite all of them.

**Formatting:**
- Use clear, well-structured language.
- Use bullet points or numbered lists when appropriate.
- Keep the answer concise but complete.

Chat history (for context continuity):
{chat_history}""",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
])


HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Give a binary score: 'yes' means the generation is grounded in the facts, 'no' means it contains statements not supported by the provided facts.

Focus on factual accuracy. Minor phrasing differences are acceptable as long as the meaning is preserved.""",
    ),
    (
        "human",
        "Retrieved facts:\n\n{documents}\n\nLLM generation:\n\n{generation}",
    ),
])


ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader assessing whether an answer addresses / is relevant to a user question.

Give a binary score: 'yes' means the answer is relevant and addresses the question, 'no' means it does not.""",
    ),
    (
        "human",
        "User question: {question}\n\nAnswer: {generation}",
    ),
])
