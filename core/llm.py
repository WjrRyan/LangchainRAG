"""
LLM wrapper for Google Gemini via LangChain.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Return a configured ChatGoogleGenerativeAI instance.

    Parameters
    ----------
    model : str, optional
        Override the default model name from config.
    temperature : float, optional
        Override the default temperature from config.
    """
    return ChatGoogleGenerativeAI(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,
    )
