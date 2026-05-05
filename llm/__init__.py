"""LLM client factory: returns OpenAI or local backend by role name."""

from config import (
    ANALYZER_MODEL,
    GENERATOR_MODEL,
    LLM_BACKEND,
    SAFETY_MODEL,
    SUMMARIZER_MODEL,
)
from llm.base_llm import BaseLLM


def get_llm(role: str) -> BaseLLM:
    """
    Factory that returns the correct LLM instance based on config.

    role: "analyzer" | "generator" | "safety" | "summarizer"

    When LLM_BACKEND="openai", uses OpenAI API with role-specific models.
    When LLM_BACKEND="local", uses local vLLM/Ollama server (same model for all roles).
    """
    if LLM_BACKEND == "openai":
        from llm.openai_llm import OpenAILLM
        model_map = {
            "analyzer": ANALYZER_MODEL,
            "generator": GENERATOR_MODEL,
            "safety": SAFETY_MODEL,
            "summarizer": SUMMARIZER_MODEL,
        }
        return OpenAILLM(model=model_map.get(role, GENERATOR_MODEL))
    elif LLM_BACKEND == "local":
        from llm.local_llm import LocalLLM
        return LocalLLM()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Use 'openai' or 'local'.")
