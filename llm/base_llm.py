from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from pydantic import BaseModel


class BaseLLM(ABC):
    """Abstract LLM interface. Every agent calls this — never calls OpenAI directly."""

    @abstractmethod
    async def generate_json(
        self,
        messages: list[dict],
        response_schema: type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> BaseModel:
        """
        Send messages, expect structured JSON output matching the Pydantic schema.
        Used by Analyzer and Safety agents.

        Must:
        1. Call the LLM with the messages
        2. Parse the response as JSON
        3. Validate against the Pydantic schema
        4. If validation fails, retry ONCE with the error appended to messages
        5. If second attempt fails, raise ValueError with details
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.75,
        max_tokens: int = 250,
    ) -> AsyncGenerator[str, None]:
        """
        Send messages, stream response tokens back.
        Used by Generator agent.

        Must yield each token as it arrives.
        """
        ...

    @abstractmethod
    async def generate_text(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> str:
        """
        Send messages, return complete text response (non-streaming).
        Used by Safety agent's LLM classifier.
        """
        ...
