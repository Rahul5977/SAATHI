"""OpenAI Async API wrapper implementing `BaseLLM`."""

from openai import AsyncOpenAI
import json
import logging
from pydantic import BaseModel
from typing import AsyncGenerator
from llm.base_llm import BaseLLM
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    def __init__(self, model: str):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    async def generate_json(
        self,
        messages: list[dict],
        response_schema: type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> BaseModel:
        raw_text = ""
        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content or ""
                parsed = json.loads(raw_text)
                return response_schema.model_validate(parsed)
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"JSON parse/validation failed (attempt 1): {e}")
                    messages = messages + [
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output was invalid: {str(e)}. "
                                "Fix the JSON and try again. Return ONLY valid JSON."
                            ),
                        },
                    ]
                else:
                    logger.error(f"JSON parse/validation failed (attempt 2): {e}")
                    raise ValueError(f"Failed to get valid JSON after 2 attempts: {e}")

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.75,
        max_tokens: int = 250,
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def generate_text(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
