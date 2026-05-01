from openai import AsyncOpenAI
import json
import logging
from pydantic import BaseModel
from typing import AsyncGenerator
from llm.base_llm import BaseLLM
from config import LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL

logger = logging.getLogger(__name__)


class LocalLLM(BaseLLM):
    def __init__(self, model: str | None = None):
        self.client = AsyncOpenAI(
            base_url=LOCAL_LLM_BASE_URL,
            api_key="not-needed",  # local servers don't need API keys
        )
        self.model = model or LOCAL_LLM_MODEL

    async def generate_json(
        self,
        messages: list[dict],
        response_schema: type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> BaseModel:
        # Local models may not support response_format, so we parse manually
        raw_text = ""
        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_text = response.choices[0].message.content or ""
                # Strip markdown code fences if present (common with local models)
                cleaned = raw_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                parsed = json.loads(cleaned)
                return response_schema.model_validate(parsed)
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Local LLM JSON parse failed (attempt 1): {e}")
                    messages = messages + [
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                f"Your output was invalid JSON: {str(e)}. "
                                "Return ONLY a valid JSON object, no markdown, no explanation."
                            ),
                        },
                    ]
                else:
                    logger.error(f"Local LLM JSON parse failed (attempt 2): {e}")
                    raise ValueError(f"Local LLM failed to produce valid JSON: {e}")

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
