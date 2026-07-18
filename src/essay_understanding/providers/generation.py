from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import Settings


class GenerationProvider(ABC):
    name: str
    available: bool = True

    @abstractmethod
    def structured(self, *, system: str, task: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        """Return one JSON object conforming to schema."""


class DisabledGenerationProvider(GenerationProvider):
    name = "disabled"
    available = False

    def structured(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError("No generation provider configured")


class OpenAICompatibleGenerationProvider(GenerationProvider):
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key

    def structured(self, *, system: str, task: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.name,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"task": task, "input": payload}, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": task, "strict": True, "schema": schema},
            },
        }
        response = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=180)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(content)


def build_generation_provider(settings: Settings) -> GenerationProvider:
    if not settings.generation_enabled:
        return DisabledGenerationProvider()
    return OpenAICompatibleGenerationProvider(settings.llm_base_url, settings.llm_model, settings.llm_api_key)

