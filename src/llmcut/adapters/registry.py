from __future__ import annotations

from llmcut.adapters.anthropic import AnthropicAdapter
from llmcut.adapters.base import ProviderAdapter
from llmcut.adapters.gemini import GeminiAdapter
from llmcut.adapters.openai import OpenAIAdapter


def adapter_for(kind: str, path: str) -> tuple[ProviderAdapter, str]:
    normalized = path.lower()
    if kind == "anthropic":
        return AnthropicAdapter(), "messages"
    if kind == "gemini":
        return GeminiAdapter(), "generate-content"
    if normalized.rstrip("/").endswith("responses"):
        return OpenAIAdapter("responses"), "responses"
    return OpenAIAdapter("chat"), "chat-completions"
