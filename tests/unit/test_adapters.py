from copy import deepcopy

import pytest

from llmcut.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter


@pytest.mark.parametrize(
    "style,payload",
    [
        (
            "chat",
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "ok",
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    }
                ],
                "tools": [{"type": "function", "function": {"name": "x"}}],
                "temperature": 0.2,
            },
        ),
        (
            "responses",
            {
                "model": "m",
                "instructions": "safe",
                "input": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "name": "x"}],
                "store": False,
            },
        ),
    ],
)
def test_openai_roundtrip_unknowns_and_ids(style: str, payload: dict[str, object]) -> None:
    adapter = OpenAIAdapter(style)
    native = adapter.to_native(adapter.from_native(deepcopy(payload)))
    assert native["model"] == "m" and native["tools"] == payload["tools"]
    assert native.get("temperature") == payload.get("temperature")
    assert native.get("store") == payload.get("store")
    if style == "chat":
        assert native["messages"][0]["tool_calls"][0]["id"] == "call_1"


def test_openai_usage_labels_cache_and_reasoning() -> None:
    usage = OpenAIAdapter().usage(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            }
        }
    )
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cached_tokens": 3,
        "reasoning_tokens": 2,
    }


def test_anthropic_roundtrip_tool_blocks_and_usage() -> None:
    payload = {
        "model": "claude",
        "max_tokens": 10,
        "system": [{"type": "text", "text": "safe", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "done"}],
            }
        ],
        "tools": [{"name": "x", "input_schema": {"type": "object"}}],
        "metadata": {"user_id": "u"},
    }
    adapter = AnthropicAdapter()
    native = adapter.to_native(adapter.from_native(payload))
    assert native["messages"][0]["content"][0]["tool_use_id"] == "toolu_1"
    assert native["metadata"] == {"user_id": "u"}
    assert (
        adapter.usage(
            {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 1,
                }
            }
        )["cached_tokens"]
        == 5
    )


def test_gemini_roundtrip_parts_functions_cache_and_usage() -> None:
    payload = {
        "model": "models/g",
        "systemInstruction": {"parts": [{"text": "safe"}]},
        "contents": [{"role": "model", "parts": [{"functionCall": {"name": "x", "args": {}}}]}],
        "tools": [{"functionDeclarations": [{"name": "x", "parameters": {"type": "object"}}]}],
        "cachedContent": "cachedContents/1",
        "safetySettings": [{"category": "X"}],
    }
    adapter = GeminiAdapter()
    native = adapter.to_native(adapter.from_native(payload))
    assert native["contents"][0]["parts"][0]["functionCall"]["name"] == "x"
    assert native["cachedContent"] == "cachedContents/1"
    assert native["safetySettings"] == payload["safetySettings"]
    assert (
        adapter.usage(
            {
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "cachedContentTokenCount": 2,
                    "thoughtsTokenCount": 1,
                }
            }
        )["reasoning_tokens"]
        == 1
    )
