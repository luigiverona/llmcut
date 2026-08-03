import json
from pathlib import Path

import pytest

from llmcut.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from llmcut.adapters.base import ProviderAdapter
from llmcut.core.optimize import Optimizer
from llmcut.policy import OptimizationMode
from llmcut.proxy.optimize import NativeOptimization, optimize_native
from llmcut.store.evidence import EvidenceStore


def run(
    payload: dict[str, object], adapter: ProviderAdapter, endpoint: str, tmp_path: Path
) -> NativeOptimization:
    body = json.dumps(payload, separators=(",", ":")).encode()
    return optimize_native(
        body, adapter, endpoint, Optimizer(EvidenceStore(tmp_path)), OptimizationMode.EXTREME
    )


@pytest.mark.parametrize(
    "adapter,endpoint,payload",
    [
        (
            OpenAIAdapter("chat"),
            "chat-completions",
            {
                "model": "gpt-test",
                "stream": True,
                "reasoning_effort": "high",
                "messages": [
                    {"role": "system", "content": "safe"},
                    {"role": "assistant", "content": [{"type": "tool_call", "id": "call_1"}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "x", "parameters": {"type": "object"}},
                    }
                ],
                "response_format": {"type": "json_object"},
            },
        ),
        (
            OpenAIAdapter("responses"),
            "responses",
            {
                "model": "gpt-test",
                "stream": False,
                "reasoning": {"effort": "high"},
                "instructions": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "safe"}],
                    }
                ],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "file_id": "file_1"}],
                    }
                ],
                "tools": [],
                "metadata": {"trace": "x"},
            },
        ),
        (
            AnthropicAdapter(),
            "messages",
            {
                "model": "claude-test",
                "max_tokens": 10,
                "stream": True,
                "thinking": {"type": "enabled", "budget_tokens": 4},
                "system": [
                    {"type": "text", "text": "safe", "cache_control": {"type": "ephemeral"}}
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "toolu_1", "name": "x", "input": {}}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                        ],
                    },
                ],
                "tools": [],
                "metadata": {"user_id": "u"},
            },
        ),
        (
            GeminiAdapter(),
            "generate-content",
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "x", "args": {"a": 1}, "id": "call_1"}}
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "x",
                                    "response": {"result": 1},
                                    "id": "call_1",
                                }
                            }
                        ],
                    },
                ],
                "systemInstruction": {"parts": [{"text": "safe"}]},
                "generationConfig": {"temperature": 0},
                "cachedContent": "cachedContents/1",
                "tools": [],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            },
        ),
    ],
)
def test_roundtrip_preserves_supported_native_semantics(
    tmp_path: Path, adapter: ProviderAdapter, endpoint: str, payload: dict[str, object]
) -> None:
    result = run(payload, adapter, endpoint, tmp_path)
    assert result.status in {"unchanged", "optimized"}
    forwarded = json.loads(result.body)
    assert forwarded == payload


def test_duplicate_tool_schema_is_reduced_end_to_end(tmp_path: Path) -> None:
    tool = {
        "type": "function",
        "function": {"name": "lookup", "description": "x" * 1000, "parameters": {"type": "object"}},
    }
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "use lookup"}],
        "tools": [tool, tool],
        "temperature": 0,
    }
    result = run(payload, OpenAIAdapter(), "chat-completions", tmp_path)
    assert result.status == "optimized"
    assert len(json.loads(result.body)["tools"]) == 1
    assert result.effective_tokens < result.original_tokens


def test_larger_or_invalid_attempt_restores_original(tmp_path: Path) -> None:
    body = b'{"model":"m","messages":[]}'
    result = optimize_native(
        body,
        OpenAIAdapter(),
        "chat-completions",
        Optimizer(EvidenceStore(tmp_path)),
        OptimizationMode.EXTREME,
    )
    assert result.status == "unchanged" and result.body == body
    invalid = b'{"model":'
    restored = optimize_native(
        invalid,
        OpenAIAdapter(),
        "chat-completions",
        Optimizer(EvidenceStore(tmp_path)),
        OptimizationMode.EXTREME,
    )
    assert restored.status == "restored" and restored.body == invalid
    assert "parsed" in (restored.fallback_reason or "")


def test_deep_json_fails_open_without_prompt_in_reason(tmp_path: Path) -> None:
    nested: object = "SENSITIVE_PROMPT"
    for _ in range(70):
        nested = [nested]
    body = json.dumps({"model": "m", "messages": nested}).encode()
    result = optimize_native(
        body,
        OpenAIAdapter(),
        "chat-completions",
        Optimizer(EvidenceStore(tmp_path)),
        OptimizationMode.EXTREME,
    )
    assert result.status == "restored" and "SENSITIVE_PROMPT" not in (result.fallback_reason or "")


def test_gemini_duplicate_function_declarations_are_safely_reduced(tmp_path: Path) -> None:
    declaration = {"name": "lookup", "description": "x" * 1000, "parameters": {"type": "object"}}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "lookup"}]}],
        "tools": [{"functionDeclarations": [declaration, declaration]}],
    }
    result = run(payload, GeminiAdapter(), "generate-content", tmp_path)
    assert result.status == "optimized"
    forwarded = json.loads(result.body)
    assert forwarded["tools"][0]["functionDeclarations"] == [declaration]
