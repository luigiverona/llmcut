from __future__ import annotations

import json
from typing import Any

from llmcut.adapters.base import ProviderAdapter
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, ModelConfiguration


class OpenAIAdapter(ProviderAdapter):
    def __init__(self, style: str = "chat") -> None:
        if style not in {"chat", "responses"}:
            raise ValueError("OpenAI style must be chat or responses")
        self.style = style

    def from_native(self, payload: dict[str, Any]) -> CanonicalRequest:
        known = {
            "model",
            "messages",
            "tools",
            "input",
            "instructions",
            "reasoning",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "stream",
        }
        blocks: list[ContextBlock] = []
        if self.style == "chat":
            for index, message in enumerate(payload.get("messages", [])):
                role = message.get("role", "user")
                kind = _role_kind(role)
                content = message.get("content", "")
                text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
                metadata = {
                    key: value for key, value in message.items() if key not in {"role", "content"}
                }
                metadata["native_content"] = content
                metadata["dedupe_safe"] = False
                blocks.append(
                    ContextBlock(
                        f"message:{index}", kind, text, "openai:messages", metadata=metadata
                    )
                )
        else:
            instructions = payload.get("instructions")
            if instructions:
                blocks.append(
                    ContextBlock(
                        "instructions",
                        BlockKind.DEVELOPER,
                        _text(instructions),
                        "openai:instructions",
                        metadata={"native_content": instructions, "dedupe_safe": False},
                    )
                )
            input_value = payload.get("input", [])
            if isinstance(input_value, str):
                input_value = [{"role": "user", "content": input_value}]
            for index, item in enumerate(input_value):
                role = item.get("role", "user")
                blocks.append(
                    ContextBlock(
                        f"input:{index}",
                        _role_kind(role),
                        _text(item.get("content", item)),
                        "openai:input",
                        metadata={
                            key: value
                            for key, value in item.items()
                            if key not in {"role", "content"}
                        }
                        | {"native_content": item.get("content", item), "dedupe_safe": False},
                    )
                )
        tools = [
            ContextBlock(
                f"tool:{index}",
                BlockKind.TOOL_DEFINITION,
                json.dumps(tool, sort_keys=True, separators=(",", ":")),
                "openai:tools",
            )
            for index, tool in enumerate(payload.get("tools", []))
        ]
        params: dict[str, Any] = {
            key: payload[key]
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens", "stream")
            if key in payload
        }
        return CanonicalRequest(
            blocks,
            ModelConfiguration(
                "openai", payload["model"], params, dict(payload.get("reasoning", {}))
            ),
            tools,
            passthrough={key: value for key, value in payload.items() if key not in known},
        )

    def to_native(self, request: CanonicalRequest) -> dict[str, Any]:
        payload = dict(request.passthrough)
        payload.pop("llmcut_original", None)
        payload.update(request.model.parameters)
        payload["model"] = request.model.model
        if request.model.reasoning:
            payload["reasoning"] = request.model.reasoning
        payload["tools"] = [json.loads(tool.content) for tool in request.tools]
        if self.style == "chat":
            payload["messages"] = [
                {
                    "role": _kind_role(block.kind),
                    "content": block.metadata.get("native_content", block.content),
                    **_public_metadata(block.metadata),
                }
                for block in request.blocks
            ]
        else:
            instructions = [
                block.content
                for block in request.blocks
                if block.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
            ]
            if instructions:
                instruction_blocks = [
                    block
                    for block in request.blocks
                    if block.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
                ]
                payload["instructions"] = (
                    instruction_blocks[0].metadata.get("native_content")
                    if len(instruction_blocks) == 1
                    else "\n".join(instructions)
                )
            payload["input"] = [
                {
                    "role": _kind_role(block.kind),
                    "content": block.metadata.get("native_content", block.content),
                    **_public_metadata(block.metadata),
                }
                for block in request.blocks
                if block.kind not in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
            ]
        return payload

    def usage(self, response: dict[str, Any]) -> dict[str, int]:
        usage = response.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "cached_tokens": details.get("cached_tokens", 0),
            "reasoning_tokens": output_details.get("reasoning_tokens", 0),
        }

    def validate_native(self, payload: dict[str, Any]) -> None:
        super().validate_native(payload)
        key = "messages" if self.style == "chat" else "input"
        if key not in payload or not isinstance(payload[key], (str, list)):
            raise ValueError(f"OpenAI {self.style} request requires {key}")


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _role_kind(role: str) -> BlockKind:
    return {
        "system": BlockKind.SYSTEM,
        "developer": BlockKind.DEVELOPER,
        "user": BlockKind.USER,
        "assistant": BlockKind.ASSISTANT,
        "tool": BlockKind.TOOL_RESULT,
    }.get(role, BlockKind.USER)


def _kind_role(kind: BlockKind) -> str:
    return {
        BlockKind.SYSTEM: "system",
        BlockKind.DEVELOPER: "developer",
        BlockKind.USER: "user",
        BlockKind.ASSISTANT: "assistant",
        BlockKind.TOOL_RESULT: "tool",
        BlockKind.TOOL_CALL: "assistant",
    }.get(kind, "user")


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    internal = {
        "native_content",
        "dedupe_safe",
        "revision",
        "confidence",
        "task_irrelevant",
        "proven_redundant",
        "represented_by_verified_structure",
        "superseded",
    }
    return {
        key: value for key, value in metadata.items() if key not in internal and value is not None
    }
