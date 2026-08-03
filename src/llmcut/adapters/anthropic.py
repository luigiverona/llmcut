from __future__ import annotations

import json
from typing import Any

from llmcut.adapters.base import ProviderAdapter
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, ModelConfiguration


class AnthropicAdapter(ProviderAdapter):
    def validate_native(self, payload: dict[str, Any]) -> None:
        super().validate_native(payload)
        if not isinstance(payload.get("messages"), list) or not isinstance(
            payload.get("max_tokens"), int
        ):
            raise ValueError("Anthropic request requires messages and max_tokens")

    def from_native(self, payload: dict[str, Any]) -> CanonicalRequest:
        known = {"model", "system", "messages", "tools", "max_tokens", "stream", "thinking"}
        blocks: list[ContextBlock] = []
        if "system" in payload:
            blocks.append(
                ContextBlock(
                    "system",
                    BlockKind.SYSTEM,
                    _text(payload["system"]),
                    "anthropic:system",
                    metadata={"native_content": payload["system"], "dedupe_safe": False},
                )
            )
        for index, message in enumerate(payload.get("messages", [])):
            role = BlockKind.ASSISTANT if message["role"] == "assistant" else BlockKind.USER
            blocks.append(
                ContextBlock(
                    f"message:{index}",
                    role,
                    _text(message.get("content", "")),
                    "anthropic:messages",
                    metadata={"native_content": message.get("content", ""), "dedupe_safe": False},
                )
            )
        tools = [
            ContextBlock(
                f"tool:{index}",
                BlockKind.TOOL_DEFINITION,
                json.dumps(tool, sort_keys=True, separators=(",", ":")),
                "anthropic:tools",
            )
            for index, tool in enumerate(payload.get("tools", []))
        ]
        return CanonicalRequest(
            blocks,
            ModelConfiguration(
                "anthropic",
                payload["model"],
                {key: payload[key] for key in ("max_tokens", "stream") if key in payload},
                dict(payload.get("thinking", {})),
            ),
            tools,
            passthrough={key: value for key, value in payload.items() if key not in known},
        )

    def to_native(self, request: CanonicalRequest) -> dict[str, Any]:
        payload = {
            key: value for key, value in request.passthrough.items() if key != "llmcut_original"
        }
        payload.update(request.model.parameters)
        payload["model"] = request.model.model
        if request.model.reasoning:
            payload["thinking"] = request.model.reasoning
        systems = [
            block.content
            for block in request.blocks
            if block.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
        ]
        if systems:
            system_blocks = [
                block
                for block in request.blocks
                if block.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
            ]
            payload["system"] = (
                system_blocks[0].metadata.get("native_content", systems[0])
                if len(system_blocks) == 1
                else "\n".join(systems)
            )
        payload["messages"] = [
            {
                "role": "assistant" if block.kind is BlockKind.ASSISTANT else "user",
                "content": block.metadata.get("native_content", block.content),
            }
            for block in request.blocks
            if block.kind not in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
        ]
        payload["tools"] = [json.loads(tool.content) for tool in request.tools]
        return payload

    def usage(self, response: dict[str, Any]) -> dict[str, int]:
        usage = response.get("usage") or {}
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cached_tokens": usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0),
            "reasoning_tokens": 0,
        }


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item.get("text", json.dumps(item, sort_keys=True))
            if isinstance(item, dict)
            else str(item)
            for item in value
        )
    return json.dumps(value, sort_keys=True)
