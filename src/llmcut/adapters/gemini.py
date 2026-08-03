from __future__ import annotations

import json
from typing import Any

from llmcut.adapters.base import ProviderAdapter
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, ModelConfiguration


class GeminiAdapter(ProviderAdapter):
    def validate_native(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload.get("contents"), list):
            raise ValueError("Gemini request requires a contents list")

    def from_native(self, payload: dict[str, Any]) -> CanonicalRequest:
        known = {
            "model",
            "systemInstruction",
            "contents",
            "tools",
            "generationConfig",
            "cachedContent",
        }
        blocks: list[ContextBlock] = []
        if payload.get("systemInstruction"):
            blocks.append(
                ContextBlock(
                    "system",
                    BlockKind.SYSTEM,
                    _parts(payload["systemInstruction"].get("parts", [])),
                    "gemini:systemInstruction",
                    metadata={"native": payload["systemInstruction"], "dedupe_safe": False},
                )
            )
        for index, content in enumerate(payload.get("contents", [])):
            kind = BlockKind.ASSISTANT if content.get("role") == "model" else BlockKind.USER
            blocks.append(
                ContextBlock(
                    f"content:{index}",
                    kind,
                    _parts(content.get("parts", [])),
                    "gemini:contents",
                    metadata={"native_parts": content.get("parts", []), "dedupe_safe": False},
                )
            )
        declarations: list[dict[str, Any]] = []
        other_tools: list[dict[str, Any]] = []
        for tool in payload.get("tools", []):
            declarations.extend(tool.get("functionDeclarations", []))
            extra = {key: value for key, value in tool.items() if key != "functionDeclarations"}
            if extra:
                other_tools.append(extra)
        tools = [
            ContextBlock(
                f"tool:{index}",
                BlockKind.TOOL_DEFINITION,
                json.dumps(tool, sort_keys=True, separators=(",", ":")),
                "gemini:tools",
            )
            for index, tool in enumerate(declarations)
        ]
        passthrough = {key: value for key, value in payload.items() if key not in known}
        if other_tools:
            passthrough["gemini_other_tools"] = other_tools
        cache = {"cachedContent": payload["cachedContent"]} if "cachedContent" in payload else {}
        return CanonicalRequest(
            blocks,
            ModelConfiguration(
                "gemini", payload.get("model", ""), dict(payload.get("generationConfig", {}))
            ),
            tools,
            cache,
            passthrough,
        )

    def to_native(self, request: CanonicalRequest) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in request.passthrough.items()
            if key not in {"llmcut_original", "gemini_other_tools"}
        }
        systems = [
            block
            for block in request.blocks
            if block.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
        ]
        if systems:
            payload["systemInstruction"] = (
                systems[0].metadata.get("native")
                if len(systems) == 1
                else {"parts": [{"text": "\n".join(x.content for x in systems)}]}
            )
        payload["contents"] = [
            {
                "role": "model" if block.kind is BlockKind.ASSISTANT else "user",
                "parts": block.metadata.get("native_parts", [{"text": block.content}]),
            }
            for block in request.blocks
            if block not in systems
        ]
        declarations = [json.loads(tool.content) for tool in request.tools]
        payload["tools"] = (
            [{"functionDeclarations": declarations}] if declarations else []
        ) + list(request.passthrough.get("gemini_other_tools", []))
        if request.model.parameters:
            payload["generationConfig"] = request.model.parameters
        payload.update(request.cache)
        return payload

    def usage(self, response: dict[str, Any]) -> dict[str, int]:
        usage = response.get("usageMetadata") or {}
        return {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "cached_tokens": usage.get("cachedContentTokenCount", 0),
            "reasoning_tokens": usage.get("thoughtsTokenCount", 0),
        }


def _parts(parts: list[dict[str, Any]]) -> str:
    return "\n".join(part.get("text", json.dumps(part, sort_keys=True)) for part in parts)
