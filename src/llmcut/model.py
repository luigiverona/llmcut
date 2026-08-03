from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class BlockKind(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_DEFINITION = "tool_definition"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ATTACHMENT = "attachment"
    REPOSITORY = "repository"
    COMMAND_OUTPUT = "command_output"
    CHECKPOINT = "checkpoint"
    SOURCE = "source"
    REPOSITORY_MAP = "repository_map"
    CONFIGURATION = "configuration"
    TEST = "test"
    DOCUMENT = "document"
    CURRENT_TASK = "current_task"


class Retention(StrEnum):
    REQUIRED = "required"
    STABLE = "stable"
    RECOVERABLE = "recoverable"
    SUPERSEDED = "superseded"
    REDUNDANT = "redundant"
    EPHEMERAL = "ephemeral"


class CountQuality(StrEnum):
    EXACT = "exact"
    PROVIDER_REPORTED = "provider-reported"
    TOKENIZER_DERIVED = "tokenizer-derived"
    ESTIMATED = "estimated"


def digest_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


@dataclass(slots=True)
class TokenCount:
    value: int
    quality: CountQuality
    method: str


@dataclass(slots=True)
class EvidenceReference:
    digest: str
    source: str
    revision: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(slots=True)
class ContextBlock:
    id: str
    kind: BlockKind
    content: str
    source: str
    priority: int = 50
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reference: EvidenceReference | None = None
    tokens: TokenCount | None = None
    digest: str = ""
    retention: Retention = Retention.REQUIRED

    def __post_init__(self) -> None:
        actual = digest_bytes(self.content.encode())
        if self.digest and self.digest != actual:
            raise ValueError(f"digest mismatch for block {self.id}")
        self.digest = actual


@dataclass(slots=True)
class ModelConfiguration:
    provider: str
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=dict)
    token_budget: int | None = None


@dataclass(slots=True)
class CanonicalRequest:
    blocks: list[ContextBlock]
    model: ModelConfiguration
    tools: list[ContextBlock] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    passthrough: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete canonical state (v0.2-compatible representation)."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize canonical state, including recovery metadata.

        Provider adapters and token counters must use ``model_bound_*`` instead.
        """
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def model_bound_dict(self) -> dict[str, Any]:
        """Return provider-neutral logical content without llmcut internal state."""

        def public(block: ContextBlock) -> dict[str, Any]:
            return {"id": block.id, "kind": block.kind.value, "content": block.content}

        return {
            "blocks": [public(item) for item in self.blocks],
            "tools": [public(item) for item in self.tools],
            "model": {
                "provider": self.model.provider,
                "model": self.model.model,
                "parameters": self.model.parameters,
                "reasoning": self.model.reasoning,
                "token_budget": self.model.token_budget,
            },
        }

    def model_bound_json(self) -> str:
        return json.dumps(self.model_bound_dict(), sort_keys=True, separators=(",", ":"))

    def evidence_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "request_id": self.request_id,
            "evidence": [
                {
                    "id": item.id,
                    "digest": item.reference.digest if item.reference else item.digest,
                    "source": item.reference.source if item.reference else item.source,
                    "revision": item.reference.revision if item.reference else None,
                    "dependencies": list(item.dependencies),
                    "retention": item.retention.value,
                }
                for item in [*self.blocks, *self.tools]
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalRequest:
        def block(item: dict[str, Any]) -> ContextBlock:
            data = dict(item)
            data["kind"] = BlockKind(data["kind"])
            data["retention"] = Retention(data.get("retention", "required"))
            if data.get("reference"):
                data["reference"] = EvidenceReference(**data["reference"])
            if data.get("tokens"):
                token = data["tokens"]
                data["tokens"] = TokenCount(
                    token["value"], CountQuality(token["quality"]), token["method"]
                )
            return ContextBlock(**data)

        return cls(
            blocks=[block(item) for item in value.get("blocks", [])],
            tools=[block(item) for item in value.get("tools", [])],
            model=ModelConfiguration(**value["model"]),
            cache=dict(value.get("cache", {})),
            passthrough=dict(value.get("passthrough", {})),
            request_id=value.get("request_id"),
        )
