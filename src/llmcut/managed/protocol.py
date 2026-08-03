from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from llmcut.errors import ProtocolError
from llmcut.model import BlockKind, ContextBlock, ModelConfiguration, Retention
from llmcut.policy import IntegrationMode, OptimizationMode

SCHEMA_VERSION = "1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(slots=True)
class Context:
    id: str
    kind: BlockKind
    content: str
    retention: Retention = Retention.RECOVERABLE
    priority: int = 50
    source_path: str | None = None
    dependencies: tuple[str, ...] = ()
    tool_call_id: str | None = None
    revision: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def source_code(cls, source: str, content: str, **kwargs: Any) -> Context:
        return cls(_source_id(source), BlockKind.SOURCE, content, source_path=source, **kwargs)

    @classmethod
    def source(cls, source: str, content: str, **kwargs: Any) -> Context:
        return cls.source_code(source, content, **kwargs)

    @classmethod
    def test(cls, source: str, content: str, **kwargs: Any) -> Context:
        return cls(_source_id(source), BlockKind.TEST, content, source_path=source, **kwargs)

    @classmethod
    def document(cls, source: str, content: str, **kwargs: Any) -> Context:
        return cls(_source_id(source), BlockKind.DOCUMENT, content, source_path=source, **kwargs)

    def to_block(self) -> ContextBlock:
        return ContextBlock(
            self.id,
            self.kind,
            self.content,
            self.source_path or f"managed:{self.id}",
            self.priority,
            list(self.dependencies),
            {"tool_call_id": self.tool_call_id, "revision": self.revision, **self.extensions},
            retention=self.retention,
        )


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    category: str = "general"
    required: bool = False
    extensions: dict[str, Any] = field(default_factory=dict)

    def transport_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(slots=True)
class ExecutionSettings:
    integration: IntegrationMode = IntegrationMode.MANAGED
    optimization: OptimizationMode = OptimizationMode.EXTREME
    max_turns: int = 8
    timeout_seconds: float = 120.0
    max_retrieval_bytes: int = 1_048_576
    max_total_tokens: int | None = None


@dataclass(slots=True)
class ManagedRequest:
    provider: str
    model: str
    task: str
    context: list[Context] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    schema_version: str = SCHEMA_VERSION
    extensions: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported managed schema_version {self.schema_version!r}")
        if not self.provider or not self.model or not self.task:
            raise ProtocolError("provider, model, and current task are required")
        sensitive = re.compile(
            r"(?:credential|api[_-]?key|authorization|password|secret|token)", re.I
        )
        if any(sensitive.search(key) for key in self.extensions):
            raise ProtocolError("request-provided credential fields are prohibited")
        if self.execution.integration is not IntegrationMode.MANAGED:
            raise ProtocolError("managed protocol requires execution.integration=managed")
        if not 1 <= self.execution.max_turns <= 64:
            raise ProtocolError("max_turns must be between 1 and 64")
        if self.execution.timeout_seconds <= 0 or self.execution.max_retrieval_bytes < 0:
            raise ProtocolError("execution bounds must be positive")
        ids: set[str] = {"current-task"}
        calls: set[str] = set()
        for item in self.context:
            if not _IDENTIFIER.fullmatch(item.id) or item.id in ids:
                raise ProtocolError(f"invalid or duplicate context id: {item.id}")
            ids.add(item.id)
            if not 0 <= item.priority <= 100:
                raise ProtocolError(f"priority outside 0..100 for {item.id}")
            if item.kind in {
                BlockKind.SYSTEM,
                BlockKind.DEVELOPER,
                BlockKind.CURRENT_TASK,
            } and item.retention not in {
                Retention.REQUIRED,
                Retention.STABLE,
                Retention.EPHEMERAL,
            }:
                raise ProtocolError(f"critical context {item.id} cannot be removable")
            if item.kind is BlockKind.TOOL_CALL:
                if not item.tool_call_id or item.tool_call_id in calls:
                    raise ProtocolError(f"tool call {item.id} needs a unique tool_call_id")
                calls.add(item.tool_call_id)
            if item.kind is BlockKind.TOOL_RESULT and item.tool_call_id not in calls:
                raise ProtocolError(f"tool result {item.id} has no preceding tool call")
        for item in self.context:
            missing = set(item.dependencies) - ids
            if missing:
                raise ProtocolError(f"unknown dependencies for {item.id}: {sorted(missing)}")
        _reject_cycles(self.context)
        names: set[str] = set()
        for tool in self.tools:
            if not _IDENTIFIER.fullmatch(tool.name) or tool.name in names:
                raise ProtocolError(f"invalid or duplicate tool name: {tool.name}")
            names.add(tool.name)
            if not isinstance(tool.input_schema, dict):
                raise ProtocolError(f"tool {tool.name} input_schema must be an object")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(self.extensions)
        value["task"] = {"content": self.task, "kind": "current"}
        value["execution"] = asdict(self.execution)
        for encoded_context, context in zip(value["context"], self.context, strict=True):
            encoded_context["source"] = encoded_context.pop("source_path")
            encoded_context.update(context.extensions)
            encoded_context.pop("extensions", None)
        for encoded_tool, tool_definition in zip(value["tools"], self.tools, strict=True):
            encoded_tool.update(tool_definition.extensions)
            encoded_tool.pop("extensions", None)
        value.pop("extensions", None)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ManagedRequest:
        if not isinstance(value, dict):
            raise ProtocolError("managed request must be an object")
        known = {
            "schema_version",
            "provider",
            "model",
            "settings",
            "task",
            "context",
            "tools",
            "execution",
        }
        task = value.get("task")
        task_text = task.get("content") if isinstance(task, dict) else task
        contexts = [
            _context_from_dict(item) for item in _object_list(value.get("context", []), "context")
        ]
        tools = [_tool_from_dict(item) for item in _object_list(value.get("tools", []), "tools")]
        execution_raw = value.get("execution", {})
        if not isinstance(execution_raw, dict):
            raise ProtocolError("execution must be an object")
        execution = ExecutionSettings(
            integration=IntegrationMode(execution_raw.get("integration", "managed")),
            optimization=OptimizationMode(execution_raw.get("optimization", "extreme")),
            max_turns=int(execution_raw.get("max_turns", 8)),
            timeout_seconds=float(execution_raw.get("timeout_seconds", 120)),
            max_retrieval_bytes=int(execution_raw.get("max_retrieval_bytes", 1_048_576)),
            max_total_tokens=execution_raw.get("max_total_tokens"),
        )
        request = cls(
            str(value.get("provider", "")),
            str(value.get("model", "")),
            str(task_text or ""),
            contexts,
            tools,
            dict(value.get("settings", {})),
            execution,
            str(value.get("schema_version", "")),
            {key: value[key] for key in value.keys() - known},
        )
        request.validate()
        return request

    def model_configuration(self) -> ModelConfiguration:
        reasoning = self.settings.get("reasoning", {})
        parameters = {key: value for key, value in self.settings.items() if key != "reasoning"}
        return ModelConfiguration(self.provider, self.model, parameters, dict(reasoning))


def _context_from_dict(value: dict[str, Any]) -> Context:
    known = {
        "id",
        "kind",
        "content",
        "retention",
        "priority",
        "source",
        "dependencies",
        "tool_call_id",
        "revision",
    }
    try:
        return Context(
            str(value["id"]),
            BlockKind(value["kind"]),
            str(value["content"]),
            Retention(value.get("retention", "recoverable")),
            int(value.get("priority", 50)),
            value.get("source"),
            tuple(value.get("dependencies", ())),
            value.get("tool_call_id"),
            value.get("revision"),
            {key: value[key] for key in value.keys() - known},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid context block: {exc}") from exc


def _tool_from_dict(value: dict[str, Any]) -> ToolDefinition:
    known = {"name", "description", "input_schema", "category", "required"}
    try:
        return ToolDefinition(
            str(value["name"]),
            str(value.get("description", "")),
            dict(value["input_schema"]),
            str(value.get("category", "general")),
            bool(value.get("required", False)),
            {key: value[key] for key in value.keys() - known},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid tool definition: {exc}") from exc


def _object_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ProtocolError(f"{name} must be a list of objects")
    return value


def _reject_cycles(context: list[Context]) -> None:
    graph = {item.id: item.dependencies for item in context}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ProtocolError("cyclic context dependencies are not allowed")
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in graph.get(identifier, ()):
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in graph:
        visit(identifier)


def _source_id(source: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "-", source).strip("-")
    return normalized[:128] or "context"
