from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from llmcut.model import digest_bytes

MAX_EVENT_TEXT = 16_384


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    kind: str
    method: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_event(message: dict[str, Any]) -> NormalizedEvent | None:
    method = str(message.get("method", ""))
    if not method:
        return None
    params = message.get("params")
    params = dict(params) if isinstance(params, dict) else {}
    if method == "thread/started":
        return NormalizedEvent("thread_started", method, {"thread_id": _id(params.get("thread"))})
    if method == "turn/started":
        return NormalizedEvent("turn_started", method, {"turn_id": _id(params.get("turn"))})
    if method == "turn/completed":
        turn = _object(params.get("turn"))
        return NormalizedEvent(
            "turn_failed" if turn.get("status") == "failed" else "turn_completed",
            method,
            {"turn_id": str(turn.get("id", "")), "status": str(turn.get("status", ""))},
        )
    if method == "thread/tokenUsage/updated":
        usage = params.get("tokenUsage") or params.get("usage") or {}
        usage_object = _object(usage)
        usage = usage_object.get("last") or usage_object.get("total") or usage_object
        return NormalizedEvent("usage_update", method, _bounded_scalars(_object(usage)))
    if method == "model/rerouted":
        return NormalizedEvent(
            "error",
            method,
            {
                "from_model": str(params.get("fromModel", "")),
                "to_model": str(params.get("toModel", "")),
            },
        )
    if method in {"item/started", "item/completed"}:
        item = _object(params.get("item"))
        item_type = str(item.get("type", ""))
        status = str(item.get("status", ""))
        if item_type == "reasoning":
            return NormalizedEvent("opaque", method, {"item_type": "reasoning", "status": status})
        if item_type == "commandExecution":
            return NormalizedEvent(
                "command_execution",
                method,
                {
                    "id": str(item.get("id", "")),
                    "command": _bounded(item.get("command")),
                    "cwd": _bounded(item.get("cwd")),
                    "status": status,
                    "exit_code": item.get("exitCode"),
                },
            )
        if item_type == "fileChange":
            raw_changes = item.get("changes")
            changes: list[Any] = raw_changes if isinstance(raw_changes, list) else []
            return NormalizedEvent(
                "file_change",
                method,
                {
                    "paths": [str(_object(value).get("path", "")) for value in changes[:256]],
                    "status": status,
                },
            )
        if item_type == "mcpToolCall":
            return NormalizedEvent(
                "mcp_result" if method == "item/completed" else "mcp_tool_call",
                method,
                {
                    "id": str(item.get("id", "")),
                    "server": str(item.get("server", "")),
                    "tool": str(item.get("tool", "")),
                    "status": status,
                    "result_bytes": len(json.dumps(item.get("result", "")).encode()),
                },
            )
    if method == "error":
        error = _object(params.get("error"))
        return NormalizedEvent("error", method, {"message": _bounded(error.get("message"))})
    if "warning" in method.lower():
        return NormalizedEvent("warning", method, _bounded_scalars(params))
    return NormalizedEvent("opaque", method, {"keys": sorted(params)[:64]})


def normalize_exec_event(message: dict[str, Any]) -> NormalizedEvent | None:
    """Normalize the public ``codex exec --json`` JSONL without retaining content."""
    event_type = str(message.get("type", ""))
    if event_type == "thread.started":
        return NormalizedEvent(
            "thread_started", event_type, {"thread_id": str(message.get("thread_id", ""))[:256]}
        )
    if event_type == "turn.started":
        return NormalizedEvent(
            "turn_started", event_type, {"turn_id": str(message.get("turn_id", ""))[:256]}
        )
    if event_type in {"turn.completed", "turn.failed"}:
        return NormalizedEvent(
            "turn_completed" if event_type.endswith("completed") else "turn_failed",
            event_type,
            {"turn_id": str(message.get("turn_id", ""))[:256], "status": event_type[5:]},
        )
    if event_type == "error":
        return NormalizedEvent(
            "error",
            event_type,
            {
                "severity": _exec_error_severity(message),
                "code": str(message.get("code", ""))[:128],
            },
        )
    if event_type in {"model.rerouted", "model.fallback"}:
        return NormalizedEvent(
            "error",
            event_type,
            {
                "severity": "fatal",
                "from_model": str(message.get("from_model", ""))[:256],
                "to_model": str(message.get("to_model", ""))[:256],
            },
        )
    if event_type not in {"item.started", "item.updated", "item.completed"}:
        return NormalizedEvent("opaque", event_type, {"keys": sorted(message)[:64]})
    item = _object(message.get("item"))
    item_type = str(item.get("type", ""))
    status = str(item.get("status", ""))[:64]
    if item_type == "reasoning":
        return NormalizedEvent(
            "opaque",
            event_type,
            {"item_type": "reasoning", "status": status, "observed_bytes": _json_size(item)},
        )
    if item_type == "agent_message":
        return NormalizedEvent(
            "agent_message", event_type, {"status": status, "observed_bytes": _json_size(item)}
        )
    if item_type == "command_execution":
        command = str(item.get("command", ""))
        output = item.get("aggregated_output", item.get("output", ""))
        return NormalizedEvent(
            "command_execution",
            event_type,
            {
                "id": str(item.get("id", ""))[:256],
                "command_digest": digest_bytes(command.encode()),
                "command_classification": _command_class(command),
                "status": status,
                "exit_code": item.get("exit_code"),
                "output_bytes": len(str(output).encode()),
            },
        )
    if item_type == "file_change":
        raw_changes = item.get("changes")
        changes = raw_changes if isinstance(raw_changes, list) else []
        return NormalizedEvent(
            "file_change",
            event_type,
            {
                "paths": [str(_object(value).get("path", ""))[:1024] for value in changes[:256]],
                "status": status,
            },
        )
    if item_type == "mcp_tool_call":
        return NormalizedEvent(
            "mcp_result" if event_type == "item.completed" else "mcp_tool_call",
            event_type,
            {
                "id": str(item.get("id", ""))[:256],
                "server": str(item.get("server", ""))[:256],
                "tool": str(item.get("tool", ""))[:256],
                "status": status,
                "result_bytes": _json_size(item.get("result")),
            },
        )
    if item_type == "error":
        return NormalizedEvent(
            "warning" if status in {"completed", "warning"} else "error",
            event_type,
            {"item_type": "error", "status": status, "observed_bytes": _json_size(item)},
        )
    return NormalizedEvent("opaque", event_type, {"item_type": item_type[:128], "status": status})


def _exec_error_severity(message: dict[str, Any]) -> str:
    value = str(message.get("severity", "")).lower()
    return value if value in {"warning", "nonfatal", "fatal"} else "fatal"


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":")).encode())
    except (TypeError, ValueError):
        return 0


def _command_class(command: str) -> str:
    try:
        from llmcut.integrations.codex.hooks.classify import classify_command

        return classify_command(command).classification.value
    except Exception:
        return "unknown"


def _id(value: Any) -> str:
    return str(_object(value).get("id", ""))


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _bounded(value: Any) -> str:
    return str(value)[:MAX_EVENT_TEXT]


def _bounded_scalars(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): item if isinstance(item, (int, float, bool)) or item is None else _bounded(item)
        for key, item in list(value.items())[:64]
    }
