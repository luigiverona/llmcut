from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

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
