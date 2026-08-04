from llmcut.integrations.codex.events import normalize_event


def test_normalizes_supported_event_surface_without_reasoning() -> None:
    messages = [
        {"method": "thread/started", "params": {"thread": {"id": "thread"}}},
        {"method": "turn/started", "params": {"turn": {"id": "turn"}}},
        {
            "method": "turn/completed",
            "params": {"turn": {"id": "turn", "status": "failed"}},
        },
        {
            "method": "thread/tokenUsage/updated",
            "params": {"usage": {"input_tokens": 12, "nested": {"safe": True}}},
        },
        {
            "method": "item/started",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "id": "command",
                    "command": ["python", "-m", "pytest"],
                    "cwd": "/workspace/repo",
                    "status": "inProgress",
                }
            },
        },
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "fileChange",
                    "changes": [{"path": "src/app.py"}],
                    "status": "completed",
                }
            },
        },
        {
            "method": "item/started",
            "params": {
                "item": {
                    "type": "mcpToolCall",
                    "id": "mcp",
                    "server": "llmcut",
                    "tool": "llmcut_plan",
                    "status": "inProgress",
                }
            },
        },
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "mcpToolCall",
                    "id": "mcp",
                    "server": "llmcut",
                    "tool": "llmcut_plan",
                    "status": "completed",
                    "result": {"ok": True},
                }
            },
        },
        {"method": "error", "params": {"error": {"message": "bounded error"}}},
        {"method": "server/warning", "params": {"message": "warning"}},
        {"method": "future/event", "params": {"new": "field"}},
    ]
    events = [normalize_event(message) for message in messages]
    assert all(event is not None for event in events)
    assert [event.kind for event in events if event] == [
        "thread_started",
        "turn_started",
        "turn_failed",
        "usage_update",
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "mcp_result",
        "error",
        "warning",
        "opaque",
    ]
    assert normalize_event({"result": {}}) is None
