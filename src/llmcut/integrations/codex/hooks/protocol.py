from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmcut.integrations.codex.hooks.config import MAX_HOOK_INPUT


@dataclass(frozen=True, slots=True)
class BashResponse:
    stdout: str
    stderr: str
    exit_code: int
    representation: str


@dataclass(frozen=True, slots=True)
class PostToolUseEvent:
    session_id: str
    turn_id: str
    cwd: Path
    command: str
    response: BashResponse
    raw: dict[str, Any]


def parse_hook_input(raw: bytes, repository_root: Path) -> PostToolUseEvent | None:
    if not raw or len(raw) > MAX_HOOK_INPUT:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("hook_event_name") != "PostToolUse":
        return None
    if value.get("tool_name") != "Bash":
        return None
    cwd_value = value.get("cwd")
    tool_input = value.get("tool_input")
    if not isinstance(cwd_value, str) or not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command")
    if not isinstance(command, str) or not command or len(command) > 131_072:
        return None
    try:
        cwd = Path(cwd_value).resolve(strict=True)
        root = repository_root.resolve(strict=True)
    except OSError:
        return None
    if cwd != root and root not in cwd.parents:
        return None
    response = parse_bash_response(value.get("tool_response"))
    if response is None:
        return None
    return PostToolUseEvent(
        str(value.get("session_id", ""))[:256],
        str(value.get("turn_id", ""))[:256],
        cwd,
        command,
        response,
        value,
    )


def parse_bash_response(value: Any) -> BashResponse | None:
    if isinstance(value, str):
        return BashResponse(value, "", 0, "combined_text_status_unavailable")
    if not isinstance(value, dict):
        return None
    stdout = value.get("stdout", "")
    stderr = value.get("stderr", "")
    exit_code = value.get("exit_code", value.get("exitCode"))
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return None
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return None
    return BashResponse(stdout, stderr, exit_code, "separate_stdout_stderr")


def replacement_response(model_content: str) -> dict[str, Any]:
    return {
        "continue": False,
        "stopReason": model_content,
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": model_content,
        },
    }
