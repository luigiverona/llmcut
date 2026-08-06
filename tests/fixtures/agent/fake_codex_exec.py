#!/usr/bin/env python3
"""Structured fake ``codex exec`` runtime that invokes the real llmcut hook subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def send(value: dict[str, object]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


if "--version" in sys.argv:
    print("codex-cli 0.146.0")
    raise SystemExit(0)

scenario = os.environ.get("LLMCUT_FAKE_SCENARIO", "success")
mode = os.environ.get("LLMCUT_EVAL_MODE", "baseline")
strategy = os.environ.get("LLMCUT_CONTEXT_STRATEGY", "off")
arguments = sys.argv[1:]
cwd = Path.cwd()
if "--cd" in arguments:
    cwd = Path(arguments[arguments.index("--cd") + 1])
prompt = sys.stdin.read()
if scenario == "malformed-jsonl":
    print("{invalid")
    raise SystemExit(0)
if scenario == "turn-failed":
    send({"type": "thread.started", "thread_id": "fake-exec-thread"})
    send({"type": "turn.failed", "error": {"message": "redacted"}})
    raise SystemExit(1)

thread_id = "fake-exec-thread"
turn_id = "fake-exec-turn"
command = "pytest -vv"
raw_output = (
    "============================= test session starts ==============================\n"
    + "tests/test_values.py::test_value[0] PASSED\n" * 300
    + "============================= 300 passed in 1.00s ==============================\n"
)
model_output = raw_output
hooks_enabled = "--enable" in arguments and arguments[arguments.index("--enable") + 1] == "hooks"
if (
    hooks_enabled
    and mode == "optimized"
    and strategy in {"post-replace", "compact-output", "hybrid"}
):
    hook_event = {
        "session_id": thread_id,
        "turn_id": turn_id,
        "cwd": str(cwd.resolve()),
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_use_id": "command-1",
        "tool_input": {"command": command},
        "tool_response": {"stdout": raw_output, "stderr": "", "exit_code": 0},
    }
    hook = subprocess.run(
        [sys.executable, "-m", "llmcut", "hook", "handle"],
        input=json.dumps(hook_event),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
        env=os.environ.copy(),
    )
    if hook.stdout:
        response = json.loads(hook.stdout)
        model_output = str(response.get("reason", raw_output))

send({"type": "thread.started", "thread_id": thread_id})
send({"type": "turn.started", "turn_id": turn_id})
send(
    {
        "type": "item.started",
        "item": {
            "id": "command-1",
            "type": "command_execution",
            "command": command,
            "status": "in_progress",
        },
    }
)
send(
    {
        "type": "item.completed",
        "item": {
            "id": "command-1",
            "type": "command_execution",
            "command": command,
            "aggregated_output": model_output,
            "exit_code": 0,
            "status": "completed",
        },
    }
)
target = cwd / "app" / "callback.py"
if target.is_file() and "validation-failure" not in scenario:
    target.write_text(target.read_text().replace("TIMEOUT_SECONDS * 1000", "TIMEOUT_SECONDS"))
    send(
        {
            "type": "item.completed",
            "item": {
                "id": "change-1",
                "type": "file_change",
                "changes": [{"path": "app/callback.py", "kind": "update"}],
                "status": "completed",
            },
        }
    )
usage = {
    "input_tokens": 80 if hooks_enabled else 100,
    "cached_input_tokens": 20,
    "output_tokens": 10,
    "reasoning_output_tokens": 2,
}
if scenario == "missing-usage":
    usage = {}
send({"type": "turn.completed", "turn_id": turn_id, "usage": usage})
