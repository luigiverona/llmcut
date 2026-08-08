#!/usr/bin/env python3
"""Structured fake ``codex exec`` runtime that invokes the real llmcut hook subprocess."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import urllib.request
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
requested_model = arguments[arguments.index("--model") + 1] if "--model" in arguments else "unknown"
config_values = [
    arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == "--config"
]
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
if hooks_enabled and os.environ.get("LLMCUT_HOOK_LEASE"):
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
    hooks_file = Path(os.environ["CODEX_HOME"]) / "hooks.json"
    hooks_document = json.loads(hooks_file.read_text())
    hook_command = hooks_document["hooks"]["PostToolUse"][-1]["hooks"][0]["command"]
    hook = subprocess.run(
        shlex.split(hook_command),
        input=json.dumps(hook_event),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
        env=os.environ.copy(),
    )
    if hook.stderr:
        sys.stderr.write(hook.stderr)
        sys.stderr.flush()
    if hook.stdout:
        response = json.loads(hook.stdout)
        model_output = str(response.get("reason", raw_output))

send({"type": "thread.started", "thread_id": thread_id})
send({"type": "turn.started", "turn_id": turn_id})
otel_environment = next(
    (
        value.split("=", 1)[1].strip('"')
        for value in config_values
        if value.startswith("otel.environment=")
    ),
    None,
)
otel_exporter = next((value for value in config_values if value.startswith("otel.exporter=")), None)
if otel_environment and otel_exporter and '"endpoint"="' in otel_exporter:
    endpoint = otel_exporter.split('"endpoint"="', 1)[1].split('"', 1)[0]
    attributes = {
        "event.name": "codex.conversation_starts",
        "conversation.id": thread_id,
        "model": requested_model,
        "reasoning_effort": "low",
        "sandbox": "workspace-write",
        "approval_policy": "never",
        "deployment.environment.name": otel_environment,
    }
    payload = json.dumps(
        {
            "resourceLogs": [
                {
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "attributes": [
                                        {"key": key, "value": {"stringValue": value}}
                                        for key, value in attributes.items()
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310 - test loopback receiver
        endpoint, data=payload, method="POST"
    )
    with urllib.request.urlopen(request, timeout=2):  # noqa: S310 - test loopback receiver
        pass
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
    "input_tokens": 80 if mode == "optimized" else 100,
    "cached_input_tokens": 20,
    "output_tokens": 10,
    "reasoning_output_tokens": 2,
}
if scenario == "missing-usage":
    usage = {}
send({"type": "turn.completed", "turn_id": turn_id, "usage": usage})
