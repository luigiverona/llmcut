#!/usr/bin/env python3
"""Scripted Codex App Server used through the production stdio JSON-RPC path."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

scenario = os.environ.get("LLMCUT_FAKE_SCENARIO", "success")
mode = os.environ.get("LLMCUT_EVAL_MODE", "baseline")
cwd = Path(".")
turns = 0


def send(value: dict[str, object]) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


for line in sys.stdin:
    if scenario == "malformed":
        print("{not-json", flush=True)
        continue
    request = json.loads(line)
    method = request.get("method")
    if method == "initialized":
        continue
    if scenario == "unexpected-exit" and method == "turn/start":
        raise SystemExit(17)
    if method == "initialize":
        if scenario == "protocol-mismatch":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32600, "message": "unsupported protocol"},
                }
            )
            continue
        result: dict[str, object] = {"platformFamily": "unix", "platformOs": "linux"}
    elif method == "thread/start":
        if request["params"].get("sandbox") not in {
            "read-only",
            "workspace-write",
            "danger-full-access",
        }:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32600, "message": "invalid sandbox"},
                }
            )
            continue
        if request["params"].get("approvalPolicy") not in {"never", "on-request", "untrusted"}:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32600, "message": "invalid approval policy"},
                }
            )
            continue
        cwd = Path(request["params"]["cwd"])
        result = {"thread": {"id": "thread-1", "sessionId": "thread-1"}}
    elif method == "turn/start":
        turns += 1
        turn_id = f"turn-{turns}"
        result = {"turn": {"id": turn_id, "status": "inProgress", "items": []}}
    elif method == "turn/interrupt":
        result = {}
    else:
        result = {}
    if "id" in request:
        send({"jsonrpc": "2.0", "id": request["id"], "result": result})
    if method != "turn/start":
        continue
    if scenario == "timeout":
        time.sleep(60)
        continue
    send(
        {
            "jsonrpc": "2.0",
            "method": "turn/started",
            "params": {"turn": {"id": turn_id, "status": "inProgress", "items": []}},
        }
    )
    send(
        {
            "jsonrpc": "2.0",
            "method": "item/completed",
            "params": {
                "item": {
                    "id": f"command-{turns}",
                    "type": "commandExecution",
                    "command": ["python", "tests/validate_callback.py"],
                    "cwd": str(cwd),
                    "status": "completed",
                    "exitCode": 0,
                }
            },
        }
    )
    should_fix = scenario not in {"validation-failure"} and not (
        scenario == "correction" and turns == 1
    )
    if should_fix:
        target = cwd / "app" / "callback.py"
        target.write_text(target.read_text().replace("TIMEOUT_SECONDS * 1000", "TIMEOUT_SECONDS"))
        send(
            {
                "jsonrpc": "2.0",
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": f"change-{turns}",
                        "type": "fileChange",
                        "changes": [
                            {"path": "app/callback.py", "kind": "update", "diff": "redacted"}
                        ],
                        "status": "completed",
                    }
                },
            }
        )
    if mode == "optimized":
        count = 2 if scenario == "repeated-mcp" else 1
        for number in range(count):
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "item/started",
                    "params": {
                        "item": {
                            "id": f"mcp-{turns}-{number}",
                            "type": "mcpToolCall",
                            "server": "llmcut",
                            "tool": "llmcut_context_get",
                            "status": "inProgress",
                            "arguments": {"context_id": "app/callback.py"},
                        }
                    },
                }
            )
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": f"mcp-{turns}-{number}",
                            "type": "mcpToolCall",
                            "server": "llmcut",
                            "tool": "llmcut_context_get",
                            "status": "completed",
                            "result": {"digest": "sha256:fixture", "bytes": 120},
                        }
                    },
                }
            )
    if scenario == "reroute":
        send(
            {
                "jsonrpc": "2.0",
                "method": "model/rerouted",
                "params": {"fromModel": "fake-codex-model", "toModel": "other-model"},
            }
        )
        continue
    if scenario != "unavailable-usage":
        input_tokens = 900 if mode == "baseline" else 500
        send(
            {
                "jsonrpc": "2.0",
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "inputTokens": input_tokens,
                        "outputTokens": 20,
                        "cachedInputTokens": 10,
                        "reasoningTokens": 5,
                    }
                },
            }
        )
    send(
        {
            "jsonrpc": "2.0",
            "method": "turn/completed",
            "params": {"turn": {"id": turn_id, "status": "completed", "items": []}},
        }
    )
