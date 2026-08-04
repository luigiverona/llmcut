from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CodexRun:
    thread_id: str
    turn_id: str
    status: str
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    changed_files: list[str] = field(default_factory=list)


class CodexAppServer:
    """Small supported JSON-RPC client for ``codex app-server``."""

    def __init__(self, executable: str = "codex") -> None:
        self.executable = executable

    async def run(
        self,
        *,
        task: str,
        cwd: Path,
        model: str,
        reasoning: str,
        sandbox: str,
        approval_policy: str,
        timeout: float = 600,
    ) -> CodexRun:
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Codex App Server stdio is unavailable")
        writer = process.stdin
        reader = process.stdout
        identifier = 0

        async def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
            nonlocal identifier
            identifier += 1
            writer.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}
                    )
                    + "\n"
                ).encode()
            )
            await writer.drain()
            while True:
                line = await reader.readline()
                if not line:
                    raise RuntimeError("Codex App Server closed unexpectedly")
                message = json.loads(line)
                if message.get("id") == identifier:
                    if "error" in message:
                        raise RuntimeError(str(message["error"]))
                    return dict(message.get("result", {}))

        events: list[dict[str, Any]] = []
        try:
            await asyncio.wait_for(
                rpc("initialize", {"clientInfo": {"name": "llmcut", "version": "0.4.0"}}), timeout
            )
            started = await asyncio.wait_for(
                rpc(
                    "thread/start",
                    {
                        "model": model,
                        "cwd": str(cwd.resolve()),
                        "approvalPolicy": approval_policy,
                        "sandbox": sandbox,
                        "config": {"model_reasoning_effort": reasoning},
                    },
                ),
                timeout,
            )
            thread_id = str(started["thread"]["id"])
            turn = await asyncio.wait_for(
                rpc(
                    "turn/start", {"threadId": thread_id, "input": [{"type": "text", "text": task}]}
                ),
                timeout,
            )
            turn_id = str(turn["turn"]["id"])
            usage = None
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout)
                if not line:
                    raise RuntimeError("Codex App Server closed before turn completion")
                event = json.loads(line)
                if "method" not in event:
                    continue
                method = str(event["method"])
                params = dict(event.get("params", {}))
                # Persist structured operational events, never raw reasoning content.
                if method in {
                    "turn/completed",
                    "turn/diff/updated",
                    "thread/tokenUsage/updated",
                    "model/rerouted",
                }:
                    events.append({"method": method, "params": params})
                if method == "thread/tokenUsage/updated":
                    usage = params.get("tokenUsage") or params.get("usage")
                if method == "model/rerouted":
                    raise RuntimeError("Codex changed the configured model; comparison invalid")
                if (
                    method == "turn/completed"
                    and str(params.get("turn", {}).get("id", params.get("turnId", ""))) == turn_id
                ):
                    return CodexRun(
                        thread_id,
                        turn_id,
                        str(params.get("turn", {}).get("status", "completed")),
                        events,
                        usage,
                    )
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
