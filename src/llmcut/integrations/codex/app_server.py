from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmcut.integrations.codex.events import NormalizedEvent, normalize_event
from llmcut.model import digest_bytes

MAX_EVENTS = 2_000
MAX_STDERR_BYTES = 64 * 1024
MAX_LINE_BYTES = 1024 * 1024

_SANDBOX_WIRE = {
    "read-only": "readOnly",
    "workspace-write": "workspaceWrite",
    "danger-full-access": "dangerFullAccess",
}
_APPROVAL_WIRE = {"never": "never", "on-request": "on-request", "untrusted": "untrusted"}


@dataclass(slots=True)
class CodexRun:
    thread_id: str
    turn_id: str
    status: str
    events: list[NormalizedEvent] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    turns: int = 0
    correction_turns: int = 0
    first_attempt_completion: bool = False
    stderr: str = ""
    request_digests: tuple[str, ...] = ()
    response_digests: tuple[str, ...] = ()


class CodexAppServer:
    """Bounded JSON-RPC client for the documented ``codex app-server`` interface."""

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
        max_turns: int = 1,
        environment: dict[str, str] | None = None,
        config_overrides: tuple[str, ...] = (),
        validation_callback: Callable[[], bool] | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> CodexRun:
        if sandbox not in _SANDBOX_WIRE or approval_policy not in _APPROVAL_WIRE:
            raise ValueError("unsupported Codex sandbox or approval policy")
        argv = [self.executable]
        for override in config_overrides:
            argv.extend(("-c", override))
        argv.append("app-server")
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
            limit=MAX_LINE_BYTES,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("Codex App Server stdio is unavailable")
        writer, reader = process.stdin, process.stdout
        deadline = time.monotonic() + timeout
        identifier = 0
        events: list[NormalizedEvent] = []
        pending: list[dict[str, Any]] = []
        requests: list[str] = []
        responses: list[str] = []
        stderr_task = asyncio.create_task(_read_stderr(process.stderr))
        thread_id = ""
        turn_id = ""
        usage: dict[str, Any] | None = None

        async def send(value: dict[str, Any]) -> None:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
            requests.append(digest_bytes(encoded.encode()))
            writer.write((encoded + "\n").encode())
            await writer.drain()

        async def read() -> dict[str, Any]:
            _check_cancel(cancellation)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex App Server evaluation timed out")
            try:
                line = await asyncio.wait_for(reader.readline(), remaining)
            except asyncio.LimitOverrunError as exc:
                raise RuntimeError("Codex App Server emitted an oversized event") from exc
            if not line:
                code = await process.wait()
                raise RuntimeError(f"Codex App Server exited unexpectedly with status {code}")
            if len(line) > MAX_LINE_BYTES:
                raise RuntimeError("Codex App Server emitted an oversized event")
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Codex App Server emitted malformed JSON") from exc
            if not isinstance(message, dict):
                raise RuntimeError("Codex App Server message must be an object")
            encoded = json.dumps(message, sort_keys=True, separators=(",", ":"))
            responses.append(digest_bytes(encoded.encode()))
            return message

        async def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
            nonlocal identifier
            identifier += 1
            await send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
            while True:
                message = await read()
                if message.get("id") == identifier:
                    if "error" in message:
                        raise RuntimeError(f"Codex App Server {method} failed: {message['error']}")
                    result = message.get("result", {})
                    if not isinstance(result, dict):
                        raise RuntimeError(f"Codex App Server {method} returned an invalid result")
                    return dict(result)
                if "method" in message:
                    pending.append(message)

        async def next_event() -> dict[str, Any]:
            return pending.pop(0) if pending else await read()

        try:
            await rpc(
                "initialize",
                {
                    "clientInfo": {"name": "llmcut", "version": "0.5.0"},
                    "capabilities": {"experimentalApi": False},
                },
            )
            await send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
            started = await rpc(
                "thread/start",
                {
                    "model": model,
                    "cwd": str(cwd.resolve()),
                    "approvalPolicy": _APPROVAL_WIRE[approval_policy],
                    "sandbox": sandbox,
                    "serviceName": "llmcut_agent_eval",
                },
            )
            thread = started.get("thread")
            if not isinstance(thread, dict) or not thread.get("id"):
                raise RuntimeError("Codex App Server thread/start omitted thread id")
            thread_id = str(thread["id"])
            turns = 0
            first_attempt = False
            status = "failed"
            prompt = task
            while turns < max_turns:
                _check_cancel(cancellation)
                turn_result = await rpc(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                        "cwd": str(cwd.resolve()),
                        "approvalPolicy": _APPROVAL_WIRE[approval_policy],
                        "sandboxPolicy": {"type": _SANDBOX_WIRE[sandbox]},
                        "model": model,
                        "effort": reasoning,
                    },
                )
                turn = turn_result.get("turn")
                if not isinstance(turn, dict) or not turn.get("id"):
                    raise RuntimeError("Codex App Server turn/start omitted turn id")
                turn_id = str(turn["id"])
                turns += 1
                while True:
                    message = await next_event()
                    normalized = normalize_event(message)
                    if normalized is not None and len(events) < MAX_EVENTS:
                        events.append(normalized)
                    if normalized and normalized.kind == "usage_update":
                        usage = dict(normalized.data)
                    if normalized and normalized.method == "model/rerouted":
                        raise RuntimeError("Codex changed the configured model; comparison invalid")
                    if normalized and normalized.kind in {"turn_completed", "turn_failed"}:
                        if normalized.data.get("turn_id") != turn_id:
                            continue
                        status = str(normalized.data.get("status", "failed"))
                        break
                validation_passed = validation_callback() if validation_callback else True
                if status == "completed" and validation_passed:
                    first_attempt = turns == 1
                    break
                if turns < max_turns:
                    prompt = (
                        "Deterministic validation failed after the prior turn. Inspect the current "
                        "worktree, correct the requested outcome, and rerun the required "
                        "validation."
                    )
            result = CodexRun(
                thread_id,
                turn_id,
                status,
                events,
                usage,
                turns,
                max(0, turns - 1),
                first_attempt,
                "",
                tuple(requests),
                tuple(responses),
            )
            await _stop(process)
            result.stderr = await stderr_task
            return result
        except (TimeoutError, asyncio.CancelledError):
            if thread_id and turn_id and process.returncode is None:
                with contextlib.suppress(BrokenPipeError, ConnectionError):
                    await send(
                        {
                            "jsonrpc": "2.0",
                            "id": identifier + 1,
                            "method": "turn/interrupt",
                            "params": {"threadId": thread_id, "turnId": turn_id},
                        }
                    )
            raise
        finally:
            await _stop(process)
            stderr = await stderr_task
            # CodexRun is returned above; stderr is intentionally only attached on diagnostics
            # raised by callers, never printed automatically or allowed to grow without bound.
            _ = stderr


async def _read_stderr(reader: asyncio.StreamReader) -> str:
    chunks = bytearray()
    while len(chunks) < MAX_STDERR_BYTES:
        value = await reader.read(min(8_192, MAX_STDERR_BYTES - len(chunks)))
        if not value:
            break
        chunks.extend(value)
    return chunks.decode(errors="replace")


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        process.kill()
        await process.wait()


def _check_cancel(cancellation: asyncio.Event | None) -> None:
    if cancellation is not None and cancellation.is_set():
        raise asyncio.CancelledError


def allowed_environment(names: tuple[str, ...], mode: str) -> dict[str, str]:
    safe_defaults = {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SYSTEMROOT", "WINDIR"}
    selected = safe_defaults | set(names)
    result = {key: value for key, value in os.environ.items() if key in selected}
    result["LLMCUT_EVAL_MODE"] = mode
    return result
