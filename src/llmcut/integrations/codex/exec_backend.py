from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmcut.integrations.codex.app_server import CodexRun
from llmcut.integrations.codex.backend import BackendCapabilities
from llmcut.integrations.codex.events import NormalizedEvent, normalize_exec_event
from llmcut.model import digest_bytes

MAX_EVENTS = 20_000
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 128 * 1024
TERMINAL_EVENTS = {"turn.completed", "turn.failed"}


@dataclass(frozen=True, slots=True)
class ExecInvocation:
    argv: tuple[str, ...]
    prompt: bytes
    executable: str
    executable_digest: str
    configuration_digest: str


def build_exec_invocation(
    executable: str,
    *,
    task: str,
    cwd: Path,
    model: str,
    reasoning: str,
    sandbox: str,
    approval_policy: str,
    config_overrides: tuple[str, ...] = (),
    hooks: bool = False,
    allow_hook_trust_bypass: bool = False,
    ephemeral: bool = True,
) -> ExecInvocation:
    """Build the sole supported argv form; the private task is stdin-only."""
    if sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
        raise ValueError(f"unsupported Codex sandbox: {sandbox}")
    if approval_policy not in {"never", "on-request", "untrusted"}:
        raise ValueError(f"unsupported Codex approval policy: {approval_policy}")
    if reasoning not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError(f"unsupported Codex reasoning effort: {reasoning}")
    resolved = shutil.which(executable) if os.sep not in executable else executable
    if resolved is None:
        raise RuntimeError(f"Codex executable is unavailable: {executable}")
    resolved_path = Path(resolved).resolve(strict=True)
    argv = [str(resolved_path)]
    if hooks:
        if not allow_hook_trust_bypass:
            raise ValueError("automated hooks require explicit hook trust bypass")
        # This placement exactly matches the runtime-proven direct-exec conformance invocation.
        argv.append("--dangerously-bypass-hook-trust")
    argv.extend(
        [
            "exec",
            "--json",
            "--color",
            "never",
            "--cd",
            str(cwd.resolve(strict=True)),
            "--model",
            model,
            "--sandbox",
            sandbox,
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--strict-config",
            "--enable" if hooks else "--disable",
            "hooks",
        ]
    )
    if ephemeral:
        argv.append("--ephemeral")
    fixed = (
        f"approval_policy={json.dumps(approval_policy)}",
        f"model_reasoning_effort={json.dumps(reasoning)}",
        "mcp_servers={}",
        'web_search="disabled"',
    )
    for value in fixed + tuple(config_overrides):
        argv.extend(("--config", value))
    argv.append("-")
    encoded_config = json.dumps(argv[1:-1], separators=(",", ":")).encode()
    return ExecInvocation(
        tuple(argv),
        task.encode(),
        str(resolved_path),
        digest_bytes(resolved_path.read_bytes()),
        digest_bytes(encoded_config),
    )


@dataclass(slots=True)
class _TurnResult:
    thread_id: str
    turn_id: str
    status: str
    events: list[NormalizedEvent]
    usage: dict[str, Any]
    stderr: str
    response_digests: tuple[str, ...]


class ExecBackend:
    """Bounded subprocess backend for the documented ``codex exec --json`` surface."""

    def __init__(self, executable: str = "codex", *, allow_hook_trust_bypass: bool = False) -> None:
        self.executable = executable
        self.allow_hook_trust_bypass = allow_hook_trust_bypass
        self._process: asyncio.subprocess.Process | None = None

    async def doctor(self) -> BackendCapabilities:
        resolved = (
            shutil.which(self.executable) if os.sep not in self.executable else self.executable
        )
        if resolved is None:
            return BackendCapabilities("exec", False, None, None, False, "codex is unavailable")
        try:
            process = await asyncio.create_subprocess_exec(
                resolved,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), 5)
        except (OSError, TimeoutError):
            return BackendCapabilities("exec", False, None, None, False, "version probe failed")
        version = stdout.decode(errors="replace").strip() if process.returncode == 0 else None
        exclusive = False
        if version:
            from llmcut.integrations.codex.hooks.capabilities import capabilities_for

            exclusive = capabilities_for(version).post_replacement == "supported"
        return BackendCapabilities(
            "exec",
            process.returncode == 0,
            version,
            version,
            True,
            None if process.returncode == 0 else "version probe failed",
            hooks=True,
            exclusive_post_tool_replacement=exclusive,
            jsonl_usage=True,
            command_events=True,
            file_change_events=True,
            resumable_turns=True,
            resolved_model_observation=False,
        )

    async def cancel(self) -> None:
        if self._process is not None:
            await _terminate(self._process)

    async def run(
        self,
        *,
        task: str,
        cwd: Path,
        model: str,
        reasoning: str,
        sandbox: str,
        approval_policy: str,
        timeout: float,
        max_turns: int,
        environment: dict[str, str],
        config_overrides: tuple[str, ...],
        validation_callback: Any = None,
        cancellation: asyncio.Event | None = None,
    ) -> CodexRun:
        hooks = any(value == "features.hooks=true" for value in config_overrides)
        invocation = build_exec_invocation(
            self.executable,
            task=task,
            cwd=cwd,
            model=model,
            reasoning=reasoning,
            sandbox=sandbox,
            approval_policy=approval_policy,
            config_overrides=tuple(
                value for value in config_overrides if value != "features.hooks=true"
            ),
            hooks=hooks,
            allow_hook_trust_bypass=self.allow_hook_trust_bypass,
            ephemeral=max_turns == 1,
        )
        deadline = time.monotonic() + timeout
        turns: list[_TurnResult] = []
        requests = [digest_bytes(invocation.prompt)]
        first = await self._run_process(
            invocation.argv,
            invocation.prompt,
            cwd,
            environment,
            deadline,
            cancellation,
        )
        turns.append(first)
        valid = validation_callback() if validation_callback else True
        while first.status == "completed" and not valid and len(turns) < max_turns:
            correction = (
                b"Deterministic validation failed after the prior turn. Inspect the current "
                b"worktree, correct the requested outcome, and rerun the required validation."
            )
            requests.append(digest_bytes(correction))
            resume_argv = _resume_argv(invocation.argv, first.thread_id)
            first = await self._run_process(
                resume_argv, correction, cwd, environment, deadline, cancellation
            )
            turns.append(first)
            valid = validation_callback() if validation_callback else True
        usage = _sum_usage([turn.usage for turn in turns])
        return CodexRun(
            turns[-1].thread_id or turns[0].thread_id,
            turns[-1].turn_id,
            turns[-1].status,
            [event for turn in turns for event in turn.events],
            usage,
            len(turns),
            max(0, len(turns) - 1),
            len(turns) == 1 and valid and turns[0].status == "completed",
            "\n".join(turn.stderr for turn in turns if turn.stderr),
            tuple(requests),
            tuple(value for turn in turns for value in turn.response_digests),
            {
                "codex_executable": invocation.executable,
                "codex_executable_digest": invocation.executable_digest,
                "configuration_digest": invocation.configuration_digest,
                "prompt_transport": "stdin",
                "jsonl_usage_authority": "turn.completed.usage",
            },
        )

    async def _run_process(
        self,
        argv: tuple[str, ...],
        prompt: bytes,
        cwd: Path,
        environment: dict[str, str],
        deadline: float,
        cancellation: asyncio.Event | None,
    ) -> _TurnResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=MAX_LINE_BYTES + 1,
        )
        self._process = process
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        stderr_task = asyncio.create_task(_bounded_stderr(process.stderr))
        events: list[NormalizedEvent] = []
        response_digests: list[str] = []
        terminal: list[dict[str, Any]] = []
        thread_id = ""
        turn_id = ""
        usage: dict[str, Any] = {}
        fatal_top_level_error = False
        try:
            process.stdin.write(prompt)
            await process.stdin.drain()
            process.stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()
            while True:
                if cancellation is not None and cancellation.is_set():
                    raise asyncio.CancelledError
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Codex exec evaluation timed out")
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), min(0.1, remaining))
                except TimeoutError:
                    continue
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    raise RuntimeError("Codex exec emitted an oversized JSONL line")
                if len(events) >= MAX_EVENTS:
                    raise RuntimeError("Codex exec exceeded the event limit")
                response_digests.append(digest_bytes(line))
                try:
                    raw = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Codex exec emitted malformed JSONL") from exc
                if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
                    raise RuntimeError("Codex exec emitted an unsupported JSONL value")
                event_type = raw["type"]
                if event_type == "error" and str(raw.get("severity", "fatal")) not in {
                    "warning",
                    "nonfatal",
                }:
                    fatal_top_level_error = True
                if event_type in {"model.rerouted", "model.fallback"}:
                    fatal_top_level_error = True
                if event_type == "thread.started":
                    thread_id = str(raw.get("thread_id", ""))
                if event_type in TERMINAL_EVENTS:
                    terminal.append(raw)
                    if event_type == "turn.completed":
                        usage = _parse_usage(raw.get("usage"))
                    turn_id = str(raw.get("turn_id", raw.get("id", "")))
                normalized = normalize_exec_event(raw)
                if normalized is not None:
                    events.append(normalized)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex exec evaluation timed out")
            code = await asyncio.wait_for(process.wait(), remaining)
            stderr = await stderr_task
            if code != 0:
                raise RuntimeError(f"Codex exec exited nonzero ({code}): {_safe_stderr(stderr)}")
            if len(terminal) != 1 or terminal[0]["type"] != "turn.completed":
                raise RuntimeError("Codex exec did not emit exactly one completed terminal turn")
            if fatal_top_level_error:
                raise RuntimeError("Codex exec emitted a fatal top-level error or model fallback")
            return _TurnResult(
                thread_id,
                turn_id,
                "completed",
                events,
                usage,
                stderr,
                tuple(response_digests),
            )
        except BaseException:
            await _terminate(process)
            stderr_task.cancel()
            with contextlib.suppress(BaseException):
                await stderr_task
            raise
        finally:
            self._process = None


def _parse_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Codex exec terminal usage is malformed")
    aliases = {
        "input_tokens": "inputTokens",
        "cached_input_tokens": "cachedInputTokens",
        "output_tokens": "outputTokens",
        "reasoning_output_tokens": "reasoningOutputTokens",
    }
    result: dict[str, Any] = {}
    for source, target in aliases.items():
        item = value.get(source)
        if item is None and source in {"cached_input_tokens", "reasoning_output_tokens"}:
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RuntimeError(f"Codex exec usage field is malformed: {source}")
        result[target] = item
    if "inputTokens" not in result or "outputTokens" not in result:
        raise RuntimeError("Codex exec required usage fields are unavailable")
    if "cachedInputTokens" in result:
        result["noncachedInputTokens"] = max(0, result["inputTokens"] - result["cachedInputTokens"])
    return result


def _sum_usage(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {key for value in values for key in value}
    return {key: sum(int(value.get(key, 0)) for value in values) for key in sorted(keys)}


def _resume_argv(initial: tuple[str, ...], thread_id: str) -> tuple[str, ...]:
    if not thread_id:
        raise RuntimeError("Codex exec resume requires a thread id")
    executable = initial[0]
    exec_index = initial.index("exec")
    prefix = list(initial[1:exec_index])
    options = list(initial[exec_index + 1 : -1])
    # Resume restores cwd/sandbox from the session and does not accept those initial-only flags.
    filtered: list[str] = []
    skip = False
    for value in options:
        if skip:
            skip = False
            continue
        if value in {"--cd", "--sandbox", "--color"}:
            skip = True
            continue
        if value == "--ephemeral":
            continue
        filtered.append(value)
    return tuple([executable, *prefix, "exec", "resume", *filtered, thread_id, "-"])


async def _bounded_stderr(stream: asyncio.StreamReader) -> str:
    data = await stream.read(MAX_STDERR_BYTES + 1)
    suffix = b"\n[stderr truncated]" if len(data) > MAX_STDERR_BYTES else b""
    return (data[:MAX_STDERR_BYTES] + suffix).decode(errors="replace")


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), 2)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), 2)


def _safe_stderr(value: str) -> str:
    # Codex stderr can contain configuration diagnostics; never echo task or command content.
    return "stderr available" if value else "no stderr"
