# ruff: noqa: E501 -- embedded fake-executable JSONL is intentionally literal.
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from llmcut.integrations.codex import exec_backend
from llmcut.integrations.codex.events import normalize_exec_event
from llmcut.integrations.codex.exec_backend import (
    ExecBackend,
    _parse_usage,
    _resume_argv,
    build_exec_invocation,
)


def _fake_codex(tmp_path: Path, body: str | None = None) -> Path:
    path = tmp_path / "fake-codex"
    source = (
        body
        or r"""#!/usr/bin/env python3
import json, pathlib, sys
if "--version" in sys.argv:
    print("codex-cli 0.146.0")
    raise SystemExit(0)
prompt = sys.stdin.read()
pathlib.Path("prompt.digest").write_text(str(len(prompt)))
events = [
 {"type":"thread.started","thread_id":"thread-1"},
 {"type":"turn.started"},
 {"type":"item.started","item":{"id":"cmd-1","type":"command_execution","command":"pytest -q","status":"in_progress"}},  # noqa: E501
 {"type":"item.completed","item":{"id":"cmd-1","type":"command_execution","command":"pytest -q","aggregated_output":"ok\n","exit_code":0,"status":"completed"}},  # noqa: E501
 {"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":25,"output_tokens":10}},
]
for event in events:
    print(json.dumps(event), flush=True)
"""
    )
    path.write_text(source)
    path.chmod(0o700)
    return path


def test_build_invocation_is_explicit_and_prompt_is_not_argv(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    invocation = build_exec_invocation(
        str(executable),
        task="private task marker",
        cwd=tmp_path,
        model="gpt-test",
        reasoning="low",
        sandbox="workspace-write",
        approval_policy="never",
    )
    joined = " ".join(invocation.argv)
    assert "private task marker" not in joined
    assert invocation.prompt == b"private task marker"
    assert "--json" in invocation.argv
    assert "--ignore-user-config" in invocation.argv
    assert "--ignore-rules" in invocation.argv
    assert "--skip-git-repo-check" in invocation.argv
    assert "--strict-config" in invocation.argv
    assert invocation.argv[-1] == "-"
    assert invocation.argv[invocation.argv.index("--disable") + 1] == "hooks"
    assert 'approval_policy="never"' in invocation.argv
    assert 'model_reasoning_effort="low"' in invocation.argv
    assert "mcp_servers={}" in invocation.argv
    with pytest.raises(RuntimeError, match="unavailable"):
        build_exec_invocation(
            "definitely-missing-codex-executable",
            task="task",
            cwd=tmp_path,
            model="model",
            reasoning="low",
            sandbox="workspace-write",
            approval_policy="never",
        )


def test_hook_invocation_requires_explicit_bypass(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    kwargs: dict[str, Any] = dict(
        task="task",
        cwd=tmp_path,
        model="gpt-test",
        reasoning="low",
        sandbox="workspace-write",
        approval_policy="never",
        hooks=True,
    )
    with pytest.raises(ValueError, match="trust bypass"):
        build_exec_invocation(str(executable), **kwargs)
    result = build_exec_invocation(str(executable), **kwargs, allow_hook_trust_bypass=True)
    assert "--dangerously-bypass-hook-trust" in result.argv
    assert result.argv[result.argv.index("--enable") + 1] == "hooks"


@pytest.mark.parametrize("field", ["sandbox", "approval_policy", "reasoning"])
def test_invocation_rejects_unsupported_setting(tmp_path: Path, field: str) -> None:
    executable = _fake_codex(tmp_path)
    kwargs: dict[str, Any] = {
        "task": "task",
        "cwd": tmp_path,
        "model": "model",
        "reasoning": "low",
        "sandbox": "workspace-write",
        "approval_policy": "never",
    }
    kwargs[field] = "unsupported"
    with pytest.raises(ValueError):
        build_exec_invocation(str(executable), **kwargs)


@pytest.mark.asyncio
async def test_exec_backend_parses_jsonl_and_usage(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    backend = ExecBackend(str(executable))
    result = await backend.run(
        task="do the task",
        cwd=tmp_path,
        model="gpt-test",
        reasoning="low",
        sandbox="workspace-write",
        approval_policy="never",
        timeout=5,
        max_turns=1,
        environment={"PATH": os.environ["PATH"]},
        config_overrides=(),
    )
    assert result.status == "completed"
    assert result.thread_id == "thread-1"
    assert result.usage == {
        "cachedInputTokens": 25,
        "inputTokens": 100,
        "noncachedInputTokens": 75,
        "outputTokens": 10,
    }
    command = next(item for item in result.events if item.kind == "command_execution")
    assert "command" not in command.data
    assert command.data["command_digest"]
    assert command.data["output_bytes"] in {0, 3}
    assert (tmp_path / "prompt.digest").read_text() == str(len("do the task"))


@pytest.mark.asyncio
async def test_doctor_reports_exec_capabilities(tmp_path: Path) -> None:
    backend = ExecBackend(str(_fake_codex(tmp_path)))
    result = await backend.doctor()
    assert result.installed
    assert result.jsonl_usage
    assert result.command_events
    assert result.hooks


@pytest.mark.asyncio
async def test_doctor_reports_missing_and_failed_executable(tmp_path: Path) -> None:
    missing = await ExecBackend("definitely-missing-codex-executable").doctor()
    assert not missing.installed
    broken = _fake_codex(tmp_path, "#!/usr/bin/env python3\nraise SystemExit(2)\n")
    failed = await ExecBackend(str(broken)).doctor()
    assert not failed.installed


@pytest.mark.asyncio
async def test_doctor_handles_launch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(*args: object, **kwargs: object) -> None:
        raise OSError("no")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    result = await ExecBackend("/bin/true").doctor()
    assert not result.installed


@pytest.mark.asyncio
async def test_cancel_delegates_to_active_process(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    async def terminate(process: object) -> None:
        seen.append(process)

    marker = object()
    backend = ExecBackend()
    backend._process = marker  # type: ignore[assignment]
    monkeypatch.setattr(exec_backend, "_terminate", terminate)
    await backend.cancel()
    assert seen == [marker]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("print('not json')", "malformed JSONL"),
        ("print('x' * (2 * 1024 * 1024 + 1))", "oversized JSONL"),
        ("print(json.dumps({'type':'thread.started','thread_id':'x'}))", "terminal turn"),
        (
            "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':'x','output_tokens':1}}))",
            "usage field",
        ),
        (
            "print(json.dumps({'type':'turn.failed'}))",
            "completed terminal turn",
        ),
        (
            "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}})); print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))",
            "exactly one completed",
        ),
        ("print(json.dumps([]))", "unsupported JSONL value"),
        (
            "print(json.dumps({'type':'error','message':'private'})); print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))",
            "fatal top-level error",
        ),
        (
            "print(json.dumps({'type':'model.fallback','from_model':'a','to_model':'b'})); print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))",
            "model fallback",
        ),
    ],
)
async def test_exec_backend_rejects_invalid_stream(tmp_path: Path, body: str, message: str) -> None:
    executable = _fake_codex(
        tmp_path,
        "#!/usr/bin/env python3\nimport json\n" + body + "\n",
    )
    backend = ExecBackend(str(executable))
    with pytest.raises(RuntimeError, match=message):
        await backend.run(
            task="task",
            cwd=tmp_path,
            model="model",
            reasoning="low",
            sandbox="workspace-write",
            approval_policy="never",
            timeout=5,
            max_turns=1,
            environment={"PATH": os.environ["PATH"]},
            config_overrides=(),
        )


@pytest.mark.asyncio
async def test_timeout_and_cancellation_clean_process(tmp_path: Path) -> None:
    executable = _fake_codex(
        tmp_path,
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
    )
    backend = ExecBackend(str(executable))
    with pytest.raises(TimeoutError):
        await backend.run(
            task="task",
            cwd=tmp_path,
            model="model",
            reasoning="low",
            sandbox="workspace-write",
            approval_policy="never",
            timeout=0.05,
            max_turns=1,
            environment={"PATH": os.environ["PATH"]},
            config_overrides=(),
        )
    assert backend._process is None

    cancellation = asyncio.Event()
    cancellation.set()
    with pytest.raises(asyncio.CancelledError):
        await backend.run(
            task="task",
            cwd=tmp_path,
            model="model",
            reasoning="low",
            sandbox="workspace-write",
            approval_policy="never",
            timeout=5,
            max_turns=1,
            environment={"PATH": os.environ["PATH"]},
            config_overrides=(),
            cancellation=cancellation,
        )


@pytest.mark.asyncio
async def test_nonzero_process_is_redacted(tmp_path: Path) -> None:
    executable = _fake_codex(
        tmp_path,
        "#!/usr/bin/env python3\nimport sys\nsys.stderr.write('private prompt marker')\nraise SystemExit(4)\n",
    )
    with pytest.raises(RuntimeError, match=r"nonzero \(4\): stderr available") as error:
        await ExecBackend(str(executable)).run(
            task="task",
            cwd=tmp_path,
            model="model",
            reasoning="low",
            sandbox="workspace-write",
            approval_policy="never",
            timeout=5,
            max_turns=1,
            environment={"PATH": os.environ["PATH"]},
            config_overrides=(),
        )
    assert "private prompt marker" not in str(error.value)
    silent = _fake_codex(tmp_path, "#!/usr/bin/env python3\nraise SystemExit(5)\n")
    with pytest.raises(RuntimeError, match="no stderr"):
        await ExecBackend(str(silent)).run(
            task="task",
            cwd=tmp_path,
            model="model",
            reasoning="low",
            sandbox="workspace-write",
            approval_policy="never",
            timeout=5,
            max_turns=1,
            environment={"PATH": os.environ["PATH"]},
            config_overrides=(),
        )


@pytest.mark.asyncio
async def test_correction_turn_resumes_and_aggregates_usage(tmp_path: Path) -> None:
    executable = _fake_codex(
        tmp_path,
        r"""#!/usr/bin/env python3
import json, sys
prompt = sys.stdin.read()
thread = "thread-resume"
print(json.dumps({"type":"thread.started","thread_id":thread}))
print(json.dumps({"type":"turn.started","turn_id":"turn"}))
print(json.dumps({"type":"turn.completed","turn_id":"turn","usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1}}))
""",
    )
    validations = iter([False, True])
    result = await ExecBackend(str(executable)).run(
        task="task",
        cwd=tmp_path,
        model="model",
        reasoning="low",
        sandbox="workspace-write",
        approval_policy="never",
        timeout=5,
        max_turns=2,
        environment={"PATH": os.environ["PATH"]},
        config_overrides=(),
        validation_callback=lambda: next(validations),
    )
    assert result.turns == 2
    assert result.correction_turns == 1
    assert not result.first_attempt_completion
    assert result.usage == {
        "cachedInputTokens": 4,
        "inputTokens": 20,
        "noncachedInputTokens": 16,
        "outputTokens": 6,
        "reasoningOutputTokens": 2,
    }


def test_usage_and_resume_helpers(tmp_path: Path) -> None:
    assert (
        _parse_usage(
            {
                "input_tokens": 4,
                "cached_input_tokens": 2,
                "output_tokens": 3,
                "reasoning_output_tokens": 1,
            }
        )["reasoningOutputTokens"]
        == 1
    )
    with pytest.raises(RuntimeError):
        _parse_usage(None)
    with pytest.raises(RuntimeError):
        _parse_usage({"input_tokens": 1})
    with pytest.raises(RuntimeError):
        _parse_usage({"input_tokens": -1, "output_tokens": 1})
    argv = ("codex", "exec", "--json", "--cd", str(tmp_path), "--sandbox", "read-only", "-")
    resumed = _resume_argv(argv, "thread")
    assert resumed[:3] == ("codex", "exec", "resume")
    assert "--cd" not in resumed and "--sandbox" not in resumed
    with pytest.raises(RuntimeError):
        _resume_argv(argv, "")
    with_ephemeral = _resume_argv(("codex", "exec", "--ephemeral", "-"), "thread")
    assert "--ephemeral" not in with_ephemeral


def test_exec_event_normalization_never_persists_content() -> None:
    reasoning = normalize_exec_event(
        {"type": "item.completed", "item": {"type": "reasoning", "text": "secret"}}
    )
    assert reasoning and "secret" not in json.dumps(reasoning.to_dict())
    message = normalize_exec_event(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "secret"}}
    )
    assert message and "secret" not in json.dumps(message.to_dict())
    warning = normalize_exec_event(
        {"type": "item.completed", "item": {"type": "error", "status": "warning"}}
    )
    assert warning and warning.kind == "warning"
    fatal = normalize_exec_event({"type": "error", "message": "secret"})
    assert fatal and fatal.data == {"severity": "fatal", "code": ""}
    unknown = normalize_exec_event({"type": "future.event", "private": "secret"})
    assert unknown and unknown.data == {"keys": ["private", "type"]}
    reroute = normalize_exec_event(
        {"type": "model.rerouted", "from_model": "requested", "to_model": "fallback"}
    )
    assert reroute and reroute.kind == "error" and reroute.data["to_model"] == "fallback"


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ({"type": "thread.started", "thread_id": "thread"}, "thread_started"),
        ({"type": "turn.started", "turn_id": "turn"}, "turn_started"),
        ({"type": "turn.completed", "turn_id": "turn"}, "turn_completed"),
        ({"type": "turn.failed", "turn_id": "turn"}, "turn_failed"),
        (
            {
                "type": "item.completed",
                "item": {
                    "id": "file",
                    "type": "file_change",
                    "status": "completed",
                    "changes": [{"path": "src/a.py"}],
                },
            },
            "file_change",
        ),
        (
            {
                "type": "item.started",
                "item": {"id": "mcp", "type": "mcp_tool_call", "server": "s", "tool": "t"},
            },
            "mcp_tool_call",
        ),
        (
            {
                "type": "item.completed",
                "item": {"id": "mcp", "type": "mcp_tool_call", "result": {"ok": True}},
            },
            "mcp_result",
        ),
        ({"type": "item.completed", "item": {"type": "future"}}, "opaque"),
    ],
)
def test_exec_event_shapes(raw: dict[str, object], kind: str) -> None:
    result = normalize_exec_event(raw)
    assert result and result.kind == kind
