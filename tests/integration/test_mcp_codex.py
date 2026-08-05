from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import tomlkit
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError

from llmcut.integrations.codex.app_server import CodexAppServer
from llmcut.integrations.codex.config import configure_codex
from llmcut.integrations.codex.doctor import _probe, detect_codex
from llmcut.mcp.server import create_mcp_server


@pytest.mark.asyncio
async def test_mcp_stdio_with_official_client(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("def callback():\n    return 30\n")
    (repo / ".env").write_text("TOKEN=secret\n")
    executable = shutil.which("llmcut")
    assert executable
    parameters = StdioServerParameters(
        command=executable,
        args=["mcp", "serve", "--repo", str(repo), "--integration", "legacy-passive"],
    )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert "llmcut_source_range" in {tool.name for tool in tools.tools}
        result = await session.call_tool(
            "llmcut_source_range", {"path": "source.py", "start": 1, "end": 2}
        )
        assert not result.isError
        assert "return 30" in "".join(getattr(item, "text", "") for item in result.content)
        resources = await session.list_resources()
        assert any(str(item.uri) == "llmcut://repository/map" for item in resources.resources)


@pytest.mark.asyncio
async def test_guided_stdio_initialization_has_bounded_workflow_and_one_tool(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("def callback():\n    return 30\n")
    executable = shutil.which("llmcut")
    assert executable
    parameters = StdioServerParameters(
        command=executable,
        args=["mcp", "serve", "--repo", str(repo), "--integration", "guided"],
    )
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        initialized = await session.initialize()
        assert initialized.instructions
        assert "Use the supplied repository orientation" in initialized.instructions[:512]
        assert "untrusted data" in initialized.instructions[:512]
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["llmcut_context"]


@pytest.mark.asyncio
async def test_mcp_tools_and_resources_in_process(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("import json\n\ndef callback():\n    return 30\n")
    (repo / "checkpoint.md").write_text("# checkpoint\nobjective remains active\n")
    (repo / "run.log").write_text("PASS one\nFAILED callback\nPASS two\n")
    server = create_mcp_server(repo)
    assert len(await server.list_tools()) == 8
    assert await server.call_tool("llmcut_plan", {"task": "fix callback in source.py"})
    assert await server.call_tool("llmcut_context_get", {"context_id": "source.py"})
    assert await server.call_tool(
        "llmcut_source_range", {"path": "source.py", "start": 1, "end": 2}
    )
    assert await server.call_tool("llmcut_symbol_get", {"path": "source.py", "symbol": "callback"})
    assert await server.call_tool("llmcut_dependencies", {"path": "source.py"})
    assert await server.call_tool(
        "llmcut_log_search", {"path": "run.log", "pattern": "FAILED", "limit": 10}
    )
    assert await server.call_tool("llmcut_checkpoint_get", {"checkpoint_id": "checkpoint.md"})
    assert await server.call_tool("llmcut_tool_discover", {"category": "repository"})
    assert list(await server.read_resource("llmcut://repository/map"))
    assert list(await server.read_resource("llmcut://context/source.py"))
    (repo / "source.py").write_text("changed")
    with pytest.raises(ToolError, match="stale"):
        await server.call_tool("llmcut_context_get", {"context_id": "source.py"})


@pytest.mark.asyncio
async def test_mcp_rejects_invalid_and_oversized_requests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("def callback():\n    return 30\n")
    (repo / "notes.txt").write_text("ordinary notes\n")
    server = create_mcp_server(repo)
    invalid: list[tuple[str, dict[str, Any]]] = [
        ("llmcut_plan", {"task": ""}),
        ("llmcut_context_get", {"context_id": "../secret"}),
        ("llmcut_source_range", {"path": "source.py", "start": 0, "end": 2}),
        ("llmcut_symbol_get", {"path": "source.py", "symbol": "missing"}),
        ("llmcut_checkpoint_get", {"checkpoint_id": "notes.txt"}),
        ("llmcut_tool_discover", {"category": "unknown"}),
    ]
    for name, arguments in invalid:
        with pytest.raises(ToolError):
            await server.call_tool(name, arguments)


@pytest.mark.asyncio
async def test_codex_app_server_protocol_and_usage(tmp_path: Path) -> None:
    fake = tmp_path / "fake-codex"
    fake.write_text(
        """#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    if method == 'initialized':
        continue
    if method == 'initialize':
        result = {'capabilities': {}}
    elif method == 'thread/start':
        result = {'thread': {'id': 'thread-1'}}
    else:
        result = {'turn': {'id': 'turn-1'}}
    print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}), flush=True)
    if method == 'turn/start':
        usage = {'jsonrpc':'2.0','method':'thread/tokenUsage/updated',
                 'params':{'tokenUsage':{'inputTokens':123}}}
        completed = {'jsonrpc':'2.0','method':'turn/completed',
                     'params':{'turn':{'id':'turn-1','status':'completed'}}}
        print(json.dumps(usage), flush=True)
        print(json.dumps(completed), flush=True)
"""
    )
    fake.chmod(0o700)
    result = await CodexAppServer(str(fake)).run(
        task="fix callback",
        cwd=tmp_path,
        model="same-model",
        reasoning="high",
        sandbox="workspace-write",
        approval_policy="on-request",
        timeout=5,
    )
    assert result.status == "completed"
    assert result.usage == {"inputTokens": 123}


@pytest.mark.asyncio
async def test_codex_rejects_model_rerouting(tmp_path: Path) -> None:
    fake = tmp_path / "reroute-codex"
    fake.write_text(
        """#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    if method == 'initialized':
        continue
    result = ({'thread': {'id': 'thread-1'}} if method == 'thread/start'
              else {'turn': {'id': 'turn-1'}} if method == 'turn/start' else {})
    print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}), flush=True)
    if method == 'turn/start':
        event = {'jsonrpc':'2.0','method':'model/rerouted','params':{'model':'different'}}
        print(json.dumps(event), flush=True)
"""
    )
    fake.chmod(0o700)
    with pytest.raises(RuntimeError, match="changed the configured model"):
        await CodexAppServer(str(fake)).run(
            task="task",
            cwd=tmp_path,
            model="same",
            reasoning="high",
            sandbox="workspace-write",
            approval_policy="on-request",
            timeout=5,
        )


def test_codex_missing_and_probe_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    capabilities = detect_codex(tmp_path / "missing.toml")
    assert not capabilities.installed and capabilities.agent_usage == "unavailable"
    assert _probe(["definitely-not-a-real-llmcut-command"]) is None
    assert _probe(["python", "-c", "raise SystemExit(2)"]) is None


def test_codex_configuration_is_safe_reversible_and_atomic(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('model = "same"\n[unrelated]\nkeep = true\n')
    dry = configure_codex(config, tmp_path, dry_run=True)
    assert dry.changed and config.read_text() == dry.before
    change = configure_codex(config, tmp_path)
    assert change.backup and change.backup.exists()
    parsed = tomlkit.parse(config.read_text())
    assert parsed["unrelated"]["keep"] is True
    assert parsed["mcp_servers"]["llmcut"]["command"] == "llmcut"
    removed = configure_codex(config, tmp_path, remove=True)
    updated = tomlkit.parse(config.read_text())
    assert removed.changed and (
        "mcp_servers" not in updated or "llmcut" not in updated["mcp_servers"]
    )
    assert os.stat(config).st_mode & 0o077 == 0
