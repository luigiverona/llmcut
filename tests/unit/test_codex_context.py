from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from llmcut.integrations.codex.context import ContextStrategy, plan_codex_context
from llmcut.integrations.codex.executor import _write_run_state
from llmcut.mcp.server import (
    RepositoryContext,
    create_mcp_server,
    load_run_state,
    serve,
    tool_schema_bytes,
)


def _repo(tmp_path: Path, count: int = 20) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "app").mkdir()
    (repo / "tests").mkdir()
    (repo / "app" / "callback.py").write_text(
        "from app.config import TIMEOUT\n\ndef callback():\n    return TIMEOUT\n"
    )
    (repo / "app" / "config.py").write_text("TIMEOUT = 30\n")
    (repo / "tests" / "test_callback.py").write_text(
        "from app.callback import callback\n\ndef test_callback():\n    assert callback()\n"
    )
    (repo / "app" / "unique.py").write_text("def singular_widget():\n    return True\n")
    for number in range(max(0, count - 3)):
        (repo / "app" / f"feature_{number}.py").write_text(
            f"def feature_{number}():\n    return {number}\n"
        )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    return repo


def test_explicit_path_and_adaptive_off(tmp_path: Path) -> None:
    repo = _repo(tmp_path, 5)
    plan = plan_codex_context(repo, "Fix app/callback.py", "adaptive")
    assert plan.selected_strategy is ContextStrategy.OFF
    assert plan.selected_files[0].path == "app/callback.py"
    assert plan.orientation_text == ""


def test_guided_plan_is_deterministic_bounded_and_metadata_only(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = plan_codex_context(repo, "Diagnose callback timeout", "guided")
    second = plan_codex_context(repo, "Diagnose callback timeout", "guided")
    assert first.to_dict() == second.to_dict()
    assert first.orientation_token_estimate <= 200
    assert "return TIMEOUT" not in first.orientation_text
    assert not any(Path(item.path).is_absolute() for item in first.selected_files)
    assert "app/config.py" in first.direct_dependencies
    adaptive_off = plan_codex_context(repo, "Fix app/callback.py", "adaptive")
    assert adaptive_off.selected_strategy is ContextStrategy.OFF
    adaptive_orientation = plan_codex_context(repo, "Fix singular widget", "adaptive")
    assert adaptive_orientation.selected_strategy is ContextStrategy.ORIENTATION
    adaptive_guided = plan_codex_context(repo, "Diagnose an unexplained regression", "adaptive")
    assert adaptive_guided.selected_strategy is ContextStrategy.GUIDED


@pytest.mark.asyncio
async def test_strategy_surfaces_and_schema_reduction(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    plan = plan_codex_context(repo, "Diagnose callback timeout", "guided")
    guided = create_mcp_server(repo, "guided", plan=plan)
    legacy = create_mcp_server(repo, "legacy-passive")
    assert [item.name for item in await guided.list_tools()] == ["llmcut_context"]
    assert len(await legacy.list_tools()) == 8
    assert tool_schema_bytes("guided") <= tool_schema_bytes("legacy-passive") * 0.3
    with pytest.raises(ValueError, match="cannot start"):
        create_mcp_server(repo, "off")
    without_plan = create_mcp_server(repo, "guided")
    with pytest.raises(Exception, match="unavailable"):
        await without_plan.call_tool("llmcut_context", {"operation": "plan"})
    assert tool_schema_bytes("off") == 0
    with pytest.raises(ToolError):
        await legacy.call_tool(
            "llmcut_log_search", {"path": "app/callback.py", "pattern": "!unsafe"}
        )


@pytest.mark.asyncio
async def test_compact_exact_retrieval_and_stale_digest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "run.log").write_text("PASS setup\nFAILED callback\n")
    (repo / "checkpoint.md").write_text("# checkpoint\nretry remains unresolved\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "evidence"], cwd=repo, check=True)
    plan = plan_codex_context(repo, "Diagnose callback timeout", "guided")
    server = create_mcp_server(repo, "guided", plan=plan)
    assert await server.call_tool("llmcut_context", {"operation": "plan"})
    assert await server.call_tool("llmcut_context", {"operation": "file", "path": "app/config.py"})
    result = await server.call_tool(
        "llmcut_context", {"operation": "range", "path": "app/callback.py", "start": 1, "end": 2}
    )
    assert result
    assert await server.call_tool(
        "llmcut_context",
        {"operation": "symbol", "path": "app/callback.py", "symbol": "callback"},
    )
    for operation in ("dependencies", "tests"):
        assert await server.call_tool(
            "llmcut_context", {"operation": operation, "path": "app/callback.py"}
        )
    assert await server.call_tool(
        "llmcut_context",
        {"operation": "log_search", "path": "run.log", "pattern": "FAILED"},
    )
    assert await server.call_tool(
        "llmcut_context", {"operation": "checkpoint", "path": "checkpoint.md"}
    )
    invalid = (
        {"operation": "invalid"},
        {"operation": "file"},
        {"operation": "range", "path": "app/callback.py", "start": 0, "end": 2},
        {"operation": "symbol", "path": "app/callback.py", "symbol": "missing"},
        {"operation": "log_search", "path": "run.log", "pattern": "!unsafe"},
        {"operation": "checkpoint", "path": "app/config.py"},
    )
    for arguments in invalid:
        with pytest.raises(ToolError):
            await server.call_tool("llmcut_context", arguments)
    wrong = create_mcp_server(
        repo,
        "guided",
        plan=replace(plan, evidence_digests=plan.evidence_digests | {"app/config.py": "bad"}),
    )
    with pytest.raises(Exception, match="does not match"):
        await wrong.call_tool("llmcut_context", {"operation": "file", "path": "app/config.py"})
    (repo / "app" / "callback.py").write_text("changed\n")
    with pytest.raises(Exception, match="stale"):
        await server.call_tool("llmcut_context", {"operation": "file", "path": "app/callback.py"})


def test_secure_digest_bound_run_state(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "source")
    state_parent = tmp_path / "private"
    state_parent.mkdir(mode=0o700)
    plan = plan_codex_context(repo, "Diagnose callback timeout", "guided")
    path = _write_run_state(state_parent, repo, plan)
    assert os.stat(path).st_mode & 0o077 == 0
    strategy, loaded = load_run_state(path, repo)
    assert strategy is ContextStrategy.GUIDED
    assert loaded and loaded.task_digest == plan.task_digest
    value = json.loads(path.read_text())
    value["digest"] = "0" * 64
    path.write_text(json.dumps(value))
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="digest"):
        load_run_state(path, repo)
    os.chmod(path, 0o644)
    with pytest.raises(ValueError, match="permissions"):
        load_run_state(path, repo)


def test_run_state_bounds_boundary_and_server_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path / "source")
    other = _repo(tmp_path / "other")
    state_parent = tmp_path / "private"
    state_parent.mkdir(mode=0o700)
    plan = plan_codex_context(repo, "Diagnose callback timeout", "guided")
    path = _write_run_state(state_parent, repo, plan)
    with pytest.raises(ValueError, match="boundary"):
        load_run_state(path, other)
    oversized = state_parent / "oversized.json"
    oversized.write_bytes(b"x" * (512 * 1024 + 1))
    os.chmod(oversized, 0o600)
    with pytest.raises(ValueError, match="exceeds"):
        load_run_state(oversized, repo)
    called: list[str] = []
    monkeypatch.setattr(
        "mcp.server.fastmcp.FastMCP.run", lambda self, transport: called.append(transport)
    )
    serve(repo, "guided", path)
    assert called == ["stdio"]
    with pytest.raises(ValueError, match="mismatch"):
        serve(repo, "legacy-passive", path)
    context = RepositoryContext(repo)
    with pytest.raises(ValueError, match="bound"):
        context.bounded("x" * (128 * 1024 + 1))


def test_invalid_strategy_and_orientation_budget(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".env").write_text("TOKEN=do-not-index\n")
    (repo / "escape.py").symlink_to(Path("/etc/passwd"))
    with pytest.raises(ValueError, match="invalid context strategy"):
        plan_codex_context(repo, "task", "invalid")
    plan = plan_codex_context(repo, "Diagnose callback timeout", "guided", orientation_budget=1)
    assert plan.selected_strategy is ContextStrategy.OFF
    assert not plan.orientation_text
    assert ".env" not in plan.evidence_digests and "escape.py" not in plan.evidence_digests
    with pytest.raises(ValueError, match="task must"):
        plan_codex_context(repo, "", "adaptive")
    with pytest.raises(ValueError, match="budget"):
        plan_codex_context(repo, "task", "adaptive", retrieval_budget=0)
