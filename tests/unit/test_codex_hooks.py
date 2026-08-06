from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmcut.cli import app
from llmcut.integrations.codex.executor import (
    _cleanup_hook_artifacts,
    _hook_metrics,
    _hook_overrides,
    _hook_replacement_verified,
    _write_evaluation_hook,
)
from llmcut.integrations.codex.hooks.classify import CommandClass, classify_command
from llmcut.integrations.codex.hooks.compact import compact_bash_result
from llmcut.integrations.codex.hooks.config import (
    HookConfig,
    hook_command,
    install_hooks,
    proposed_document,
    remove_hooks,
)
from llmcut.integrations.codex.hooks.handler import append_metrics, handle_hook
from llmcut.integrations.codex.hooks.protocol import parse_bash_response, parse_hook_input
from llmcut.integrations.codex.hooks.state import HookEvidenceStore, exact_lines, render_exact


def _event(repo: Path, output: str, *, code: int = 1, command: str = "pytest -q") -> bytes:
    return json.dumps(
        {
            "session_id": "session",
            "turn_id": "turn",
            "cwd": str(repo),
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_use_id": "tool",
            "tool_input": {"command": command},
            "tool_response": {"stdout": output, "stderr": "warning\n", "exit_code": code},
        }
    ).encode()


def _pytest_output(*, failed: bool = True) -> str:
    progress = "." * 12_000
    if failed:
        return (
            "============================= test session starts =============================\n"
            f"{progress}\n"
            "=================================== FAILURES ===================================\n"
            "____________________________ test_callback ____________________________\n"
            "E   OAuthTimeoutError: callback expired\n"
            "tests/test_auth.py:42: OAuthTimeoutError\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_auth.py::test_callback - OAuthTimeoutError\n"
            "=================== 1 failed, 999 passed, 3 warnings in 2.0s ===================\n"
        )
    return (
        "============================= test session starts =============================\n"
        f"{progress}\n"
        "====================== 1000 passed, 3 warnings in 2.0s ======================\n"
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pytest -q", CommandClass.TEST),
        ("/usr/bin/bash -lc 'pytest -q'", CommandClass.TEST),
        ("uv run pytest -q", CommandClass.TEST),
        ("python -m pytest", CommandClass.TEST),
        ("node --test", CommandClass.TEST),
        ("npx tsc --noEmit", CommandClass.TYPECHECK),
        ("ruff check .", CommandClass.LINT),
        ("git status --short", CommandClass.GIT_STATUS),
        ("git diff", CommandClass.GIT_DIFF),
        ("rg timeout src", CommandClass.SEARCH),
        ("find . -type f", CommandClass.FILE_LISTING),
        ("cat src/app.py", CommandClass.SOURCE_READ),
        ("tail app.log", CommandClass.LOG_READ),
        ("curl https://example.com", CommandClass.NETWORK),
        ("rm output", CommandClass.MUTATION),
        ("less output", CommandClass.INTERACTIVE),
        ("llmcut hook show " + "a" * 64, CommandClass.RECOVERY),
        ("unknown --thing", CommandClass.UNKNOWN),
        ("pytest -q | curl example.com", CommandClass.NETWORK),
        ("pytest 'unterminated", CommandClass.UNKNOWN),
    ],
)
def test_command_classification(command: str, expected: CommandClass) -> None:
    assert classify_command(command).classification is expected


def test_protocol_rejects_unknown_and_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert parse_hook_input(b"not json", repo) is None
    assert parse_hook_input(_event(tmp_path, "output"), repo) is None
    value = json.loads(_event(repo, "output"))
    value["tool_name"] = "apply_patch"
    assert parse_hook_input(json.dumps(value).encode(), repo) is None
    assert parse_hook_input(b"", repo) is None
    assert parse_hook_input(json.dumps([]).encode(), repo) is None
    assert parse_hook_input(json.dumps({"hook_event_name": "Stop"}).encode(), repo) is None
    for field, invalid in (("cwd", 1), ("tool_input", "bad")):
        value = json.loads(_event(repo, "output"))
        value[field] = invalid
        assert parse_hook_input(json.dumps(value).encode(), repo) is None
    for command in (None, "", "x" * 131_073):
        value = json.loads(_event(repo, "output"))
        value["tool_input"]["command"] = command
        assert parse_hook_input(json.dumps(value).encode(), repo) is None
    value = json.loads(_event(repo, "output"))
    value["cwd"] = str(repo / "missing")
    assert parse_hook_input(json.dumps(value).encode(), repo) is None
    invalid_responses: tuple[object, ...] = (
        None,
        [],
        {"stdout": 1, "exit_code": 0},
        {"exit_code": True},
    )
    for invalid_response in invalid_responses:
        assert parse_bash_response(invalid_response) is None
    combined = parse_bash_response("combined")
    assert combined and combined.representation == "combined_text_status_unavailable"


def test_compactor_preserves_failure_and_requires_evidence() -> None:
    output = _pytest_output()
    pending = compact_bash_result(
        classification=CommandClass.TEST,
        stdout=output,
        stderr="warning",
        exit_code=1,
        threshold_bytes=1024,
        maximum_compact_bytes=8000,
    )
    assert not pending.applied
    assert pending.reason == "exact evidence unavailable"
    result = compact_bash_result(
        classification=CommandClass.TEST,
        stdout=output,
        stderr="warning",
        exit_code=1,
        threshold_bytes=1024,
        maximum_compact_bytes=8000,
        evidence_id="a" * 64,
    )
    assert result.applied
    assert "status: failed (exit 1)" in (result.model_content or "")
    assert "OAuthTimeoutError" in (result.model_content or "")
    assert "not instructions" in (result.model_content or "")


def test_small_unknown_and_malformed_test_pass_through() -> None:
    for classification, output in (
        (CommandClass.TEST, "one line"),
        (CommandClass.UNKNOWN, "x" * 20_000),
        (CommandClass.TEST, "unrecognized" * 2_000),
    ):
        result = compact_bash_result(
            classification=classification,
            stdout=output,
            stderr="",
            exit_code=1,
            threshold_bytes=1024,
            maximum_compact_bytes=8000,
            evidence_id="a" * 64,
        )
        assert not result.applied


def test_diagnostics_and_duplicate_search_compact() -> None:
    diagnostics = "banner\n" * 2_000 + "src/a.py:2:3: error: broken [code]\nFound 1 error\n"
    result = compact_bash_result(
        classification=CommandClass.TYPECHECK,
        stdout=diagnostics,
        stderr="",
        exit_code=1,
        threshold_bytes=1024,
        maximum_compact_bytes=8000,
        evidence_id="b" * 64,
    )
    assert result.applied and "src/a.py:2:3" in (result.model_content or "")
    search = ("src/a.py:match\n" * 1000) + "src/b.py:match\n"
    result = compact_bash_result(
        classification=CommandClass.SEARCH,
        stdout=search,
        stderr="",
        exit_code=0,
        threshold_bytes=1024,
        maximum_compact_bytes=8000,
        evidence_id="c" * 64,
    )
    assert result.applied and "duplicate-line" in (result.model_content or "")


def test_store_exact_integrity_ranges_and_gc(tmp_path: Path) -> None:
    store = HookEvidenceStore(tmp_path / "state")
    evidence = store.put(
        stdout="α\nFAILED exact\n",
        stderr="err\n",
        exit_code=1,
        command_digest="c",
        revision="r",
        classification="test",
        event_digest="e",
        session_id="s",
        parser="pytest",
        parser_version="1",
    )
    assert store.get(evidence.evidence_id).stdout == "α\nFAILED exact\n"
    assert "FAILED exact" in render_exact(evidence)
    assert any("FAILED" in line for line in exact_lines(evidence))
    mode = stat.S_IMODE((tmp_path / "state").stat().st_mode)
    assert mode == 0o700
    with pytest.raises(ValueError, match="invalid hook evidence id"):
        store.get("../escape")
    target = tmp_path / "state" / evidence.evidence_id / "stdout"
    target.write_text("tampered")
    with pytest.raises(ValueError, match="digest"):
        store.get(evidence.evidence_id)
    old = time.time() - 100
    os.utime(target.parent, (old, old))
    result = store.collect(maximum_age_seconds=1, maximum_total_bytes=0, dry_run=True)
    assert evidence.evidence_id in result["removed"]


def test_handler_compacts_and_fails_open(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    config = HookConfig(repo, tmp_path / "state", threshold_bytes=1024, maximum_compact_bytes=8000)
    response, metrics = handle_hook(_event(repo, _pytest_output()), config)
    assert response and response["decision"] == "block"
    assert metrics["applied"] is True
    assert metrics["evidence_created"] is True
    response, metrics = handle_hook(b"bad", config)
    assert response is None and metrics["fallback_reason"]
    response, metrics = handle_hook(
        _event(repo, _pytest_output(), command="llmcut hook show " + "a" * 64), config
    )
    assert response is None and "recursively" in str(metrics["fallback_reason"])


def test_configuration_merge_remove_and_permissions(tmp_path: Path) -> None:
    target = tmp_path / "codex" / "hooks.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"hooks": {"Stop": [{"hooks": []}]}}))
    preview = install_hooks(target, dry_run=True)
    assert preview["changed"] is True
    install_hooks(target)
    installed = json.loads(target.read_text())
    assert installed["hooks"]["Stop"]
    assert len(installed["hooks"]["PostToolUse"]) == 1
    install_hooks(target)
    assert len(json.loads(target.read_text())["hooks"]["PostToolUse"]) == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    remove_hooks(target)
    assert "Stop" in json.loads(target.read_text())["hooks"]
    assert "PostToolUse" not in json.loads(target.read_text())["hooks"]
    assert proposed_document()["hooks"]["PostToolUse"][0]["matcher"] == "^Bash$"


def test_real_hook_subprocess_model_visible_replacement(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = dict(os.environ)
    environment.update(
        {"LLMCUT_HOOK_REPO": str(repo), "LLMCUT_HOOK_STATE": str(tmp_path / "state")}
    )
    result = subprocess.run(
        [sys.executable, "-m", "llmcut", "hook", "handle"],
        input=_event(repo, _pytest_output()),
        capture_output=True,
        env=environment,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    replacement = json.loads(result.stdout)
    assert replacement["decision"] == "block"
    assert "OAuthTimeoutError" in replacement["reason"]


def test_recovery_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCUT_HOOK_STATE", str(tmp_path / "state"))
    evidence = HookEvidenceStore(tmp_path / "state").put(
        stdout="one\nFAILED two\nthree\n",
        stderr="err\n",
        exit_code=1,
        command_digest="c",
        revision="r",
        classification="test",
        event_digest="e",
        session_id="s",
        parser="pytest",
        parser_version="1",
    )
    runner = CliRunner()
    assert "FAILED two" in runner.invoke(app, ["hook", "show", evidence.evidence_id]).stdout
    assert (
        runner.invoke(app, ["hook", "search", evidence.evidence_id, "--pattern", "FAILED"]).stdout
        == "FAILED two\n"
    )
    assert (
        runner.invoke(
            app, ["hook", "range", evidence.evidence_id, "--start", "1", "--end", "2"]
        ).exit_code
        == 0
    )
    assert (
        json.loads(runner.invoke(app, ["hook", "info", evidence.evidence_id]).stdout)["exit_code"]
        == 1
    )


def test_hook_management_cli_and_gc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("LLMCUT_HOOK_STATE", str(tmp_path / "state"))
    runner = CliRunner()
    config = runner.invoke(app, ["agent", "codex", "hooks", "config"])
    assert config.exit_code == 0 and json.loads(config.stdout)["mutates_files"] is False
    preview = runner.invoke(app, ["agent", "codex", "hooks", "install", "--dry-run"])
    assert preview.exit_code == 0
    assert json.loads(preview.stdout)["persistent_trust_installed"] is False
    assert runner.invoke(app, ["agent", "codex", "hooks", "install"]).exit_code == 0
    assert (tmp_path / "codex" / "hooks.json").is_file()
    removed = runner.invoke(app, ["agent", "codex", "hooks", "remove"])
    assert removed.exit_code == 0 and json.loads(removed.stdout)["changed"] is True
    missing = runner.invoke(app, ["agent", "codex", "hooks", "remove"])
    assert missing.exit_code == 0 and json.loads(missing.stdout)["changed"] is False
    gc = runner.invoke(app, ["hook", "gc", "--dry-run"])
    assert gc.exit_code == 0 and json.loads(gc.stdout)["dry_run"] is True
    assert runner.invoke(app, ["hook", "gc", "--maximum-age", "-1"]).exit_code != 0
    doctor = runner.invoke(app, ["agent", "codex", "hooks", "doctor"])
    assert doctor.exit_code == 0
    report = json.loads(doctor.stdout)
    assert report["exclusive_model_replacement_verified"] is True
    assert report["direct_exec_probe_ready"] is True
    assert report["evaluation_ready"] is False


def test_protocol_response_variants_and_config_bounds(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    value = json.loads(_event(repo, "output"))
    value["tool_response"] = "combined"
    parsed = parse_hook_input(json.dumps(value).encode(), repo)
    assert parsed and parsed.response.representation == "combined_text_status_unavailable"
    for response in (
        None,
        {"stdout": 1, "stderr": "", "exit_code": 0},
        {"stdout": "", "stderr": "", "exit_code": True},
    ):
        value["tool_response"] = response
        assert parse_hook_input(json.dumps(value).encode(), repo) is None
    value["tool_response"] = {"stdout": "", "stderr": "", "exitCode": 2}
    parsed = parse_hook_input(json.dumps(value).encode(), repo)
    assert parsed and parsed.response.exit_code == 2
    with pytest.raises(ValueError, match="repository"):
        HookConfig(repo, repo / "state").validate()
    with pytest.raises(ValueError, match="threshold"):
        HookConfig(repo, tmp_path / "state", threshold_bytes=1).validate()
    with pytest.raises(ValueError, match="compact"):
        HookConfig(repo, tmp_path / "state", maximum_compact_bytes=1).validate()


def test_configuration_rejects_invalid_shapes_and_dry_remove(tmp_path: Path) -> None:
    target = tmp_path / "hooks.json"
    target.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        install_hooks(target)
    target.write_text(json.dumps({"hooks": []}))
    with pytest.raises(ValueError, match="hooks field"):
        install_hooks(target)
    target.write_text(json.dumps({"hooks": {"PostToolUse": {}}}))
    with pytest.raises(ValueError, match="array"):
        install_hooks(target)
    target.write_text(json.dumps({"hooks": {"PostToolUse": [{"matcher": "other"}]}}))
    install_hooks(target)
    assert remove_hooks(target, dry_run=True)["changed"] is True


def test_additional_classifier_and_compactor_branches() -> None:
    assert classify_command("FOO=1 pytest -q").classification is CommandClass.TEST
    assert classify_command("FOO=1").classification is CommandClass.UNKNOWN
    assert classify_command("pytest -q && ruff check .").classification is CommandClass.UNKNOWN
    assert classify_command("npm test").classification is CommandClass.TEST
    assert classify_command("npm install").classification is CommandClass.PACKAGE_MANAGER
    assert classify_command("make build").classification is CommandClass.BUILD
    successful = compact_bash_result(
        classification=CommandClass.TEST,
        stdout=_pytest_output(failed=False),
        stderr="",
        exit_code=0,
        threshold_bytes=1024,
        maximum_compact_bytes=8000,
        evidence_id="sha256:" + "d" * 64,
    )
    assert successful.applied and "status: succeeded" in (successful.model_content or "")
    ansi = _pytest_output().replace("FAILURES", "\x1b[31mFAILURES\x1b[0m")
    result = compact_bash_result(
        classification=CommandClass.TEST,
        stdout=ansi,
        stderr="",
        exit_code=1,
        threshold_bytes=1024,
        maximum_compact_bytes=8000,
        evidence_id="sha256:" + "e" * 64,
    )
    assert result.applied


def test_compactor_conservative_projection_fallbacks() -> None:
    evidence = "sha256:" + "f" * 64
    cases = (
        (
            CommandClass.TEST,
            "test session starts\n"
            + "." * 9000
            + "\n================ 1 failed in 1s ================\n",
            1,
            8000,
        ),
        (
            CommandClass.TEST,
            "================ FAILURES ================\nE failure\n"
            + "z" * 9000
            + "\n================ 1 failed in 1s ================\n",
            1,
            8000,
        ),
        (CommandClass.TYPECHECK, "no diagnostics\n" * 1000, 1, 8000),
        (CommandClass.TYPECHECK, ("a.py:1: error: bad\n" * 1000), 1, 3000),
        (CommandClass.SEARCH, "\n".join(f"unique{index}" for index in range(2000)), 0, 8000),
        (CommandClass.SEARCH, ("duplicate\n" * 1000) + ("x" * 3000), 0, 3000),
    )
    for classification, output, code, limit in cases:
        result = compact_bash_result(
            classification=classification,
            stdout=output,
            stderr="",
            exit_code=code,
            threshold_bytes=1024,
            maximum_compact_bytes=limit,
            evidence_id=evidence,
        )
        assert not result.applied
    too_small_to_benefit = compact_bash_result(
        classification=CommandClass.SEARCH,
        stdout="a\na\n",
        stderr="",
        exit_code=0,
        threshold_bytes=1,
        maximum_compact_bytes=8000,
        evidence_id=evidence,
    )
    assert too_small_to_benefit.reason == "projection is not beneficial"


def test_handler_small_result_records_pass_through(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    response, metrics = handle_hook(_event(repo, "small", code=0), HookConfig(repo, tmp_path / "s"))
    assert response is None
    assert metrics["reason"] == "below threshold"


def test_handler_cli_metrics_and_store_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = tmp_path / "state"
    metrics = tmp_path / "metrics" / "events.jsonl"
    monkeypatch.setenv("LLMCUT_HOOK_REPO", str(repo))
    monkeypatch.setenv("LLMCUT_HOOK_STATE", str(state))
    monkeypatch.setenv("LLMCUT_HOOK_METRICS", str(metrics))
    result = CliRunner().invoke(
        app, ["hook", "handle"], input=_event(repo, _pytest_output()).decode()
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["decision"] == "block"
    recorded = json.loads(metrics.read_text().splitlines()[0])
    assert recorded["applied"] is True
    append_metrics(metrics, {"event_supported": True})
    assert len(metrics.read_text().splitlines()) == 2
    store = HookEvidenceStore(state)
    evidence = store.put(
        stdout="same",
        stderr="",
        exit_code=0,
        command_digest="c",
        revision="r",
        classification="test",
        event_digest="e",
        session_id="s",
        parser="pytest",
        parser_version="1",
    )
    assert (
        store.put(
            stdout="same",
            stderr="",
            exit_code=0,
            command_digest="c",
            revision="r",
            classification="test",
            event_digest="e",
            session_id="s",
            parser="pytest",
            parser_version="1",
        ).evidence_id
        == evidence.evidence_id
    )
    (state / "junk").write_text("ignored")
    active = store.collect(
        maximum_age_seconds=0,
        maximum_total_bytes=0,
        active_ids={evidence.evidence_id},
    )
    assert evidence.evidence_id not in active["removed"]
    collected = store.collect(maximum_age_seconds=0, maximum_total_bytes=0)
    assert evidence.evidence_id in collected["removed"]


def test_invalid_store_metadata_and_noop_removal(tmp_path: Path) -> None:
    store = HookEvidenceStore(tmp_path / "state")
    evidence = store.put(
        stdout="value",
        stderr="",
        exit_code=0,
        command_digest="c",
        revision="r",
        classification="test",
        event_digest="e",
        session_id="s",
        parser="pytest",
        parser_version="1",
    )
    metadata = tmp_path / "state" / evidence.evidence_id / "metadata.json"
    metadata.write_text(json.dumps({"exit_code": "bad"}))
    with pytest.raises(ValueError, match="metadata"):
        store.get(evidence.evidence_id)
    target = tmp_path / "hooks.json"
    target.write_text(json.dumps({"hooks": {"Stop": []}}))
    assert remove_hooks(target)["changed"] is False


def test_hook_command_fallback_and_handler_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(sys, "argv", ["python"])
    assert "-m llmcut hook handle" in hook_command()
    repo = tmp_path / "repo"
    repo.mkdir()
    response, metrics = handle_hook(
        _event(repo, _pytest_output()), HookConfig(repo, repo / "state")
    )
    assert response is None and "ValueError" in str(metrics["fallback_reason"])


def test_evaluation_hook_configuration_metrics_and_cleanup(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    config = _write_evaluation_hook(worktree)
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert json.loads(config.read_text())["hooks"]["PostToolUse"][0]["matcher"] == "^Bash$"
    assert _hook_overrides() == ("features.hooks=true",)
    assert _hook_replacement_verified("fake_codex.py", "fake")
    assert not _hook_replacement_verified("codex", "sdk")
    with pytest.raises(RuntimeError, match="without an existing"):
        _write_evaluation_hook(worktree)

    state = tmp_path / ".hook-evidence"
    state.mkdir(mode=0o700)
    (state / "entry").write_text("metadata")
    metrics = tmp_path / ".hook-metrics.jsonl"
    metrics.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "event_supported": True,
                        "applied": True,
                        "classification": "test",
                        "original_bytes": 12000,
                        "compact_bytes": 1000,
                        "original_tokens_estimate": 4000,
                        "compact_tokens_estimate": 333,
                        "parser": "pytest",
                    }
                ),
                "malformed",
                json.dumps(
                    {
                        "event_supported": True,
                        "applied": False,
                        "classification": "recovery",
                        "original_bytes": 100,
                        "compact_bytes": 100,
                    }
                ),
            )
        )
    )
    observation = _hook_metrics(metrics)
    assert observation["compacted_events"] == 1
    assert observation["recovery_calls"] == 1
    assert observation["parsers"] == ["pytest"]
    _cleanup_hook_artifacts(state, metrics, tmp_path)
    assert not state.exists() and not metrics.exists()
    config.unlink()
    config.parent.rmdir()


def test_hook_capability_is_surface_and_version_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(["codex", "--version"], 0, "codex-cli 0.146.0\n", "")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    assert _hook_replacement_verified("codex", "app-server")
    completed = subprocess.CompletedProcess(["codex", "--version"], 1, "", "failure")
    assert not _hook_replacement_verified("codex", "app-server")

    def unavailable(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("missing")

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert not _hook_replacement_verified("codex", "app-server")
