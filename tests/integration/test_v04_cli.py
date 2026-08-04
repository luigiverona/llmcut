from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from llmcut.cli import _run_evaluator, app
from llmcut.measurement import request_digest, response_digest

runner = CliRunner()


def _capture(root: Path) -> Path:
    root.mkdir()
    request = {"model": "same", "messages": [{"role": "user", "content": "task"}]}
    response = {"choices": [{"message": {"content": "done"}}]}
    (root / "request.json").write_text(json.dumps(request))
    (root / "response.json").write_text(json.dumps(response))
    manifest = {
        "schema_version": "1",
        "capture_id": "cli-capture",
        "provider": "openai",
        "model": "same",
        "endpoint": "chat.completions",
        "persistence": {"prompt_content": True},
        "turns": [
            {
                "request": {"digest": request_digest(request), "content_location": "request.json"},
                "response": {
                    "digest": response_digest(response),
                    "content_location": "response.json",
                },
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_token_and_capture_commands(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"messages": [{"content": "a larger provider payload"}]}))
    second.write_text(json.dumps({"messages": [{"content": "small"}]}))
    counted = runner.invoke(
        app,
        ["tokens", "count", "--provider", "openai", "--model", "same", "--input", str(first)],
    )
    assert counted.exit_code == 0 and "request_digest" in counted.stdout
    compared = runner.invoke(
        app,
        ["tokens", "compare", str(first), str(second), "--provider", "openai", "--model", "same"],
    )
    assert compared.exit_code == 0 and "reduction_percent" in compared.stdout
    capture = _capture(tmp_path / "capture")
    for command in ("inspect", "verify", "replay", "redact"):
        result = runner.invoke(app, ["capture", command, str(capture)])
        assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["tokens", "verify", str(capture)]).exit_code == 0
    deleted = runner.invoke(app, ["capture", "delete", str(capture)])
    assert deleted.exit_code == 0 and not capture.exists()


def test_mcp_and_codex_inspection_commands(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("value = 1\n")
    for command in ("inspect", "doctor", "config"):
        result = runner.invoke(app, ["mcp", command, "--repo", str(repo)])
        assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["agent", "codex", "doctor"]).exit_code == 0
    assert runner.invoke(app, ["agent", "codex", "config", "--repo", str(repo)]).exit_code == 0
    config = tmp_path / "codex.toml"
    initialized = runner.invoke(
        app,
        [
            "agent",
            "codex",
            "init",
            "--repo",
            str(repo),
            "--config",
            str(config),
            "--dry-run",
        ],
    )
    assert initialized.exit_code == 0 and not config.exists()
    suite = Path("tests/fixtures/agent/suite.toml")
    evaluation = runner.invoke(
        app,
        ["agent", "eval", "--agent", "codex", "--suite", str(suite), "--dry-run"],
    )
    assert evaluation.exit_code == 0 and "subscription usage: unavailable" in evaluation.stdout


def test_executable_eval_cli() -> None:
    result = runner.invoke(app, ["eval", "--corpus", "tests/fixtures/benchmarks/suite.toml"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    assert value["statistics"]["passed"]
    assert value["statistics"]["negative_cases"] == 1


def test_external_evaluator_success_and_failure(tmp_path: Path) -> None:
    @dataclass
    class Result:
        value: int = 1

    success = SimpleNamespace(
        evaluator_command=["python", "-c", "import sys; sys.exit(0)"], timeout=5
    )
    failure = SimpleNamespace(
        evaluator_command=["python", "-c", "import sys; sys.exit(3)"], timeout=5
    )
    assert _run_evaluator(success, Result(), tmp_path) == (True, None)
    assert _run_evaluator(failure, Result(), tmp_path) == (False, "evaluator exited 3")
