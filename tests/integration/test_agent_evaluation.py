from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmcut.captures import verify_capture
from llmcut.cli import app
from llmcut.integrations.codex.events import normalize_event
from llmcut.integrations.codex.executor import CodexEvaluator
from llmcut.integrations.codex.suite import load_suite

SUITE = Path("tests/fixtures/agent/suite.toml").resolve()
runner = CliRunner()


def test_non_dry_run_cli_executes_ab_and_writes_capture(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    capture = tmp_path / "capture"
    result = runner.invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--suite",
            str(SUITE),
            "--repetitions",
            "1",
            "--order",
            "optimized-first",
            "--format",
            "json",
            "--output",
            str(output),
            "--capture",
            str(capture),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report["summary"]["passed"]
    assert report["order"][0]["mode"] == "optimized"
    assert report["claims"]["payload_reduction"] is False
    assert report["claims"]["agent_input_tokens"] == "measured_reduction"
    assert report["claims"]["subscription_usage"] == "not_measured"
    assert report["comparison_design"] == "standard-baseline"
    assert report["tasks"][0]["comparisons"][0]["settings_parity"]
    assert report["tasks"][0]["comparisons"][0]["core_execution_parity"] == "passed"
    assert report["tasks"][0]["modes"]["optimized"]["retrieval_calls"] == 1
    assert verify_capture(capture).turns == 2
    assert output.stat().st_mode & 0o077 == 0


def test_unavailable_usage_and_correction_turns(tmp_path: Path) -> None:
    unavailable = tmp_path / "unavailable.json"
    result = runner.invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--suite",
            str(SUITE),
            "--repetitions",
            "1",
            "--format",
            "json",
            "--output",
            str(unavailable),
        ],
        env={"LLMCUT_FAKE_SCENARIO": "unavailable-usage"},
    )
    assert result.exit_code == 0, result.output
    report = json.loads(unavailable.read_text())
    assert report["summary"]["agent_usage_comparisons"] == 0
    assert all(run["agent_usage_quality"] == "unavailable" for run in report["tasks"][0]["runs"])
    correction = tmp_path / "correction.json"
    result = runner.invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--suite",
            str(SUITE),
            "--repetitions",
            "1",
            "--format",
            "json",
            "--output",
            str(correction),
        ],
        env={"LLMCUT_FAKE_SCENARIO": "correction"},
    )
    assert result.exit_code == 0, result.output
    runs = json.loads(correction.read_text())["tasks"][0]["runs"]
    assert all(run["correction_turns"] == 1 for run in runs)
    assert all(not run["first_attempt_completion"] for run in runs)
    assert all(len(run["validation"]) == 2 for run in runs)


@pytest.mark.parametrize(
    "scenario",
    ["validation-failure", "malformed", "unexpected-exit", "reroute", "protocol-mismatch"],
)
def test_cli_failure_scenarios_exit_nonzero_and_clean_processes(
    tmp_path: Path, scenario: str
) -> None:
    output = tmp_path / f"{scenario}.json"
    result = runner.invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--suite",
            str(SUITE),
            "--repetitions",
            "1",
            "--fail-fast",
            "--format",
            "json",
            "--output",
            str(output),
        ],
        env={"LLMCUT_FAKE_SCENARIO": scenario},
    )
    assert result.exit_code == 1
    report = json.loads(output.read_text())
    assert not report["summary"]["passed"]
    assert report["evaluation_root"] is None


def test_repeated_mcp_calls_and_settings_mismatch() -> None:
    repeated = runner.invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--suite",
            str(SUITE),
            "--repetitions",
            "1",
            "--format",
            "json",
        ],
        env={"LLMCUT_FAKE_SCENARIO": "repeated-mcp"},
    )
    assert repeated.exit_code == 0
    report = json.loads(repeated.stdout)
    optimized = next(run for run in report["tasks"][0]["runs"] if run["mode"] == "optimized")
    assert optimized["repeated_mcp_calls"] == 1

    suite = load_suite(SUITE)
    changed_task = replace(suite.tasks[0], optimized={"context": "managed-mcp", "model": "other"})
    changed_suite = replace(suite, repetitions=1, tasks=(changed_task,))
    evaluation = asyncio.run(CodexEvaluator(changed_suite).run())
    comparison = evaluation.tasks[0]["comparisons"][0]
    assert not comparison["eligible"] and not comparison["settings_parity"]


def test_timeout_cancellation_and_event_redaction() -> None:
    suite = replace(load_suite(SUITE), repetitions=1, timeout_seconds=1)
    original = __import__("os").environ.get("LLMCUT_FAKE_SCENARIO")
    __import__("os").environ["LLMCUT_FAKE_SCENARIO"] = "timeout"
    try:
        evaluation = asyncio.run(CodexEvaluator(suite, fail_fast=True).run())
    finally:
        if original is None:
            __import__("os").environ.pop("LLMCUT_FAKE_SCENARIO", None)
        else:
            __import__("os").environ["LLMCUT_FAKE_SCENARIO"] = original
    run = evaluation.tasks[0]["runs"][0]
    assert run["timed_out"] and not run["quality_passed"]

    cancellation = asyncio.Event()
    cancellation.set()
    cancelled = asyncio.run(
        CodexEvaluator(replace(suite, timeout_seconds=5), fail_fast=True).run(cancellation)
    )
    assert cancelled.tasks[0]["runs"][0]["cancelled"]

    reasoning = normalize_event(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "reasoning",
                    "summary": "private summary",
                    "content": "hidden chain of thought",
                }
            },
        }
    )
    assert reasoning and reasoning.data == {"item_type": "reasoning", "status": ""}
    assert "private" not in json.dumps(reasoning.to_dict())


def test_evaluator_override_bounds_ordering_and_registered_root(tmp_path: Path) -> None:
    suite = replace(load_suite(SUITE), repetitions=2)
    with pytest.raises(ValueError, match="repetitions override"):
        CodexEvaluator(suite, repetitions=21)
    with pytest.raises(ValueError, match="ordering override"):
        CodexEvaluator(suite, order="invalid")
    with pytest.raises(ValueError, match="timeout override"):
        CodexEvaluator(suite, timeout=8_000)

    optimized = CodexEvaluator(suite, order="optimized-first").plan()
    assert optimized.order[0]["mode"] == "optimized"
    alternating = CodexEvaluator(suite, order="alternating").plan()
    assert [item["mode"] for item in alternating.order] == [
        "baseline",
        "optimized",
        "optimized",
        "baseline",
    ]

    root = tmp_path / "explicit-evaluation"
    evaluation = asyncio.run(
        CodexEvaluator(
            replace(suite, repetitions=1), evaluation_root=root, keep_worktrees=True
        ).run()
    )
    assert evaluation.evaluation_root == str(root.resolve())
    assert (root / suite.tasks[0].id / "repetition-1" / "baseline").is_dir()
    shutil.rmtree(root)
