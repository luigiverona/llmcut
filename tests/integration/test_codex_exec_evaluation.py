from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from llmcut.cli import app
from llmcut.integrations.codex.executor import CodexEvaluator, _reconcile_hooks
from llmcut.integrations.codex.suite import load_suite

SUITE = Path(__file__).parents[1] / "fixtures" / "agent" / "exec-hook-suite.toml"


def test_fake_exec_evaluator_runs_real_hook_and_reconciles(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    capture = tmp_path / "capture"
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--backend",
            "exec",
            "--suite",
            str(SUITE),
            "--allow-hook-trust-bypass",
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
    runs = report["tasks"][0]["runs"]
    baseline = next(item for item in runs if item["mode"] == "baseline")
    optimized = next(item for item in runs if item["mode"] == "optimized")
    assert baseline["hook_observation"]["activation"] == "disabled"
    assert baseline["hook_observation"]["hook_events"] == 0
    hook = optimized["hook_observation"]
    assert hook["activation"] == "observed"
    assert hook["validity"] == "valid"
    assert hook["compacted_events"] == 1
    assert hook["matched_hook_events"] == 1
    assert hook["unmatched_codex_events"] == 0
    assert optimized["agent_usage"]["inputTokens"] < baseline["agent_usage"]["inputTokens"]
    assert optimized["quality_passed"] and baseline["quality_passed"]
    manifest = json.loads((capture / "manifest.json").read_text())
    assert manifest["provider"] == "codex-exec"
    assert manifest["endpoint"] == "exec-jsonl"


def test_exec_suite_requires_explicit_trust_bypass() -> None:
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "eval",
            "--agent",
            "codex",
            "--backend",
            "exec",
            "--suite",
            str(SUITE),
        ],
    )
    assert result.exit_code == 3
    assert "explicit --allow-hook-trust-bypass" in result.output


def test_exec_suite_rejects_unobservable_model_and_inherited_config() -> None:
    suite = load_suite(SUITE)
    require_model = replace(
        suite,
        execution=replace(suite.execution, require_resolved_model_observation=True),
    )
    try:
        asyncio.run(CodexEvaluator(require_model, allow_hook_trust_bypass=True).run())
    except RuntimeError as exc:
        assert "resolved-model observation" in str(exc)
    else:
        raise AssertionError("unobservable resolved model was accepted")
    inherited = replace(
        suite,
        execution=replace(suite.execution, ignore_user_config=False),
    )
    try:
        asyncio.run(CodexEvaluator(inherited, allow_hook_trust_bypass=True).run())
    except RuntimeError as exc:
        assert "ignore_user_config=true" in str(exc)
    else:
        raise AssertionError("inherited exec configuration was accepted")


def test_hook_reconciliation_rejects_duplicates_and_malformed_events() -> None:
    digest = "sha256:abc"
    events: list[dict[str, Any]] = [
        {"kind": "opaque", "method": "item.completed", "data": {}},
        {"kind": "command_execution", "method": "item.completed", "data": "bad"},
        {
            "kind": "command_execution",
            "method": "item.completed",
            "data": {"command_digest": digest},
        },
    ]
    result = _reconcile_hooks(
        events, {"observation": "observed", "command_digests": [digest, digest]}
    )
    assert result["matched_hook_events"] == 1
    assert result["unmatched_hook_events"] == 1
    assert result["reconciliation"] == "partially_observed"
    sdk = _reconcile_hooks(
        [
            {
                "kind": "command_execution",
                "method": "item/completed",
                "data": {"command": "pytest -q"},
            }
        ],
        {"observation": "observed", "command_digests": []},
    )
    assert sdk["unmatched_codex_events"] == 1
