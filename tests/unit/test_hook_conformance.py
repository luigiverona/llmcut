from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmcut.cli import app
from llmcut.integrations.codex.hooks.conformance import (
    Canaries,
    PostVariant,
    evaluate_returned,
    handle_conformance_hook,
    post_response,
    run_fake_matrix,
    run_live_post_matrix,
    simulated_model_result,
)


def test_post_response_variants_are_isolated() -> None:
    compact = "COMPACT_ONLY_value"
    values = {variant: post_response(variant, compact) for variant in PostVariant}
    assert values[PostVariant.NONE] == (None, 0, "")
    assert values[PostVariant.ADDITIONAL][0] == {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": compact,
        }
    }
    assert values[PostVariant.CONTINUE_REASON][0] == {
        "continue": False,
        "stopReason": compact,
    }
    continue_context = values[PostVariant.CONTINUE_CONTEXT][0]
    continue_reason = values[PostVariant.CONTINUE_REASON][0]
    assert continue_context is not None and "stopReason" not in continue_context
    assert continue_reason is not None and "hookSpecificOutput" not in continue_reason
    assert values[PostVariant.BLOCK_REASON][0] == {"decision": "block", "reason": compact}
    assert values[PostVariant.EXIT_TWO] == (None, 2, compact)
    assert all("suppressOutput" not in json.dumps(value) for value in values.values())


def test_fake_matrix_canary_semantics() -> None:
    results = {item.variant: item for item in run_fake_matrix()}
    assert results[PostVariant.NONE.value].state == "nonexclusive"
    assert results[PostVariant.ADDITIONAL.value].state == "nonexclusive"
    for variant in {
        PostVariant.NONE,
        PostVariant.ADDITIONAL,
        PostVariant.CONTINUE_REASON,
        PostVariant.CONTINUE_CONTEXT,
        PostVariant.CONTINUE_BOTH,
    }:
        assert results[variant.value].state == "nonexclusive"
    for variant in {PostVariant.BLOCK_REASON, PostVariant.BLOCK_CONTEXT, PostVariant.EXIT_TWO}:
        assert results[variant.value].exclusive


def test_canary_evaluation_is_strict_and_inconclusive() -> None:
    canaries = Canaries("ORIGINAL_HEAD_a", "ORIGINAL_MIDDLE_b", "ORIGINAL_TAIL_c", "COMPACT_ONLY_d")
    returned = json.dumps(
        {
            "original_head": None,
            "original_middle": None,
            "original_tail": None,
            "compact_only": canaries.compact_only,
        }
    )
    assert evaluate_returned(
        PostVariant.CONTINUE_REASON,
        returned,
        canaries,
        hook_invoked=True,
        command_exit_code=0,
        original_bytes=10,
        replacement_bytes=2,
    ).exclusive
    assert (
        evaluate_returned(
            PostVariant.CONTINUE_REASON,
            "not json",
            canaries,
            hook_invoked=True,
            command_exit_code=0,
            original_bytes=10,
            replacement_bytes=2,
        ).state
        == "inconclusive"
    )
    assert (
        evaluate_returned(
            PostVariant.CONTINUE_REASON,
            "[]",
            canaries,
            hook_invoked=True,
            command_exit_code=0,
            original_bytes=10,
            replacement_bytes=2,
        ).state
        == "inconclusive"
    )


def test_real_conformance_handler_subprocess_contract(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    marker = tmp_path / "marker"
    state.write_text(
        json.dumps(
            {
                "variant": PostVariant.CONTINUE_REASON.value,
                "compact": "COMPACT_ONLY_exact",
                "marker": str(marker),
            }
        )
    )
    os.chmod(state, 0o600)
    raw = json.dumps({"hook_event_name": "PostToolUse"}).encode()
    response, code, stderr = handle_conformance_hook(raw, state)
    assert code == 0 and not stderr and marker.is_file()
    assert response == {"continue": False, "stopReason": "COMPACT_ONLY_exact"}
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        handle_conformance_hook(raw, state)
    assert handle_conformance_hook(b"{}", state) == (None, 0, "")
    os.chmod(state, 0o644)
    with pytest.raises(ValueError, match="restrictive"):
        handle_conformance_hook(raw, state)
    os.chmod(state, 0o600)
    state.write_text("[]")
    with pytest.raises(ValueError, match="invalid probe"):
        handle_conformance_hook(raw, state)


def test_fake_runtime_executes_real_hook_and_replaces_model_result(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    marker = tmp_path / "marker"
    compact = "COMPACT_ONLY_subprocess"
    state.write_text(
        json.dumps(
            {
                "variant": PostVariant.BLOCK_REASON.value,
                "compact": compact,
                "marker": str(marker),
            }
        )
    )
    os.chmod(state, 0o600)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmcut",
            "hook",
            "conformance-handle",
            "--state",
            str(state),
        ],
        input=json.dumps({"hook_event_name": "PostToolUse"}),
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    response = json.loads(process.stdout)
    original = "ORIGINAL_HEAD_hidden\nORIGINAL_MIDDLE_hidden\nORIGINAL_TAIL_hidden"
    model_visible = response["reason"] if response["decision"] == "block" else original
    assert process.returncode == 0
    assert model_visible == compact
    assert "ORIGINAL_" not in model_visible
    assert marker.is_file()


def test_conformance_cli_validation_and_capabilities(tmp_path: Path) -> None:
    runner = CliRunner()
    capabilities = runner.invoke(app, ["agent", "codex", "hooks", "capabilities"])
    assert capabilities.exit_code == 0
    report = json.loads(capabilities.stdout)
    assert report["post_replacement"] == "supported"
    assert report["pre_rewrite"] == "unverified"
    assert report["probe_digest"].startswith("sha256:")
    assert runner.invoke(app, ["agent", "codex", "hooks", "probe"]).exit_code != 0
    assert (
        runner.invoke(
            app,
            [
                "agent",
                "codex",
                "hooks",
                "probe",
                "--post-tool-use",
                "--allow-hook-trust-bypass",
                "--format",
                "bad",
            ],
        ).exit_code
        != 0
    )
    state = tmp_path / "state.json"
    marker = tmp_path / "marker"
    state.write_text(
        json.dumps(
            {
                "variant": PostVariant.EXIT_TWO.value,
                "compact": "COMPACT_ONLY_cli",
                "marker": str(marker),
            }
        )
    )
    os.chmod(state, 0o600)
    result = runner.invoke(
        app,
        ["hook", "conformance-handle", "--state", str(state)],
        input=json.dumps({"hook_event_name": "PostToolUse"}),
    )
    assert result.exit_code == 2 and "COMPACT_ONLY_cli" in result.stderr


def test_live_harness_with_fake_hook_capable_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    repo.mkdir()
    output.mkdir(mode=0o700)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, "codex-cli fake-1\n", "")
        state = next(output.rglob("state.json"))
        value = json.loads(state.read_text())
        script = next(output.rglob("emit.py"))
        tokens = re.findall(r"(?:ORIGINAL_(?:HEAD|MIDDLE|TAIL))_[0-9a-f]+", script.read_text())
        canaries = Canaries(tokens[0], tokens[1], tokens[2], value["compact"])
        variant = PostVariant(value["variant"])
        response, code, stderr = handle_conformance_hook(
            json.dumps({"hook_event_name": "PostToolUse"}).encode(), state
        )
        assert code == 0 and not stderr and response is not None
        returned = json.dumps(simulated_model_result(variant, canaries))
        stdout = "\n".join(
            (
                json.dumps(
                    {
                        "item": {
                            "type": "command_execution",
                            "status": "completed",
                            "exit_code": 0,
                        }
                    }
                ),
                json.dumps({"item": {"type": "agent_message", "text": returned}}),
                json.dumps({"usage": {"input_tokens": 123}}),
            )
        )
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    results = run_live_post_matrix(
        executable="fake-codex",
        output_dir=output,
        repository=repo,
        variants=(PostVariant.BLOCK_REASON,),
    )
    assert len(results) == 1 and results[0].exclusive
    assert results[0].runtime_version == "codex-cli fake-1"
    assert results[0].capture_digest
    assert not (repo / ".codex").exists()
    assert not list(output.rglob("state.json"))
    assert not list(output.rglob("emit.py"))


def test_live_harness_rejects_existing_codex_layer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".codex").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without an existing"):
        run_live_post_matrix(repository=repo, output_dir=tmp_path / "out", variants=())
