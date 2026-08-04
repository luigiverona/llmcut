from __future__ import annotations

import asyncio
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmcut.integrations.codex.auth import authentication_preflight
from llmcut.integrations.codex.backend import (
    AppServerBackend,
    FakeBackend,
    SDKBackend,
    _sdk_notification,
    codex_agent_environment,
    create_backend,
    mcp_environment,
    validation_environment,
)
from llmcut.integrations.codex.executor import CodexEvaluator, _mcp_overrides, _safe_error
from llmcut.integrations.codex.suite import load_suite


def test_codex_validation_and_mcp_environments_exclude_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/private/home")
    monkeypatch.setenv("CODEX_HOME", "/private/codex")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "secret-token")
    monkeypatch.setenv("TEST_SETTING", "visible")

    agent = codex_agent_environment((), "baseline", "existing-session", None)
    validation = validation_environment((), "baseline")
    explicitly_allowed = validation_environment(("HOME",), "baseline")
    mcp = mcp_environment()

    assert agent["HOME"] == "/private/home" and agent["CODEX_HOME"] == "/private/codex"
    assert "CODEX_ACCESS_TOKEN" not in agent
    assert "HOME" not in validation and "CODEX_HOME" not in validation
    assert explicitly_allowed["HOME"] == "/private/home"
    assert "HOME" not in mcp and "CODEX_ACCESS_TOKEN" not in mcp


def test_explicit_auth_variable_reaches_only_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/private/home")
    monkeypatch.setenv("LLMCUT_TEST_API_KEY", "secret-value")
    agent = codex_agent_environment((), "optimized", "api-key", "LLMCUT_TEST_API_KEY")
    assert agent["LLMCUT_TEST_API_KEY"] == "secret-value"
    assert "HOME" not in agent
    assert "LLMCUT_TEST_API_KEY" not in validation_environment((), "optimized")
    with pytest.raises(RuntimeError, match="unavailable"):
        codex_agent_environment((), "optimized", "api-key", "MISSING_KEY")


def test_authentication_preflight_reports_category_without_identity(tmp_path: Path) -> None:
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nprintf 'Logged in using ChatGPT\\n'\n")
    executable.chmod(0o700)
    status = authentication_preflight(executable=str(executable))
    report = status.to_dict()
    assert status.authenticated and status.method == "chatgpt"
    assert "email" not in report and "token" not in report
    assert authentication_preflight(mode="none").automation_ready is False


def test_authentication_modes_require_only_named_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NAMED_ACCESS", "private")
    status = authentication_preflight(mode="access-token", env_var="NAMED_ACCESS")
    assert status.authenticated and status.method == "access-token"
    assert "private" not in str(status.to_dict())
    missing = authentication_preflight(mode="api-key", env_var="MISSING")
    assert not missing.authenticated and missing.diagnostic
    with pytest.raises(ValueError, match="unsupported"):
        authentication_preflight(mode="bad")


def test_authentication_preflight_handles_missing_and_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("llmcut.integrations.codex.auth.shutil.which", lambda _: None)
    assert "unavailable" in str(authentication_preflight().diagnostic)
    broken = tmp_path / "codex"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o700)
    failed = authentication_preflight(executable=str(broken))
    assert not failed.authenticated and failed.method == "unknown"
    monkeypatch.setattr(
        "llmcut.integrations.codex.auth.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("codex", 10)),
    )
    timed_out = authentication_preflight(executable=str(broken))
    assert timed_out.diagnostic == "login status unavailable"


def test_backend_factory_and_capabilities() -> None:
    assert isinstance(create_backend("sdk"), SDKBackend)
    assert isinstance(create_backend("app-server"), AppServerBackend)
    assert isinstance(create_backend("fake", "/bin/false"), FakeBackend)
    with pytest.raises(ValueError, match="requires"):
        create_backend("fake")
    with pytest.raises(ValueError, match="unsupported"):
        create_backend("other")
    capability = asyncio.run(SDKBackend().doctor())
    assert capability.installed and capability.version == "0.144.4"


def test_backend_diagnostics_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    app_server = AppServerBackend("/bin/false")
    assert asyncio.run(app_server.doctor()).name == "app-server"
    assert asyncio.run(app_server.cancel()) is None
    backend = SDKBackend()

    def missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert not asyncio.run(backend.doctor()).installed

    class Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = Client()
    backend._active_client = client
    asyncio.run(backend.cancel())
    assert client.closed


def test_sdk_rejects_invalid_execution_settings(tmp_path: Path) -> None:
    backend = SDKBackend()
    with pytest.raises(ValueError, match="sandbox"):
        backend._run_sync("task", tmp_path, "model", "low", "bad", "never", 1, {}, (), None)
    with pytest.raises(ValueError, match="reasoning"):
        backend._run_sync("task", tmp_path, "model", "bad", "read-only", "never", 1, {}, (), None)


class SampleEnum(Enum):
    value_name = "value"


@dataclass
class SamplePayload:
    snake_name: str
    nested_value: list[SampleEnum]


def test_sdk_notification_normalizes_supported_and_opaque_payloads() -> None:
    event = _sdk_notification(
        SimpleNamespace(method="sample", payload=SamplePayload("x", [SampleEnum.value_name]))
    )
    assert event == {
        "method": "sample",
        "params": {"snakeName": "x", "nestedValue": ["value"]},
    }
    wrapped = _sdk_notification(
        SimpleNamespace(method="sample", payload={"params": {"thread_id": "thread"}})
    )
    assert wrapped["params"] == {"threadId": "thread"}
    opaque = _sdk_notification(SimpleNamespace(method="future", payload=object()))
    assert opaque["params"] == {"valueType": "object"}


def test_standard_baseline_uses_identical_prompt_and_no_mcp() -> None:
    suite = replace(load_suite(Path("tests/fixtures/agent/suite.toml")), repetitions=1)
    evaluator = CodexEvaluator(suite, backend="sdk")
    task = suite.tasks[0]
    assert evaluator._prompt(task, "baseline", task.repository) == task.prompt
    assert evaluator._prompt(task, "optimized", task.repository) == task.prompt
    assert evaluator._mcp_overrides_for_mode(task.repository, "baseline") == ()
    assert evaluator._mcp_overrides_for_mode(task.repository, "optimized")
    assert "mcp_servers.llmcut.env_vars=[]" in _mcp_overrides(task.repository, "optimized")


def test_live_suite_is_standard_sdk_and_resolves_allowlisted_repositories() -> None:
    suite = load_suite(Path("tests/live/codex/suite.toml"))
    assert suite.execution.backend == "sdk"
    assert suite.execution.comparison_design == "standard-baseline"
    assert suite.repetitions == 3 and len(suite.tasks) >= 3
    assert all(task.repository.is_dir() for task in suite.tasks)


def test_sensitive_error_values_are_redacted() -> None:
    rendered = _safe_error(
        RuntimeError("authorization=Bearer-private CODEX_ACCESS_TOKEN=secretvalue sk-abcdefghi")
    )
    assert "private" not in rendered and "secretvalue" not in rendered
    assert "sk-abcdefghi" not in rendered
    assert rendered.count("[REDACTED]") >= 2


def test_environment_reports_names_not_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_SETTING", "private-value")
    suite = load_suite(Path("tests/fixtures/agent/suite.toml"))
    execution = replace(suite.execution, environment_allowlist=("TEST_SETTING",))
    report = CodexEvaluator(replace(suite, execution=execution)).plan().to_dict()
    assert "private-value" not in str(report)
    assert report["environment"]["allowlisted_names"] == ["TEST_SETTING"]
    assert os.environ["TEST_SETTING"] == "private-value"
