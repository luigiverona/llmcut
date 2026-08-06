from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from llmcut.integrations.codex.app_server import CodexAppServer, CodexRun
from llmcut.integrations.codex.events import NormalizedEvent, normalize_event
from llmcut.model import digest_bytes


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    name: str
    installed: bool
    version: str | None
    runtime_version: str | None
    usage_events: bool
    detail: str | None = None
    hooks: bool = False
    exclusive_post_tool_replacement: bool = False
    jsonl_usage: bool = False
    command_events: bool = False
    file_change_events: bool = False
    resumable_turns: bool = False
    developer_instructions: bool = True
    mcp: bool = True
    resolved_model_observation: bool = False


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    hooks: bool = False
    exclusive_post_tool_replacement: bool = False
    jsonl_usage: bool = False
    command_events: bool = False
    file_change_events: bool = False
    resumable_turns: bool = False
    developer_instructions: bool = False
    mcp: bool = False


def requirements_for_strategy(strategy: str) -> BackendRequirements:
    if strategy in {"compact-output", "post-replace", "hybrid", "adaptive"}:
        return BackendRequirements(
            hooks=True,
            exclusive_post_tool_replacement=True,
            jsonl_usage=True,
            command_events=True,
        )
    if strategy == "orientation":
        return BackendRequirements(developer_instructions=True)
    if strategy in {"guided", "guided-mcp", "legacy-passive"}:
        return BackendRequirements(mcp=True, developer_instructions=strategy != "legacy-passive")
    return BackendRequirements()


def validate_backend_requirements(strategy: str, capabilities: BackendCapabilities) -> None:
    requirements = requirements_for_strategy(strategy)
    missing = [
        name
        for name in (
            "hooks",
            "exclusive_post_tool_replacement",
            "jsonl_usage",
            "command_events",
            "file_change_events",
            "resumable_turns",
            "developer_instructions",
            "mcp",
        )
        if getattr(requirements, name) and not getattr(capabilities, name)
    ]
    if missing:
        detail = ", ".join(missing)
        suggestion = " Use --backend exec." if requirements.hooks else ""
        raise RuntimeError(
            f"Intervention {strategy} requires backend capabilities: {detail}. "
            f"Backend {capabilities.name} does not provide them.{suggestion}"
        )


class CodexBackend(Protocol):
    async def doctor(self) -> BackendCapabilities: ...

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
        validation_callback: Callable[[], bool] | None,
        cancellation: asyncio.Event | None,
    ) -> CodexRun: ...

    async def cancel(self) -> None: ...


class AppServerBackend:
    def __init__(self, executable: str = "codex") -> None:
        self.executable = executable

    async def doctor(self) -> BackendCapabilities:
        return BackendCapabilities("app-server", True, None, None, True)

    async def run(self, **kwargs: Any) -> CodexRun:
        return await CodexAppServer(self.executable).run(**kwargs)

    async def cancel(self) -> None:
        return None


class SDKBackend:
    """Official ``openai-codex`` SDK backend using its pinned runtime by default."""

    def __init__(
        self, executable: str | None = None, *, allow_hook_trust_bypass: bool = False
    ) -> None:
        self.executable = executable
        self.allow_hook_trust_bypass = allow_hook_trust_bypass
        self._active_turn: Any = None
        self._active_client: Any = None

    async def doctor(self) -> BackendCapabilities:
        try:
            version = importlib.metadata.version("openai-codex")
            runtime = importlib.metadata.version("openai-codex-cli-bin")
        except importlib.metadata.PackageNotFoundError:
            return BackendCapabilities("sdk", False, None, None, False, "install llmcut[codex]")
        return BackendCapabilities("sdk", True, version, runtime, True)

    async def cancel(self) -> None:
        client = self._active_client
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()

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
        validation_callback: Callable[[], bool] | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> CodexRun:
        operation = asyncio.create_task(
            asyncio.to_thread(
                self._run_sync,
                task,
                cwd,
                model,
                reasoning,
                sandbox,
                approval_policy,
                max_turns,
                environment,
                config_overrides,
                validation_callback,
            )
        )
        deadline = asyncio.get_running_loop().time() + timeout
        while not operation.done():
            if cancellation is not None and cancellation.is_set():
                await self.cancel()
                raise asyncio.CancelledError
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                await self.cancel()
                raise TimeoutError("Codex SDK evaluation timed out")
            await asyncio.wait({operation}, timeout=min(0.1, remaining))
        return await operation

    def _run_sync(
        self,
        task: str,
        cwd: Path,
        model: str,
        reasoning: str,
        sandbox: str,
        approval_policy: str,
        max_turns: int,
        environment: dict[str, str],
        config_overrides: tuple[str, ...],
        validation_callback: Callable[[], bool] | None,
    ) -> CodexRun:
        try:
            from openai_codex import (
                ApprovalMode,
                Codex,
                CodexConfig,
                Sandbox,
            )
            from openai_codex.generated.v2_all import ReasoningEffort
        except ImportError as exc:
            raise RuntimeError("Codex SDK is unavailable; install llmcut[codex]") from exc

        sandbox_map = {
            "read-only": Sandbox.read_only,
            "workspace-write": Sandbox.workspace_write,
            "danger-full-access": Sandbox.full_access,
        }
        approval_map = {
            "never": ApprovalMode.deny_all,
            "on-request": ApprovalMode.auto_review,
            "untrusted": ApprovalMode.auto_review,
        }
        if sandbox not in sandbox_map or approval_policy not in approval_map:
            raise ValueError("unsupported Codex sandbox or approval policy")
        supported_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if reasoning not in supported_efforts:
            raise ValueError(f"unsupported Codex reasoning effort: {reasoning}")
        effort = ReasoningEffort(reasoning)
        launch_args: tuple[str, ...] | None = None
        if self.allow_hook_trust_bypass:
            executable = self.executable or _installed_codex_binary()
            values = [executable, "--dangerously-bypass-hook-trust"]
            for override in config_overrides:
                values.extend(("--config", override))
            values.extend(("app-server", "--listen", "stdio://"))
            launch_args = tuple(values)
        config = CodexConfig(
            codex_bin=self.executable,
            launch_args_override=launch_args,
            config_overrides=config_overrides,
            cwd=str(cwd.resolve()),
            env=environment,
            client_name="llmcut",
            client_title="llmcut agent evaluation",
            client_version="0.5.0",
        )
        events: list[NormalizedEvent] = []
        request_digests: list[str] = []
        response_digests: list[str] = []
        usage: dict[str, Any] | None = None
        turn_id = ""
        turns = 0
        first_attempt = False
        status = "failed"
        prompt = task
        try:
            with Codex(config) as codex:
                self._active_client = codex
                thread = codex.thread_start(
                    model=model,
                    cwd=str(cwd.resolve()),
                    sandbox=sandbox_map[sandbox],
                    approval_mode=approval_map[approval_policy],
                    service_name="llmcut_agent_eval",
                )
                while turns < max_turns:
                    request_digests.append(digest_bytes(prompt.encode()))
                    handle = thread.turn(
                        prompt,
                        cwd=str(cwd.resolve()),
                        model=model,
                        effort=effort,
                        sandbox=sandbox_map[sandbox],
                        approval_mode=approval_map[approval_policy],
                    )
                    self._active_turn = handle
                    turn_id = str(handle.id)
                    turns += 1
                    completed_payload: dict[str, Any] = {}

                    try:
                        for notification in handle.stream():
                            raw = _sdk_notification(notification)
                            response_digests.append(digest_bytes(repr(raw).encode()))
                            normalized = normalize_event(raw)
                            if normalized is not None:
                                events.append(normalized)
                                if normalized.kind == "usage_update":
                                    usage = dict(normalized.data)
                                if normalized.kind in {"turn_completed", "turn_failed"}:
                                    status = str(normalized.data.get("status", "failed"))
                                    completed_payload = normalized.data
                    finally:
                        self._active_turn = None
                    if completed_payload.get("turn_id") not in {None, "", turn_id}:
                        raise RuntimeError("Codex SDK completed an unexpected turn")
                    validation_passed = validation_callback() if validation_callback else True
                    if status == "completed" and validation_passed:
                        first_attempt = turns == 1
                        break
                    if turns < max_turns:
                        prompt = (
                            "Deterministic validation failed after the prior turn. Inspect the "
                            "current worktree, correct the requested outcome, and rerun the "
                            "required validation."
                        )
                result = CodexRun(
                    str(thread.id),
                    turn_id,
                    status,
                    events,
                    usage,
                    turns,
                    max(0, turns - 1),
                    first_attempt,
                    "",
                    tuple(request_digests),
                    tuple(response_digests),
                )
        finally:
            self._active_client = None
            self._active_turn = None
        return result


class FakeBackend(SDKBackend):
    """SDK-backed test transport using an explicit local fake Codex runtime."""

    async def doctor(self) -> BackendCapabilities:
        base = await super().doctor()
        return BackendCapabilities(
            "fake",
            base.installed,
            base.version,
            base.runtime_version,
            base.usage_events,
            base.detail,
            hooks=True,
            exclusive_post_tool_replacement=True,
            jsonl_usage=True,
            command_events=True,
            file_change_events=True,
            resumable_turns=True,
        )


def create_backend(
    name: str, executable: str = "codex", *, allow_hook_trust_bypass: bool = False
) -> CodexBackend:
    if name == "sdk":
        return SDKBackend(
            None if executable == "codex" else executable,
            allow_hook_trust_bypass=allow_hook_trust_bypass,
        )
    if name == "app-server":
        return AppServerBackend(executable)
    if name == "exec":
        from llmcut.integrations.codex.exec_backend import ExecBackend

        return ExecBackend(executable, allow_hook_trust_bypass=allow_hook_trust_bypass)
    if name == "fake":
        if executable == "codex":
            raise ValueError("fake backend requires a configured test executable")
        return FakeBackend(executable, allow_hook_trust_bypass=allow_hook_trust_bypass)
    raise ValueError(f"unsupported Codex backend: {name}")


def _installed_codex_binary() -> str:
    from importlib.resources import files

    return str(files("codex_cli_bin").joinpath("bin", "codex"))


def _sdk_notification(notification: Any) -> dict[str, Any]:
    payload = notification.payload
    if is_dataclass(payload) and not isinstance(payload, type):
        params = asdict(payload)
    elif callable(getattr(payload, "model_dump", None)):
        params = payload.model_dump(by_alias=True, mode="json", warnings=False)
    elif isinstance(payload, dict):
        params = dict(payload)
    else:
        params = {"value_type": type(payload).__name__}
    if set(params) == {"params"} and isinstance(params["params"], dict):
        params = params["params"]
    return {"method": str(notification.method), "params": _camelize(params)}


def _camelize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            parts = str(key).split("_")
            result[parts[0] + "".join(part.title() for part in parts[1:])] = _camelize(item)
        return result
    if isinstance(value, list):
        return [_camelize(item) for item in value]
    return value


_CODEX_DISCOVERY_ENV = {
    "HOME",
    "USERPROFILE",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "PATH",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "SSL_CERT_FILE",
    "CODEX_CA_CERTIFICATE",
}
_SAFE_RUNTIME_ENV = {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SYSTEMROOT", "WINDIR"}


def codex_agent_environment(
    names: tuple[str, ...], mode: str, auth_mode: str, auth_env_var: str | None
) -> dict[str, str]:
    selected = (
        _CODEX_DISCOVERY_ENV
        if auth_mode == "existing-session"
        else _SAFE_RUNTIME_ENV | {"SSL_CERT_FILE", "CODEX_CA_CERTIFICATE"}
    ) | set(names)
    if auth_mode in {"api-key", "access-token"}:
        if not auth_env_var or auth_env_var not in os.environ:
            raise RuntimeError(
                f"authentication environment variable is unavailable: {auth_env_var}"
            )
        selected.add(auth_env_var)
    result = {key: value for key, value in os.environ.items() if key in selected}
    result["LLMCUT_EVAL_MODE"] = mode
    return result


def validation_environment(names: tuple[str, ...], mode: str) -> dict[str, str]:
    selected = _SAFE_RUNTIME_ENV | set(names)
    result = {key: value for key, value in os.environ.items() if key in selected}
    result["LLMCUT_EVAL_MODE"] = mode
    return result


def mcp_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in _SAFE_RUNTIME_ENV}
