from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmcut.model import digest_bytes

SCHEMA_VERSION = "1"
ORDERS = {"baseline-first", "optimized-first", "alternating", "random"}
SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}
APPROVALS = {"never", "on-request", "untrusted"}
BACKENDS = {"sdk", "app-server", "exec", "fake"}
AUTH_MODES = {"existing-session", "api-key", "access-token", "none"}
COMPARISON_DESIGNS = {
    "standard-baseline",
    "tool-parity-baseline",
    "hook-parity-baseline",
    "synthetic-full-context",
}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    model: str
    reasoning_effort: str
    sandbox: str
    approval_policy: str
    environment_allowlist: tuple[str, ...] = ()
    backend: str = "sdk"
    auth_mode: str = "existing-session"
    auth_env_var: str | None = None
    comparison_design: str = "standard-baseline"
    context_strategy: str = "adaptive"
    orientation_budget: int = 200
    retrieval_budget: int = 4_096
    require_hook_activation: bool = False
    require_resolved_model_observation: bool = False
    ignore_user_config: bool = True
    ignore_rules: bool = True
    ephemeral: bool = True


@dataclass(frozen=True, slots=True)
class AgentTask:
    id: str
    repository: Path
    starting_ref: str
    prompt: str
    validation: tuple[tuple[str, ...], ...]
    allowed_changes: tuple[str, ...]
    forbidden_changes: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()
    max_turns: int = 2
    baseline: dict[str, Any] = field(default_factory=dict)
    optimized: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentSuite:
    schema_version: str
    agent: str
    repetitions: int
    order: str
    seed: int
    timeout_seconds: float
    execution: ExecutionConfig
    tasks: tuple[AgentTask, ...]
    executable: str = "codex"

    @property
    def digest(self) -> str:
        return digest_bytes(self.canonical_json().encode())

    def canonical_json(self) -> str:
        return json.dumps(_suite_dict(self), sort_keys=True, separators=(",", ":"))


def load_suite(path: Path) -> AgentSuite:
    if not path.is_file():
        raise ValueError(f"agent suite does not exist: {path}")
    raw = tomllib.loads(path.read_text())
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported agent suite schema_version")
    if raw.get("agent") != "codex":
        raise ValueError("agent suite requires agent=codex")
    repetitions = int(raw.get("repetitions", 3))
    if not 1 <= repetitions <= 20:
        raise ValueError("repetitions must be between 1 and 20")
    order = str(raw.get("order", "random"))
    if order not in ORDERS:
        raise ValueError(f"unsupported ordering policy: {order}")
    timeout = float(raw.get("timeout_seconds", 900))
    if not 1 <= timeout <= 7_200:
        raise ValueError("timeout_seconds must be between 1 and 7200")
    execution_raw = _object(raw.get("execution"), "execution")
    execution = ExecutionConfig(
        model=_required(execution_raw, "model"),
        reasoning_effort=_required(execution_raw, "reasoning_effort"),
        sandbox=str(execution_raw.get("sandbox", "workspace-write")),
        approval_policy=str(execution_raw.get("approval_policy", "never")),
        environment_allowlist=tuple(
            _strings(execution_raw.get("environment_allowlist", []), "environment_allowlist")
        ),
        backend=str(execution_raw.get("backend", "sdk")),
        auth_mode=str(execution_raw.get("auth_mode", "existing-session")),
        auth_env_var=(
            str(execution_raw["auth_env_var"]) if execution_raw.get("auth_env_var") else None
        ),
        comparison_design=str(execution_raw.get("comparison_design", "standard-baseline")),
        context_strategy=str(execution_raw.get("context_strategy", "adaptive")),
        orientation_budget=int(execution_raw.get("orientation_budget", 200)),
        retrieval_budget=int(execution_raw.get("retrieval_budget", 4_096)),
        require_hook_activation=bool(execution_raw.get("require_hook_activation", False)),
        require_resolved_model_observation=bool(
            execution_raw.get("require_resolved_model_observation", False)
        ),
        ignore_user_config=bool(execution_raw.get("ignore_user_config", True)),
        ignore_rules=bool(execution_raw.get("ignore_rules", True)),
        ephemeral=bool(execution_raw.get("ephemeral", True)),
    )
    if execution.sandbox not in SANDBOXES:
        raise ValueError(f"unsupported sandbox: {execution.sandbox}")
    if execution.approval_policy not in APPROVALS:
        raise ValueError(f"unsupported approval policy: {execution.approval_policy}")
    if execution.reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {execution.reasoning_effort}")
    if execution.backend not in BACKENDS:
        raise ValueError(f"unsupported Codex backend: {execution.backend}")
    if execution.auth_mode not in AUTH_MODES:
        raise ValueError(f"unsupported authentication mode: {execution.auth_mode}")
    if execution.auth_mode in {"api-key", "access-token"} and not execution.auth_env_var:
        raise ValueError("explicit authentication requires auth_env_var")
    if execution.auth_env_var and not _IDENTIFIER.fullmatch(execution.auth_env_var):
        raise ValueError("auth_env_var must be an environment variable name")
    if execution.comparison_design not in COMPARISON_DESIGNS:
        raise ValueError(f"unsupported comparison design: {execution.comparison_design}")
    from llmcut.integrations.codex.context import ContextStrategy

    ContextStrategy.parse(execution.context_strategy)
    if not 0 <= execution.orientation_budget <= 2_000:
        raise ValueError("orientation_budget must be between 0 and 2000")
    if not 1 <= execution.retrieval_budget <= 128_000:
        raise ValueError("retrieval_budget must be between 1 and 128000")
    base = path.parent.resolve()
    repository_base = _repository_root(base, raw.get("repository_root"))
    tasks: list[AgentTask] = []
    identifiers: set[str] = set()
    for item in raw.get("tasks", []):
        value = _object(item, "task")
        identifier = _required(value, "id")
        if not _IDENTIFIER.fullmatch(identifier) or identifier in identifiers:
            raise ValueError(f"invalid or duplicate task id: {identifier}")
        identifiers.add(identifier)
        repository = _within(repository_base, _required(value, "repository"))
        if not repository.is_dir() or repository.is_symlink():
            raise ValueError(f"task repository is unavailable: {repository}")
        max_turns = int(value.get("max_turns", 2))
        if not 1 <= max_turns <= 8:
            raise ValueError("task max_turns must be between 1 and 8")
        validation = _commands(value.get("validation"))
        tasks.append(
            AgentTask(
                identifier,
                repository,
                str(value.get("starting_ref", "HEAD")),
                _required(value, "prompt"),
                validation,
                tuple(_safe_paths(value.get("allowed_changes", []))),
                tuple(_safe_paths(value.get("forbidden_changes", []))),
                tuple(_safe_paths(value.get("required_files", []))),
                max_turns,
                _object(value.get("baseline", {}), "baseline"),
                _object(value.get("optimized", {}), "optimized"),
            )
        )
    if not tasks:
        raise ValueError("agent suite requires at least one task")
    executable = str(raw.get("codex_executable", "codex"))
    local_executable = (base / executable).resolve()
    if local_executable.is_file():
        executable = str(local_executable)
    elif "/" in executable or "\\" in executable:
        executable_path = _within(base, executable)
        if not executable_path.is_file():
            raise ValueError("configured Codex executable is unavailable")
        executable = str(executable_path)
    return AgentSuite(
        SCHEMA_VERSION,
        "codex",
        repetitions,
        order,
        int(raw.get("seed", 1729)),
        timeout,
        execution,
        tuple(tasks),
        executable,
    )


def _commands(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("validation requires one or more argv arrays")
    if all(isinstance(item, str) for item in value):
        value = [value]
    commands: list[tuple[str, ...]] = []
    for command in value:
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) and x for x in command)
        ):
            raise ValueError("validation commands must be non-empty argv arrays")
        if len(command) > 64 or any(len(item) > 4_096 or "\0" in item for item in command):
            raise ValueError("validation command exceeds configured bounds")
        commands.append(tuple(command))
    return tuple(commands)


def _within(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("suite path escapes suite root")
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("suite path escapes suite root")
    return resolved


def _repository_root(base: Path, value: Any) -> Path:
    if value is None:
        return base
    if not isinstance(value, str) or not value:
        raise ValueError("repository_root must be a non-empty relative path")
    candidate = (base / value).resolve()
    checkout = next(
        (parent for parent in (base, *base.parents) if (parent / ".git").exists()), None
    )
    if checkout is None or (candidate != checkout and checkout not in candidate.parents):
        raise ValueError("repository_root must remain inside the Git checkout")
    if not candidate.is_dir() or candidate.is_symlink():
        raise ValueError("repository_root is unavailable")
    return candidate


def _safe_paths(value: Any) -> list[str]:
    result = _strings(value, "path list")
    for item in result:
        path = Path(item)
        if path.is_absolute() or ".." in path.parts or not item:
            raise ValueError(f"unsafe task path: {item}")
    return result


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string array")
    return list(value)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table")
    return dict(value)


def _required(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result or len(result) > 16_384:
        raise ValueError(f"{key} must be a bounded non-empty string")
    return result


def _suite_dict(value: AgentSuite) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "agent": value.agent,
        "repetitions": value.repetitions,
        "order": value.order,
        "seed": value.seed,
        "timeout_seconds": value.timeout_seconds,
        "execution": {
            "model": value.execution.model,
            "reasoning_effort": value.execution.reasoning_effort,
            "sandbox": value.execution.sandbox,
            "approval_policy": value.execution.approval_policy,
            "environment_allowlist": value.execution.environment_allowlist,
            "backend": value.execution.backend,
            "auth_mode": value.execution.auth_mode,
            "auth_env_var": value.execution.auth_env_var,
            "comparison_design": value.execution.comparison_design,
            "context_strategy": value.execution.context_strategy,
            "orientation_budget": value.execution.orientation_budget,
            "retrieval_budget": value.execution.retrieval_budget,
            "require_hook_activation": value.execution.require_hook_activation,
            "require_resolved_model_observation": (
                value.execution.require_resolved_model_observation
            ),
            "ignore_user_config": value.execution.ignore_user_config,
            "ignore_rules": value.execution.ignore_rules,
            "ephemeral": value.execution.ephemeral,
        },
        "tasks": [
            {
                "id": item.id,
                "repository": str(item.repository),
                "starting_ref": item.starting_ref,
                "prompt": item.prompt,
                "validation": item.validation,
                "allowed_changes": item.allowed_changes,
                "forbidden_changes": item.forbidden_changes,
                "required_files": item.required_files,
                "max_turns": item.max_turns,
                "baseline": item.baseline,
                "optimized": item.optimized,
            }
            for item in value.tasks
        ],
        "executable": value.executable,
    }
