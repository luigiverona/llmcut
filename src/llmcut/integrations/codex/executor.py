from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from llmcut.index import RepositoryIndex
from llmcut.integrations.codex.app_server import CodexRun
from llmcut.integrations.codex.auth import AuthenticationStatus, authentication_preflight
from llmcut.integrations.codex.backend import (
    CodexBackend,
    codex_agent_environment,
    create_backend,
    validate_backend_requirements,
    validation_environment,
)
from llmcut.integrations.codex.context import (
    CodexContextPlan,
    ContextStrategy,
    plan_codex_context,
)
from llmcut.integrations.codex.doctor import detect_codex
from llmcut.integrations.codex.suite import AgentSuite, AgentTask
from llmcut.model import digest_bytes
from llmcut.tokens.estimate import ConservativeEstimator

MAX_OUTPUT_BYTES = 64 * 1024
MAX_DIFF_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidationResult:
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    failed_tests: int
    skipped_tests: int
    warnings: int


@dataclass(slots=True)
class AgentRunResult:
    task_id: str
    repetition: int
    mode: str
    order_index: int
    starting_commit: str
    starting_manifest_digest: str
    settings_digest: str
    parity_digest: str
    status: str
    duration_seconds: float
    validation: list[ValidationResult]
    validation_passed: bool
    changed_files: tuple[str, ...]
    unrelated_changes: tuple[str, ...]
    forbidden_changes: tuple[str, ...]
    missing_required_files: tuple[str, ...]
    git_status: str
    diff_stat: str
    diff_digest: str
    final_worktree_digest: str
    turns: int
    correction_turns: int
    first_attempt_completion: bool
    tool_calls: int
    mcp_calls: int
    retrieval_calls: int
    repeated_mcp_calls: int
    payload_estimate: int
    agent_usage: dict[str, Any] | None
    agent_usage_quality: str
    subscription_usage: str
    timed_out: bool = False
    cancelled: bool = False
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    request_digests: tuple[str, ...] = ()
    response_digests: tuple[str, ...] = ()
    worktree: str | None = None
    backend: str = "sdk"
    user_task_payload: int = 0
    llmcut_mcp_payload: int = 0
    observable_agent_payload: int | None = None
    comparison_design: str = "standard-baseline"
    context_strategy: str = "off"
    orientation_tokens: int = 0
    schema_tokens: int = 0
    developer_instruction_tokens: int = 0
    retrieval_request_tokens: int = 0
    retrieval_result_tokens: int = 0
    context_decision: dict[str, Any] = field(default_factory=dict)
    discovery_observation: dict[str, Any] = field(default_factory=dict)
    hook_observation: dict[str, Any] = field(default_factory=dict)
    requested_settings: dict[str, Any] = field(default_factory=dict)
    resolved_model_observation: str = "unavailable"

    @property
    def quality_passed(self) -> bool:
        return (
            self.status == "completed"
            and self.validation_passed
            and not self.unrelated_changes
            and not self.forbidden_changes
            and not self.missing_required_files
            and not self.timed_out
            and not self.cancelled
            and self.hook_observation.get("validity") != "invalid"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"quality_passed": self.quality_passed}


@dataclass(slots=True)
class AgentEvaluation:
    schema_version: str
    run_id: str
    agent: str
    codex_version: str | None
    suite_digest: str
    seed: int
    started_at: str
    completed_at: str
    environment: dict[str, Any]
    order: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    summary: dict[str, Any]
    claims: dict[str, Any]
    dry_run: bool = False
    evaluation_root: str | None = None
    backend: str = "sdk"
    sdk_version: str | None = None
    runtime_version: str | None = None
    authentication: dict[str, Any] = field(default_factory=dict)
    comparison_design: str = "standard-baseline"
    pilot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodexEvaluator:
    def __init__(
        self,
        suite: AgentSuite,
        *,
        repetitions: int | None = None,
        order: str | None = None,
        seed: int | None = None,
        timeout: float | None = None,
        keep_worktrees: bool = False,
        fail_fast: bool = False,
        evaluation_root: Path | None = None,
        backend: str | None = None,
        auth_mode: str | None = None,
        auth_env_var: str | None = None,
        context_strategy: str | None = None,
        pilot: bool = False,
        allow_hook_trust_bypass: bool = False,
    ) -> None:
        self.suite = suite
        self.repetitions = repetitions or suite.repetitions
        self.order_policy = order or suite.order
        self.seed = suite.seed if seed is None else seed
        self.timeout = timeout or suite.timeout_seconds
        self.keep_worktrees = keep_worktrees
        self.fail_fast = fail_fast
        self._provided_root = evaluation_root
        self.backend_name = backend or suite.execution.backend
        self.auth_mode = auth_mode or suite.execution.auth_mode
        self.auth_env_var = auth_env_var or suite.execution.auth_env_var
        self.context_strategy_override = (
            ContextStrategy.parse(context_strategy) if context_strategy is not None else None
        )
        self.context_strategy = self.context_strategy_override or ContextStrategy.parse(
            suite.execution.context_strategy
        )
        self.pilot = pilot
        self.allow_hook_trust_bypass = allow_hook_trust_bypass
        self._backend: CodexBackend = create_backend(
            self.backend_name,
            suite.executable,
            allow_hook_trust_bypass=allow_hook_trust_bypass,
        )
        if not 1 <= self.repetitions <= 20:
            raise ValueError("repetitions override must be between 1 and 20")
        if self.order_policy not in {"baseline-first", "optimized-first", "alternating", "random"}:
            raise ValueError("invalid ordering override")
        if not 1 <= self.timeout <= 7_200:
            raise ValueError("timeout override must be between 1 and 7200")

    def plan(self) -> AgentEvaluation:
        capabilities = detect_codex() if self.suite.executable == "codex" else None
        order = self._order()
        tasks = [
            {
                "id": task.id,
                "repository": str(task.repository),
                "starting_ref": task.starting_ref,
                "validation": [list(command) for command in task.validation],
                "allowed_changes": list(task.allowed_changes),
                "required_files": list(task.required_files),
                "max_turns": task.max_turns,
            }
            for task in self.suite.tasks
        ]
        now = datetime.now(UTC).isoformat()
        return AgentEvaluation(
            "1",
            uuid.uuid4().hex,
            "codex",
            capabilities.version if capabilities else "configured-test-transport",
            self.suite.digest,
            self.seed,
            now,
            now,
            self._environment_report(),
            order,
            tasks,
            {"planned_runs": len(order), "executed_runs": 0},
            {
                "payload_reduction": False,
                "agent_usage_reduction": False,
                "subscription_reduction": False,
                "observable_payload": "not_measured",
                "agent_input_tokens": "not_measured",
                "provider_input_tokens": "not_measured",
                "subscription_usage": "not_measured",
                "task_quality": "not_measured",
            },
            True,
            backend=self.backend_name,
            comparison_design=self.suite.execution.comparison_design,
            pilot=self.pilot,
        )

    async def run(self, cancellation: asyncio.Event | None = None) -> AgentEvaluation:
        started_at = datetime.now(UTC).isoformat()
        capabilities = detect_codex() if self.suite.executable == "codex" else None
        backend_capabilities = await self._backend.doctor()
        if not backend_capabilities.installed:
            raise RuntimeError(
                backend_capabilities.detail or "selected Codex backend is unavailable"
            )
        if self.suite.executable == "codex":
            validate_backend_requirements(self.context_strategy.value, backend_capabilities)
        if self.suite.execution.require_resolved_model_observation and not (
            backend_capabilities.resolved_model_observation
        ):
            raise RuntimeError(
                "release suite requires resolved-model observation, but the selected backend "
                "does not expose it"
            )
        if (
            self.suite.execution.require_hook_activation
            and self.context_strategy.intervention.output_compaction
            and not self.allow_hook_trust_bypass
        ):
            raise RuntimeError(
                "automated hook evaluation requires explicit --allow-hook-trust-bypass"
            )
        if self.backend_name == "exec" and (
            not self.suite.execution.ignore_user_config or not self.suite.execution.ignore_rules
        ):
            raise RuntimeError(
                "exec evaluation requires ignore_user_config=true and ignore_rules=true"
            )
        authentication = self._authentication_status()
        if self.suite.executable == "codex" and not authentication.automation_ready:
            raise RuntimeError(authentication.diagnostic or "Codex authentication is unavailable")
        root = self._evaluation_root()
        registered: list[tuple[Path, Path]] = []
        results: list[AgentRunResult] = []
        order = self._order()
        try:
            seeds = {
                task.id: _prepare_seed(root / task.id / "seed", task) for task in self.suite.tasks
            }
            for order_index, entry in enumerate(order):
                task = next(item for item in self.suite.tasks if item.id == entry["task_id"])
                seed, commit, manifest = seeds[task.id]
                worktree = root / task.id / f"repetition-{entry['repetition']}" / str(entry["mode"])
                _add_worktree(seed, worktree, commit)
                registered.append((seed, worktree))
                result = await self._run_one(
                    task,
                    int(entry["repetition"]),
                    str(entry["mode"]),
                    order_index,
                    worktree,
                    commit,
                    manifest,
                    cancellation,
                )
                results.append(result)
                if self.fail_fast and not result.quality_passed:
                    break
            task_reports = _task_reports(self.suite.tasks, results)
            summary = _summary(task_reports, results)
            summary["pilot"] = self.pilot
            summary["release_statistics_eligible"] = not self.pilot
            completed_at = datetime.now(UTC).isoformat()
            return AgentEvaluation(
                "1",
                uuid.uuid4().hex,
                "codex",
                capabilities.version if capabilities else "configured-test-transport",
                self.suite.digest,
                self.seed,
                started_at,
                completed_at,
                self._environment_report(),
                order,
                task_reports,
                summary,
                {
                    "payload_reduction": False,
                    "agent_usage_reduction": summary.get("agent_input_claim")
                    == "measured_reduction",
                    "subscription_reduction": False,
                    "observable_payload": "not_measured",
                    "agent_input_tokens": summary.get("agent_input_claim", "not_measured"),
                    "provider_input_tokens": "not_measured",
                    "subscription_usage": "not_measured",
                    "task_quality": "measured_no_regression"
                    if summary.get("passed")
                    else "invalid_comparison",
                },
                False,
                str(root) if self.keep_worktrees else None,
                self.backend_name,
                backend_capabilities.version,
                backend_capabilities.runtime_version,
                authentication.to_dict(),
                self.suite.execution.comparison_design,
                self.pilot,
            )
        finally:
            if not self.keep_worktrees:
                _cleanup(root, registered)

    async def _run_one(
        self,
        task: AgentTask,
        repetition: int,
        mode: str,
        order_index: int,
        worktree: Path,
        commit: str,
        manifest: str,
        cancellation: asyncio.Event | None,
    ) -> AgentRunResult:
        settings = self._settings(task, mode)
        settings_digest = digest_bytes(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        )
        parity = {
            key: value
            for key, value in settings.items()
            if key not in {"context_strategy", "hook_trust_bypass"}
        }
        parity_digest = digest_bytes(
            json.dumps(parity, sort_keys=True, separators=(",", ":")).encode()
        )
        validations: list[ValidationResult] = []

        def validate() -> bool:
            current = _validate(
                task,
                worktree,
                validation_environment(self.suite.execution.environment_allowlist, mode),
                self.timeout,
            )
            validations.extend(current)
            return all(item.exit_code == 0 for item in current)

        prompt = self._prompt(task, mode, worktree)
        requested_strategy = self._context_strategy(task, mode)
        plan = plan_codex_context(
            worktree,
            task.prompt,
            requested_strategy,
            orientation_budget=self.suite.execution.orientation_budget,
            retrieval_budget=self.suite.execution.retrieval_budget,
            state_dir=worktree.parent / ".context-index",
        )
        state_path: Path | None = None
        mcp_overrides: tuple[str, ...] = ()
        if plan.selected_strategy is ContextStrategy.GUIDED:
            state_path = _write_run_state(worktree.parent, worktree, plan)
            mcp_overrides = _mcp_overrides(worktree, plan.selected_strategy.value, state_path)
        elif plan.selected_strategy is ContextStrategy.LEGACY_PASSIVE:
            mcp_overrides = _mcp_overrides(worktree, plan.selected_strategy.value)
        developer_overrides = (
            (f"developer_instructions={json.dumps(plan.orientation_text)}",)
            if plan.orientation_text
            else ()
        )
        hook_overrides: tuple[str, ...] = ()
        hook_state: Path | None = None
        hook_metrics: Path | None = None
        hook_config: Path | None = None
        if plan.selected_strategy.intervention.output_compaction:
            if not _hook_replacement_verified(self.suite.executable, self.backend_name):
                raise RuntimeError(
                    "hook output compaction is release-blocked: exclusive model-facing replacement "
                    "was not demonstrated by the installed Codex runtime"
                )
            if not self.allow_hook_trust_bypass:
                raise RuntimeError(
                    "hook intervention requires explicit --allow-hook-trust-bypass "
                    "during evaluation"
                )
            hook_state = worktree.parent / ".hook-evidence"
            hook_metrics = worktree.parent / ".hook-metrics.jsonl"
            hook_config = _write_evaluation_hook(worktree)
            hook_overrides = _hook_overrides()
        started = time.monotonic()
        run: CodexRun | None = None
        error = None
        timed_out = False
        cancelled = False
        try:
            agent_environment = codex_agent_environment(
                self.suite.execution.environment_allowlist,
                mode,
                self.auth_mode,
                self.auth_env_var,
            )
            agent_environment["LLMCUT_CONTEXT_STRATEGY"] = plan.selected_strategy.value
            if hook_state is not None and hook_metrics is not None:
                from llmcut.integrations.codex.hooks.config import definition_digest

                agent_environment.update(
                    {
                        "LLMCUT_HOOK_REPO": str(worktree),
                        "LLMCUT_HOOK_STATE": str(hook_state),
                        "LLMCUT_HOOK_METRICS": str(hook_metrics),
                        "LLMCUT_HOOK_DEFINITION_DIGEST": definition_digest(),
                    }
                )
            run = await self._backend.run(
                task=prompt,
                cwd=worktree,
                model=str(settings["model"]),
                reasoning=str(settings["reasoning_effort"]),
                sandbox=str(settings["sandbox"]),
                approval_policy=str(settings["approval_policy"]),
                timeout=self.timeout,
                max_turns=task.max_turns,
                environment=agent_environment,
                config_overrides=developer_overrides + mcp_overrides + hook_overrides,
                validation_callback=validate,
                cancellation=cancellation,
            )
        except TimeoutError as exc:
            timed_out, error = True, str(exc)
        except asyncio.CancelledError:
            cancelled, error = True, "evaluation cancelled"
        except Exception as exc:
            error = _safe_error(exc)
        finally:
            if state_path is not None:
                state_path.unlink(missing_ok=True)
            if hook_config is not None:
                hook_config.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    hook_config.parent.rmdir()
        if not validations:
            validations.extend(
                _validate(
                    task,
                    worktree,
                    validation_environment(self.suite.execution.environment_allowlist, mode),
                    self.timeout,
                )
            )
        duration = time.monotonic() - started
        state = _worktree_state(worktree, task)
        events = [item.to_dict() for item in run.events] if run else []
        mcp_events = [item for item in events if item["kind"] in {"mcp_tool_call", "mcp_result"}]
        calls = [item for item in mcp_events if item["kind"] == "mcp_tool_call"]
        call_keys = [(item["data"].get("server"), item["data"].get("tool")) for item in calls]
        task_payload = (
            ConservativeEstimator().count(task.prompt, model=self.suite.execution.model).value
        )
        mcp_payload = sum(int(item["data"].get("result_bytes", 0)) // 3 for item in mcp_events)
        hook_observation = _hook_metrics(hook_metrics)
        hook_observation.update(_reconcile_hooks(events, hook_observation))
        if plan.selected_strategy.intervention.output_compaction:
            from llmcut.integrations.codex.hooks.config import definition_digest

            definition_valid = hook_observation.get("hook_definition_digests") == [
                definition_digest()
            ]
            commands_observed = int(hook_observation.get("codex_completed_commands", 0))
            hooks_observed = int(hook_observation.get("hook_events", 0))
            hook_observation["activation"] = (
                "observed"
                if hooks_observed
                else "no_eligible_commands"
                if not commands_observed
                else "project_layer_untrusted"
                if hook_config is not None
                else "hook_observation_missing"
            )
            hook_observation["validity"] = (
                "valid"
                if (hooks_observed and definition_valid)
                or (not commands_observed and not hooks_observed)
                else "invalid"
            )
        else:
            hook_observation["activation"] = "disabled"
            hook_observation["validity"] = "valid"
        _cleanup_hook_artifacts(hook_state, hook_metrics, worktree.parent)
        return AgentRunResult(
            task.id,
            repetition,
            mode,
            order_index,
            commit,
            manifest,
            settings_digest,
            parity_digest,
            run.status if run else "failed",
            duration,
            validations,
            all(item.exit_code == 0 for item in validations[-len(task.validation) :]),
            state["changed"],
            tuple(sorted(set(state["changed"]) - set(task.allowed_changes))),
            tuple(sorted(set(state["changed"]) & set(task.forbidden_changes))),
            tuple(item for item in task.required_files if not (worktree / item).is_file()),
            state["status"],
            state["stat"],
            digest_bytes(state["diff"].encode()),
            _tree_digest(worktree),
            run.turns if run else 0,
            run.correction_turns if run else 0,
            run.first_attempt_completion if run else False,
            sum(item["kind"] == "command_execution" for item in events),
            len(calls),
            sum(str(item["data"].get("tool", "")).startswith("llmcut_") for item in calls),
            len(call_keys) - len(set(call_keys)),
            task_payload,
            run.usage if run else None,
            "agent_reported" if run and run.usage else "unavailable",
            "unavailable",
            timed_out,
            cancelled,
            error,
            events,
            run.request_digests if run else (),
            run.response_digests if run else (),
            str(worktree) if self.keep_worktrees else None,
            self.backend_name,
            task_payload,
            mcp_payload,
            None,
            self.suite.execution.comparison_design,
            plan.selected_strategy.value,
            plan.orientation_token_estimate,
            plan.mcp_schema_estimate,
            plan.orientation_token_estimate,
            sum(
                max(1, len(json.dumps(item["data"], sort_keys=True).encode()) // 3)
                for item in calls
            ),
            sum(int(item["data"].get("result_bytes", 0)) // 3 for item in mcp_events),
            asdict(plan.adaptive_decision),
            _discovery_metrics(events),
            hook_observation,
            settings | (run.backend_metadata if run else {}),
            "unavailable",
        )

    def _authentication_status(self) -> AuthenticationStatus:
        if self.suite.executable != "codex":
            return AuthenticationStatus(True, "unknown", "unknown", "default", True)
        return authentication_preflight(
            mode=self.auth_mode,
            env_var=self.auth_env_var,
        )

    def _prompt(self, task: AgentTask, mode: str, worktree: Path) -> str:
        if (
            self.suite.execution.comparison_design == "synthetic-full-context"
            and mode == "baseline"
        ):
            return _baseline_prompt(task, worktree)
        return task.prompt

    def _mcp_overrides_for_mode(self, worktree: Path, mode: str) -> tuple[str, ...]:
        design = self.suite.execution.comparison_design
        if design == "standard-baseline" and mode == "baseline":
            return ()
        strategy = "legacy-passive" if mode == "optimized" else "legacy-passive"
        return _mcp_overrides(worktree, strategy)

    def _context_strategy(self, task: AgentTask, mode: str) -> ContextStrategy:
        if mode == "baseline" and self.suite.execution.comparison_design == "standard-baseline":
            return ContextStrategy.OFF
        if self.context_strategy_override is not None:
            return self.context_strategy_override
        overrides = task.baseline if mode == "baseline" else task.optimized
        value = overrides.get(
            "context_strategy",
            overrides.get("context", self.suite.execution.context_strategy),
        )
        if (
            mode == "optimized"
            and "context_strategy" not in overrides
            and "context" not in overrides
        ):
            value = self.context_strategy.value
        aliases = {
            "baseline": "off",
            "optimized": "legacy-passive",
            "managed-mcp": "legacy-passive",
        }
        return ContextStrategy.parse(aliases.get(str(value), str(value)))

    def _settings(self, task: AgentTask, mode: str) -> dict[str, Any]:
        overrides = task.baseline if mode == "baseline" else task.optimized
        executable = Path(self.suite.executable).resolve()
        return {
            "task_digest": digest_bytes(task.prompt.encode()),
            "model": overrides.get("model", self.suite.execution.model),
            "reasoning_effort": overrides.get(
                "reasoning_effort", self.suite.execution.reasoning_effort
            ),
            "sandbox": overrides.get("sandbox", self.suite.execution.sandbox),
            "approval_policy": overrides.get(
                "approval_policy", self.suite.execution.approval_policy
            ),
            "environment_allowlist": self.suite.execution.environment_allowlist,
            "timeout": self.timeout,
            "max_turns": task.max_turns,
            "validation": task.validation,
            "context_strategy": self._context_strategy(task, mode).value,
            "backend": self.backend_name,
            "codex_executable": str(executable),
            "codex_executable_digest": (
                digest_bytes(executable.read_bytes()) if executable.is_file() else "unavailable"
            ),
            "hook_trust_bypass": bool(
                self.allow_hook_trust_bypass
                and self._context_strategy(task, mode).intervention.output_compaction
            ),
            "ignore_user_config": self.suite.execution.ignore_user_config,
            "ignore_rules": self.suite.execution.ignore_rules,
            "ephemeral": self.suite.execution.ephemeral,
        }

    def _order(self) -> list[dict[str, Any]]:
        generator = random.Random(self.seed)  # noqa: S311 - recorded deterministic ordering
        result: list[dict[str, Any]] = []
        for task in self.suite.tasks:
            for repetition in range(1, self.repetitions + 1):
                if self.order_policy == "baseline-first":
                    modes = ["baseline", "optimized"]
                elif self.order_policy == "optimized-first":
                    modes = ["optimized", "baseline"]
                elif self.order_policy == "alternating":
                    modes = (
                        ["baseline", "optimized"] if repetition % 2 else ["optimized", "baseline"]
                    )
                else:
                    modes = ["baseline", "optimized"]
                    generator.shuffle(modes)
                result.extend(
                    {"task_id": task.id, "repetition": repetition, "mode": mode} for mode in modes
                )
        return result

    def _evaluation_root(self) -> Path:
        if self._provided_root is not None:
            root = self._provided_root.resolve()
            root.mkdir(parents=True, exist_ok=False, mode=0o700)
            return root
        return Path(tempfile.mkdtemp(prefix="llmcut-agent-eval-"))

    def _environment_report(self) -> dict[str, Any]:
        return {
            "allowlisted_names": list(self.suite.execution.environment_allowlist),
            "values_persisted": False,
            "model": self.suite.execution.model,
            "reasoning_effort": self.suite.execution.reasoning_effort,
            "sandbox": self.suite.execution.sandbox,
            "approval_policy": self.suite.execution.approval_policy,
            "timeout_seconds": self.timeout,
            "backend": self.backend_name,
            "authentication_mode": self.auth_mode,
            "comparison_design": self.suite.execution.comparison_design,
            "context_strategy": self.context_strategy.value,
            "pilot": self.pilot,
            "codex_agent_environment": "authentication discovery variables allowed; values omitted",
            "validation_environment": "suite allowlist plus safe runtime defaults; values omitted",
            "mcp_environment": "Codex MCP allowlist; credential variables not forwarded",
        }


def _prepare_seed(destination: Path, task: AgentTask) -> tuple[Path, str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (task.repository / ".git").exists():
        _run(
            ["git", "clone", "--no-hardlinks", "--quiet", str(task.repository), str(destination)],
            destination.parent,
        )
    else:
        shutil.copytree(
            task.repository,
            destination,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", ".llmcut", "__pycache__", "*.pyc"),
        )
        if any(path.is_symlink() for path in destination.rglob("*")):
            raise ValueError("benchmark repository contains a symlink")
        _run(["git", "init", "-q"], destination)
        _run(["git", "config", "user.email", "agent-eval@llmcut.invalid"], destination)
        _run(["git", "config", "user.name", "llmcut agent eval"], destination)
        _run(["git", "add", "."], destination)
        _run(["git", "commit", "-qm", "materialize benchmark fixture"], destination)
    resolved = _run(
        ["git", "rev-parse", "--verify", f"{task.starting_ref}^{{commit}}"], destination
    ).stdout.strip()
    _run(["git", "checkout", "--quiet", resolved], destination)
    if _run(["git", "status", "--porcelain"], destination).stdout:
        raise ValueError("benchmark starting revision is not clean")
    return destination, resolved, _tracked_manifest(destination)


def _add_worktree(seed: Path, worktree: Path, commit: str) -> None:
    worktree.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if worktree.exists():
        raise ValueError("evaluation worktree already exists")
    _run(["git", "worktree", "add", "--detach", "--quiet", str(worktree), commit], seed)
    if _run(["git", "rev-parse", "HEAD"], worktree).stdout.strip() != commit:
        raise ValueError("evaluation worktree started at an unexpected commit")
    if _tracked_manifest(worktree) != _tracked_manifest(seed):
        raise ValueError("baseline/optimized tracked-file parity failed")


def _cleanup(root: Path, registered: list[tuple[Path, Path]]) -> None:
    root = root.resolve()
    if not root.name.startswith("llmcut-agent-eval-") and not registered:
        raise ValueError("refusing to clean an unregistered evaluation root")
    for seed, worktree in reversed(registered):
        resolved = worktree.resolve()
        if root not in resolved.parents:
            raise ValueError("registered worktree escaped evaluation root")
        _run(["git", "worktree", "remove", "--force", str(resolved)], seed, check=False)
    if root.exists() and (root.name.startswith("llmcut-agent-eval-") or registered):
        shutil.rmtree(root)


def _baseline_prompt(task: AgentTask, worktree: Path) -> str:
    records = RepositoryIndex(worktree, worktree.parent / ".baseline-index").build()
    chunks = [task.prompt, "\nFull baseline repository context follows exactly:\n"]
    for record in records:
        path = worktree / record.path
        if not record.binary and path.is_file() and not path.is_symlink():
            chunks.extend((f"\n--- {record.path} ---\n", path.read_text(errors="replace")))
    return "".join(chunks)


def _mcp_overrides(worktree: Path, strategy: str, run_state: Path | None = None) -> tuple[str, ...]:
    values = ["mcp", "serve", "--repo", str(worktree), "--integration", strategy]
    if run_state is not None:
        values.extend(("--run-state", str(run_state)))
    args = json.dumps(values)
    return (
        'mcp_servers.llmcut.command="llmcut"',
        f"mcp_servers.llmcut.args={args}",
        "mcp_servers.llmcut.env_vars=[]",
        "mcp_servers.llmcut.required=true",
    )


def _write_run_state(parent: Path, worktree: Path, plan: CodexContextPlan) -> Path:
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(parent, 0o700)
    payload = {
        "strategy": plan.selected_strategy.value,
        "task_digest": plan.task_digest,
        "repository_revision": plan.repository_revision,
        "repository_root": str(worktree.resolve()),
        "plan": plan.to_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    envelope = json.dumps(
        {"digest": digest_bytes(canonical), "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    path = parent / f".llmcut-run-state-{uuid.uuid4().hex}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(envelope)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _discovery_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    commands = [
        str(item["data"].get("command", ""))
        for item in events
        if item.get("kind") == "command_execution"
    ]
    reads: list[str] = []
    searches = 0
    listings = 0
    for command in commands:
        if re.search(r"(^|\s)(rg|grep|find)(\s|$)", command):
            searches += 1
        if re.search(r"(^|\s)(ls|tree)(\s|$)", command):
            listings += 1
        match = re.search(r"(?:cat|sed\s+-n\s+\S+)\s+([A-Za-z0-9_./-]+)", command)
        if match:
            reads.append(match.group(1))
    first = "unavailable"
    if commands:
        first = "shell_search" if searches else "directory_listing" if listings else "shell_command"
    return {
        "state": "partially_observed" if commands else "unavailable",
        "shell_commands": len(commands),
        "file_reads": len(reads),
        "searches": searches,
        "directory_listings": listings,
        "first_repository_discovery_action": first,
        "unique_files_inspected": len(set(reads)),
        "repeated_file_reads": len(reads) - len(set(reads)),
        "command_output_bytes": "unavailable",
        "files_inspected_before_first_edit": "partially_observed" if reads else "unavailable",
    }


def _validate(
    task: AgentTask, cwd: Path, environment: dict[str, str], timeout: float
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for command in task.validation:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = str(exc.stdout or ""), str(exc.stderr or ""), 124
        combined = stdout + "\n" + stderr
        results.append(
            ValidationResult(
                command,
                code,
                time.monotonic() - started,
                stdout[:MAX_OUTPUT_BYTES],
                stderr[:MAX_OUTPUT_BYTES],
                len(re.findall(r"\bFAILED\b|\bfailures?\b", combined, re.I)),
                len(re.findall(r"\bSKIPPED\b", combined, re.I)),
                len(re.findall(r"\bWARNINGS?\b", combined, re.I)),
            )
        )
        if code:
            break
    return results


def _worktree_state(cwd: Path, task: AgentTask) -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain=v1"], cwd).stdout[:MAX_OUTPUT_BYTES]
    changed = tuple(_run(["git", "diff", "--name-only", "HEAD"], cwd).stdout.splitlines())
    stat = _run(["git", "diff", "--stat", "HEAD"], cwd).stdout[:MAX_OUTPUT_BYTES]
    diff = _run(["git", "diff", "--binary", "HEAD"], cwd).stdout[:MAX_DIFF_BYTES]
    return {"status": status, "changed": changed, "stat": stat, "diff": diff}


def _tracked_manifest(cwd: Path) -> str:
    files = _run(["git", "ls-files", "-z"], cwd).stdout.split("\0")
    chunks = [name.encode() + b"\0" + (cwd / name).read_bytes() for name in files if name]
    return digest_bytes(b"\0".join(chunks))


def _tree_digest(cwd: Path) -> str:
    chunks = []
    for path in sorted(cwd.rglob("*")):
        relative = path.relative_to(cwd)
        if (
            path.is_file()
            and not path.is_symlink()
            and not {".git", ".llmcut-eval-index", "__pycache__"} & set(relative.parts)
            and path.suffix != ".pyc"
        ):
            chunks.append(str(relative).encode() + b"\0" + path.read_bytes())
    return digest_bytes(b"\0".join(chunks))


def _run(argv: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=120, check=False)
    if check and result.returncode:
        raise ValueError(f"command failed ({argv[0]}): {result.stderr[:2_048]}")
    return result


def _task_reports(
    tasks: tuple[AgentTask, ...], results: list[AgentRunResult]
) -> list[dict[str, Any]]:
    reports = []
    for task in tasks:
        task_results = [item for item in results if item.task_id == task.id]
        modes = {
            mode: _aggregate([item for item in task_results if item.mode == mode])
            for mode in ("baseline", "optimized")
        }
        pairs = []
        for repetition in sorted({item.repetition for item in task_results}):
            baseline = next(
                (
                    item
                    for item in task_results
                    if item.repetition == repetition and item.mode == "baseline"
                ),
                None,
            )
            optimized = next(
                (
                    item
                    for item in task_results
                    if item.repetition == repetition and item.mode == "optimized"
                ),
                None,
            )
            if not baseline or not optimized:
                pairs.append(
                    {
                        "repetition": repetition,
                        "eligible": False,
                        "exclusion_reason": "incomplete pair",
                    }
                )
                continue
            settings_parity = baseline.parity_digest == optimized.parity_digest
            quality_parity = baseline.quality_passed == optimized.quality_passed
            eligible = settings_parity and baseline.quality_passed and optimized.quality_passed
            payload_reduction = (
                (baseline.payload_estimate - optimized.payload_estimate)
                / baseline.payload_estimate
                * 100
                if eligible
                and baseline.payload_estimate
                and baseline.comparison_design == "synthetic-full-context"
                else None
            )
            pairs.append(
                {
                    "repetition": repetition,
                    "eligible": eligible,
                    "settings_parity": settings_parity,
                    "core_execution_parity": "passed" if settings_parity else "failed",
                    "user_task_parity": baseline.user_task_payload == optimized.user_task_payload,
                    "repository_parity": (
                        baseline.starting_manifest_digest == optimized.starting_manifest_digest
                    ),
                    "validation_parity": baseline.validation_passed == optimized.validation_passed,
                    "intervention_difference": "llmcut context integration",
                    "optimized_context_strategy": optimized.context_strategy,
                    "comparison_design": baseline.comparison_design,
                    "quality_parity": quality_parity,
                    "payload_reduction_percent": payload_reduction,
                    "observable_payload_difference": None,
                    "agent_usage_reduction_percent": _usage_reduction(baseline, optimized)
                    if eligible
                    else None,
                    "duration_difference_seconds": optimized.duration_seconds
                    - baseline.duration_seconds,
                    "success_difference": int(optimized.quality_passed)
                    - int(baseline.quality_passed),
                    "first_attempt_difference": int(optimized.first_attempt_completion)
                    - int(baseline.first_attempt_completion),
                    "correction_turn_difference": optimized.correction_turns
                    - baseline.correction_turns,
                    "exclusion_reason": None
                    if eligible
                    else "settings mismatch or deterministic quality failure",
                }
            )
        reports.append(
            {
                "id": task.id,
                "modes": modes,
                "comparisons": pairs,
                "runs": [item.to_dict() for item in task_results],
            }
        )
    return reports


def _aggregate(results: list[AgentRunResult]) -> dict[str, Any]:
    durations = [item.duration_seconds for item in results]
    payloads = [item.payload_estimate for item in results]
    usage = [item.agent_usage for item in results if item.agent_usage]
    return {
        "repetitions_attempted": len(results),
        "successful_completions": sum(item.status == "completed" for item in results),
        "validation_successes": sum(item.validation_passed for item in results),
        "first_attempt_successes": sum(item.first_attempt_completion for item in results),
        "failures": sum(not item.quality_passed for item in results),
        "timeouts": sum(item.timed_out for item in results),
        "cancellations": sum(item.cancelled for item in results),
        "median_duration": median(durations) if durations else None,
        "p25_duration": _percentile(durations, 0.25),
        "p75_duration": _percentile(durations, 0.75),
        "minimum_duration": min(durations, default=None),
        "maximum_duration": max(durations, default=None),
        "median_payload_tokens": median(payloads) if payloads else None,
        "median_observable_payload": None,
        "median_agent_input_tokens": _usage_median(usage, ("inputTokens", "input_tokens")),
        "median_output_tokens": _usage_median(usage, ("outputTokens", "output_tokens")),
        "median_cached_tokens": _usage_median(usage, ("cachedInputTokens", "cached_tokens")),
        "correction_turns": sum(item.correction_turns for item in results),
        "tool_calls": sum(item.tool_calls for item in results),
        "mcp_calls": sum(item.mcp_calls for item in results),
        "median_mcp_calls": median([item.mcp_calls for item in results]) if results else None,
        "retrieval_calls": sum(item.retrieval_calls for item in results),
        "median_retrieval_calls": median([item.retrieval_calls for item in results])
        if results
        else None,
        "median_correction_turns": median([item.correction_turns for item in results])
        if results
        else None,
        "timeout_rate": sum(item.timed_out for item in results) / len(results) if results else 0.0,
        "unrelated_change_rate": sum(bool(item.unrelated_changes) for item in results)
        / len(results)
        if results
        else 0.0,
    }


def _summary(tasks: list[dict[str, Any]], results: list[AgentRunResult]) -> dict[str, Any]:
    from llmcut.mcp.server import tool_schema_bytes

    comparisons = [pair for task in tasks for pair in task["comparisons"]]
    eligible = [pair for pair in comparisons if pair.get("eligible")]
    reductions = [
        float(pair["payload_reduction_percent"])
        for pair in eligible
        if pair.get("payload_reduction_percent") is not None
    ]
    agent = [pair for pair in eligible if pair.get("agent_usage_reduction_percent") is not None]
    agent_reductions = [float(pair["agent_usage_reduction_percent"]) for pair in agent]
    summary: dict[str, Any] = {
        "runs": len(results),
        "quality_successes": sum(item.quality_passed for item in results),
        "regressions": sum(
            item.mode == "optimized" and not item.quality_passed for item in results
        ),
        "eligible_comparisons": len(eligible),
        "excluded_comparisons": len(comparisons) - len(eligible),
        "median_payload_reduction_percent": median(reductions) if reductions else None,
        "agent_usage_comparisons": len(agent),
        "agent_usage_reductions": sum(value > 0 for value in agent_reductions),
        "agent_input_claim": (
            "not_measured"
            if not agent_reductions
            else "measured_reduction"
            if median(agent_reductions) > 0
            else "measured_no_reduction"
        ),
        "paired_agent_input_differences_percent": agent_reductions,
        "agent_input_minimum_reduction_percent": min(agent_reductions, default=None),
        "agent_input_maximum_reduction_percent": max(agent_reductions, default=None),
        "agent_input_iqr_percent": (
            [_percentile(agent_reductions, 0.25), _percentile(agent_reductions, 0.75)]
            if agent_reductions
            else None
        ),
        "subscription_usage": "unavailable",
        "passed": bool(results)
        and all(item.quality_passed for item in results)
        and len(eligible) == len(comparisons),
    }
    non_control = [task for task in tasks if "control" not in str(task["id"]).lower()]
    representative_pairs = [
        pair
        for task in non_control
        for pair in task["comparisons"]
        if pair.get("eligible") and pair.get("agent_usage_reduction_percent") is not None
    ]
    representative_reductions = [
        float(pair["agent_usage_reduction_percent"]) for pair in representative_pairs
    ]
    task_medians = {
        str(task["id"]): median(values)
        for task in non_control
        if (
            values := [
                float(pair["agent_usage_reduction_percent"])
                for pair in task["comparisons"]
                if pair.get("eligible") and pair.get("agent_usage_reduction_percent") is not None
            ]
        )
    }
    control_runs = [item for item in results if "control" in item.task_id.lower()]
    control_safe = all(
        item.mode != "optimized" or item.context_strategy == "off" for item in control_runs
    )
    schema_safe = all(
        item.context_strategy != "guided"
        or item.schema_tokens
        <= max(1, (tool_schema_bytes(ContextStrategy.LEGACY_PASSIVE) + 2) // 3) * 0.3
        for item in results
    )
    intervention_observed = any(
        item.mode == "optimized"
        and (
            item.retrieval_calls > 0
            or item.orientation_tokens > 0
            or int(item.hook_observation.get("compacted_events", 0)) > 0
        )
        for item in results
    )
    summary.update(
        {
            "representative_tasks": len(non_control),
            "representative_agent_usage_comparisons": len(representative_pairs),
            "representative_median_agent_input_reduction_percent": (
                median(representative_reductions) if representative_reductions else None
            ),
            "representative_positive_pair_rate": (
                sum(value > 0 for value in representative_reductions)
                / len(representative_reductions)
                if representative_reductions
                else None
            ),
            "representative_task_medians": task_medians,
            "no_benefit_control_safe": control_safe,
            "guided_schema_reduction_passed": schema_safe,
            "context_intervention_observed": intervention_observed,
            "release_measurement_gates_passed": bool(representative_reductions)
            and len(non_control) >= 4
            and all(len(task["comparisons"]) >= 3 for task in non_control)
            and median(representative_reductions) >= 5
            and sum(value > 0 for value in representative_reductions)
            / len(representative_reductions)
            >= 0.6
            and all(value >= -5 for value in task_medians.values())
            and control_safe
            and schema_safe
            and intervention_observed,
        }
    )
    return summary


def _usage_median(values: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    result = []
    for value in values:
        found = next((value[key] for key in keys if isinstance(value.get(key), (int, float))), None)
        if found is not None:
            result.append(float(found))
    return median(result) if result else None


def _hook_overrides() -> tuple[str, ...]:
    from llmcut.integrations.codex.hooks.config import project_hook_overrides

    return project_hook_overrides()


def _hook_replacement_verified(executable: str, backend: str) -> bool:
    from llmcut.integrations.codex.hooks.capabilities import capabilities_for

    if executable != "codex":
        return True  # The configured fake runtime executes the real hook subprocess in tests.
    if backend == "sdk":
        return False  # Direct-exec conformance does not transfer to the App Server surface.
    else:
        try:
            result = subprocess.run(
                [executable, "--version"], text=True, capture_output=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return False
        runtime = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return capabilities_for(runtime).post_replacement == "supported"


def _write_evaluation_hook(worktree: Path) -> Path:
    from llmcut.integrations.codex.hooks.config import proposed_document

    directory = worktree / ".codex"
    target = directory / "hooks.json"
    if directory.exists():
        raise RuntimeError(
            "evaluation hook isolation requires a disposable fixture without an existing "
            ".codex directory"
        )
    directory.mkdir(mode=0o700)
    payload = json.dumps(proposed_document(), sort_keys=True, separators=(",", ":"))
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.write("\n")
    except BaseException:
        target.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return target


def _hook_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file() or path.is_symlink():
        return {
            "observation": "unavailable" if path is not None else "observed",
            "hook_events": 0,
            "eligible_events": 0,
            "compacted_events": 0,
            "pass_through_events": 0,
            "original_result_bytes": 0,
            "model_facing_result_bytes": 0,
            "recovery_calls": 0,
            "recovery_bytes": 0,
        }
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines()[:10_000]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    compacted = [item for item in events if item.get("applied") is True]
    recoveries = [item for item in events if item.get("classification") == "recovery"]
    return {
        "observation": "observed",
        "hook_events": len(events),
        "eligible_events": sum(item.get("event_supported") is True for item in events),
        "compacted_events": len(compacted),
        "pass_through_events": len(events) - len(compacted),
        "original_result_bytes": sum(int(item.get("original_bytes", 0)) for item in events),
        "model_facing_result_bytes": sum(int(item.get("compact_bytes", 0)) for item in events),
        "estimated_original_tokens": sum(
            int(item.get("original_tokens_estimate", 0)) for item in events
        ),
        "estimated_compact_tokens": sum(
            int(item.get("compact_tokens_estimate", 0)) for item in events
        ),
        "recovery_calls": len(recoveries),
        "recovery_bytes": sum(int(item.get("original_bytes", 0)) for item in recoveries),
        "parsers": sorted({str(item.get("parser")) for item in compacted if item.get("parser")}),
        "command_digests": [
            str(item["command_digest"])
            for item in events
            if isinstance(item.get("command_digest"), str)
        ],
        "hook_definition_digests": sorted(
            {
                str(item["hook_definition_digest"])
                for item in events
                if isinstance(item.get("hook_definition_digest"), str)
            }
        ),
    }


def _reconcile_hooks(events: list[dict[str, Any]], observation: dict[str, Any]) -> dict[str, Any]:
    commands: list[str] = []
    for event in events:
        if event.get("kind") != "command_execution" or event.get("method") not in {
            "item.completed",
            "item/completed",
        }:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        digest = data.get("command_digest")
        if not isinstance(digest, str) and isinstance(data.get("command"), str):
            digest = digest_bytes(str(data["command"]).encode())
        if isinstance(digest, str):
            commands.append(digest)
    hook_values = observation.get("command_digests", [])
    hooks = [str(value) for value in hook_values] if isinstance(hook_values, list) else []
    unmatched_commands = list(commands)
    unmatched_hooks: list[str] = []
    matched = 0
    for value in hooks:
        if value in unmatched_commands:
            unmatched_commands.remove(value)
            matched += 1
        else:
            unmatched_hooks.append(value)
    return {
        "codex_completed_commands": len(commands),
        "matched_hook_events": matched,
        "unmatched_codex_events": len(unmatched_commands),
        "unmatched_hook_events": len(unmatched_hooks),
        "reconciliation": (
            "observed"
            if observation.get("observation") == "observed" and not unmatched_hooks
            else "partially_observed"
        ),
    }


def _cleanup_hook_artifacts(state: Path | None, metrics: Path | None, parent: Path) -> None:
    root = parent.resolve(strict=True)
    if metrics is not None and metrics.parent.resolve(strict=True) == root:
        metrics.unlink(missing_ok=True)
    if state is None or state.is_symlink() or state.parent.resolve(strict=True) != root:
        return
    if state.is_dir():
        shutil.rmtree(state)


def _usage_reduction(baseline: AgentRunResult, optimized: AgentRunResult) -> float | None:
    if not baseline.agent_usage or not optimized.agent_usage:
        return None
    before = _usage_median([baseline.agent_usage], ("inputTokens", "input_tokens"))
    after = _usage_median([optimized.agent_usage], ("inputTokens", "input_tokens"))
    return (before - after) / before * 100 if before and after is not None else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _safe_error(error: BaseException) -> str:
    rendered = str(error)[:2_048]
    rendered = re.sub(
        r"(?i)(authorization|cookie|api[-_ ]?key|access[-_ ]?token|password)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        rendered,
    )
    rendered = re.sub(r"\b(?:sk-|Bearer\s+)[A-Za-z0-9._-]{8,}", "[REDACTED]", rendered)
    return rendered
