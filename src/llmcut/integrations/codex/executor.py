from __future__ import annotations

import asyncio
import json
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
from llmcut.integrations.codex.app_server import CodexAppServer, CodexRun, allowed_environment
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
    claims: dict[str, bool]
    dry_run: bool = False
    evaluation_root: str | None = None

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
    ) -> None:
        self.suite = suite
        self.repetitions = repetitions or suite.repetitions
        self.order_policy = order or suite.order
        self.seed = suite.seed if seed is None else seed
        self.timeout = timeout or suite.timeout_seconds
        self.keep_worktrees = keep_worktrees
        self.fail_fast = fail_fast
        self._provided_root = evaluation_root
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
            },
            True,
        )

    async def run(self, cancellation: asyncio.Event | None = None) -> AgentEvaluation:
        started_at = datetime.now(UTC).isoformat()
        capabilities = detect_codex() if self.suite.executable == "codex" else None
        if capabilities is not None and (not capabilities.installed or not capabilities.app_server):
            raise RuntimeError("Codex App Server is unavailable; run `llmcut agent codex doctor`")
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
                    "payload_reduction": bool(summary.get("eligible_comparisons")),
                    "agent_usage_reduction": bool(
                        capabilities and summary.get("agent_usage_comparisons")
                    ),
                    "subscription_reduction": False,
                },
                False,
                str(root) if self.keep_worktrees else None,
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
        parity = {key: value for key, value in settings.items() if key != "context_strategy"}
        parity_digest = digest_bytes(
            json.dumps(parity, sort_keys=True, separators=(",", ":")).encode()
        )
        validations: list[ValidationResult] = []

        def validate() -> bool:
            current = _validate(
                task,
                worktree,
                allowed_environment(self.suite.execution.environment_allowlist, mode),
                self.timeout,
            )
            validations.extend(current)
            return all(item.exit_code == 0 for item in current)

        prompt = task.prompt if mode == "optimized" else _baseline_prompt(task, worktree)
        mcp_overrides = _mcp_overrides(worktree, mode)
        started = time.monotonic()
        run: CodexRun | None = None
        error = None
        timed_out = False
        cancelled = False
        try:
            run = await CodexAppServer(self.suite.executable).run(
                task=prompt,
                cwd=worktree,
                model=str(settings["model"]),
                reasoning=str(settings["reasoning_effort"]),
                sandbox=str(settings["sandbox"]),
                approval_policy=str(settings["approval_policy"]),
                timeout=self.timeout,
                max_turns=task.max_turns,
                environment=allowed_environment(self.suite.execution.environment_allowlist, mode),
                config_overrides=mcp_overrides,
                validation_callback=validate,
                cancellation=cancellation,
            )
        except TimeoutError as exc:
            timed_out, error = True, str(exc)
        except asyncio.CancelledError:
            cancelled, error = True, "evaluation cancelled"
        except Exception as exc:
            error = str(exc)[:2_048]
        if not validations:
            validations.extend(
                _validate(
                    task,
                    worktree,
                    allowed_environment(self.suite.execution.environment_allowlist, mode),
                    self.timeout,
                )
            )
        duration = time.monotonic() - started
        state = _worktree_state(worktree, task)
        events = [item.to_dict() for item in run.events] if run else []
        mcp_events = [item for item in events if item["kind"] in {"mcp_tool_call", "mcp_result"}]
        calls = [item for item in mcp_events if item["kind"] == "mcp_tool_call"]
        call_keys = [(item["data"].get("server"), item["data"].get("tool")) for item in calls]
        payload = ConservativeEstimator().count(prompt, model=self.suite.execution.model).value
        payload += sum(int(item["data"].get("result_bytes", 0)) // 3 for item in mcp_events)
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
            payload,
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
        )

    def _settings(self, task: AgentTask, mode: str) -> dict[str, Any]:
        overrides = task.baseline if mode == "baseline" else task.optimized
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
            "context_strategy": overrides.get("context", mode),
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


def _mcp_overrides(worktree: Path, mode: str) -> tuple[str, ...]:
    args = json.dumps(["mcp", "serve", "--repo", str(worktree), "--integration", mode])
    return (
        'mcp_servers.llmcut.command="llmcut"',
        f"mcp_servers.llmcut.args={args}",
        "mcp_servers.llmcut.required=true",
    )


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
                if eligible and baseline.payload_estimate
                else None
            )
            pairs.append(
                {
                    "repetition": repetition,
                    "eligible": eligible,
                    "settings_parity": settings_parity,
                    "quality_parity": quality_parity,
                    "payload_reduction_percent": payload_reduction,
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
        "minimum_duration": min(durations, default=None),
        "maximum_duration": max(durations, default=None),
        "median_payload_tokens": median(payloads) if payloads else None,
        "median_agent_input_tokens": _usage_median(usage, ("inputTokens", "input_tokens")),
        "median_output_tokens": _usage_median(usage, ("outputTokens", "output_tokens")),
        "median_cached_tokens": _usage_median(usage, ("cachedInputTokens", "cached_tokens")),
        "correction_turns": sum(item.correction_turns for item in results),
        "tool_calls": sum(item.tool_calls for item in results),
        "mcp_calls": sum(item.mcp_calls for item in results),
        "retrieval_calls": sum(item.retrieval_calls for item in results),
        "unrelated_change_rate": sum(bool(item.unrelated_changes) for item in results)
        / len(results)
        if results
        else 0.0,
    }


def _summary(tasks: list[dict[str, Any]], results: list[AgentRunResult]) -> dict[str, Any]:
    comparisons = [pair for task in tasks for pair in task["comparisons"]]
    eligible = [pair for pair in comparisons if pair.get("eligible")]
    reductions = [
        float(pair["payload_reduction_percent"])
        for pair in eligible
        if pair.get("payload_reduction_percent") is not None
    ]
    agent = [pair for pair in eligible if pair.get("agent_usage_reduction_percent") is not None]
    return {
        "runs": len(results),
        "quality_successes": sum(item.quality_passed for item in results),
        "regressions": sum(
            item.mode == "optimized" and not item.quality_passed for item in results
        ),
        "eligible_comparisons": len(eligible),
        "excluded_comparisons": len(comparisons) - len(eligible),
        "median_payload_reduction_percent": median(reductions) if reductions else None,
        "agent_usage_comparisons": len(agent),
        "subscription_usage": "unavailable",
        "passed": bool(results)
        and all(item.quality_passed for item in results)
        and len(eligible) == len(comparisons),
    }


def _usage_median(values: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    result = []
    for value in values:
        found = next((value[key] for key in keys if isinstance(value.get(key), (int, float))), None)
        if found is not None:
            result.append(float(found))
    return median(result) if result else None


def _usage_reduction(baseline: AgentRunResult, optimized: AgentRunResult) -> float | None:
    if not baseline.agent_usage or not optimized.agent_usage:
        return None
    before = _usage_median([baseline.agent_usage], ("inputTokens", "input_tokens"))
    after = _usage_median([optimized.agent_usage], ("inputTokens", "input_tokens"))
    return (before - after) / before * 100 if before and after is not None else None
