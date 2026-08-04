from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median, quantiles
from typing import Any

from llmcut.adapters import OpenAIAdapter
from llmcut.managed.protocol import Context, ManagedRequest
from llmcut.measurement import TokenMeasurement, count_payload
from llmcut.model import BlockKind, CanonicalRequest, Retention, digest_bytes
from llmcut.tokens.registry import CounterRegistry


@dataclass(frozen=True, slots=True)
class ExecutableResult:
    task_id: str
    starting_revision: str
    baseline: TokenMeasurement
    optimized: TokenMeasurement
    retrieval_tokens: int
    continuation_tokens: int
    total_optimized_tokens: int
    reduction_percent: float
    validation_passed: bool
    changed_files: tuple[str, ...]
    unrelated_files: tuple[str, ...]
    fallback: str | None
    measurement_trust: str = "locally_counted"

    @property
    def eligible(self) -> bool:
        return self.validation_passed and not self.unrelated_files

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["baseline"] = self.baseline.to_dict()
        value["optimized"] = self.optimized.to_dict()
        value["eligible"] = self.eligible
        return value


def evaluate_suite(path: Path) -> tuple[list[ExecutableResult], dict[str, Any]]:
    suite = tomllib.loads(path.read_text())
    results = [_evaluate_case(path.parent, value) for value in suite.get("tasks", [])]
    return results, release_statistics(results)


def _evaluate_case(base: Path, value: dict[str, Any]) -> ExecutableResult:
    fixture = (base / str(value["repository"])).resolve()
    if not fixture.is_dir():
        raise ValueError(f"benchmark repository is missing: {fixture}")
    with tempfile.TemporaryDirectory(prefix="llmcut-benchmark-") as temporary:
        seed = Path(temporary) / "seed"
        shutil.copytree(fixture, seed)
        _run(["git", "init", "-q"], seed)
        _run(["git", "config", "user.email", "benchmark@llmcut.invalid"], seed)
        _run(["git", "config", "user.name", "llmcut benchmark"], seed)
        _run(["git", "add", "."], seed)
        _run(["git", "commit", "-qm", "seed benchmark"], seed)
        revision = _run(["git", "rev-parse", "HEAD"], seed).stdout.strip()
        baseline_worktree = Path(temporary) / "baseline"
        optimized_worktree = Path(temporary) / "optimized"
        _run(["git", "worktree", "add", "--detach", str(baseline_worktree), revision], seed)
        _run(["git", "worktree", "add", "--detach", str(optimized_worktree), revision], seed)
        if _tree_digest(baseline_worktree) != _tree_digest(optimized_worktree):
            raise ValueError("baseline and optimized worktrees do not share identical files")
        patch = (fixture / str(value["patch"])).read_text()
        for worktree in (baseline_worktree, optimized_worktree):
            applied = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=worktree,
                input=patch,
                text=True,
                capture_output=True,
                check=False,
            )
            if applied.returncode:
                raise ValueError(f"benchmark patch failed: {applied.stderr}")
        command = [str(item) for item in value["validation"]]
        baseline_validation = _run(command, baseline_worktree, check=False)
        optimized_validation = _run(command, optimized_worktree, check=False)
        changed = tuple(
            _run(["git", "diff", "--name-only"], optimized_worktree).stdout.splitlines()
        )
        allowed = set(str(item) for item in value.get("allowed_changes", []))
        unrelated = tuple(sorted(set(changed) - allowed))
        baseline_payload, optimized_payload, continuation_payload = _payloads(fixture, value)
        registry = CounterRegistry()
        baseline_count = count_payload(
            registry, "openai", str(value.get("model", "offline-model")), baseline_payload
        )
        optimized_count = count_payload(
            registry, "openai", str(value.get("model", "offline-model")), optimized_payload
        )
        retrieval_payload = {
            "operation": "context.expand",
            "results": [
                {"source": relative, "content": (fixture / relative).read_text(errors="replace")}
                for relative in value.get("retrieve", [])
            ],
        }
        retrieval = (
            count_payload(
                registry,
                "openai",
                str(value.get("model", "offline-model")),
                retrieval_payload,
            ).value
            if retrieval_payload["results"]
            else 0
        )
        continuation = (
            count_payload(
                registry,
                "openai",
                str(value.get("model", "offline-model")),
                continuation_payload,
            ).value
            if continuation_payload is not None
            else 0
        )
        total = optimized_count.value + retrieval + continuation
        fallback = None
        if bool(value.get("no_savings", False)) and total < baseline_count.value:
            total = baseline_count.value
            fallback = "baseline retained by no-savings control"
        reduction = (
            (baseline_count.value - total) / baseline_count.value * 100
            if baseline_count.value
            else 0.0
        )
        return ExecutableResult(
            str(value["id"]),
            revision,
            baseline_count,
            optimized_count,
            retrieval,
            continuation,
            total,
            reduction,
            baseline_validation.returncode == 0 and optimized_validation.returncode == 0,
            changed,
            unrelated,
            fallback,
        )


def _payloads(
    fixture: Path, value: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    files = [
        path
        for path in sorted(fixture.rglob("*"))
        if path.is_file()
        and path.name != str(value["patch"])
        and path.suffix != ".pyc"
        and not {".git", ".llmcut", "__pycache__"} & set(path.relative_to(fixture).parts)
    ]
    contexts = [
        Context(
            str(path.relative_to(fixture)).replace("/", ":"),
            _kind(path),
            path.read_text(errors="replace"),
            Retention.REQUIRED
            if str(path.relative_to(fixture)) in value.get("required", [])
            else Retention.RECOVERABLE,
            source_path=str(path.relative_to(fixture)),
        )
        for path in files
    ]
    request = ManagedRequest(
        "openai", str(value.get("model", "offline-model")), str(value["task"]), contexts
    )
    task = Context(
        "current-task",
        BlockKind.CURRENT_TASK,
        request.task,
        Retention.REQUIRED,
        100,
        "managed:task",
    ).to_block()
    baseline = CanonicalRequest(
        [item.to_block() for item in contexts] + [task], request.model_configuration()
    )
    required = set(str(item) for item in value.get("required", []))
    selected = [item.to_block() for item in contexts if item.source_path in required] + [task]
    optimized = CanonicalRequest(selected, request.model_configuration())
    adapter = OpenAIAdapter()
    continuation = adapter.to_native(baseline) if value.get("retrieve") else None
    return adapter.to_native(baseline), adapter.to_native(optimized), continuation


def _kind(path: Path) -> BlockKind:
    if "test" in path.name.lower():
        return BlockKind.TEST
    if path.suffix in {".toml", ".json", ".yaml", ".yml"}:
        return BlockKind.CONFIGURATION
    if path.suffix in {".md", ".txt", ".log"}:
        return BlockKind.DOCUMENT
    return BlockKind.SOURCE


def _tree_digest(root: Path) -> str:
    chunks = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            chunks.append(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest_bytes(b"\0".join(chunks))


def _run(argv: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.lower().endswith(("token", "key", "secret", "password"))
    }
    result = subprocess.run(
        argv, cwd=cwd, text=True, capture_output=True, timeout=120, check=False, env=environment
    )
    if check and result.returncode:
        raise ValueError(f"command failed ({argv[0]}): {result.stderr}")
    return result


def release_statistics(results: list[ExecutableResult]) -> dict[str, Any]:
    eligible = [item for item in results if item.eligible]
    excluded = [item for item in results if not item.eligible]
    reductions = [item.reduction_percent for item in eligible]
    positive = [value for value in reductions if value > 0]
    quartiles = quantiles(reductions, n=4, method="inclusive") if len(reductions) > 1 else [0.0] * 3
    return {
        "eligible_cases": len(eligible),
        "excluded_cases": [
            {"task_id": item.task_id, "reason": "validation or unrelated changes"}
            for item in excluded
        ],
        "saving_cases": len(positive),
        "no_savings_cases": sum(value == 0 for value in reductions),
        "negative_cases": sum(value < 0 for value in reductions),
        "median_reduction_all_eligible": median(reductions) if reductions else 0.0,
        "median_reduction_positive": median(positive) if positive else 0.0,
        "p25_reduction": quartiles[0],
        "p75_reduction": quartiles[2],
        "maximum_reduction": max(reductions, default=0.0),
        "quality_success_rate": len(eligible) / len(results) if results else 0.0,
        "passed": len(eligible) >= 5
        and len(positive) >= 4
        and median(reductions) >= 20
        and median(positive) >= 25
        and not excluded,
    }
