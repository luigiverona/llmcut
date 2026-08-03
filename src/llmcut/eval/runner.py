from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llmcut.core.optimize import Optimizer
from llmcut.eval.corpus import CorpusCase
from llmcut.model import CanonicalRequest
from llmcut.policy import Policy

Executor = Callable[[CanonicalRequest], tuple[dict[str, Any], dict[str, int]]]


@dataclass(slots=True)
class EvaluationResult:
    task_id: str
    baseline_complete: bool
    optimized_complete: bool
    invariant_parity: bool
    settings_identical: bool
    baseline_input_tokens: int
    optimized_input_tokens: int
    output_tokens: int
    cached_tokens: int
    recovery_tokens: int
    retries: int
    baseline_latency: float
    optimized_latency: float
    regression: bool
    attempted_input_tokens: int = 0
    effective_input_tokens: int = 0
    fallback_reason: str | None = None
    evaluator_passed: bool = True


def run_case(
    case: CorpusCase, optimizer: Optimizer, execute: Executor, policy: Policy | None = None
) -> EvaluationResult:
    started = time.perf_counter()
    baseline, baseline_usage = execute(case.request)
    baseline_latency = time.perf_counter() - started
    optimized_request, report = optimizer.optimize(case.request, policy)
    if (
        case.request.model.provider,
        case.request.model.model,
        case.request.model.parameters,
        case.request.model.reasoning,
    ) != (
        optimized_request.model.provider,
        optimized_request.model.model,
        optimized_request.model.parameters,
        optimized_request.model.reasoning,
    ):
        raise ValueError("optimizer changed provider/model/settings")
    attempted_tokens = report.optimized_tokens
    fallback_reason = None
    execution_request = optimized_request
    if report.optimized_tokens >= report.original_tokens:
        execution_request = case.request
        fallback_reason = "attempted optimization was not smaller"
    started = time.perf_counter()
    optimized, usage = execute(execution_request)
    optimized_latency = time.perf_counter() - started
    parity = _normalize(baseline) == _normalize(optimized) and all(
        optimized.get(key) == value for key, value in case.expected_invariants.items()
    )
    if case.expected_output is not None:
        parity = parity and _normalize(optimized) == _normalize(case.expected_output)
    complete = bool(optimized.get("complete", True))
    baseline_complete = bool(baseline.get("complete", True))
    return EvaluationResult(
        case.task_id,
        baseline_complete,
        complete,
        parity,
        True,
        baseline_usage.get("input_tokens", report.original_tokens),
        usage.get("input_tokens", report.optimized_tokens),
        usage.get("output_tokens", 0),
        usage.get("cached_tokens", 0),
        usage.get("recovery_tokens", 0),
        usage.get("retries", 0),
        baseline_latency,
        optimized_latency,
        baseline_complete and (not complete or not parity),
        attempted_tokens,
        usage.get(
            "input_tokens", report.original_tokens if fallback_reason else report.optimized_tokens
        ),
        fallback_reason,
    )


def _normalize(value: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value, sort_keys=True))
    if not isinstance(normalized, dict):
        raise TypeError("evaluation response must be an object")
    return normalized
