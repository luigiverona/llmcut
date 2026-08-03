from __future__ import annotations

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
    started = time.perf_counter()
    optimized, usage = execute(optimized_request)
    optimized_latency = time.perf_counter() - started
    parity = all(optimized.get(key) == value for key, value in case.expected_invariants.items())
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
    )
