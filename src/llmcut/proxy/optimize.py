from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from llmcut.adapters.base import ProviderAdapter
from llmcut.core.optimize import Optimizer
from llmcut.policy import OptimizationMode, Policy
from llmcut.report import OptimizationReport
from llmcut.tokens.base import TokenCounter
from llmcut.tokens.estimate import ConservativeEstimator


@dataclass(slots=True)
class NativeOptimization:
    body: bytes
    status: str
    endpoint_format: str
    original_tokens: int
    attempted_tokens: int
    effective_tokens: int
    count_quality: str
    fallback_reason: str | None
    omitted_blocks: int
    duration_seconds: float
    report: OptimizationReport | None = None


def optimize_native(
    body: bytes,
    adapter: ProviderAdapter,
    endpoint_format: str,
    optimizer: Optimizer,
    mode: OptimizationMode,
    counter: TokenCounter | None = None,
) -> NativeOptimization:
    started = time.perf_counter()
    counter = counter or ConservativeEstimator()
    original_count = counter.count(body.decode("utf-8", errors="strict"))

    def fallback(status: str, reason: str, attempted: int | None = None) -> NativeOptimization:
        return NativeOptimization(
            body,
            status,
            endpoint_format,
            original_count.value,
            attempted if attempted is not None else original_count.value,
            original_count.value,
            original_count.quality.value,
            reason,
            0,
            time.perf_counter() - started,
        )

    try:
        payload = _bounded_json(body)
        adapter.validate_native(payload)
        canonical = adapter.from_native(payload)
        reconstructed = adapter.to_native(canonical)
        if not adapter.semantically_equal(payload, reconstructed):
            return fallback("restored", "native round-trip was not semantically equivalent")
        optimized, report = optimizer.optimize(canonical, Policy(mode=mode))
        candidate = adapter.to_native(optimized)
        candidate_body = json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode()
        attempted = counter.count(candidate_body.decode()).value
        if not _safe_candidate(payload, candidate, endpoint_format):
            result = fallback(
                "restored", "optimized reconstruction failed structural safety checks", attempted
            )
            result.report = report
            return result
        omitted = (
            len(canonical.blocks)
            + len(canonical.tools)
            - len(optimized.blocks)
            - len(optimized.tools)
        )
        if attempted >= original_count.value:
            result = fallback("unchanged", "optimized native request was not smaller", attempted)
            result.omitted_blocks = omitted
            result.report = report
            return result
        return NativeOptimization(
            candidate_body,
            "optimized",
            endpoint_format,
            original_count.value,
            attempted,
            attempted,
            original_count.quality.value,
            None,
            omitted,
            time.perf_counter() - started,
            report,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, ValueError):
        return fallback("restored", "request could not be safely parsed or optimized")


def _bounded_json(body: bytes, max_depth: int = 64) -> dict[str, Any]:
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("provider request must be a JSON object")
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > max_depth:
            raise ValueError("JSON nesting exceeds safety limit")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _safe_candidate(original: dict[str, Any], candidate: dict[str, Any], endpoint: str) -> bool:
    context_keys = {
        "chat-completions": {"messages", "tools"},
        "responses": {"input", "tools"},
        "messages": {"messages", "tools"},
        "generate-content": {"contents", "tools"},
    }[endpoint]
    if {key: value for key, value in original.items() if key not in context_keys} != {
        key: value for key, value in candidate.items() if key not in context_keys
    }:
        return False
    primary = {
        "chat-completions": "messages",
        "responses": "input",
        "messages": "messages",
        "generate-content": "contents",
    }[endpoint]
    if original.get(primary, []) != candidate.get(primary, []):
        return False
    return _exact_duplicate_reduction(
        original.get("tools", []), candidate.get("tools", []), endpoint
    )


def _exact_duplicate_reduction(original: Any, candidate: Any, endpoint: str) -> bool:
    if original == candidate:
        return True
    if not isinstance(original, list) or not isinstance(candidate, list):
        return False
    if endpoint == "generate-content":
        # Gemini groups functionDeclarations inside tool containers; regrouping is safe only when
        # every non-function tool is unchanged and declarations are exact first-occurrence dedupes.
        def flatten(items: list[Any]) -> tuple[list[Any], list[Any]]:
            declarations: list[Any] = []
            others: list[Any] = []
            for item in items:
                if not isinstance(item, dict):
                    others.append(item)
                    continue
                declarations.extend(item.get("functionDeclarations", []))
                extra = {key: value for key, value in item.items() if key != "functionDeclarations"}
                if extra:
                    others.append(extra)
            return declarations, others

        old, old_other = flatten(original)
        new, new_other = flatten(candidate)
        return old_other == new_other and new == _unique(old)
    return candidate == _unique(original)


def _unique(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    serialized: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, separators=(",", ":"))
        if key not in serialized:
            serialized.add(key)
            result.append(item)
    return result
