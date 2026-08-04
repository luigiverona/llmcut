from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median, quantiles
from typing import Any

from llmcut.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from llmcut.managed.protocol import Context, ManagedRequest
from llmcut.managed.runtime import ManagedRuntime, ProviderCall
from llmcut.measurement import TokenMeasurement, count_payload
from llmcut.model import BlockKind, CanonicalRequest, Retention
from llmcut.tokens.estimate import ConservativeEstimator
from llmcut.tokens.registry import CounterRegistry


@dataclass(slots=True)
class ManagedEvaluation:
    task_id: str
    baseline_tokens: int
    initial_tokens: int
    retrieval_tokens: int
    continuation_tokens: int
    total_tokens: int
    output_tokens: int
    cached_tokens: int
    retrievals: int
    turns: int
    quality_passed: bool
    fallback: str | None
    baseline_measurement: TokenMeasurement | None = None
    managed_measurements: tuple[TokenMeasurement, ...] = ()
    exclusion_reason: str | None = None

    @property
    def reduction_percent(self) -> float:
        return (
            (self.baseline_tokens - self.total_tokens) / self.baseline_tokens * 100
            if self.baseline_tokens
            else 0.0
        )

    @property
    def saving(self) -> bool:
        return self.quality_passed and self.total_tokens < self.baseline_tokens

    @property
    def eligible(self) -> bool:
        return (
            self.quality_passed
            and self.exclusion_reason is None
            and (self.baseline_measurement is None or self.baseline_measurement.eligible)
            and all(item.eligible for item in self.managed_measurements)
        )


async def evaluate_managed(
    task_id: str,
    request: ManagedRequest,
    runtime: ManagedRuntime,
    provider_call: ProviderCall,
    *,
    expected_output: Any | None = None,
    required_facts: tuple[str, ...] = (),
) -> ManagedEvaluation:
    full_blocks = [item.to_block() for item in request.context]
    full_blocks.append(
        Context(
            "current-task",
            BlockKind.CURRENT_TASK,
            request.task,
            Retention.REQUIRED,
            100,
            "managed:task",
        ).to_block()
    )
    full = CanonicalRequest(full_blocks, request.model_configuration(), [])
    native = _adapter(request.provider).to_native(full)
    baseline_response = await provider_call(
        request.provider, native, request.execution.timeout_seconds
    )
    baseline_output = _output(request.provider, baseline_response)
    baseline_measurement = count_payload(
        CounterRegistry(estimate=ConservativeEstimator()), request.provider, request.model, native
    )
    baseline_tokens = baseline_measurement.value
    managed = await runtime.run(request)
    quality = _normalize(baseline_output) == _normalize(managed.output)
    if expected_output is not None:
        quality = quality and _normalize(managed.output) == _normalize(expected_output)
    rendered = json.dumps(managed.output, sort_keys=True)
    quality = quality and all(fact in rendered for fact in required_facts)
    usage = managed.usage
    payload_values = [item.value for item in managed.payload_measurements]
    initial_payload = payload_values[0] if payload_values else usage.initial_input_tokens
    continuation_payload = sum(payload_values[1:])
    return ManagedEvaluation(
        task_id,
        baseline_tokens,
        initial_payload,
        usage.retrieval_request_tokens + usage.retrieval_result_tokens,
        continuation_payload,
        initial_payload + continuation_payload,
        usage.output_tokens,
        usage.cached_tokens,
        len(managed.retrievals),
        managed.turns,
        quality,
        managed.fallback,
        baseline_measurement,
        managed.payload_measurements,
    )


def release_targets(results: list[ManagedEvaluation]) -> dict[str, Any]:
    eligible = [item for item in results if item.eligible]
    excluded = [item for item in results if not item.eligible]
    saving = [item for item in eligible if item.saving]
    no_saving = [item for item in eligible if not item.saving]
    quality = all(item.quality_passed for item in results)
    all_reductions = [item.reduction_percent for item in eligible]
    saving_reductions = [item.reduction_percent for item in saving]
    quartiles = (
        quantiles(all_reductions, n=4, method="inclusive") if len(all_reductions) > 1 else [0.0] * 3
    )
    return {
        "eligible_cases": len(eligible),
        "excluded_cases": [
            {"task_id": item.task_id, "reason": item.exclusion_reason or "quality failed"}
            for item in excluded
        ],
        "positive_saving_cases": len(saving),
        "no_savings_cases": len(no_saving),
        "regressions": sum(not item.quality_passed for item in results),
        "median_reduction_across_all_eligible_cases": median(all_reductions)
        if all_reductions
        else 0.0,
        "median_reduction_across_saving_cases": median(saving_reductions)
        if saving_reductions
        else 0.0,
        "p25_reduction": quartiles[0],
        "p75_reduction": quartiles[2],
        "maximum_reduction": max(all_reductions, default=0.0),
        "all_quality_passed": quality,
        "passed": len(eligible) >= 5
        and len(saving) >= 4
        and median(all_reductions) >= 20
        and bool(saving_reductions)
        and median(saving_reductions) >= 25
        and quality,
    }


async def evaluate_recorded_corpus(path: Path, runtime_factory: Any) -> list[ManagedEvaluation]:
    results: list[ManagedEvaluation] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        try:
            request = ManagedRequest.from_dict(value["managed_request"])
            responses = [
                value["recorded_baseline_response"],
                *value["recorded_managed_responses"],
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid managed corpus record at line {number}: {exc}") from exc

        async def recorded(
            _: str,
            __: dict[str, Any],
            ___: float,
            queue: list[Any] = responses,
        ) -> dict[str, Any]:
            if not queue:
                raise ValueError("recorded managed corpus exhausted provider responses")
            response = queue.pop(0)
            if not isinstance(response, dict):
                raise ValueError("recorded provider response must be an object")
            return response

        runtime = runtime_factory(recorded)
        results.append(
            await evaluate_managed(
                str(value["task_id"]),
                request,
                runtime,
                recorded,
                expected_output=value.get("expected_output"),
                required_facts=tuple(value.get("required_facts", ())),
            )
        )
    return results


def _adapter(provider: str) -> Any:
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider == "gemini":
        return GeminiAdapter()
    return OpenAIAdapter()


def _output(provider: str, response: dict[str, Any]) -> Any:
    if provider == "openai":
        return ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    if provider == "anthropic":
        return "".join(
            str(item.get("text", ""))
            for item in response.get("content", [])
            if item.get("type") == "text"
        )
    parts = ((response.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(str(item.get("text", "")) for item in parts)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    return json.loads(json.dumps(value, sort_keys=True))
