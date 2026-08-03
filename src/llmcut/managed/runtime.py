from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from llmcut.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from llmcut.errors import ExecutionError, RetrievalError
from llmcut.managed.planner import ContextPlan, ContextPlanner
from llmcut.managed.protocol import ManagedRequest
from llmcut.managed.retrieval import RetrievalResult, RetrievalService
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock
from llmcut.store.evidence import EvidenceStore
from llmcut.tokens.base import TokenCounter
from llmcut.tokens.estimate import ConservativeEstimator

ProviderCall = Callable[[str, dict[str, Any], float], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ManagedUsage:
    baseline_input_tokens: int = 0
    initial_input_tokens: int = 0
    retrieval_request_tokens: int = 0
    retrieval_result_tokens: int = 0
    continuation_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_input_tokens: int = 0
    count_quality: str = "estimated"


@dataclass(slots=True)
class RetrievalEvent:
    operation: str
    context_id: str
    source: str
    digest: str
    result_tokens: int
    cached: bool


@dataclass(slots=True)
class ManagedResult:
    run_id: str
    status: str
    provider: str
    model: str
    output: Any = None
    usage: ManagedUsage = field(default_factory=ManagedUsage)
    turns: int = 0
    retrievals: tuple[RetrievalEvent, ...] = ()
    fallback: str | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    planning_seconds: float = 0.0
    provider_seconds: float = 0.0

    @property
    def saving(self) -> bool:
        return self.usage.total_input_tokens < self.usage.baseline_input_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"saving": self.saving}


class ManagedRuntime:
    def __init__(
        self,
        evidence: EvidenceStore,
        provider_call: ProviderCall | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self.evidence = evidence
        self.provider_call = provider_call
        self.counter = counter or ConservativeEstimator()
        self.planner = ContextPlanner(evidence, self.counter)

    async def plan(self, request: ManagedRequest) -> ContextPlan:
        return self.planner.plan(request)

    async def run(
        self,
        request: ManagedRequest,
        *,
        dry_run: bool = False,
        cancellation: asyncio.Event | None = None,
    ) -> ManagedResult:
        started = time.monotonic()
        plan = self.planner.plan(request)
        planning_seconds = time.monotonic() - started
        provider_seconds = 0.0
        usage = ManagedUsage(
            baseline_input_tokens=plan.baseline_tokens,
            initial_input_tokens=plan.initial_tokens,
            total_input_tokens=plan.initial_tokens,
            count_quality=plan.count_quality,
        )
        identifier = uuid.uuid4().hex
        if dry_run:
            return ManagedResult(
                identifier,
                "planned",
                request.provider,
                request.model,
                usage=usage,
                plan=plan.diagnostic_dict(),
                fallback=plan.fallback,
                planning_seconds=planning_seconds,
            )
        if self.provider_call is None:
            raise ExecutionError("managed execution requires a configured provider transport")
        retrieval = RetrievalService(self.evidence, request, plan)
        working = CanonicalRequest.from_dict(plan.request.to_dict())
        _add_retrieval_tools(working, plan.retrieval_operations, request.provider)
        events: list[RetrievalEvent] = []
        repeated: set[str] = set()
        output: Any = None
        for turn in range(1, request.execution.max_turns + 1):
            _check_bounds(request, started, cancellation)
            native = _adapter(request.provider).to_native(working)
            remaining = max(0.001, request.execution.timeout_seconds - (time.monotonic() - started))
            try:
                provider_started = time.monotonic()
                response = await asyncio.wait_for(
                    self.provider_call(request.provider, native, remaining), timeout=remaining
                )
                provider_seconds += time.monotonic() - provider_started
            except TimeoutError as exc:
                raise ExecutionError("managed execution timed out") from exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ExecutionError("managed provider request failed") from exc
            provider_usage = _adapter(request.provider).usage(response)
            input_tokens = (
                provider_usage["input_tokens"]
                or self.counter.count(
                    json.dumps(native, sort_keys=True, separators=(",", ":")), model=request.model
                ).value
            )
            if turn == 1:
                usage.initial_input_tokens = input_tokens
            else:
                usage.continuation_input_tokens += input_tokens
            usage.total_input_tokens = usage.initial_input_tokens + usage.continuation_input_tokens
            usage.output_tokens += provider_usage["output_tokens"]
            usage.reasoning_tokens += provider_usage["reasoning_tokens"]
            usage.cached_tokens += provider_usage["cached_tokens"]
            calls = _tool_calls(request.provider, response)
            if not calls:
                output = _output(request.provider, response)
                return ManagedResult(
                    identifier,
                    "completed",
                    request.provider,
                    request.model,
                    output,
                    usage,
                    turn,
                    tuple(events),
                    plan.fallback,
                    plan.diagnostic_dict(),
                    planning_seconds,
                    provider_seconds,
                )
            working.blocks.append(_assistant_block(request.provider, response, turn))
            for call_id, operation, arguments in calls:
                call_key = (
                    operation + ":" + json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                )
                if call_key in repeated:
                    raise ExecutionError("repeated identical retrieval request detected")
                repeated.add(call_key)
                request_tokens = self.counter.count(call_key, model=request.model).value
                usage.retrieval_request_tokens += request_tokens
                try:
                    result = retrieval.execute(operation, arguments)
                except RetrievalError as exc:
                    raise ExecutionError(f"invalid managed retrieval: {exc}") from exc
                result_tokens = self.counter.count(
                    result.model_content(), model=request.model
                ).value
                usage.retrieval_result_tokens += result_tokens
                events.append(_event(result, result_tokens))
                working.blocks.append(
                    _retrieval_block(request.provider, result, call_id, operation, len(events))
                )
                if (
                    sum(
                        len(item.content.encode())
                        for item in working.blocks
                        if item.kind is BlockKind.TOOL_RESULT
                    )
                    > request.execution.max_retrieval_bytes
                ):
                    raise ExecutionError("maximum managed retrieval volume exceeded")
            if (
                request.execution.max_total_tokens is not None
                and usage.total_input_tokens > request.execution.max_total_tokens
            ):
                raise ExecutionError("managed execution token bound exceeded")
        raise ExecutionError("managed execution turn limit reached")


def _adapter(provider: str) -> Any:
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider == "gemini":
        return GeminiAdapter()
    if provider == "openai":
        return OpenAIAdapter()
    raise ExecutionError(f"unsupported configured provider: {provider}")


def _add_retrieval_tools(
    request: CanonicalRequest, operations: tuple[str, ...], provider: str
) -> None:
    for operation in operations:
        schema = {
            "name": operation,
            "description": f"Retrieve exact locally retained evidence using {operation}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "maxLength": 128},
                    "name": {"type": "string", "maxLength": 128},
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                    "pattern": {"type": "string", "maxLength": 256},
                    "symbol": {"type": "string", "maxLength": 256},
                    "dependency": {"type": "string", "maxLength": 128},
                },
                "additionalProperties": False,
            },
        }
        if provider == "anthropic":
            value = {
                "name": operation,
                "description": schema["description"],
                "input_schema": schema["parameters"],
            }
        elif provider == "openai":
            value = {"type": "function", "function": schema}
        else:
            value = schema
        request.tools.append(
            ContextBlock(
                f"retrieval-tool:{operation}",
                BlockKind.TOOL_DEFINITION,
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                "managed:retrieval-tools",
            )
        )


def _tool_calls(provider: str, response: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    if provider == "openai":
        message = (response.get("choices") or [{}])[0].get("message") or {}
        for item in message.get("tool_calls", []):
            function = item.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ExecutionError("malformed provider tool-call arguments") from exc
            if not isinstance(arguments, dict):
                raise ExecutionError("provider tool-call arguments must be an object")
            calls.append((str(item.get("id", "")), str(function.get("name", "")), arguments))
    elif provider == "anthropic":
        for item in response.get("content", []):
            if item.get("type") == "tool_use":
                arguments = item.get("input", {})
                if not isinstance(arguments, dict):
                    raise ExecutionError("provider tool-call arguments must be an object")
                calls.append((str(item.get("id", "")), str(item.get("name", "")), arguments))
    else:
        parts = ((response.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        for index, item in enumerate(parts):
            if "functionCall" in item:
                call = item["functionCall"]
                arguments = call.get("args", {})
                if not isinstance(arguments, dict):
                    raise ExecutionError("provider tool-call arguments must be an object")
                calls.append(
                    (str(call.get("id", f"call-{index}")), str(call.get("name", "")), arguments)
                )
    if any(not call_id or not name for call_id, name, _ in calls):
        raise ExecutionError("malformed provider retrieval call")
    return calls


def _output(provider: str, response: dict[str, Any]) -> Any:
    if provider == "openai":
        return ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    if provider == "anthropic":
        content = response.get("content", [])
        return "".join(str(item.get("text", "")) for item in content if item.get("type") == "text")
    parts = ((response.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(str(item.get("text", "")) for item in parts)


def _assistant_block(provider: str, response: dict[str, Any], turn: int) -> ContextBlock:
    if provider == "openai":
        message = dict((response.get("choices") or [{}])[0].get("message") or {})
        content = message.pop("content", None)
        message.pop("role", None)
        return ContextBlock(
            f"assistant:{turn}",
            BlockKind.ASSISTANT,
            content or "",
            "managed:provider",
            metadata={"native_content": content, **message},
        )
    if provider == "anthropic":
        content = response.get("content", [])
        return ContextBlock(
            f"assistant:{turn}",
            BlockKind.ASSISTANT,
            json.dumps(content, sort_keys=True, separators=(",", ":")),
            "managed:provider",
            metadata={"native_content": content},
        )
    parts = ((response.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return ContextBlock(
        f"assistant:{turn}",
        BlockKind.ASSISTANT,
        json.dumps(parts, sort_keys=True, separators=(",", ":")),
        "managed:provider",
        metadata={"native_parts": parts},
    )


def _retrieval_block(
    provider: str,
    result: RetrievalResult,
    call_id: str,
    operation: str,
    number: int,
) -> ContextBlock:
    metadata: dict[str, Any] = {"tool_call_id": call_id, "name": operation}
    if provider == "anthropic":
        metadata["native_content"] = [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": result.model_content(),
            }
        ]
    elif provider == "gemini":
        metadata["native_parts"] = [
            {
                "functionResponse": {
                    "name": operation,
                    "response": {"content": result.model_content()},
                }
            }
        ]
    return ContextBlock(
        f"retrieval:{number}",
        BlockKind.TOOL_RESULT,
        result.model_content(),
        result.source,
        metadata=metadata,
    )


def _event(result: RetrievalResult, tokens: int) -> RetrievalEvent:
    return RetrievalEvent(
        result.operation, result.context_id, result.source, result.digest, tokens, result.cached
    )


def _check_bounds(
    request: ManagedRequest, started: float, cancellation: asyncio.Event | None
) -> None:
    if cancellation is not None and cancellation.is_set():
        raise asyncio.CancelledError
    if time.monotonic() - started >= request.execution.timeout_seconds:
        raise ExecutionError("managed execution timed out")
