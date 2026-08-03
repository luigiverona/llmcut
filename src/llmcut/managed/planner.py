from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from llmcut.managed.protocol import Context, ManagedRequest, ToolDefinition
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, Retention, digest_bytes
from llmcut.policy import OptimizationMode
from llmcut.store.evidence import EvidenceStore
from llmcut.tokens.base import TokenCounter
from llmcut.tokens.estimate import ConservativeEstimator


@dataclass(frozen=True, slots=True)
class PlanDecision:
    context_id: str
    included: bool
    reason: str
    confidence: float


@dataclass(slots=True)
class ContextPlan:
    request: CanonicalRequest
    deferred: tuple[str, ...]
    stable_prefix: bytes
    dynamic_suffix: bytes
    retrieval_operations: tuple[str, ...]
    selected_tools: tuple[str, ...]
    deferred_tools: tuple[str, ...]
    decisions: tuple[PlanDecision, ...]
    baseline_tokens: int
    initial_tokens: int
    count_quality: str
    stable_tokens: int = 0
    dynamic_tokens: int = 0
    fallback: str | None = None
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def stable_prefix_digest(self) -> str:
        return digest_bytes(self.stable_prefix)

    def diagnostic_dict(self) -> dict[str, Any]:
        return {
            "initial_context_ids": [item.id for item in self.request.blocks],
            "deferred_context_ids": list(self.deferred),
            "selected_tools": list(self.selected_tools),
            "deferred_tools": list(self.deferred_tools),
            "retrieval_operations": list(self.retrieval_operations),
            "decisions": [asdict(item) for item in self.decisions],
            "baseline_tokens": self.baseline_tokens,
            "initial_tokens": self.initial_tokens,
            "count_quality": self.count_quality,
            "stable_prefix_digest": self.stable_prefix_digest,
            "stable_prefix_tokens": self.stable_tokens,
            "dynamic_suffix_tokens": self.dynamic_tokens,
            "fallback": self.fallback,
        }


class ContextPlanner:
    def __init__(self, evidence: EvidenceStore, counter: TokenCounter | None = None) -> None:
        self.evidence = evidence
        self.counter = counter or ConservativeEstimator()

    def plan(self, request: ManagedRequest, mode: OptimizationMode | None = None) -> ContextPlan:
        request.validate()
        mode = mode or request.execution.optimization
        contexts = list(request.context)
        by_id = {item.id: item for item in contexts}
        selected: set[str] = set()
        reasons: dict[str, tuple[str, float]] = {}
        task_terms = _terms(request.task)
        open_calls: dict[str, set[str]] = {}
        for item in contexts:
            if item.kind is BlockKind.TOOL_CALL and item.tool_call_id:
                open_calls[item.tool_call_id] = {item.id}
            elif item.kind is BlockKind.TOOL_RESULT and item.tool_call_id in open_calls:
                open_calls[item.tool_call_id].add(item.id)
                open_calls.pop(item.tool_call_id)
        active_ids = set().union(*open_calls.values()) if open_calls else set()
        for item in contexts:
            required = item.retention in {Retention.REQUIRED, Retention.STABLE, Retention.EPHEMERAL}
            named = bool(task_terms & (_terms(item.source_path or "") | _terms(item.id)))
            current = item.kind is BlockKind.CURRENT_TASK
            continuity = item.id in active_ids
            if required or current or continuity:
                selected.add(item.id)
                reasons[item.id] = ("required policy, task, or tool continuity", 1.0)
            elif named:
                selected.add(item.id)
                reasons[item.id] = ("directly named by current task", 0.95)
            elif mode is OptimizationMode.STRICT:
                selected.add(item.id)
                reasons[item.id] = ("strict mode retains all context", 1.0)
            elif mode is OptimizationMode.PARITY and item.priority >= 50:
                selected.add(item.id)
                reasons[item.id] = ("parity mode retained medium/high priority context", 0.85)
            elif item.retention is Retention.REDUNDANT:
                reasons[item.id] = ("caller-declared redundant; exact evidence retained", 1.0)
            elif item.retention is Retention.SUPERSEDED:
                reasons[item.id] = ("superseded context deferred behind verified evidence", 0.95)
            else:
                reasons[item.id] = ("recoverable secondary evidence deferred", 0.9)

        # Dependencies of selected content are included transitively unless explicitly recoverable;
        # recoverable dependencies remain available via dependency.get.
        changed = True
        while changed:
            changed = False
            for identifier in tuple(selected):
                for dependency in by_id[identifier].dependencies:
                    item = by_id[dependency]
                    if dependency not in selected and item.retention is not Retention.RECOVERABLE:
                        selected.add(dependency)
                        reasons[dependency] = (f"dependency of {identifier}", 1.0)
                        changed = True

        blocks: list[ContextBlock] = []
        evidence: dict[str, str] = {}
        for item in contexts:
            reference = self.evidence.put(
                item.content,
                item.source_path or f"managed:{item.id}",
                item.revision,
                {"context_id": item.id, "kind": item.kind.value},
            )
            evidence[item.id] = reference.digest
            if self.evidence.get(reference.digest) != item.content:
                selected.add(item.id)
                reasons[item.id] = (
                    "exact retrieval unavailable after persistence redaction; retained",
                    1.0,
                )
            block = item.to_block()
            block.reference = reference
            if item.id in selected:
                blocks.append(block)
        task = Context(
            "current-task",
            BlockKind.CURRENT_TASK,
            request.task,
            Retention.REQUIRED,
            100,
            "managed:task",
        ).to_block()
        blocks.append(task)

        selected_tools, deferred_tools = _select_tools(request.tools, request.task)
        tools = [_tool_block(item) for item in selected_tools]
        canonical = CanonicalRequest(blocks, request.model_configuration(), tools)
        stable_blocks = [
            item
            for item in blocks
            if item.retention is Retention.STABLE
            or item.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
        ]
        dynamic_blocks = [item for item in blocks if item not in stable_blocks]
        stable_prefix = _serialize_blocks(stable_blocks, tools)
        dynamic_suffix = _serialize_blocks(dynamic_blocks, [])
        stable_count = self.counter.count(stable_prefix.decode(), model=request.model).value
        dynamic_count = self.counter.count(dynamic_suffix.decode(), model=request.model).value
        all_blocks = [item.to_block() for item in contexts] + [task]
        baseline = CanonicalRequest(
            all_blocks, request.model_configuration(), [_tool_block(item) for item in request.tools]
        )
        baseline_count = self.counter.count(baseline.model_bound_json(), model=request.model)
        initial_count = self.counter.count(canonical.model_bound_json(), model=request.model)
        fallback = None
        if initial_count.value >= baseline_count.value:
            canonical = baseline
            blocks = all_blocks
            selected_tools = list(request.tools)
            deferred_tools = []
            selected = {item.id for item in contexts}
            initial_count = baseline_count
            fallback = "baseline request is not larger than managed plan"
        deferred = tuple(item.id for item in contexts if item.id not in selected)
        operations = _operations([by_id[item] for item in deferred], bool(deferred_tools))
        decisions = tuple(
            PlanDecision(item.id, item.id in selected, *reasons[item.id]) for item in contexts
        )
        return ContextPlan(
            canonical,
            deferred,
            stable_prefix,
            dynamic_suffix,
            operations,
            tuple(item.name for item in selected_tools),
            tuple(item.name for item in deferred_tools),
            decisions,
            baseline_count.value,
            initial_count.value,
            initial_count.quality.value,
            stable_count,
            dynamic_count,
            fallback,
            evidence,
        )


def _select_tools(
    tools: list[ToolDefinition], task: str
) -> tuple[list[ToolDefinition], list[ToolDefinition]]:
    if len(tools) <= 8:
        return list(tools), []
    terms = _terms(task)
    selected = [
        item
        for item in tools
        if item.required or bool(terms & (_terms(item.name) | _terms(item.category)))
    ]
    if not selected:
        selected = [item for item in tools if item.required]
    deferred = [item for item in tools if item not in selected]
    return selected, deferred


def _tool_block(tool: ToolDefinition) -> ContextBlock:
    return ContextBlock(
        f"tool:{tool.name}",
        BlockKind.TOOL_DEFINITION,
        json.dumps(tool.transport_dict(), sort_keys=True, separators=(",", ":")),
        "managed:tools",
        metadata={"category": tool.category},
        retention=Retention.REQUIRED if tool.required else Retention.RECOVERABLE,
    )


def _operations(deferred: list[Context], tools_deferred: bool) -> tuple[str, ...]:
    result = {"evidence.get", "context.expand"} if deferred else set()
    kinds = {item.kind for item in deferred}
    if kinds & {BlockKind.SOURCE, BlockKind.TEST, BlockKind.CONFIGURATION}:
        result.update({"source.range", "symbol.get", "dependency.get"})
    if BlockKind.COMMAND_OUTPUT in kinds:
        result.update({"log.search", "log.range"})
    if BlockKind.REPOSITORY_MAP in kinds or BlockKind.REPOSITORY in kinds:
        result.add("repository.map")
    if tools_deferred:
        result.add("tool.discover")
    return tuple(sorted(result))


def _serialize_blocks(blocks: list[ContextBlock], tools: list[ContextBlock]) -> bytes:
    value = {
        "blocks": [
            {"id": item.id, "kind": item.kind.value, "content": item.content} for item in blocks
        ],
        "tools": [{"id": item.id, "content": item.content} for item in tools],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _terms(value: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", value)}
