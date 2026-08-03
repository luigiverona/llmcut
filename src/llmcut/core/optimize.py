from __future__ import annotations

from dataclasses import asdict

from llmcut.core.dedupe import deduplicate
from llmcut.core.pack import stable_partition
from llmcut.core.policies import selection_policy
from llmcut.model import CanonicalRequest, ContextBlock
from llmcut.policy import Policy
from llmcut.report import Decision, OptimizationReport
from llmcut.store.evidence import EvidenceStore
from llmcut.tokens.base import TokenCounter
from llmcut.tokens.estimate import ConservativeEstimator


class Optimizer:
    def __init__(self, evidence: EvidenceStore, counter: TokenCounter | None = None) -> None:
        self.evidence = evidence
        self.counter = counter or ConservativeEstimator()

    def optimize(
        self, request: CanonicalRequest, policy: Policy | None = None
    ) -> tuple[CanonicalRequest, OptimizationReport]:
        policy = policy or Policy()
        policy.validate()
        original_json = request.to_json()
        working = CanonicalRequest.from_dict(request.to_dict())
        original_ref = self.evidence.put(
            original_json, f"request:{request.request_id or 'anonymous'}"
        )
        kept, duplicates = deduplicate(working.blocks)
        tools, tool_duplicates = deduplicate(working.tools)
        report = OptimizationReport(mode=policy.mode.value, fallback=policy.fallback)
        report.duplicates = [asdict(item) for item in [*duplicates, *tool_duplicates]]
        selector = selection_policy(policy)
        selected: list[ContextBlock] = []
        for block in kept:
            block.reference = self.evidence.put(
                block.content, block.source, str(block.metadata.get("revision", "")) or None
            )
            block.tokens = self.counter.count(block.content, model=working.model.model)
            decision = selector.block(block)
            include, reason, confidence = decision.include, decision.reason, decision.confidence
            # Fail open: only a high-confidence redundancy decision may exclude content.
            if confidence < policy.confidence_threshold:
                include, reason = True, "low confidence: full content restored"
            if include:
                selected.append(block)
            report.decisions.append(
                Decision(
                    block.id,
                    include,
                    reason,
                    confidence,
                    block.reference.digest,
                    block.tokens.value,
                    block.tokens.quality.value,
                )
            )
        selected_tools: list[ContextBlock] = []
        for tool in tools:
            tool.reference = self.evidence.put(tool.content, tool.source)
            tool.tokens = self.counter.count(tool.content, model=working.model.model)
            decision = selector.tool(tool)
            if decision.confidence < policy.confidence_threshold or decision.include:
                selected_tools.append(tool)
        optimized = CanonicalRequest(
            selected,
            working.model,
            selected_tools,
            dict(working.cache),
            dict(working.passthrough),
            working.request_id,
        )
        optimized.passthrough["llmcut_original"] = original_ref.digest
        original_count = self.counter.count(original_json, model=working.model.model)
        optimized_count = self.counter.count(optimized.to_json(), model=working.model.model)
        report.original_tokens = original_count.value
        report.optimized_tokens = optimized_count.value
        report.attempted_tokens = optimized_count.value
        report.effective_tokens = optimized_count.value
        report.optimization_overhead_tokens = max(0, optimized_count.value - original_count.value)
        report.count_quality = optimized_count.quality.value
        partitions, report.stable_prefix_digest = stable_partition(optimized)
        report.stable_tokens = self.counter.count("".join(partitions[:3])).value
        report.dynamic_tokens = self.counter.count("".join(partitions[3:])).value
        report.potential_cacheable_tokens = report.stable_tokens
        return optimized, report
