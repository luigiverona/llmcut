from __future__ import annotations

from dataclasses import dataclass

from llmcut.model import BlockKind, ContextBlock
from llmcut.policy import OptimizationMode, Policy


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    include: bool
    reason: str
    confidence: float


class SelectionPolicy:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def block(self, block: ContextBlock) -> PolicyDecision:
        return PolicyDecision(True, "strict mode retains all unique content", 1.0)

    def tool(self, block: ContextBlock) -> PolicyDecision:
        return PolicyDecision(True, "tool retained", 1.0)


class ParitySelectionPolicy(SelectionPolicy):
    def block(self, block: ContextBlock) -> PolicyDecision:
        if block.metadata.get("proven_redundant") is True:
            return PolicyDecision(False, "caller-proven redundancy with recoverable evidence", 1.0)
        if block.kind is BlockKind.CHECKPOINT and block.metadata.get("superseded") is True:
            return PolicyDecision(False, "verified superseded checkpoint", 1.0)
        if (
            block.kind is BlockKind.COMMAND_OUTPUT
            and block.metadata.get("represented_by_verified_structure") is True
        ):
            return PolicyDecision(
                False, "raw command output has verified structured representation", 1.0
            )
        return PolicyDecision(True, "parity policy found no proven recoverable omission", 0.5)


class ExtremeSelectionPolicy(ParitySelectionPolicy):
    def block(self, block: ContextBlock) -> PolicyDecision:
        parity = super().block(block)
        if not parity.include:
            return parity
        if (
            block.kind is BlockKind.REPOSITORY
            and block.metadata.get("task_irrelevant") is True
            and block.metadata.get("confidence") == "high"
            and block.reference is not None
        ):
            return PolicyDecision(False, "deterministically task-irrelevant repository range", 0.95)
        return PolicyDecision(True, "extreme policy failed open on uncertain content", 0.5)

    def tool(self, block: ContextBlock) -> PolicyDecision:
        if (
            block.metadata.get("task_irrelevant") is True
            and block.metadata.get("confidence") == "high"
            and block.reference is not None
        ):
            return PolicyDecision(False, "task-scoped tool omitted with immediate recovery", 0.95)
        return PolicyDecision(True, "tool relevance uncertain; retained", 0.5)


def selection_policy(policy: Policy) -> SelectionPolicy:
    if policy.mode is OptimizationMode.EXTREME:
        return ExtremeSelectionPolicy(policy)
    if policy.mode is OptimizationMode.PARITY:
        return ParitySelectionPolicy(policy)
    return SelectionPolicy(policy)
