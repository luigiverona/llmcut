from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Decision:
    block_id: str
    included: bool
    reason: str
    confidence: float
    evidence_digest: str
    tokens: int
    token_quality: str


@dataclass(slots=True)
class OptimizationReport:
    mode: str
    decisions: list[Decision] = field(default_factory=list)
    duplicates: list[dict[str, str]] = field(default_factory=list)
    original_tokens: int = 0
    optimized_tokens: int = 0
    attempted_tokens: int = 0
    effective_tokens: int = 0
    optimization_overhead_tokens: int = 0
    fallback_reason: str | None = None
    restoration_overhead_tokens: int = 0
    count_quality: str = "estimated"
    fallback: str = "full_context"
    stable_prefix_digest: str = ""
    stable_tokens: int = 0
    dynamic_tokens: int = 0
    potential_cacheable_tokens: int = 0
    actual_cached_tokens: int | None = None
    parity_policy_enforced: bool = True
    outcome_parity_established: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
