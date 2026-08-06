from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int
    delay_seconds: float


DEFAULT_POLICY: RetryPolicy = {"attempts": 3, "delay_seconds": 0.25}


def next_delay(policy: RetryPolicy, attempt: int) -> float:
    return policy.delay_seconds * attempt
