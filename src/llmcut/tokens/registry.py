from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any

from llmcut.model import CountQuality, TokenCount, digest_bytes
from llmcut.tokens.base import TokenCounter
from llmcut.tokens.estimate import ConservativeEstimator

CountEndpoint = Callable[[dict[str, Any]], int]


@dataclass(slots=True)
class CounterRegistry:
    """Select endpoint, official, compatible, then conservative counts."""

    estimate: TokenCounter = field(default_factory=ConservativeEstimator)
    endpoint_timeout: float = 2.0
    _endpoints: dict[str, CountEndpoint] = field(default_factory=dict)
    _tokenizers: dict[tuple[str, str], TokenCounter] = field(default_factory=dict)
    _compatible: dict[str, TokenCounter] = field(default_factory=dict)
    _cache: dict[str, TokenCount] = field(default_factory=dict)
    version: str = "counter-registry-v1"

    def register_endpoint(self, provider: str, endpoint: CountEndpoint) -> None:
        self._endpoints[provider] = endpoint

    def register_official_tokenizer(self, provider: str, model: str, counter: TokenCounter) -> None:
        self._tokenizers[(provider, model)] = counter

    def register_compatible(self, provider: str, counter: TokenCounter) -> None:
        self._compatible[provider] = counter

    def count_transport(self, provider: str, model: str, payload: dict[str, Any]) -> TokenCount:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = digest_bytes(f"{provider}\0{model}\0{serialized}".encode())
        if key in self._cache:
            return self._cache[key]
        count = self._provider_count(provider, payload)
        if count is None:
            counter = self._tokenizers.get((provider, model))
            if counter is not None:
                measured = counter.count(serialized, model=model)
                count = TokenCount(
                    measured.value, CountQuality.TOKENIZER_DERIVED, "official model tokenizer"
                )
            elif provider in self._compatible:
                measured = self._compatible[provider].count(serialized, model=model)
                count = TokenCount(
                    measured.value,
                    CountQuality.TOKENIZER_DERIVED,
                    "documented compatible tokenizer",
                )
            else:
                count = self.estimate.count(serialized, model=model)
        self._cache[key] = count
        return count

    def _provider_count(self, provider: str, payload: dict[str, Any]) -> TokenCount | None:
        endpoint = self._endpoints.get(provider)
        if endpoint is None:
            return None
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(endpoint, payload)
            try:
                value = future.result(timeout=self.endpoint_timeout)
            except (TimeoutError, OSError, ValueError):
                future.cancel()
                return None
        if value < 0:
            return None
        return TokenCount(value, CountQuality.PROVIDER_REPORTED, "provider count endpoint")
