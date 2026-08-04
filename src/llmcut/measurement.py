from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from llmcut.model import CountQuality, digest_bytes
from llmcut.tokens.registry import CounterRegistry


class MeasurementQuality(StrEnum):
    PROVIDER_REPORTED = "provider_reported"
    PROVIDER_COUNT_ENDPOINT = "provider_count_endpoint"
    OFFICIAL_TOKENIZER = "official_tokenizer"
    COMPATIBLE_TOKENIZER = "compatible_tokenizer"
    ESTIMATED = "estimated"


class MeasurementTrust(StrEnum):
    UNTRUSTED_FIXTURE = "untrusted_fixture"
    VERIFIED_CAPTURE = "verified_capture"
    LIVE_PROVIDER = "live_provider"
    LOCALLY_COUNTED = "locally_counted"


class MeasurementLayer(StrEnum):
    PAYLOAD = "payload"
    PROVIDER = "provider"
    AGENT = "agent"
    SUBSCRIPTION = "subscription"


@dataclass(frozen=True, slots=True)
class TokenMeasurement:
    value: int
    quality: MeasurementQuality
    source: str
    provider: str
    model: str
    request_digest: str
    response_digest: str | None
    counter_version: str
    timestamp: str
    trust: MeasurementTrust
    layer: MeasurementLayer

    @property
    def eligible(self) -> bool:
        return self.trust is not MeasurementTrust.UNTRUSTED_FIXTURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def request_digest(payload: dict[str, Any]) -> str:
    return digest_bytes(canonical_payload(payload))


def response_digest(payload: dict[str, Any]) -> str:
    return digest_bytes(canonical_payload(payload))


def count_payload(
    registry: CounterRegistry, provider: str, model: str, payload: dict[str, Any]
) -> TokenMeasurement:
    count = registry.count_transport(provider, model, payload)
    if count.quality is CountQuality.PROVIDER_REPORTED:
        quality = MeasurementQuality.PROVIDER_COUNT_ENDPOINT
        trust = MeasurementTrust.LIVE_PROVIDER
    elif count.quality is CountQuality.TOKENIZER_DERIVED:
        quality = (
            MeasurementQuality.COMPATIBLE_TOKENIZER
            if "compatible" in count.method
            else MeasurementQuality.OFFICIAL_TOKENIZER
        )
        trust = MeasurementTrust.LOCALLY_COUNTED
    else:
        quality = MeasurementQuality.ESTIMATED
        trust = MeasurementTrust.LOCALLY_COUNTED
    return TokenMeasurement(
        count.value,
        quality,
        count.method,
        provider,
        model,
        request_digest(payload),
        None,
        registry.version,
        datetime.now(UTC).isoformat(),
        trust,
        MeasurementLayer.PAYLOAD,
    )


def provider_measurement(
    *,
    value: int,
    provider: str,
    model: str,
    request: dict[str, Any],
    response: dict[str, Any],
    trust: MeasurementTrust,
    source: str = "provider response usage",
) -> TokenMeasurement:
    return TokenMeasurement(
        value,
        MeasurementQuality.PROVIDER_REPORTED,
        source,
        provider,
        model,
        request_digest(request),
        response_digest(response),
        "provider-response-v1",
        datetime.now(UTC).isoformat(),
        trust,
        MeasurementLayer.PROVIDER,
    )
