from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from llmcut.integrations.codex.otel import (
    LocalOtelReceiver,
    _attributes,
    _parse_otlp,
    _toml_inline,
    _value,
    otel_overrides,
)


def _payload(*, environment: str = "llmcut-run") -> bytes:
    attributes = [
        ("event.name", "codex.conversation_starts"),
        ("conversation.id", "thread-1"),
        ("model", "gpt-test"),
        ("reasoning_effort", "low"),
        ("sandbox", "workspace-write"),
        ("approval_policy", "never"),
        ("deployment.environment.name", environment),
        ("prompt", "SECRET PROMPT"),
        ("tool.output", "PRIVATE SOURCE"),
    ]
    return json.dumps(
        {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.version",
                                "value": {"stringValue": "codex-cli test"},
                            }
                        ]
                    },
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "timeUnixNano": "123",
                                    "attributes": [
                                        {"key": key, "value": {"stringValue": value}}
                                        for key, value in attributes
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    ).encode()


def test_loopback_otel_receiver_retains_only_model_metadata() -> None:
    with LocalOtelReceiver() as receiver:
        request = urllib.request.Request(  # noqa: S310 - fixed loopback receiver
            receiver.endpoint,
            data=_payload(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310 - loopback
            assert response.status == 200
        observation = receiver.conversation_start("llmcut-run")
        assert observation is not None
        assert observation.model == "gpt-test"
        assert observation.reasoning == "low"
        assert observation.sandbox == "workspace-write"
        assert observation.approval_policy == "never"
        assert observation.conversation_id == "thread-1"
        assert "SECRET" not in json.dumps(observation.to_dict())
        assert "PRIVATE" not in json.dumps(observation.to_dict())


def test_otel_receiver_rejects_nonloopback_malformed_and_oversized() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalOtelReceiver("0.0.0.0")
    with LocalOtelReceiver() as receiver:
        malformed = urllib.request.Request(  # noqa: S310 - fixed loopback receiver
            receiver.endpoint, data=b"bad", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(malformed, timeout=2)  # noqa: S310 - loopback
        assert error.value.code == 400
        oversized = urllib.request.Request(  # noqa: S310 - fixed loopback receiver
            receiver.endpoint,
            data=b"x" * 1_048_577,
            method="POST",
        )
        with pytest.raises((urllib.error.HTTPError, urllib.error.URLError)):
            urllib.request.urlopen(oversized, timeout=2)  # noqa: S310 - loopback
        assert receiver.observations == []


def test_otel_receiver_rejects_wrong_path_and_empty_body() -> None:
    with LocalOtelReceiver() as receiver:
        wrong = urllib.request.Request(  # noqa: S310 - fixed loopback receiver
            receiver.endpoint.replace("/v1/logs", "/other"), data=b"{}", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(wrong, timeout=2)  # noqa: S310 - loopback
        assert error.value.code == 413
        assert receiver.conversation_start("missing") is None


def test_otel_parser_ignores_unexpected_shapes_and_supports_body_event() -> None:
    assert _parse_otlp([]) == []
    assert _parse_otlp({"resourceLogs": [None, {"scopeLogs": [None]}]}) == []
    payload = {
        "resourceLogs": [
            {
                "resource": {"attributes": "invalid"},
                "scopeLogs": [
                    {
                        "logRecords": [
                            None,
                            {
                                "body": {"stringValue": "codex.conversation_starts"},
                                "timeUnixNano": "invalid",
                                "attributes": [
                                    None,
                                    {"key": "model", "value": {"stringValue": "gpt-test"}},
                                    {"key": "ignored", "value": {"arrayValue": []}},
                                ],
                            },
                            {
                                "attributes": [
                                    {"key": "event.name", "value": {"stringValue": "other"}}
                                ]
                            },
                        ]
                    }
                ],
            }
        ]
    }
    observations = _parse_otlp(payload)
    assert len(observations) == 1 and observations[0].model == "gpt-test"
    assert observations[0].timestamp is None
    assert _attributes("bad") == {}
    assert _value("plain") == "plain"
    assert _value({"intValue": 2}) == 2
    assert _value({"arrayValue": []}) is None
    assert _toml_inline(["value"]) == '["value"]'


def test_otel_overrides_are_private_and_correlated() -> None:
    values = otel_overrides("http://127.0.0.1:4318/v1/logs", "llmcut-run")
    assert "otel.log_user_prompt=false" in values
    assert all("SECRET" not in value for value in values)
    with pytest.raises(ValueError, match="loopback"):
        otel_overrides("https://example.com/v1/logs", "llmcut-run")
    with pytest.raises(ValueError, match="correlation"):
        otel_overrides("http://127.0.0.1:4318/v1/logs", "other")
