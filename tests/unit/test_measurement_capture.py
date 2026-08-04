from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmcut.captures import redact_capture, verify_capture
from llmcut.measurement import MeasurementTrust, count_payload, request_digest, response_digest
from llmcut.tokens.registry import CounterRegistry


def _capture(root: Path) -> Path:
    requests = root / "requests"
    responses = root / "responses"
    requests.mkdir(parents=True)
    responses.mkdir()
    request = {"model": "same", "messages": [{"role": "user", "content": "task"}]}
    response = {"choices": [{"message": {"content": "done"}}], "usage": {"prompt_tokens": 999}}
    (requests / "turn-1.json").write_text(json.dumps(request))
    (responses / "turn-1.json").write_text(json.dumps(response))
    manifest = {
        "schema_version": "1",
        "capture_id": "capture-1",
        "provider": "openai",
        "model": "same",
        "endpoint": "chat.completions",
        "persistence": {"prompt_content": True},
        "turns": [
            {
                "request": {
                    "digest": request_digest(request),
                    "content_location": "requests/turn-1.json",
                },
                "response": {
                    "digest": response_digest(response),
                    "content_location": "responses/turn-1.json",
                },
                "usage": {"input_tokens": 999, "quality": "provider_reported"},
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_capture_digest_binding_and_tampering(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "capture")
    assert verify_capture(capture).turns == 1
    response = capture / "responses" / "turn-1.json"
    value = json.loads(response.read_text())
    value["usage"]["prompt_tokens"] = 1
    response.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="response digest mismatch"):
        verify_capture(capture)


def test_payload_count_ignores_fixture_usage() -> None:
    registry = CounterRegistry()
    request = {"model": "same", "messages": [{"role": "user", "content": "exact payload"}]}
    first = count_payload(registry, "openai", "same", request)
    fixture_response = {"usage": {"prompt_tokens": 1}}
    fixture_response["usage"]["prompt_tokens"] = 9_999_999
    second = count_payload(registry, "openai", "same", request)
    assert first.value == second.value
    assert first.request_digest == second.request_digest
    assert first.trust is MeasurementTrust.LOCALLY_COUNTED


def test_capture_redaction_removes_credentials_and_reasoning(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "capture")
    request = capture / "requests" / "turn-1.json"
    value = json.loads(request.read_text())
    value.update({"authorization": "Bearer secret", "reasoning_content": "private reasoning"})
    request.write_text(json.dumps(value))
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["turns"][0]["request"]["digest"] = request_digest(value)
    (capture / "manifest.json").write_text(json.dumps(manifest))
    assert redact_capture(capture) == 2
    verify_capture(capture)
    rendered = request.read_text()
    assert "secret" not in rendered and "private reasoning" not in rendered


def test_counter_provenance_priorities() -> None:
    registry = CounterRegistry()
    registry.register_endpoint("openai", lambda _: 12)
    measured = count_payload(registry, "openai", "same", {"messages": []})
    assert measured.value == 12
    assert measured.quality.value == "provider_count_endpoint"
    assert measured.trust.value == "live_provider"


def test_capture_rejects_malformed_manifests_and_locations(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        verify_capture(manifest)
    manifest.write_text(json.dumps({"schema_version": "2", "turns": [{}]}))
    with pytest.raises(ValueError, match="schema version"):
        verify_capture(manifest)
    manifest.write_text(json.dumps({"schema_version": "1", "turns": []}))
    with pytest.raises(ValueError, match="at least one"):
        verify_capture(manifest)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "provider": "openai",
                "model": "same",
                "turns": [
                    {
                        "provider": "anthropic",
                        "request": {"content_location": "../escape.json", "digest": "sha256:x"},
                        "response": {},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        verify_capture(manifest)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "provider": "openai",
                "model": "same",
                "turns": [
                    {
                        "provider": "anthropic",
                        "request": {},
                        "response": {},
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="provider identity"):
        verify_capture(manifest)
