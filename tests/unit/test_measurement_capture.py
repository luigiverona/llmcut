from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llmcut.captures import delete_capture, redact_capture, verify_capture, write_agent_capture
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["turns"][0]["request"].update(digest="bad"), "request digest"),
        (lambda value: value["turns"][0].update(model="other"), "model identity"),
        (
            lambda value: value["turns"][0]["request"].update(content_location="requests/missing"),
            "request is missing",
        ),
        (
            lambda value: value["turns"][0]["response"].update(
                content_location="responses/missing"
            ),
            "response is missing",
        ),
    ],
)
def test_capture_rejects_binding_failures(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], object], message: str
) -> None:
    capture = _capture(tmp_path / "capture")
    manifest = json.loads((capture / "manifest.json").read_text())
    mutation(manifest)
    (capture / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        verify_capture(capture)


def test_agent_capture_empty_runs_artifact_and_safe_deletion(tmp_path: Path) -> None:
    capture = write_agent_capture(
        {
            "run_id": "run",
            "codex_version": "configured-test-transport",
            "environment": {"model": "same"},
            "tasks": [],
            "authorization": "secret",
        },
        tmp_path / "capture",
    )
    assert verify_capture(capture).turns == 1
    manifest = json.loads((capture / "manifest.json").read_text())
    assert manifest["capture_provenance"] == "untrusted_fixture"
    artifact = capture / "artifacts" / "evaluation.json"
    assert "secret" not in artifact.read_text()
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="artifact.*digest"):
        verify_capture(capture)
    with pytest.raises(ValueError, match="explicit capture directory"):
        delete_capture(tmp_path / "missing")

    capture = write_agent_capture({"run_id": "second", "tasks": []}, tmp_path / "capture-2")
    delete_capture(capture)
    assert not capture.exists()


def test_agent_capture_rejects_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "capture"
    destination.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        write_agent_capture({}, destination)


def test_capture_rejects_unbound_usage_and_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    value: dict[str, Any] = {
        "schema_version": "1",
        "provider": "openai",
        "model": "same",
        "turns": [
            {
                "request": {},
                "response": {},
                "usage": {"quality": "provider_reported"},
            }
        ],
    }
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="bound request"):
        verify_capture(manifest)
    value["turns"][0]["usage"] = {"quality": "estimated"}
    value["artifacts"] = [{}]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="no content location"):
        verify_capture(manifest)


def test_redaction_skips_metadata_only_turn_and_live_capture_runs(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "metadata")
    manifest = json.loads((capture / "manifest.json").read_text())
    manifest["turns"][0]["request"].pop("content_location")
    (capture / "manifest.json").write_text(json.dumps(manifest))
    assert redact_capture(capture) == 0

    live = write_agent_capture(
        {
            "run_id": "live",
            "codex_version": "codex 1.0",
            "environment": {"model": "same"},
            "tasks": [
                {
                    "runs": [
                        {
                            "request_digests": ["sha256:request"],
                            "response_digests": ["sha256:response"],
                            "agent_usage": {"inputTokens": 4},
                            "agent_usage_quality": "agent_reported",
                        }
                    ]
                }
            ],
        },
        tmp_path / "live",
    )
    assert json.loads((live / "manifest.json").read_text())["capture_provenance"] == "live_agent"


def test_agent_capture_rejects_non_object_and_cleans_temporary_directory(tmp_path: Path) -> None:
    invalid: Any = []
    with pytest.raises(ValueError, match="must be an object"):
        write_agent_capture(invalid, tmp_path / "bad")
    assert not list(tmp_path.glob(".llmcut-capture-*"))
