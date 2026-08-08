from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_OTLP_BODY = 1_048_576
MAX_OTLP_REQUESTS = 64
ALLOWED_FIELDS = {
    "event.name": "event_name",
    "name": "event_name",
    "conversation.id": "conversation_id",
    "conversation_id": "conversation_id",
    "thread.id": "conversation_id",
    "model": "model",
    "reasoning_effort": "reasoning",
    "reasoning.effort": "reasoning",
    "sandbox": "sandbox",
    "sandbox_policy": "sandbox",
    "approval_policy": "approval_policy",
    "app.version": "codex_version",
    "service.version": "codex_version",
    "deployment.environment.name": "environment",
    "environment": "environment",
}


@dataclass(frozen=True, slots=True)
class ModelObservation:
    event_name: str
    conversation_id: str | None = None
    model: str | None = None
    reasoning: str | None = None
    sandbox: str | None = None
    approval_policy: str | None = None
    codex_version: str | None = None
    environment: str | None = None
    timestamp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalOtelReceiver:
    """Bounded loopback-only OTLP/HTTP JSON receiver retaining allowlisted metadata."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        if host != "127.0.0.1":
            raise ValueError("OTLP receiver must bind to IPv4 loopback")
        self.observations: list[ModelObservation] = []
        self.rejected_requests = 0
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
                if self.path != "/v1/logs" or len(receiver.observations) >= MAX_OTLP_REQUESTS:
                    receiver.rejected_requests += 1
                    self.send_error(413)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    size = 0
                if size <= 0 or size > MAX_OTLP_BODY:
                    receiver.rejected_requests += 1
                    self.send_error(413)
                    return
                raw = self.rfile.read(size)
                try:
                    payload = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    receiver.rejected_requests += 1
                    self.send_error(400)
                    return
                receiver.observations.extend(_parse_otlp(payload))
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        host = str(self._server.server_address[0])
        port = int(self._server.server_address[1])
        return f"http://{host}:{port}/v1/logs"

    def __enter__(self) -> LocalOtelReceiver:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def conversation_start(self, environment: str) -> ModelObservation | None:
        return next(
            (
                item
                for item in self.observations
                if item.event_name == "codex.conversation_starts"
                and item.environment == environment
            ),
            None,
        )


def otel_overrides(endpoint: str, environment: str) -> tuple[str, ...]:
    if not endpoint.startswith("http://127.0.0.1:") or not endpoint.endswith("/v1/logs"):
        raise ValueError("OTLP endpoint must be loopback HTTP logs endpoint")
    if not environment.startswith("llmcut-") or len(environment) > 96:
        raise ValueError("invalid OTLP run correlation environment")
    exporter = {"otlp-http": {"endpoint": endpoint, "protocol": "json", "headers": {}}}
    return (
        f"otel.environment={json.dumps(environment)}",
        "otel.log_user_prompt=false",
        f"otel.exporter={_toml_inline(exporter)}",
        'otel.metrics_exporter="none"',
        'otel.trace_exporter="none"',
    )


def _parse_otlp(payload: Any) -> list[ModelObservation]:
    if not isinstance(payload, dict):
        return []
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for resource_log in payload.get("resourceLogs", []):
        if not isinstance(resource_log, dict):
            continue
        resource = _attributes(resource_log.get("resource", {}).get("attributes", []))
        for scope_log in resource_log.get("scopeLogs", []):
            if not isinstance(scope_log, dict):
                continue
            for record in scope_log.get("logRecords", []):
                if isinstance(record, dict):
                    records.append((resource, record))
    observations = []
    for resource, record in records:
        values = resource | _attributes(record.get("attributes", []))
        body = _value(record.get("body"))
        if isinstance(body, str) and body.startswith("codex."):
            values.setdefault("event.name", body)
        selected = {
            mapped: str(values[key])[:256]
            for key, mapped in ALLOWED_FIELDS.items()
            if key in values and isinstance(values[key], (str, int, float, bool))
        }
        event_name = selected.pop("event_name", "")
        if not event_name.startswith("codex."):
            continue
        timestamp_value = record.get("timeUnixNano")
        timestamp = int(str(timestamp_value)) if str(timestamp_value).isdigit() else None
        observations.append(ModelObservation(event_name, timestamp=timestamp, **selected))
    return observations[:MAX_OTLP_REQUESTS]


def _attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    result = {}
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            result[item["key"]] = _value(item.get("value"))
    return result


def _value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for name in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if name in value:
            return value[name]
    return None


def _toml_inline(value: Any) -> str:
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                f"{json.dumps(str(key))}={_toml_inline(item)}" for key, item in value.items()
            )
            + "}"
        )
    return json.dumps(value)
