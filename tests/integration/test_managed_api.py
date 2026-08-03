from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from llmcut.cli import app
from llmcut.config import Config, ProviderConfig
from llmcut.proxy.app import create_app


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1",
        "provider": "openai",
        "model": "mock-model",
        "settings": {"reasoning": {"effort": "high"}},
        "task": {"content": "Fix callback.py", "kind": "current"},
        "context": [
            {
                "id": "callback",
                "kind": "source",
                "source": "src/callback.py",
                "content": "def callback(): pass",
                "retention": "recoverable",
            },
            {
                "id": "other",
                "kind": "document",
                "content": "unrelated " * 100,
                "retention": "recoverable",
            },
        ],
        "tools": [],
        "execution": {"integration": "managed", "optimization": "extreme"},
    }


def test_managed_plan_and_status(tmp_path: Path) -> None:
    config = Config(state_dir=tmp_path / "state")
    with TestClient(create_app(config)) as client:
        response = client.post("/managed/plan", json=_payload())
        assert response.status_code == 200
        value = response.json()
        assert value["status"] == "planned"
        assert "other" in value["plan"]["deferred_context_ids"]
        status = client.get(f"/managed/runs/{value['run_id']}")
        assert status.status_code == 200 and status.json()["run_id"] == value["run_id"]
        assert client.get("/managed/runs/not-a-run").status_code == 404
        metrics = json.dumps(client.get("/metrics").json())
        assert "Fix callback" not in metrics and "def callback" not in metrics


def test_managed_auth_body_and_schema_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(state_dir=tmp_path / "state", max_request_bytes=2000)
    monkeypatch.setenv("LLMCUT_MANAGED_TOKEN", "local-secret")
    with TestClient(create_app(config)) as client:
        assert client.post("/managed/plan", json=_payload()).status_code == 401
        headers = {"authorization": "Bearer local-secret"}
        assert client.post("/managed/plan", json=_payload(), headers=headers).status_code == 200
        assert client.post("/managed/plan", content=b"x" * 2001, headers=headers).status_code == 413
        malicious = _payload() | {"api_key": "request-secret"}
        response = client.post("/managed/plan", json=malicious, headers=headers)
        assert response.status_code == 400 and "credential" in response.text


def test_managed_cli_dry_run(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/managed/repository-task.json").resolve()
    result = CliRunner().invoke(
        app,
        ["run", "--request", str(fixture), "--repo", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "planned"


def test_managed_cli_evaluation_targets(tmp_path: Path) -> None:
    corpus = Path("tests/fixtures/eval/managed.jsonl").resolve()
    result = CliRunner().invoke(app, ["eval", "--corpus", str(corpus), "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    assert value["targets"]["passed"]
    assert value["targets"]["positive_saving_cases"] == 5
    heavy = next(item for item in value["cases"] if item["task_id"] == "retrieval-heavy-control")
    assert not heavy["saving"] and heavy["total_tokens"] > heavy["baseline_tokens"]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_managed_http_run_with_mock_upstream(tmp_path: Path, provider: str) -> None:
    config = Config(
        state_dir=tmp_path / "state",
        providers={"mock": ProviderConfig("mock", provider, "https://mock.invalid/v1", "")},
    )
    application = create_app(config)

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if provider == "anthropic":
                return {
                    "content": [{"type": "text", "text": "complete"}],
                    "usage": {"input_tokens": 30, "output_tokens": 2},
                }
            if provider == "gemini":
                return {
                    "candidates": [{"content": {"parts": [{"text": "complete"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 30,
                        "candidatesTokenCount": 2,
                    },
                }
            return {
                "choices": [{"message": {"role": "assistant", "content": "complete"}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 2},
            }

    class Upstream:
        async def post(self, *_: object, **__: object) -> Response:
            return Response()

        async def aclose(self) -> None:
            return None

    with TestClient(application) as client:
        application.state.client = Upstream()
        payload = _payload()
        payload["provider"] = provider
        if provider == "anthropic":
            payload["settings"] = {"max_tokens": 128, "reasoning": {"effort": "high"}}
        response = client.post("/managed/run", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["output"] == "complete"
