import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llmcut.config import Config, ProviderConfig
from llmcut.errors import ConfigurationError
from llmcut.proxy.app import create_app
from llmcut.proxy.security import external_bind_warning, filtered_headers, upstream_url


def upstream_app(captured: list[dict[str, object]]) -> Starlette:
    async def endpoint(request: Request) -> Response:
        body = await request.body()
        captured.append(
            {"path": request.url.path, "body": json.loads(body), "headers": dict(request.headers)}
        )
        if request.url.path.endswith("stream"):

            async def chunks() -> AsyncIterator[bytes]:
                yield b"data: one\n\n"
                yield b"data: two\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return JSONResponse({"ok": True, "usage": {"input_tokens": 1}})

    return Starlette(routes=[Route("/{path:path}", endpoint, methods=["POST"])])


@pytest.mark.parametrize(
    "provider,path,payload",
    [
        ("openai", "chat/completions", {"model": "m", "messages": []}),
        ("anthropic", "v1/messages", {"model": "m", "messages": [], "max_tokens": 1}),
        ("gemini", "v1beta/models/g:generateContent", {"contents": [{"parts": [{"text": "hi"}]}]}),
    ],
)
async def test_nonstreaming_provider_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    path: str,
    payload: dict[str, object],
) -> None:
    captured: list[dict[str, object]] = []
    config = Config(
        state_dir=tmp_path / ".llmcut",
        providers={provider: ProviderConfig(provider, provider, "https://mock.local", "TEST_KEY")},
    )
    monkeypatch.setenv("TEST_KEY", "supersecret")
    app = create_app(config)
    async with app.router.lifespan_context(app):
        await app.state.client.aclose()
        app.state.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app(captured))
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            response = await client.post(f"/{provider}/{path}", json=payload)
    assert response.status_code == 200 and response.json()["ok"]
    assert captured[0]["body"] == payload
    headers = captured[0]["headers"]
    assert isinstance(headers, dict)
    assert any("supersecret" in str(value) for value in headers.values())
    files = list((tmp_path / ".llmcut").glob("*"))
    assert all(b"supersecret" not in file.read_bytes() for file in files if file.is_file())


async def test_streaming_passthrough(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []
    config = Config(
        state_dir=tmp_path,
        providers={"openai": ProviderConfig("openai", "openai", "https://mock.local", "")},
    )
    app = create_app(config)
    async with app.router.lifespan_context(app):
        await app.state.client.aclose()
        app.state.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app(captured))
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            response = await client.post("/openai/stream", json={"stream": True})
    assert response.content == b"data: one\n\ndata: two\n\n"


async def test_proxy_optimizes_before_upstream_and_sets_diagnostics(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []
    tool = {
        "type": "function",
        "function": {"name": "lookup", "description": "x" * 1200, "parameters": {"type": "object"}},
    }
    config = Config(
        state_dir=tmp_path,
        providers={"openai": ProviderConfig("openai", "openai", "https://mock.local", "")},
    )
    app = create_app(config)
    async with app.router.lifespan_context(app):
        await app.state.client.aclose()
        app.state.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app(captured))
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            response = await client.post(
                "/openai/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "lookup"}],
                    "tools": [tool, tool],
                    "stream": False,
                },
            )
    upstream_body = captured[0]["body"]
    assert isinstance(upstream_body, dict) and len(upstream_body["tools"]) == 1
    assert response.headers["x-llmcut-status"] == "optimized"
    assert int(response.headers["x-llmcut-optimized-tokens"]) < int(
        response.headers["x-llmcut-original-tokens"]
    )
    assert response.headers["x-llmcut-count-quality"] == "estimated"


async def test_diagnostic_headers_can_be_disabled(tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []
    config = Config(
        state_dir=tmp_path,
        diagnostic_headers=False,
        providers={"openai": ProviderConfig("openai", "openai", "https://mock.local", "")},
    )
    app = create_app(config)
    async with app.router.lifespan_context(app):
        await app.state.client.aclose()
        app.state.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=upstream_app(captured))
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            response = await client.post(
                "/openai/chat/completions", json={"model": "m", "messages": []}
            )
    assert "x-llmcut-status" not in response.headers


async def test_allowlist_body_limit_health_and_metrics(tmp_path: Path) -> None:
    config = Config(state_dir=tmp_path, max_request_bytes=3, providers={})
    app = create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        assert (await client.post("/unknown/x", content=b"{}")).status_code == 404
        assert (await client.get("/health")).json()["status"] == "ok"
        assert "runs" in (await client.get("/metrics")).json()


async def test_configured_provider_body_limit(tmp_path: Path) -> None:
    config = Config(
        state_dir=tmp_path,
        max_request_bytes=3,
        providers={"openai": ProviderConfig("openai", "openai", "https://mock.local", "")},
    )
    app = create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://local"
    ) as client:
        response = await client.post("/openai/chat/completions", content=b"1234")
    assert response.status_code == 413 and "limit" in response.text


def test_security_helpers() -> None:
    assert "connection" not in filtered_headers({"connection": "close", "x-ok": "yes"})
    assert upstream_url("https://safe.example/v1", "messages").startswith("https://safe.example/")
    with pytest.raises(ConfigurationError):
        upstream_url("https://safe.example", "https://evil.example/x")
    assert external_bind_warning("0.0.0.0") and external_bind_warning("127.0.0.1") is None
