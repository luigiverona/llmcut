from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llmcut.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter, adapter_for
from llmcut.config import Config
from llmcut.core.optimize import Optimizer
from llmcut.policy import OptimizationMode
from llmcut.proxy.optimize import NativeOptimization, optimize_native
from llmcut.proxy.security import filtered_headers, upstream_url
from llmcut.store.evidence import EvidenceStore
from llmcut.store.metrics import MetricsStore


def create_app(config: Config) -> Starlette:
    evidence = EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)
    metrics = MetricsStore(evidence.db)
    optimizer = Optimizer(evidence)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(timeout=config.timeout_seconds, follow_redirects=False)
        yield
        await app.state.client.aclose()

    async def health(_: Request) -> Response:
        try:
            evidence.db.integrity_check()
            return JSONResponse({"status": "ok", "version": "0.2.0"})
        except Exception:
            return JSONResponse({"status": "degraded"}, status_code=503)

    async def stats(_: Request) -> Response:
        return JSONResponse(metrics.summary())

    async def forward(request: Request) -> Response:
        provider_name = request.path_params["provider"]
        provider = config.providers.get(provider_name)
        if provider is None:
            return JSONResponse(
                {"error": {"message": "provider is not configured", "type": "configuration_error"}},
                status_code=404,
            )
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > config.max_request_bytes:
            return JSONResponse(
                {"error": {"message": "request body exceeds configured limit"}}, status_code=413
            )
        body = await request.body()
        if len(body) > config.max_request_bytes:
            return JSONResponse(
                {"error": {"message": "request body exceeds configured limit"}}, status_code=413
            )
        path = request.path_params["path"]
        try:
            target = upstream_url(provider.base_url, path)
        except Exception:
            return JSONResponse({"error": {"message": "unsafe upstream route"}}, status_code=400)
        headers = filtered_headers(dict(request.headers))
        credential = os.environ.get(provider.credential_env) if provider.credential_env else None
        if credential:
            if provider.kind == "anthropic":
                headers["x-api-key"] = credential
            elif provider.kind == "gemini":
                headers["x-goog-api-key"] = credential
            else:
                headers["authorization"] = f"Bearer {credential}"
        headers.update(provider.headers)
        adapter, endpoint_format = adapter_for(provider.kind, path)
        try:
            mode = OptimizationMode(config.mode)
        except ValueError:
            mode = OptimizationMode.EXTREME
        optimization = optimize_native(body, adapter, endpoint_format, optimizer, mode)
        forwarded_body = optimization.body
        stream_requested = _stream_requested(forwarded_body, path)
        client: httpx.AsyncClient = request.app.state.client
        upstream_started = time.perf_counter()
        try:
            upstream_request = client.build_request(
                request.method,
                target,
                content=forwarded_body,
                headers=headers,
                params=request.query_params,
            )
            upstream = await client.send(upstream_request, stream=stream_requested)
        except httpx.TimeoutException:
            return JSONResponse(
                {"error": {"message": "configured upstream timed out", "type": "timeout_error"}},
                status_code=504,
            )
        except httpx.HTTPError:
            return JSONResponse(
                {
                    "error": {
                        "message": "configured upstream request failed",
                        "type": "upstream_error",
                    }
                },
                status_code=502,
            )
        response_headers = filtered_headers(dict(upstream.headers))
        upstream_seconds = time.perf_counter() - upstream_started
        metric_id = metrics.record_request(
            {
                "provider": provider.kind,
                "endpoint_format": endpoint_format,
                "mode": mode.value,
                "integration_mode": config.integration_mode,
                "original_tokens": optimization.original_tokens,
                "attempted_tokens": optimization.attempted_tokens,
                "effective_tokens": optimization.effective_tokens,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "count_quality": optimization.count_quality,
                "optimization_seconds": optimization.duration_seconds,
                "upstream_seconds": upstream_seconds,
                "fallback": int(optimization.status != "optimized"),
                "fallback_reason": optimization.fallback_reason,
                "omitted_blocks": optimization.omitted_blocks,
                "recovery_events": 0,
                "retries": 0,
                "evaluated_parity": None,
            }
        )
        if config.diagnostic_headers:
            response_headers.update(_diagnostic_headers(optimization, mode))
        if stream_requested:

            async def chunks() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                chunks(),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
            )
        content = await upstream.aread()
        await upstream.aclose()
        _record_usage(metrics, provider.kind, content, metric_id)
        return Response(
            content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    routes = [
        Route("/health", health),
        Route("/metrics", stats),
        Route(
            "/{provider}/{path:path}", forward, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
        ),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def _stream_requested(body: bytes, path: str) -> bool:
    if "stream" in path.lower():
        return True
    try:
        value = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return False
    return value.get("stream") is True


def _record_usage(
    metrics: MetricsStore, provider_kind: str, content: bytes, metric_id: str | None = None
) -> None:
    try:
        response = json.loads(content)
        adapter = {
            "anthropic": AnthropicAdapter(),
            "gemini": GeminiAdapter(),
        }.get(provider_kind, OpenAIAdapter())
        usage = adapter.usage(response)
        if any(usage.values()):
            with metrics.db.connect() as conn:
                conn.execute(
                    "INSERT INTO usage(provider,input_tokens,output_tokens,cached_tokens,"
                    "reasoning_tokens,raw) VALUES(?,?,?,?,?,?)",
                    (
                        provider_kind,
                        usage["input_tokens"],
                        usage["output_tokens"],
                        usage["cached_tokens"],
                        usage["reasoning_tokens"],
                        json.dumps(usage, sort_keys=True),
                    ),
                )
                if metric_id is not None:
                    conn.execute(
                        "UPDATE request_metrics SET output_tokens=?,cached_tokens=?,"
                        "reasoning_tokens=? WHERE id=?",
                        (
                            usage["output_tokens"],
                            usage["cached_tokens"],
                            usage["reasoning_tokens"],
                            metric_id,
                        ),
                    )
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        # Usage observation is optional and must never alter provider response semantics.
        return


def _diagnostic_headers(result: NativeOptimization, mode: OptimizationMode) -> dict[str, str]:
    return {
        "x-llmcut-status": result.status,
        "x-llmcut-mode": mode.value,
        "x-llmcut-original-tokens": str(result.original_tokens),
        "x-llmcut-optimized-tokens": str(result.effective_tokens),
        "x-llmcut-count-quality": result.count_quality,
    }
