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
from llmcut.errors import LlmcutError
from llmcut.managed.protocol import ManagedRequest
from llmcut.managed.runtime import ManagedRuntime
from llmcut.policy import OptimizationMode
from llmcut.proxy.optimize import NativeOptimization, optimize_native
from llmcut.proxy.security import filtered_headers, upstream_url
from llmcut.store.evidence import EvidenceStore
from llmcut.store.metrics import MetricsStore


def create_app(config: Config) -> Starlette:
    evidence = EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)
    metrics = MetricsStore(evidence.db)
    optimizer = Optimizer(evidence)
    completed_runs: dict[str, dict[str, object]] = {}

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(timeout=config.timeout_seconds, follow_redirects=False)
        yield
        await app.state.client.aclose()

    async def health(_: Request) -> Response:
        try:
            evidence.db.integrity_check()
            from llmcut import __version__

            return JSONResponse({"status": "ok", "version": __version__})
        except Exception:
            return JSONResponse({"status": "degraded"}, status_code=503)

    async def stats(_: Request) -> Response:
        return JSONResponse(metrics.summary())

    async def managed(request: Request) -> Response:
        auth_error = _managed_auth(request, config)
        if auth_error is not None:
            return auth_error
        body = await request.body()
        if len(body) > config.max_request_bytes:
            return JSONResponse(
                {"error": {"message": "request body exceeds configured limit"}}, status_code=413
            )
        try:
            payload = _bounded_json_body(body)
            managed_request = ManagedRequest.from_dict(payload)

            async def provider_call(
                kind: str, native: dict[str, object], timeout: float
            ) -> dict[str, object]:
                providers = [item for item in config.providers.values() if item.kind == kind]
                if len(providers) != 1:
                    raise ValueError("managed provider must resolve to one configured upstream")
                provider = providers[0]
                headers = dict(provider.headers)
                credential = (
                    os.environ.get(provider.credential_env) if provider.credential_env else None
                )
                if credential:
                    if kind == "anthropic":
                        headers["x-api-key"] = credential
                    elif kind == "gemini":
                        headers["x-goog-api-key"] = credential
                    else:
                        headers["authorization"] = f"Bearer {credential}"
                client: httpx.AsyncClient = request.app.state.client
                response = await client.post(
                    provider.base_url, json=native, headers=headers, timeout=timeout
                )
                response.raise_for_status()
                value = response.json()
                if not isinstance(value, dict):
                    raise ValueError("provider response must be an object")
                return value

            runtime = ManagedRuntime(evidence, provider_call)
            result = await runtime.run(managed_request, dry_run=request.url.path == "/managed/plan")
            rendered = result.to_dict()
            usage = result.usage
            metrics.record_managed(
                {
                    "id": result.run_id,
                    "integration_mode": "managed",
                    "optimization_mode": managed_request.execution.optimization.value,
                    "provider": result.provider,
                    "model": result.model,
                    "baseline_tokens": usage.baseline_input_tokens,
                    "initial_tokens": usage.initial_input_tokens,
                    "retrieval_request_tokens": usage.retrieval_request_tokens,
                    "retrieval_result_tokens": usage.retrieval_result_tokens,
                    "continuation_tokens": usage.continuation_input_tokens,
                    "total_effective_tokens": usage.total_input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "count_quality": usage.count_quality,
                    "planning_seconds": result.planning_seconds,
                    "provider_seconds": result.provider_seconds,
                    "retrieval_count": len(result.retrievals),
                    "fallback": int(result.fallback is not None),
                    "quality_state": "not_evaluated",
                    "completed": int(result.status == "completed"),
                }
            )
            completed_runs[result.run_id] = rendered
            if len(completed_runs) > 128:
                completed_runs.pop(next(iter(completed_runs)))
            return JSONResponse(rendered)
        except (LlmcutError, ValueError, TypeError, KeyError) as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "managed_request_error"}}, status_code=400
            )

    async def managed_run_status(request: Request) -> Response:
        auth_error = _managed_auth(request, config)
        if auth_error is not None:
            return auth_error
        value = completed_runs.get(request.path_params["run_id"])
        if value is None:
            return JSONResponse({"error": {"message": "managed run not found"}}, status_code=404)
        return JSONResponse(value)

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
        Route("/managed/run", managed, methods=["POST"]),
        Route("/managed/plan", managed, methods=["POST"]),
        Route("/managed/runs/{run_id}", managed_run_status, methods=["GET"]),
        Route(
            "/{provider}/{path:path}", forward, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
        ),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def _bounded_json_body(body: bytes, max_depth: int = 64) -> dict[str, object]:
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("managed request must be a JSON object")
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > max_depth:
            raise ValueError("managed request nesting exceeds safety limit")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _managed_auth(request: Request, config: Config) -> JSONResponse | None:
    expected = os.environ.get(config.managed_bearer_token_env)
    if expected is None:
        return None
    import hmac

    supplied = request.headers.get("authorization", "")
    if not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], expected):
        return JSONResponse(
            {"error": {"message": "managed endpoint authentication failed"}}, status_code=401
        )
    return None


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
