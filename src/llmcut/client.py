from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

from llmcut.config import Config, load_config
from llmcut.errors import ConfigurationError
from llmcut.managed.protocol import ManagedRequest
from llmcut.managed.runtime import ManagedResult, ManagedRuntime, ProviderCall
from llmcut.store.evidence import EvidenceStore


class AsyncClient:
    def __init__(self, runtime: ManagedRuntime) -> None:
        self._runtime = runtime

    @classmethod
    def from_config(cls, project: Path | None = None) -> AsyncClient:
        config = load_config(project)
        store = EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)
        return cls(ManagedRuntime(store, _transport(config)))

    async def run(
        self,
        request: ManagedRequest,
        mode: str | None = None,
        *,
        dry_run: bool = False,
        timeout: float | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> ManagedResult:
        if mode is not None:
            request.execution.optimization = request.execution.optimization.__class__(mode)
        if timeout is not None:
            request.execution.timeout_seconds = timeout
        return await self._runtime.run(request, dry_run=dry_run, cancellation=cancellation)


class Client:
    def __init__(self, runtime: ManagedRuntime) -> None:
        self._async = AsyncClient(runtime)

    @classmethod
    def from_config(cls, project: Path | None = None) -> Client:
        config = load_config(project)
        store = EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)
        return cls(ManagedRuntime(store, _transport(config)))

    def run(
        self,
        request: ManagedRequest,
        mode: str | None = None,
        *,
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> ManagedResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._async.run(request, mode, dry_run=dry_run, timeout=timeout))
        raise RuntimeError("Client.run cannot be used inside an event loop; use AsyncClient")


def _transport(config: Config) -> ProviderCall:
    async def call(provider_name: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        matches = [item for item in config.providers.values() if item.kind == provider_name]
        if len(matches) != 1:
            raise ConfigurationError(f"managed provider {provider_name!r} must resolve uniquely")
        provider = matches[0]
        headers = dict(provider.headers)
        credential = os.environ.get(provider.credential_env) if provider.credential_env else None
        if credential:
            if provider.kind == "anthropic":
                headers["x-api-key"] = credential
            elif provider.kind == "gemini":
                headers["x-goog-api-key"] = credential
            else:
                headers["authorization"] = f"Bearer {credential}"
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(provider.base_url, json=payload, headers=headers)
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict):
            raise ValueError("provider response must be an object")
        return value

    return call
