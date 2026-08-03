from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmcut.errors import ConfigurationError

DEFAULT_CONFIG = """# llmcut v0.3.0 — generated safe defaults
mode = "extreme"
retention_days = 30
persist_prompt_content = true

[proxy]
host = "127.0.0.1"
port = 8765
max_request_bytes = 10485760
timeout_seconds = 120.0
diagnostic_headers = true
integration_mode = "transparent"
managed_bearer_token_env = "LLMCUT_MANAGED_TOKEN"

[modes.extreme]
quality_floor = "baseline"
allow_lossy_context = false
allow_model_change = false
allow_reasoning_change = false
allow_validation_reduction = false
retain_original_context = true
fallback = "full_context"
tool_loading = "lazy"
history = "referenced_checkpoint"
logs = "virtualized"
cache = "maximum"
"""


@dataclass(slots=True)
class ProviderConfig:
    name: str
    kind: str
    base_url: str
    credential_env: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Config:
    mode: str = "extreme"
    state_dir: Path = Path(".llmcut")
    host: str = "127.0.0.1"
    port: int = 8765
    max_request_bytes: int = 10 * 1024 * 1024
    timeout_seconds: float = 120.0
    diagnostic_headers: bool = True
    integration_mode: str = "transparent"
    retention_days: int = 30
    persist_prompt_content: bool = True
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    managed_bearer_token_env: str = "LLMCUT_MANAGED_TOKEN"  # noqa: S105 -- environment name


def load_config(project: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    project = (project or Path.cwd()).resolve()
    merged: dict[str, Any] = {}
    user = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "llmcut/config.toml"
    for path in (user, project / ".llmcut/config.toml"):
        if path.is_file():
            _deep_merge(merged, tomllib.loads(path.read_text()))
    env: dict[str, Any] = {}
    mappings = {
        "LLMCUT_MODE": "mode",
        "LLMCUT_HOST": "host",
        "LLMCUT_PORT": "port",
        "LLMCUT_MAX_REQUEST_BYTES": "max_request_bytes",
        "LLMCUT_TIMEOUT_SECONDS": "timeout_seconds",
    }
    for key, target in mappings.items():
        if key in os.environ:
            env[target] = os.environ[key]
    proxy = dict(merged.get("proxy", {}))
    proxy.update(
        {
            key: value
            for key, value in env.items()
            if key in {"host", "port", "max_request_bytes", "timeout_seconds"}
        }
    )
    if overrides:
        proxy.update(
            {key: value for key, value in overrides.items() if key in proxy or key == "host"}
        )
    provider_configs = {}
    for name, value in merged.get("provider", {}).items():
        provider_configs[name] = ProviderConfig(
            name,
            value["kind"],
            value["base_url"],
            value.get("credential_env", ""),
            dict(value.get("headers", {})),
        )
    mode = str((overrides or {}).get("mode", env.get("mode", merged.get("mode", "extreme"))))
    config = Config(
        mode=mode,
        state_dir=project / ".llmcut",
        host=str(proxy.get("host", "127.0.0.1")),
        port=int(proxy.get("port", 8765)),
        max_request_bytes=int(proxy.get("max_request_bytes", 10 * 1024 * 1024)),
        timeout_seconds=float(proxy.get("timeout_seconds", 120)),
        diagnostic_headers=bool(proxy.get("diagnostic_headers", True)),
        integration_mode=str(proxy.get("integration_mode", "transparent")),
        retention_days=int(merged.get("retention_days", 30)),
        persist_prompt_content=bool(merged.get("persist_prompt_content", True)),
        providers=provider_configs,
        managed_bearer_token_env=str(proxy.get("managed_bearer_token_env", "LLMCUT_MANAGED_TOKEN")),
    )
    if config.max_request_bytes < 1 or config.timeout_seconds <= 0:
        raise ConfigurationError("proxy size and timeout limits must be positive")
    if config.integration_mode not in {"transparent", "managed"}:
        raise ConfigurationError("integration_mode must be transparent or managed")
    return config


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
