from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CodexCapabilities:
    installed: bool
    executable: str | None
    version: str | None
    app_server: bool
    mcp: bool
    config_path: str
    llmcut_configured: bool
    agent_usage: str
    subscription_usage: str = "subscription_unavailable"
    sdk_installed: bool = False
    sdk_version: str | None = None
    sdk_runtime_version: str | None = None
    selected_backend: str = "sdk"
    authentication_available: bool = False
    authentication_method: str = "unknown"
    model_setting_support: bool = True
    live_evaluation_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def detect_codex(config_path: Path | None = None) -> CodexCapabilities:
    from llmcut.integrations.codex.auth import authentication_preflight

    executable = shutil.which("codex")
    target = config_path or default_config_path()
    configured = target.is_file() and "[mcp_servers.llmcut]" in target.read_text(errors="replace")
    if executable is None:
        return CodexCapabilities(
            False, None, None, False, False, str(target), configured, "unavailable"
        )
    version: str | None = _probe([executable, "--version"])
    app_server = _probe([executable, "app-server", "--help"]) is not None
    mcp = _probe([executable, "mcp", "--help"]) is not None
    sdk_version: str | None
    runtime_version: str | None
    try:
        sdk_version = importlib.metadata.version("openai-codex")
        runtime_version = importlib.metadata.version("openai-codex-cli-bin")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = runtime_version = None
    auth = authentication_preflight(executable=executable)
    return CodexCapabilities(
        True,
        executable,
        version,
        app_server,
        mcp,
        str(target),
        configured,
        "agent_reported_when thread/tokenUsage/updated is emitted",
        sdk_installed=sdk_version is not None,
        sdk_version=sdk_version,
        sdk_runtime_version=runtime_version,
        authentication_available=auth.authenticated,
        authentication_method=auth.method,
        live_evaluation_ready=bool(sdk_version and auth.automation_ready and mcp),
    )


def _probe(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (
        (result.stdout or result.stderr).strip().splitlines()[0]
        if (result.stdout or result.stderr).strip()
        else "available"
    )
