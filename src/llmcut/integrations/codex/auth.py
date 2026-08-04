from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from llmcut.integrations.codex.backend import codex_agent_environment


@dataclass(frozen=True, slots=True)
class AuthenticationStatus:
    authenticated: bool
    method: str
    credential_store: str
    codex_home: str
    automation_ready: bool
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def authentication_preflight(
    *,
    executable: str | None = None,
    mode: str = "existing-session",
    env_var: str | None = None,
) -> AuthenticationStatus:
    if mode not in {"existing-session", "api-key", "access-token", "none"}:
        raise ValueError(f"unsupported authentication mode: {mode}")
    codex_home = "configured" if os.environ.get("CODEX_HOME") else "default"
    if mode == "none":
        return AuthenticationStatus(False, "unknown", "unknown", codex_home, False)
    if mode in {"api-key", "access-token"}:
        available = bool(env_var and os.environ.get(env_var))
        method = "api" if mode == "api-key" else "access-token"
        return AuthenticationStatus(
            available,
            method,
            "unknown",
            codex_home,
            available,
            None if available else "configured environment variable is unavailable",
        )
    binary = executable or shutil.which("codex")
    if not binary:
        return AuthenticationStatus(
            False, "unknown", "unknown", codex_home, False, "Codex executable is unavailable"
        )
    try:
        completed = subprocess.run(
            [binary, "login", "status"],
            env=codex_agent_environment((), "preflight", "existing-session", None),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return AuthenticationStatus(
            False, "unknown", "unknown", codex_home, False, "login status unavailable"
        )
    rendered = (completed.stdout or completed.stderr).strip().lower()
    authenticated = completed.returncode == 0 and "logged in" in rendered
    if "chatgpt" in rendered:
        method = "chatgpt"
    elif "api" in rendered:
        method = "api"
    elif "access token" in rendered:
        method = "access-token"
    else:
        method = "unknown"
    return AuthenticationStatus(
        authenticated,
        method,
        "auto",
        codex_home,
        authenticated,
        None if authenticated else "run `codex login` and retry",
    )
