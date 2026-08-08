from __future__ import annotations

import json
import os
import secrets
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from llmcut.model import digest_bytes

LEASE_VERSION = 1
LeaseMode = Literal["observe", "compact"]


@dataclass(frozen=True, slots=True)
class HookLease:
    schema_version: int
    lease_id: str
    token_digest: str
    mode: LeaseMode
    repository_root: str
    repository_revision: str
    evaluation_run_id: str
    allowed_cwd: str
    hook_definition_digest: str
    created_at: int
    expires_at: int
    state_root: str
    metrics_path: str


@dataclass(frozen=True, slots=True)
class LeaseActivation:
    lease: HookLease
    token: str
    registry: Path

    def environment(self) -> dict[str, str]:
        return {
            "LLMCUT_HOOK_LEASE": self.lease.lease_id,
            "LLMCUT_HOOK_LEASE_TOKEN": self.token,
            "LLMCUT_HOOK_LEASE_ROOT": str(self.registry),
            "LLMCUT_HOOK_RUN_ID": self.lease.evaluation_run_id,
        }


def create_lease(
    registry: Path,
    *,
    mode: LeaseMode,
    repository_root: Path,
    repository_revision: str,
    evaluation_run_id: str,
    allowed_cwd: Path,
    hook_definition_digest: str,
    state_root: Path,
    metrics_path: Path,
    lifetime_seconds: int = 900,
) -> LeaseActivation:
    if mode not in {"observe", "compact"}:
        raise ValueError("invalid hook lease mode")
    if not 1 <= lifetime_seconds <= 3600:
        raise ValueError("invalid hook lease lifetime")
    root = _protected_directory(registry)
    repository = _regular_directory(repository_root)
    cwd = _regular_directory(allowed_cwd)
    if cwd != repository and repository not in cwd.parents:
        raise ValueError("lease cwd escapes repository")
    state = state_root.resolve()
    metrics = metrics_path.resolve()
    if repository == state or repository in state.parents:
        raise ValueError("lease state must remain outside repository")
    if repository == metrics.parent or repository in metrics.parents:
        raise ValueError("lease metrics must remain outside repository")
    lease_id = secrets.token_hex(32)
    token = secrets.token_urlsafe(48)
    now = int(time.time())
    lease = HookLease(
        LEASE_VERSION,
        lease_id,
        digest_bytes(token.encode()),
        mode,
        str(repository),
        repository_revision[:128],
        evaluation_run_id[:256],
        str(cwd),
        hook_definition_digest[:128],
        now,
        now + lifetime_seconds,
        str(state),
        str(metrics),
    )
    descriptor, temporary = tempfile.mkstemp(prefix=".lease-", dir=root)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(asdict(lease), stream, sort_keys=True, separators=(",", ":"))
        os.chmod(temporary, 0o600)
        os.replace(temporary, root / f"{lease_id}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return LeaseActivation(lease, token, root)


def load_lease(
    registry: Path,
    lease_id: str,
    token: str,
    evaluation_run_id: str,
    *,
    now: int | None = None,
) -> HookLease:
    if len(lease_id) != 64 or any(character not in "0123456789abcdef" for character in lease_id):
        raise ValueError("invalid hook lease id")
    root = _protected_directory(registry, create=False)
    target = root / f"{lease_id}.json"
    if target.is_symlink() or not target.is_file():
        raise ValueError("hook lease is unavailable")
    if stat.S_IMODE(target.stat().st_mode) & 0o077:
        raise ValueError("hook lease permissions are unsafe")
    value = json.loads(target.read_text())
    lease = HookLease(**value)
    current = int(time.time()) if now is None else now
    if lease.schema_version != LEASE_VERSION or lease.lease_id != lease_id:
        raise ValueError("hook lease binding is invalid")
    if not secrets.compare_digest(lease.token_digest, digest_bytes(token.encode())):
        raise ValueError("hook lease token is invalid")
    if not secrets.compare_digest(lease.evaluation_run_id, evaluation_run_id):
        raise ValueError("hook lease run binding is invalid")
    if current > lease.expires_at:
        raise ValueError("hook lease expired")
    _regular_directory(Path(lease.repository_root))
    _regular_directory(Path(lease.allowed_cwd))
    return lease


def remove_lease(activation: LeaseActivation) -> None:
    target = activation.registry / f"{activation.lease.lease_id}.json"
    if target.parent == activation.registry and not target.is_symlink():
        target.unlink(missing_ok=True)


def _protected_directory(path: Path, *, create: bool = True) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or expanded.is_symlink():
        raise ValueError("hook lease registry must be an absolute non-symlink path")
    if create:
        expanded.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(expanded, 0o700)
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise ValueError("hook lease registry permissions are unsafe")
    return resolved


def _regular_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("hook lease path must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("hook lease path must be a directory")
    return resolved
