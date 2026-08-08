from __future__ import annotations

import fcntl
import json
import os
import secrets
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from llmcut.integrations.codex.hooks.config import (
    bridge_command,
    bridge_definition_digest,
    bridge_document,
)
from llmcut.model import digest_bytes


@dataclass(frozen=True, slots=True)
class UserHookTransaction:
    transaction_id: str
    target: str
    original_existed: bool
    original_digest: str | None
    merged_digest: str
    backup_path: str | None
    bridge_digest: str
    pid: int
    timestamp: int
    lease_id: str
    changed: bool


def user_transaction_root() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "llmcut" / "codex-user-hooks"


def user_hook_status(target: Path, transaction_root: Path) -> dict[str, Any]:
    present = target.is_file() and not target.is_symlink()
    configured = False
    digest = None
    if present:
        raw = target.read_bytes()
        digest = digest_bytes(raw)
        configured = _contains_bridge(_parse(raw))
    journals = sorted(path.stem for path in _transaction_root(transaction_root).glob("*.json"))
    return {
        "target_present": present,
        "bridge_configured": configured,
        "current_digest": digest,
        "bridge_definition_digest": bridge_definition_digest(),
        "pending_transactions": journals,
    }


def install_user_bridge(
    target: Path,
    transaction_root: Path,
    *,
    lease_id: str,
    dry_run: bool = False,
    persistent: bool = False,
) -> dict[str, Any]:
    target = _validate_target(target)
    root = _transaction_root(transaction_root)
    with _lock(root):
        existed = target.exists()
        original = target.read_bytes() if existed else b"{}\n"
        current = _parse(original)
        merged = _merge_bridge(current)
        encoded = (json.dumps(merged, indent=2, sort_keys=True) + "\n").encode()
        changed = encoded != original
        result: dict[str, Any] = {
            "changed": changed,
            "dry_run": dry_run,
            "persistent": persistent,
            "original_digest": digest_bytes(original) if existed else None,
            "merged_digest": digest_bytes(encoded),
            "bridge_definition_digest": bridge_definition_digest(),
        }
        if dry_run:
            return result
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.parent.is_symlink():
            raise ValueError("Codex home must not be a symlink")
        transaction_id = secrets.token_hex(24)
        backup: Path | None = None
        if existed and changed:
            backup = root / f"{transaction_id}.backup"
            _write_atomic(backup, original)
        if not persistent:
            transaction = UserHookTransaction(
                transaction_id,
                str(target),
                existed,
                digest_bytes(original) if existed else None,
                digest_bytes(encoded),
                str(backup) if backup else None,
                bridge_definition_digest(),
                os.getpid(),
                int(time.time()),
                lease_id[:128],
                changed,
            )
            _write_atomic(
                root / f"{transaction_id}.json",
                (json.dumps(asdict(transaction), sort_keys=True) + "\n").encode(),
            )
            result["transaction_id"] = transaction_id
        if changed:
            _write_atomic(target, encoded)
        return result


def remove_user_bridge(
    target: Path, transaction_root: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    target = _validate_target(target)
    root = _transaction_root(transaction_root)
    with _lock(root):
        if not target.exists():
            return {"changed": False, "dry_run": dry_run}
        original = target.read_bytes()
        cleaned = _remove_bridge(_parse(original))
        encoded = (json.dumps(cleaned, indent=2, sort_keys=True) + "\n").encode()
        changed = encoded != original
        if changed and not dry_run:
            _write_atomic(target, encoded)
        return {"changed": changed, "dry_run": dry_run}


def restore_user_bridge(transaction_root: Path, transaction_id: str) -> dict[str, Any]:
    root = _transaction_root(transaction_root)
    with _lock(root):
        return _restore_locked(root, transaction_id)


def _restore_locked(root: Path, transaction_id: str) -> dict[str, Any]:
    journal = root / f"{transaction_id}.json"
    transaction = _read_transaction(journal)
    target = _validate_target(Path(transaction.target))
    related = [
        item
        for item in _pending_transactions(root)
        if item.transaction_id != transaction_id and item.target == transaction.target
    ]
    if transaction.changed and related:
        return {
            "transaction_id": transaction_id,
            "cleanup": "deferred",
            "active_references": len(related),
        }
    if not transaction.changed:
        journal.unlink()
        owners = [item for item in related if item.changed]
        remaining = [item for item in related if not item.changed]
        result: dict[str, Any] = {"transaction_id": transaction_id, "cleanup": "complete"}
        if len(owners) == 1 and not remaining:
            result["owner_cleanup"] = _restore_locked(root, owners[0].transaction_id)
        return result
    current = target.read_bytes() if target.exists() else b""
    current_digest = digest_bytes(current)
    result = {"transaction_id": transaction_id, "cleanup": "complete"}
    if current_digest == transaction.merged_digest:
        if transaction.original_existed:
            backup = Path(transaction.backup_path or "")
            if backup.is_symlink() or not backup.is_file():
                raise ValueError("user hook backup is unavailable")
            original = backup.read_bytes()
            if digest_bytes(original) != transaction.original_digest:
                raise ValueError("user hook backup digest mismatch")
            _write_atomic(target, original)
        else:
            target.unlink(missing_ok=True)
    else:
        try:
            cleaned = _remove_bridge(_parse(current))
        except (ValueError, json.JSONDecodeError):
            return {**result, "cleanup": "incomplete", "reason": "current file is ambiguous"}
        _write_atomic(target, (json.dumps(cleaned, indent=2, sort_keys=True) + "\n").encode())
        result["cleanup"] = "surgical"
    journal.unlink()
    if transaction.backup_path:
        Path(transaction.backup_path).unlink(missing_ok=True)
    return result


def _read_transaction(path: Path) -> UserHookTransaction:
    if path.is_symlink() or not path.is_file():
        raise ValueError("user hook transaction is unavailable")
    return UserHookTransaction(**json.loads(path.read_text()))


def _pending_transactions(root: Path) -> list[UserHookTransaction]:
    return [
        _read_transaction(path) for path in sorted(root.glob("*.json")) if not path.is_symlink()
    ]


def recover_user_bridges(transaction_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    root = _transaction_root(transaction_root)
    pending = sorted(path.stem for path in root.glob("*.json") if not path.is_symlink())
    if dry_run:
        return {"pending": pending, "dry_run": True}
    results = [restore_user_bridge(root, transaction_id) for transaction_id in pending]
    return {"pending": pending, "results": results, "dry_run": False}


def _merge_bridge(current: dict[str, Any]) -> dict[str, Any]:
    result = cast(dict[str, Any], json.loads(json.dumps(current)))
    hooks = result.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("existing hooks field must be an object")
    groups = hooks.setdefault("PostToolUse", [])
    if not isinstance(groups, list):
        raise ValueError("existing PostToolUse hooks must be an array")
    if not _contains_bridge(result):
        groups.append(bridge_document()["hooks"]["PostToolUse"][0])
    return result


def _remove_bridge(current: dict[str, Any]) -> dict[str, Any]:
    result = cast(dict[str, Any], json.loads(json.dumps(current)))
    hooks = result.get("hooks")
    if not isinstance(hooks, dict):
        return result
    groups = hooks.get("PostToolUse")
    if not isinstance(groups, list):
        return result
    retained = []
    for group in groups:
        if not isinstance(group, dict):
            retained.append(group)
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            retained.append(group)
            continue
        kept = [
            handler
            for handler in handlers
            if not isinstance(handler, dict) or handler.get("command") != bridge_command()
        ]
        if kept:
            copy = dict(group)
            copy["hooks"] = kept
            retained.append(copy)
    if retained:
        hooks["PostToolUse"] = retained
    else:
        hooks.pop("PostToolUse", None)
    return result


def _contains_bridge(current: dict[str, Any]) -> bool:
    hooks = current.get("hooks")
    groups = hooks.get("PostToolUse") if isinstance(hooks, dict) else None
    return bool(
        isinstance(groups, list)
        and any(
            isinstance(handler, dict) and handler.get("command") == bridge_command()
            for group in groups
            if isinstance(group, dict) and isinstance(group.get("hooks"), list)
            for handler in group["hooks"]
        )
    )


def _parse(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("hooks.json must contain an object")
    return value


def _validate_target(target: Path) -> Path:
    expanded = target.expanduser()
    if not expanded.is_absolute() or expanded.is_symlink():
        raise ValueError("user hooks path must be absolute and not a symlink")
    if expanded.exists() and (
        not expanded.is_file() or stat.S_IMODE(expanded.stat().st_mode) & 0o077
    ):
        raise ValueError("user hooks file must be a restrictive regular file")
    return expanded


def _transaction_root(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or expanded.is_symlink():
        raise ValueError("transaction root must be absolute and not a symlink")
    expanded.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(expanded, 0o700)
    return expanded.resolve(strict=True)


def _write_atomic(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    path = root / "mutation.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
