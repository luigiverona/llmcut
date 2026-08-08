from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from llmcut.model import digest_bytes

DEFAULT_THRESHOLD = 8_192
MAX_HOOK_INPUT = 8 * 1024 * 1024
MAX_COMPACT_BYTES = 24_000


class HookSource(StrEnum):
    PROJECT_HOOKS_JSON = "project_hooks_json"
    PROJECT_CONFIG_TOML = "project_config_toml"
    CLI_OVERRIDE = "cli_override"
    PROFILE = "profile"
    USER_HOOKS_JSON = "user_hooks_json"


@dataclass(frozen=True, slots=True)
class HookSourceDiagnostic:
    source: HookSource
    configured: bool
    config_layer_active: bool | None
    requires_project_trust: bool
    requires_definition_trust: bool
    hook_observed: bool | None
    runtime_version: str | None


@dataclass(frozen=True, slots=True)
class ProjectTrustOverride:
    value: str
    trusted_path_digest: str
    scope: str = "invocation_only"


@dataclass(frozen=True, slots=True)
class HookConfig:
    repository_root: Path
    state_root: Path
    threshold_bytes: int = DEFAULT_THRESHOLD
    maximum_compact_bytes: int = MAX_COMPACT_BYTES
    maximum_store_bytes: int = 256 * 1024 * 1024
    maximum_age_seconds: int = 7 * 24 * 60 * 60

    def validate(self) -> HookConfig:
        repository = self.repository_root.resolve(strict=True)
        state = self.state_root.resolve()
        if not 1_024 <= self.threshold_bytes <= 16 * 1024 * 1024:
            raise ValueError("invalid hook compaction threshold")
        if not 2_048 <= self.maximum_compact_bytes <= 256 * 1024:
            raise ValueError("invalid compact output bound")
        if state == repository or repository in state.parents:
            raise ValueError("hook state must not be stored in the repository")
        return self


def hook_command() -> str:
    invoked = Path(sys.argv[0])
    installed = str(invoked.resolve()) if invoked.name == "llmcut" and invoked.is_file() else None
    installed = installed or shutil.which("llmcut")
    if installed:
        return f'"{Path(installed).resolve()}" hook handle'
    executable = Path(sys.executable).resolve()
    return f'"{executable}" -m llmcut hook handle'


def proposed_document() -> dict[str, Any]:
    return {
        "description": "llmcut exact recoverable Bash output compaction",
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "^Bash$",
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_command(),
                            "timeout": 15,
                            "statusMessage": "Compacting recoverable Bash output",
                            "additionalContextLimit": 9000,
                        }
                    ],
                }
            ]
        },
    }


def definition_digest(document: dict[str, Any] | None = None) -> str:
    payload = json.dumps(document or proposed_document(), sort_keys=True, separators=(",", ":"))
    return digest_bytes(payload.encode())


def inline_overrides() -> tuple[str, ...]:
    """Render a CLI-only hook source for conformance diagnostics."""
    command = json.dumps(hook_command())
    value = (
        'hooks.PostToolUse=[{matcher="^Bash$",hooks=[{type="command",command='
        f'{command},timeout=15,statusMessage="Compacting recoverable Bash output",'
        "additionalContextLimit=9000}]}]"
    )
    return ("features.hooks=true", value)


def project_hook_overrides() -> tuple[str, ...]:
    """Enable the authoritative project ``hooks.json`` source without duplicating it."""
    return ("features.hooks=true",)


def project_trust_override(worktree: Path) -> ProjectTrustOverride:
    """Render exact, invocation-scoped trust for one canonical disposable worktree."""
    expanded = worktree.expanduser()
    if not expanded.is_absolute():
        raise ValueError("trusted worktree must be an absolute path")
    if expanded.is_symlink():
        raise ValueError("trusted worktree must not be a symlink")
    canonical = expanded.resolve(strict=True)
    if not canonical.is_dir():
        raise ValueError("trusted worktree must be a directory")
    rendered = str(canonical)
    if any(character in rendered for character in ("\x00", "\n", "\r")):
        raise ValueError("trusted worktree contains an unsafe character")
    # JSON strings are valid TOML basic strings and safely quote dots, spaces,
    # Unicode, and embedded quotes in a dotted key segment.
    value = f'projects.{json.dumps(rendered, ensure_ascii=False)}.trust_level="trusted"'
    return ProjectTrustOverride(value, digest_bytes(rendered.encode()))


def install_hooks(target: Path, *, dry_run: bool = False) -> dict[str, Any]:
    target = target.expanduser().resolve()
    proposed = proposed_document()
    existing: dict[str, Any] = {}
    if target.exists():
        parsed = json.loads(target.read_text())
        if not isinstance(parsed, dict):
            raise ValueError("existing hook configuration must be a JSON object")
        existing = parsed
    merged = _merge(existing, proposed)
    result = {
        "target": str(target),
        "digest": definition_digest(merged),
        "changed": merged != existing,
    }
    if dry_run or merged == existing:
        return result
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    backup = target.with_suffix(target.suffix + ".bak")
    if target.exists():
        shutil.copy2(target, backup)
        os.chmod(backup, 0o600)
    descriptor, name = tempfile.mkstemp(prefix=".hooks-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(merged, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(name, 0o600)
        json.loads(Path(name).read_text())
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    return result


def remove_hooks(target: Path, *, dry_run: bool = False) -> dict[str, Any]:
    target = target.expanduser().resolve()
    if not target.exists():
        return {"target": str(target), "changed": False}
    parsed = json.loads(target.read_text())
    cleaned = _remove(parsed)
    if cleaned == parsed or dry_run:
        return {"target": str(target), "changed": cleaned != parsed}
    descriptor, name = tempfile.mkstemp(prefix=".hooks-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(cleaned, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(name, 0o600)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return {"target": str(target), "changed": True}


def _merge(current: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    result = cast(dict[str, Any], json.loads(json.dumps(current)))
    hooks = result.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("existing hooks field must be an object")
    groups = hooks.setdefault("PostToolUse", [])
    if not isinstance(groups, list):
        raise ValueError("existing PostToolUse hooks must be an array")
    wanted = proposed["hooks"]["PostToolUse"][0]
    commands = [
        handler.get("command")
        for group in groups
        if isinstance(group, dict)
        for handler in group.get("hooks", [])
        if isinstance(handler, dict)
    ]
    if hook_command() not in commands:
        groups.append(wanted)
    return result


def _remove(current: dict[str, Any]) -> dict[str, Any]:
    result = cast(dict[str, Any], json.loads(json.dumps(current)))
    hooks = result.get("hooks")
    if not isinstance(hooks, dict):
        return result
    groups = hooks.get("PostToolUse")
    if not isinstance(groups, list):
        return result
    retained = []
    for group in groups:
        handlers = group.get("hooks") if isinstance(group, dict) else None
        if not isinstance(handlers, list):
            retained.append(group)
            continue
        kept = [
            item
            for item in handlers
            if not isinstance(item, dict) or item.get("command") != hook_command()
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
