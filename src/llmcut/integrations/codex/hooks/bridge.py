from __future__ import annotations

import os
import time
from pathlib import Path

from llmcut.integrations.codex.hooks.classify import classify_command
from llmcut.integrations.codex.hooks.config import MAX_HOOK_INPUT, HookConfig
from llmcut.integrations.codex.hooks.handler import append_metrics, handle_hook
from llmcut.integrations.codex.hooks.lease import HookLease, load_lease
from llmcut.integrations.codex.hooks.protocol import parse_hook_input
from llmcut.model import digest_bytes


def bridge_hook(raw: bytes, environment: dict[str, str] | None = None) -> dict[str, object] | None:
    """Run the static user bridge; absent or invalid activation always fails open."""
    env = os.environ if environment is None else environment
    lease_id = env.get("LLMCUT_HOOK_LEASE")
    token = env.get("LLMCUT_HOOK_LEASE_TOKEN")
    root = env.get("LLMCUT_HOOK_LEASE_ROOT")
    run_id = env.get("LLMCUT_HOOK_RUN_ID")
    if lease_id is None or token is None or root is None or run_id is None:
        return None
    try:
        if len(raw) > MAX_HOOK_INPUT:
            return None
        lease = load_lease(Path(root), lease_id, token, run_id)
        event = parse_hook_input(raw, Path(lease.repository_root))
        if event is None or event.cwd != Path(lease.allowed_cwd):
            return None
        if lease.mode == "observe":
            append_metrics(Path(lease.metrics_path), _observe_metrics(raw, lease, event))
            return None
        response, metrics = handle_hook(
            raw,
            HookConfig(Path(lease.repository_root), Path(lease.state_root)),
        )
        metrics["bridge_mode"] = "compact"
        metrics["lease_digest"] = digest_bytes(lease.lease_id.encode())
        metrics["hook_definition_digest"] = lease.hook_definition_digest
        append_metrics(Path(lease.metrics_path), metrics)
        return response
    except Exception:
        return None


def _observe_metrics(raw: bytes, lease: HookLease, event: object) -> dict[str, object]:
    from llmcut.integrations.codex.hooks.protocol import PostToolUseEvent

    assert isinstance(event, PostToolUseEvent)
    started = time.monotonic_ns()
    classified = classify_command(event.command)
    original_bytes = len(event.response.stdout.encode()) + len(event.response.stderr.encode())
    return {
        "event_supported": True,
        "applied": False,
        "bridge_mode": "observe",
        "classification": classified.classification.value,
        "command_digest": digest_bytes(event.command.encode()),
        "session_digest": digest_bytes(event.session_id.encode()),
        "turn_digest": digest_bytes(event.turn_id.encode()),
        "event_digest": digest_bytes(raw),
        "original_bytes": original_bytes,
        "compact_bytes": original_bytes,
        "original_tokens_estimate": max(1, original_bytes // 3),
        "compact_tokens_estimate": max(1, original_bytes // 3),
        "exit_code": event.response.exit_code,
        "lease_digest": digest_bytes(lease.lease_id.encode()),
        "hook_definition_digest": lease.hook_definition_digest,
        "duration_ms": (time.monotonic_ns() - started) / 1_000_000,
    }
