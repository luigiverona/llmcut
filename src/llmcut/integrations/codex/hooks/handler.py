from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from llmcut.integrations.codex.hooks.classify import CommandClass, classify_command
from llmcut.integrations.codex.hooks.compact import compact_bash_result
from llmcut.integrations.codex.hooks.config import HookConfig
from llmcut.integrations.codex.hooks.protocol import parse_hook_input, replacement_response
from llmcut.integrations.codex.hooks.state import HookEvidenceStore
from llmcut.model import digest_bytes


def handle_hook(
    raw: bytes, config: HookConfig
) -> tuple[dict[str, object] | None, dict[str, object]]:
    started = time.monotonic_ns()
    metrics: dict[str, object] = {"event_supported": False, "applied": False}
    try:
        config.validate()
        event = parse_hook_input(raw, config.repository_root)
        if event is None:
            metrics["fallback_reason"] = "malformed or unsupported event"
            return None, metrics
        metrics["event_supported"] = True
        classified = classify_command(event.command)
        metrics["classification"] = classified.classification.value
        if (
            classified.classification is CommandClass.RECOVERY
            or os.environ.get("LLMCUT_HOOK_RECOVERY") == "1"
        ):
            metrics["fallback_reason"] = "recovery output never recursively compacted"
            return None, metrics
        response = event.response
        original_bytes = len(response.stdout.encode()) + len(response.stderr.encode())
        metrics["original_bytes"] = original_bytes
        preliminary = compact_bash_result(
            classification=classified.classification,
            stdout=response.stdout,
            stderr=response.stderr,
            exit_code=response.exit_code,
            threshold_bytes=config.threshold_bytes,
            maximum_compact_bytes=config.maximum_compact_bytes,
        )
        if preliminary.reason != "exact evidence unavailable":
            metrics.update(asdict(preliminary))
            return None, metrics
        store = HookEvidenceStore(config.state_root)
        evidence = store.put(
            stdout=response.stdout,
            stderr=response.stderr,
            exit_code=response.exit_code,
            command_digest=digest_bytes(event.command.encode()),
            revision=_revision(config.repository_root),
            classification=classified.classification.value,
            event_digest=digest_bytes(raw),
            session_id=event.session_id,
            parser=preliminary.parser,
            parser_version=preliminary.parser_version,
        )
        result = compact_bash_result(
            classification=classified.classification,
            stdout=response.stdout,
            stderr=response.stderr,
            exit_code=response.exit_code,
            threshold_bytes=config.threshold_bytes,
            maximum_compact_bytes=config.maximum_compact_bytes,
            evidence_id=evidence.evidence_id,
        )
        metrics.update(asdict(result))
        if not result.applied or result.model_content is None:
            return None, metrics
        metrics["evidence_created"] = True
        return replacement_response(result.model_content), metrics
    except Exception as exc:
        metrics["fallback_reason"] = f"hook failure: {type(exc).__name__}"
        return None, metrics
    finally:
        metrics["duration_ms"] = (time.monotonic_ns() - started) / 1_000_000


def _revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def append_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        stream.write(json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n")
