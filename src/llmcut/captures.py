from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from llmcut.measurement import request_digest, response_digest

CAPTURE_SCHEMA_VERSION = "1"
_SENSITIVE = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|password", re.I)
_REASONING = re.compile(r"reasoning_content|chain_of_thought|thinking", re.I)


@dataclass(frozen=True, slots=True)
class CaptureVerification:
    capture_id: str
    turns: int
    prompt_content: bool


def _safe_location(root: Path, location: str) -> Path:
    relative = PurePosixPath(location)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("capture content location escapes capture root")
    target = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in target.parents:
        raise ValueError("capture content location escapes capture root")
    return target


def load_capture(path: Path) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    manifest_path = root / "manifest.json" if path.is_dir() else path
    value = json.loads(manifest_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("capture manifest must be an object")
    if value.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported capture schema version")
    if not isinstance(value.get("turns"), list) or not value["turns"]:
        raise ValueError("capture must contain at least one ordered turn")
    return cast(dict[str, Any], value)


def verify_capture(path: Path) -> CaptureVerification:
    root = path if path.is_dir() else path.parent
    value = load_capture(path)
    for number, turn in enumerate(value["turns"], 1):
        request = turn.get("request", {})
        response = turn.get("response", {})
        if request.get("content_location"):
            request_path = _safe_location(root, str(request["content_location"]))
            if not request_path.is_file():
                raise ValueError(f"capture turn {number} request is missing")
            payload = json.loads(request_path.read_text())
            if request_digest(payload) != request.get("digest"):
                raise ValueError(f"capture turn {number} request digest mismatch")
        if response.get("content_location"):
            response_path = _safe_location(root, str(response["content_location"]))
            if not response_path.is_file():
                raise ValueError(f"capture turn {number} response is missing")
            payload = json.loads(response_path.read_text())
            if response_digest(payload) != response.get("digest"):
                raise ValueError(f"capture turn {number} response digest mismatch")
        usage = turn.get("usage")
        if (
            usage
            and usage.get("quality") == "provider_reported"
            and (not request.get("digest") or not response.get("digest"))
        ):
            raise ValueError("provider usage requires bound request and response digests")
        if turn.get("provider") not in {None, value.get("provider")}:
            raise ValueError("capture provider identity mismatch")
        if turn.get("model") not in {None, value.get("model")}:
            raise ValueError("capture model identity mismatch")
    for number, artifact in enumerate(value.get("artifacts", []), 1):
        location = artifact.get("content_location")
        if not location:
            raise ValueError(f"capture artifact {number} has no content location")
        target = _safe_location(root, str(location))
        if not target.is_file() or request_digest(json.loads(target.read_text())) != artifact.get(
            "digest"
        ):
            raise ValueError(f"capture artifact {number} digest mismatch")
    return CaptureVerification(
        str(value.get("capture_id", "")),
        len(value["turns"]),
        bool(value.get("persistence", {}).get("prompt_content", False)),
    )


def redact_capture(path: Path) -> int:
    root = path if path.is_dir() else path.parent
    value = load_capture(path)
    changed = 0
    for turn in value["turns"]:
        for side, digest_fn in (("request", request_digest), ("response", response_digest)):
            item = turn.get(side, {})
            location = item.get("content_location")
            if not location:
                continue
            target = _safe_location(root, str(location))
            payload = json.loads(target.read_text())
            redacted, count = _redact(payload)
            changed += count
            target.write_text(json.dumps(redacted, sort_keys=True, separators=(",", ":")))
            os.chmod(target, 0o600)
            item["digest"] = digest_fn(redacted)
    value["redaction"] = {"applied": True, "version": "deterministic-v1"}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.chmod(manifest, 0o600)
    return changed


def _redact(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            if _SENSITIVE.search(key):
                result[key] = "[REDACTED]"
                count += 1
            elif _REASONING.search(key):
                count += 1
            else:
                result[key], nested = _redact(item)
                count += nested
        return result, count
    if isinstance(value, list):
        items: list[Any] = []
        count = 0
        for item in value:
            redacted, nested = _redact(item)
            items.append(redacted)
            count += nested
        return items, count
    return value, 0


def delete_capture(path: Path) -> None:
    if not path.is_dir() or not (path / "manifest.json").is_file():
        raise ValueError("capture deletion requires an explicit capture directory")
    root = path.resolve()
    if root == root.parent or len(root.parts) < 3:
        raise ValueError("refusing broad capture deletion target")
    verify_capture(root)
    shutil.rmtree(root)


def write_agent_capture(evaluation: dict[str, Any], destination: Path) -> Path:
    """Write a metadata-only, redacted, digest-verifiable agent evaluation capture."""
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("capture destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".llmcut-capture-", dir=destination.parent))
    os.chmod(temporary, 0o700)
    try:
        redacted, _ = _redact(evaluation)
        if not isinstance(redacted, dict):
            raise ValueError("agent evaluation capture must be an object")
        artifact_dir = temporary / "artifacts"
        artifact_dir.mkdir(mode=0o700)
        artifact = artifact_dir / "evaluation.json"
        artifact.write_text(json.dumps(redacted, sort_keys=True, separators=(",", ":")))
        os.chmod(artifact, 0o600)
        artifact_payload = json.loads(artifact.read_text())
        runs = [
            run
            for task in redacted.get("tasks", [])
            for run in task.get("runs", [])
            if isinstance(run, dict)
        ]
        turns = []
        from llmcut.integrations.codex.hooks.config import definition_digest

        for number, run in enumerate(runs, 1):
            requests = run.get("request_digests") or []
            responses = run.get("response_digests") or []
            turns.append(
                {
                    "sequence": number,
                    "request": {"digest": requests[-1] if requests else request_digest({})},
                    "response": {"digest": responses[-1] if responses else response_digest({})},
                    "usage": {
                        "input_tokens": (run.get("agent_usage") or {}).get("inputTokens", 0),
                        "quality": run.get("agent_usage_quality", "unavailable"),
                    },
                    "settings_digest": run.get("settings_digest"),
                    "worktree_revision": run.get("starting_commit"),
                    "validation_passed": run.get("validation_passed"),
                    "tool_calls": run.get("tool_calls", 0),
                    "mcp_calls": run.get("mcp_calls", 0),
                    "hook": {
                        "definition_digest": definition_digest(),
                        "trust_mode": (
                            "documented_one_off_bypass"
                            if (run.get("hook_observation") or {}).get("hook_events", 0)
                            else "disabled"
                        ),
                        "observation": run.get("hook_observation", {}),
                    },
                }
            )
        if not turns:
            turns.append(
                {
                    "sequence": 1,
                    "request": {"digest": request_digest({})},
                    "response": {"digest": response_digest({})},
                    "usage": {"input_tokens": 0, "quality": "unavailable"},
                }
            )
        manifest = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_id": str(redacted.get("run_id", "")),
            "provider": "codex-app-server",
            "model": str(redacted.get("environment", {}).get("model", "unknown")),
            "endpoint": "app-server-stdio",
            "capture_provenance": (
                "untrusted_fixture"
                if redacted.get("codex_version") == "configured-test-transport"
                else "live_agent"
                if any(run.get("agent_usage") for run in runs)
                else "locally_counted"
            ),
            "persistence": {"prompt_content": False, "source_content": False},
            "redaction": {"applied": True, "version": "deterministic-v1"},
            "turns": turns,
            "artifacts": [
                {
                    "kind": "agent_evaluation",
                    "digest": request_digest(artifact_payload),
                    "content_location": "artifacts/evaluation.json",
                }
            ],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.chmod(manifest_path, 0o600)
        os.replace(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary)
        raise
