from __future__ import annotations

import json
import os
import re
import shutil
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
