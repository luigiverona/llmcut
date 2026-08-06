from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmcut.model import digest_bytes

EVIDENCE_VERSION = 1


@dataclass(frozen=True, slots=True)
class HookEvidence:
    evidence_id: str
    stdout: str
    stderr: str
    exit_code: int
    metadata: dict[str, Any]


class HookEvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def put(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int,
        command_digest: str,
        revision: str,
        classification: str,
        event_digest: str,
        session_id: str,
        parser: str,
        parser_version: str,
    ) -> HookEvidence:
        payload = _payload(stdout, stderr, exit_code)
        evidence_id = digest_bytes(payload)
        target = self.root / evidence_id
        if target.exists():
            return self.get(evidence_id)
        temporary = Path(tempfile.mkdtemp(prefix=".evidence-", dir=self.root))
        os.chmod(temporary, 0o700)
        metadata: dict[str, Any] = {
            "schema_version": EVIDENCE_VERSION,
            "evidence_id": evidence_id,
            "command_digest": command_digest,
            "revision": revision,
            "classification": classification,
            "event_digest": event_digest,
            "session_digest": digest_bytes(session_id.encode()),
            "timestamp": int(time.time()),
            "parser": parser,
            "parser_version": parser_version,
            "stdout_bytes": len(stdout.encode()),
            "stderr_bytes": len(stderr.encode()),
            "exit_code": exit_code,
        }
        try:
            for name, content in (("stdout", stdout), ("stderr", stderr)):
                path = temporary / name
                path.write_bytes(content.encode())
                os.chmod(path, 0o600)
            manifest = temporary / "metadata.json"
            manifest.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
            os.chmod(manifest, 0o600)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return HookEvidence(evidence_id, stdout, stderr, exit_code, metadata)

    def get(self, evidence_id: str) -> HookEvidence:
        _validate_id(evidence_id)
        target = (self.root / evidence_id).resolve(strict=True)
        if target.parent != self.root or target.is_symlink():
            raise ValueError("evidence path escapes store")
        stdout = _safe_read(target, "stdout")
        stderr = _safe_read(target, "stderr")
        metadata = json.loads(_safe_read(target, "metadata.json"))
        exit_code = metadata.get("exit_code")
        if not isinstance(metadata, dict) or not isinstance(exit_code, int):
            raise ValueError("invalid hook evidence metadata")
        if digest_bytes(_payload(stdout, stderr, exit_code)) != evidence_id:
            raise ValueError("hook evidence digest verification failed")
        return HookEvidence(evidence_id, stdout, stderr, exit_code, metadata)

    def info(self, evidence_id: str) -> dict[str, Any]:
        return dict(self.get(evidence_id).metadata)

    def collect(
        self,
        *,
        maximum_age_seconds: int,
        maximum_total_bytes: int,
        dry_run: bool = False,
        active_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        active = active_ids or set()
        now = int(time.time())
        entries: list[tuple[Path, int, int]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or path.is_symlink() or not _valid_id(path.name):
                continue
            size = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
            entries.append((path, int(path.stat().st_mtime), size))
        total = sum(item[2] for item in entries)
        removed: list[str] = []
        for path, modified, size in sorted(entries, key=lambda item: item[1]):
            expired = now - modified > maximum_age_seconds
            over = total > maximum_total_bytes
            if path.name in active or not (expired or over):
                continue
            removed.append(path.name)
            total -= size
            if not dry_run:
                shutil.rmtree(path)
        return {"removed": removed, "remaining_bytes": total, "dry_run": dry_run}


def render_exact(evidence: HookEvidence) -> str:
    return (
        f"exit_code: {evidence.exit_code}\n"
        "--- stdout ---\n"
        f"{evidence.stdout}"
        "\n--- stderr ---\n"
        f"{evidence.stderr}"
    )


def exact_lines(evidence: HookEvidence) -> list[str]:
    return render_exact(evidence).splitlines(keepends=True)


def _payload(stdout: str, stderr: str, exit_code: int) -> bytes:
    return json.dumps(
        {"stdout": stdout, "stderr": stderr, "exit_code": exit_code},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def _validate_id(value: str) -> None:
    if not _valid_id(value):
        raise ValueError("invalid hook evidence id")


def _valid_id(value: str) -> bool:
    prefix = "sha256:"
    return (
        value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and all(character in "0123456789abcdef" for character in value[len(prefix) :])
    )


def _safe_read(root: Path, name: str) -> str:
    path = (root / name).resolve(strict=True)
    if path.parent != root or path.is_symlink() or not path.is_file():
        raise ValueError("invalid hook evidence file")
    return path.read_text()
