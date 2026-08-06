from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from llmcut.model import digest_bytes


class PostVariant(StrEnum):
    NONE = "none"
    ADDITIONAL = "additional-context"
    CONTINUE_REASON = "continue-stop-reason"
    CONTINUE_CONTEXT = "continue-additional-context"
    CONTINUE_BOTH = "continue-both"
    BLOCK_REASON = "block-reason"
    BLOCK_CONTEXT = "block-reason-additional-context"
    EXIT_TWO = "exit-two"


@dataclass(frozen=True, slots=True)
class Canaries:
    original_head: str
    original_middle: str
    original_tail: str
    compact_only: str

    @classmethod
    def create(cls) -> Canaries:
        return cls(
            *(
                _canary(prefix)
                for prefix in ("ORIGINAL_HEAD", "ORIGINAL_MIDDLE", "ORIGINAL_TAIL", "COMPACT_ONLY")
            )
        )


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    variant: str
    state: str
    original_head_present: bool | None
    original_middle_present: bool | None
    original_tail_present: bool | None
    compact_only_present: bool | None
    hook_invoked: bool
    command_exit_code: int | None
    original_bytes: int
    replacement_bytes: int
    agent_usage: dict[str, int]
    detail: str
    runtime_version: str = "unavailable"
    trust_mode: str = "documented_one_off_bypass"
    capture_digest: str | None = None

    @property
    def exclusive(self) -> bool:
        return self.state == "exclusive"


def post_response(variant: PostVariant, compact: str) -> tuple[dict[str, Any] | None, int, str]:
    context = {"hookEventName": "PostToolUse", "additionalContext": compact}
    if variant is PostVariant.NONE:
        return None, 0, ""
    if variant is PostVariant.ADDITIONAL:
        return {"hookSpecificOutput": context}, 0, ""
    if variant is PostVariant.CONTINUE_REASON:
        return {"continue": False, "stopReason": compact}, 0, ""
    if variant is PostVariant.CONTINUE_CONTEXT:
        return {"continue": False, "hookSpecificOutput": context}, 0, ""
    if variant is PostVariant.CONTINUE_BOTH:
        return {"continue": False, "stopReason": compact, "hookSpecificOutput": context}, 0, ""
    if variant is PostVariant.BLOCK_REASON:
        return {"decision": "block", "reason": compact}, 0, ""
    if variant is PostVariant.BLOCK_CONTEXT:
        return {"decision": "block", "reason": compact, "hookSpecificOutput": context}, 0, ""
    return None, 2, compact


def simulated_model_result(variant: PostVariant, canaries: Canaries) -> dict[str, str | None]:
    """Model the documented semantics for offline protocol tests, not release evidence."""
    original = {
        "original_head": canaries.original_head,
        "original_middle": canaries.original_middle,
        "original_tail": canaries.original_tail,
    }
    if variant is PostVariant.NONE:
        return {**original, "compact_only": None}
    if variant is PostVariant.ADDITIONAL:
        return {**original, "compact_only": canaries.compact_only}
    if variant is PostVariant.CONTINUE_REASON:
        return {**original, "compact_only": None}
    if variant in {PostVariant.CONTINUE_CONTEXT, PostVariant.CONTINUE_BOTH}:
        return {**original, "compact_only": canaries.compact_only}
    return {
        "original_head": None,
        "original_middle": None,
        "original_tail": None,
        "compact_only": canaries.compact_only,
    }


def evaluate_returned(
    variant: PostVariant,
    returned: str,
    canaries: Canaries,
    *,
    hook_invoked: bool,
    command_exit_code: int | None,
    original_bytes: int,
    replacement_bytes: int,
    usage: dict[str, int] | None = None,
) -> ConformanceResult:
    try:
        value = json.loads(returned)
    except json.JSONDecodeError:
        return ConformanceResult(
            variant.value,
            "inconclusive",
            None,
            None,
            None,
            None,
            hook_invoked,
            command_exit_code,
            original_bytes,
            replacement_bytes,
            usage or {},
            "agent response was not JSON",
        )
    if not isinstance(value, dict):
        return ConformanceResult(
            variant.value,
            "inconclusive",
            None,
            None,
            None,
            None,
            hook_invoked,
            command_exit_code,
            original_bytes,
            replacement_bytes,
            usage or {},
            "agent response was not an object",
        )
    present = tuple(value.get(name) == expected for name, expected in asdict(canaries).items())
    missing_originals = all(
        value.get(name) is None for name in ("original_head", "original_middle", "original_tail")
    )
    compact = value.get("compact_only") == canaries.compact_only
    exclusive = hook_invoked and missing_originals and compact
    state = (
        "exclusive"
        if exclusive
        else "nonexclusive"
        if hook_invoked and any(present)
        else "inconclusive"
    )
    return ConformanceResult(
        variant.value,
        state,
        present[0],
        present[1],
        present[2],
        present[3],
        hook_invoked,
        command_exit_code,
        original_bytes,
        replacement_bytes,
        usage or {},
        "canary comparison completed",
    )


def read_probe_state(path: Path) -> tuple[PostVariant, str, Path]:
    target = path.resolve(strict=True)
    if target.is_symlink() or stat.S_IMODE(target.stat().st_mode) & 0o077:
        raise ValueError("probe state must be a restrictive regular file")
    value = json.loads(target.read_text())
    if not isinstance(value, dict):
        raise ValueError("invalid probe state")
    variant = PostVariant(str(value.get("variant")))
    compact = str(value.get("compact", ""))
    marker = Path(str(value.get("marker", ""))).resolve()
    if not compact.startswith("COMPACT_ONLY_") or marker.parent != target.parent:
        raise ValueError("invalid probe state binding")
    return variant, compact, marker


def handle_conformance_hook(raw: bytes, state_path: Path) -> tuple[dict[str, Any] | None, int, str]:
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("hook_event_name") != "PostToolUse":
        return None, 0, ""
    variant, compact, marker = read_probe_state(state_path)
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    return post_response(variant, compact)


def run_fake_matrix() -> list[ConformanceResult]:
    results = []
    for variant in PostVariant:
        canaries = Canaries.create()
        response, _, _ = post_response(variant, canaries.compact_only)
        returned = json.dumps(simulated_model_result(variant, canaries))
        results.append(
            evaluate_returned(
                variant,
                returned,
                canaries,
                hook_invoked=variant is PostVariant.NONE
                or response is not None
                or variant is PostVariant.EXIT_TWO,
                command_exit_code=0,
                original_bytes=32_000,
                replacement_bytes=len(canaries.compact_only.encode()),
            )
        )
    return results


def run_live_post_matrix(
    *,
    executable: str = "codex",
    output_dir: Path | None = None,
    repository: Path | None = None,
    variants: tuple[PostVariant, ...] | None = None,
) -> list[ConformanceResult]:
    destination = (
        output_dir.resolve() if output_dir else Path(tempfile.mkdtemp(prefix="llmcut-hook-probe-"))
    )
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    repo = (repository or Path.cwd()).resolve(strict=True)
    codex_dir = repo / ".codex"
    if codex_dir.exists():
        raise RuntimeError("runtime probe requires a repository without an existing .codex layer")
    codex_dir.mkdir(mode=0o700)
    results: list[ConformanceResult] = []
    version = _runtime_version(executable)
    try:
        for variant in variants or tuple(PostVariant):
            result = _run_live_variant(destination, repo, codex_dir, executable, variant)
            bound = replace(result, runtime_version=version)
            digest = digest_bytes(
                json.dumps(asdict(bound), sort_keys=True, separators=(",", ":")).encode()
            )
            results.append(replace(bound, capture_digest=digest))
    finally:
        (codex_dir / "hooks.json").unlink(missing_ok=True)
        codex_dir.rmdir()
    return results


def _run_live_variant(
    root: Path, repository: Path, codex_dir: Path, executable: str, variant: PostVariant
) -> ConformanceResult:
    run = root / f"{variant.value}-{secrets.token_hex(8)}"
    run.mkdir(mode=0o700)
    canaries = Canaries.create()
    script = run / "emit.py"
    lines = [
        "import sys",
        f"print({canaries.original_head!r})",
        "print('x' * 12000)",
        f"print({canaries.original_middle!r})",
        "print('y' * 12000)",
        f"print({canaries.original_tail!r})",
    ]
    script.write_text("\n".join(lines) + "\n")
    os.chmod(script, 0o600)
    marker = run / "hook-invoked"
    state = run / "state.json"
    state.write_text(
        json.dumps(
            {"variant": variant.value, "compact": canaries.compact_only, "marker": str(marker)}
        )
    )
    os.chmod(state, 0o600)
    configured = os.environ.get("LLMCUT_PROBE_EXECUTABLE")
    resolved = configured or shutil.which("llmcut")
    if not resolved:
        raise RuntimeError("llmcut executable is unavailable for hook probe")
    llmcut = Path(resolved).resolve(strict=True)
    command = f'"{llmcut}" hook conformance-handle --state "{state}"'
    hooks = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "^Bash$",
                    "hooks": [{"type": "command", "command": command, "timeout": 10}],
                }
            ]
        }
    }
    hook_file = codex_dir / "hooks.json"
    hook_file.write_text(json.dumps(hooks))
    os.chmod(hook_file, 0o600)
    prompt = (
        f"Run exactly: {os.fspath(Path(sys.executable).resolve())} {script}. "
        "Do not inspect files or run anything else. After the command, return only one JSON "
        "object with keys original_head, original_middle, original_tail, compact_only. For each "
        "key, reproduce the exact visible token beginning with that uppercase prefix, or null "
        "when no such token is visible."
    )
    process = subprocess.run(
        [
            executable,
            "--dangerously-bypass-hook-trust",
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            prompt,
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    messages: list[str] = []
    usage: dict[str, int] = {}
    command_exit: int | None = None
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if isinstance(item, dict) and item.get("type") == "agent_message":
            messages.append(str(item.get("text", "")))
        if (
            isinstance(item, dict)
            and item.get("type") == "command_execution"
            and item.get("status") in {"completed", "failed"}
        ):
            command_exit = item.get("exit_code") if isinstance(item.get("exit_code"), int) else None
        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
            usage = {
                str(key): int(value)
                for key, value in event["usage"].items()
                if isinstance(value, int)
            }
    original_bytes = 24_000 + sum(
        len(value.encode()) + 1
        for value in (canaries.original_head, canaries.original_middle, canaries.original_tail)
    )
    returned = messages[-1] if messages else ""
    result = evaluate_returned(
        variant,
        returned,
        canaries,
        hook_invoked=marker.exists(),
        command_exit_code=command_exit,
        original_bytes=original_bytes,
        replacement_bytes=len(canaries.compact_only.encode()),
        usage=usage,
    )
    metadata = {
        **asdict(result),
        "runtime_output_digest": digest_bytes(process.stdout.encode()),
        "runtime_error_digest": digest_bytes(process.stderr.encode()),
    }
    report = run / "result.json"
    report.write_text(json.dumps(metadata, sort_keys=True, indent=2))
    os.chmod(report, 0o600)
    script.unlink(missing_ok=True)
    state.unlink(missing_ok=True)
    hook_file.unlink(missing_ok=True)
    return result


def _canary(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _runtime_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"], text=True, capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"
