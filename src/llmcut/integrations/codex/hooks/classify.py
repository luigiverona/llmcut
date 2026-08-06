from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import StrEnum


class CommandClass(StrEnum):
    TEST = "test"
    TYPECHECK = "typecheck"
    LINT = "lint"
    BUILD = "build"
    GIT_STATUS = "git_status"
    GIT_DIFF = "git_diff"
    SEARCH = "search"
    FILE_LISTING = "file_listing"
    SOURCE_READ = "source_read"
    LOG_READ = "log_read"
    PACKAGE_MANAGER = "package_manager"
    MUTATION = "mutation"
    NETWORK = "network"
    INTERACTIVE = "interactive"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClassifiedCommand:
    classification: CommandClass
    commands: tuple[tuple[str, ...], ...]
    reason: str


def classify_command(command: str, *, _depth: int = 0) -> ClassifiedCommand:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return ClassifiedCommand(CommandClass.UNKNOWN, (), "shell tokenization failed")
    segments: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in tokens:
        if token and set(token) <= {";", "&", "|"}:
            if current:
                segments.append(tuple(current))
                current = []
        else:
            current.append(token)
    if current:
        segments.append(tuple(current))
    if not segments:
        return ClassifiedCommand(CommandClass.UNKNOWN, (), "empty command")
    if len(segments) == 1 and _depth == 0:
        wrapped = _unwrap_shell(segments[0])
        if wrapped is not None:
            return classify_command(wrapped, _depth=1)
    classes = tuple(_classify_segment(segment) for segment in segments)
    unsafe = {
        CommandClass.MUTATION,
        CommandClass.NETWORK,
        CommandClass.INTERACTIVE,
        CommandClass.UNKNOWN,
    }
    selected = next((item for item in classes if item in unsafe), None)
    if selected is not None:
        return ClassifiedCommand(
            selected, tuple(segments), "compound command contains unsafe class"
        )
    if len(set(classes)) != 1:
        return ClassifiedCommand(CommandClass.UNKNOWN, tuple(segments), "mixed command classes")
    return ClassifiedCommand(classes[0], tuple(segments), "recognized executable")


def _classify_segment(argv: tuple[str, ...]) -> CommandClass:
    words = list(argv)
    while words and ("=" in words[0] and not words[0].startswith(("/", "./"))):
        words.pop(0)
    if not words:
        return CommandClass.UNKNOWN
    executable = words[0].rsplit("/", 1)[-1]
    lower = [item.lower() for item in words]
    if (
        executable == "llmcut"
        and len(lower) > 2
        and lower[1] == "hook"
        and lower[2] in {"show", "range", "search", "info"}
    ):
        return CommandClass.RECOVERY
    if executable in {"curl", "wget", "ssh", "scp", "nc", "ncat"}:
        return CommandClass.NETWORK
    if executable in {"vim", "vi", "nano", "less", "more", "top", "htop"}:
        return CommandClass.INTERACTIVE
    if executable in {"rm", "mv", "cp", "install", "chmod", "chown", "tee", "dd", "truncate"}:
        return CommandClass.MUTATION
    if (
        executable in {"pytest", "py.test"}
        or lower[:3] == ["python", "-m", "pytest"]
        or (executable == "uv" and lower[1:3] == ["run", "pytest"])
    ):
        return CommandClass.TEST
    if executable in {"node"} and "--test" in lower:
        return CommandClass.TEST
    if executable in {"npm", "pnpm", "yarn"} and "test" in lower[1:3]:
        return CommandClass.TEST
    if executable in {"mypy", "basedpyright", "pyright", "tsc"} or (
        executable == "npx" and "tsc" in lower[1:3]
    ):
        return CommandClass.TYPECHECK
    if executable in {"ruff", "eslint", "flake8"}:
        return CommandClass.LINT
    if executable == "git" and len(lower) > 1 and lower[1] == "status":
        return CommandClass.GIT_STATUS
    if executable == "git" and len(lower) > 1 and lower[1] in {"diff", "show"}:
        return CommandClass.GIT_DIFF
    if executable in {"rg", "grep"}:
        return CommandClass.SEARCH
    if executable in {"find", "ls", "tree"}:
        return CommandClass.FILE_LISTING
    if executable in {"cat", "sed", "head", "tail", "bat"}:
        return (
            CommandClass.LOG_READ
            if any(item.endswith((".log", ".out")) for item in words)
            else CommandClass.SOURCE_READ
        )
    if executable in {"npm", "pnpm", "yarn", "pip", "uv", "poetry"}:
        return CommandClass.PACKAGE_MANAGER
    if executable in {"make", "cmake", "cargo", "go"} and "build" in lower:
        return CommandClass.BUILD
    return CommandClass.UNKNOWN


def _unwrap_shell(argv: tuple[str, ...]) -> str | None:
    if len(argv) != 3:
        return None
    executable = argv[0].rsplit("/", 1)[-1]
    if executable not in {"bash", "sh"} or argv[1] not in {"-c", "-lc"}:
        return None
    return argv[2]
