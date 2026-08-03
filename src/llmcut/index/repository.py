from __future__ import annotations

import json
import mimetypes
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llmcut.index.symbols import parse_source
from llmcut.model import digest_bytes

SECRET_NAMES = {".env", ".env.local", ".npmrc", ".pypirc", "credentials", "id_rsa", "id_ed25519"}
VENDORED_PARTS = {"node_modules", "vendor", ".venv", "dist", "build"}
LOCK_NAMES = {"uv.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"}


@dataclass(slots=True)
class FileRecord:
    path: str
    size: int
    language: str
    digest: str
    status: str
    imports: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    parser: str = "generic"
    binary: bool = False
    generated: bool = False
    vendored: bool = False
    lock_file: bool = False
    tests: list[str] = field(default_factory=list)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


class RepositoryIndex:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def build(self, include_untracked: bool = False) -> list[FileRecord]:
        tracked = _git(self.repo, "ls-files", "-z")
        if tracked.returncode == 0:
            paths = {item for item in tracked.stdout.split("\0") if item}
            if include_untracked:
                others = _git(self.repo, "ls-files", "--others", "--exclude-standard", "-z")
                paths.update(item for item in others.stdout.split("\0") if item)
        else:
            paths = {
                str(path.relative_to(self.repo)) for path in self.repo.rglob("*") if path.is_file()
            }
        status_result = _git(self.repo, "status", "--porcelain=v1", "-z")
        status = {
            entry[3:]: entry[:2] for entry in status_result.stdout.split("\0") if len(entry) > 3
        }
        records: list[FileRecord] = []
        for relative in sorted(paths):
            if not self._safe(relative):
                continue
            path = self.repo / relative
            if path.is_symlink() or not path.is_file():
                continue
            raw = path.read_bytes()
            binary = b"\0" in raw[:8192]
            content = "" if binary else raw.decode("utf-8", errors="replace")
            parsed = parse_source(relative, content) if not binary else parse_source(relative, "")
            language = _language(path)
            records.append(
                FileRecord(
                    relative,
                    len(raw),
                    language,
                    digest_bytes(raw),
                    status.get(relative, "tracked"),
                    parsed.imports,
                    parsed.symbols,
                    parsed.parser,
                    binary,
                    _generated(relative, content),
                    bool(set(Path(relative).parts) & VENDORED_PARTS),
                    path.name in LOCK_NAMES,
                )
            )
        self._associate_tests(records)
        return records

    def save(self, records: list[FileRecord], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.write_text(
            json.dumps([asdict(item) for item in records], sort_keys=True, indent=2)
        )
        os.chmod(destination, 0o600)

    def _safe(self, relative: str) -> bool:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.name in SECRET_NAMES:
            return False
        try:
            resolved = (self.repo / path).resolve()
            resolved.relative_to(self.repo)
        except (OSError, ValueError):
            return False
        return not any(part in {".git", ".llmcut", ".ssh", ".gnupg"} for part in path.parts)

    @staticmethod
    def _associate_tests(records: list[FileRecord]) -> None:
        tests = [
            item
            for item in records
            if item.path.startswith("tests/") or "test" in Path(item.path).stem
        ]
        for record in records:
            stem = Path(record.path).stem.removeprefix("test_")
            record.tests = sorted(
                test.path
                for test in tests
                if stem in Path(test.path).stem
                or any(stem in imported.split(".") for imported in test.imports)
            )


def _language(path: Path) -> str:
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".toml": "TOML",
        ".md": "Markdown",
    }
    return mapping.get(path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "unknown")


def _generated(path: str, content: str) -> bool:
    return "generated" in Path(path).name.lower() or "@generated" in content[:500].lower()
