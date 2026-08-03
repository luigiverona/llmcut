from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llmcut.index.symbols import PARSER_VERSION, SymbolRange, parse_source
from llmcut.model import digest_bytes
from llmcut.store.database import Database

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
    symbol_ranges: list[SymbolRange] = field(default_factory=list)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


class RepositoryIndex:
    def __init__(self, repo: Path, state_dir: Path | None = None) -> None:
        self.repo = repo.resolve()
        self.database = Database((state_dir or self.repo / ".llmcut") / "state.db")
        self.database.initialize()
        identity = _git(self.repo, "rev-parse", "--show-toplevel").stdout.strip() or str(self.repo)
        self.repository_id = digest_bytes(identity.encode())
        self.cache_hits = 0
        self.cache_misses = 0

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
        staged = _git(self.repo, "ls-files", "-s", "-z")
        blob_oids: dict[str, str] = {}
        for entry in staged.stdout.split("\0"):
            if "\t" in entry:
                metadata, filename = entry.split("\t", 1)
                fields = metadata.split()
                if len(fields) >= 2:
                    blob_oids[filename] = fields[1]
        records: list[FileRecord] = []
        for relative in sorted(paths):
            if not self._safe(relative):
                continue
            path = self.repo / relative
            if path.is_symlink() or not path.is_file():
                continue
            clean = status.get(relative, "tracked") == "tracked"
            cached = self._cached(relative, blob_oids.get(relative)) if clean else None
            if cached is not None:
                records.append(cached)
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            raw = path.read_bytes()
            binary = b"\0" in raw[:8192]
            content = "" if binary else raw.decode("utf-8", errors="replace")
            parsed = parse_source(relative, content) if not binary else parse_source(relative, "")
            language = _language(path)
            record = FileRecord(
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
                [],
                parsed.ranges,
            )
            records.append(record)
            self._store(relative, blob_oids.get(relative), record)
        self._remove_deleted(paths)
        self._associate_tests(records)
        for record in records:
            self._store(record.path, blob_oids.get(record.path), record)
        return records

    def stats(self) -> dict[str, int]:
        return {"cache_hits": self.cache_hits, "cache_misses": self.cache_misses}

    def _cached(self, path: str, blob_oid: str | None) -> FileRecord | None:
        if blob_oid is None:
            return None
        with self.database.connect() as db:
            row = db.execute(
                "SELECT record_json FROM repository_index WHERE repository_id=? "
                "AND path=? AND blob_oid=? AND parser_version=?",
                (self.repository_id, path, blob_oid, PARSER_VERSION),
            ).fetchone()
        return _record(json.loads(row[0])) if row else None

    def _store(self, path: str, blob_oid: str | None, record: FileRecord) -> None:
        with self.database.connect() as db:
            db.execute(
                "INSERT INTO repository_index VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(repository_id,path) DO UPDATE SET blob_oid=excluded.blob_oid,"
                "parser_version=excluded.parser_version,record_json=excluded.record_json,"
                "updated_at=excluded.updated_at",
                (
                    self.repository_id,
                    path,
                    blob_oid,
                    PARSER_VERSION,
                    json.dumps(asdict(record), sort_keys=True),
                    int(time.time()),
                ),
            )

    def _remove_deleted(self, current_paths: set[str]) -> None:
        with self.database.connect() as db:
            existing = {
                row[0]
                for row in db.execute(
                    "SELECT path FROM repository_index WHERE repository_id=?", (self.repository_id,)
                )
            }
            for path in existing - current_paths:
                db.execute(
                    "DELETE FROM repository_index WHERE repository_id=? AND path=?",
                    (self.repository_id, path),
                )

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


def _record(value: dict[str, Any]) -> FileRecord:
    data = dict(value)
    data["symbol_ranges"] = [SymbolRange(**item) for item in data.get("symbol_ranges", [])]
    return FileRecord(**data)
