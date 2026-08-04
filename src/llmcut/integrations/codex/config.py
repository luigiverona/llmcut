from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomlkit


@dataclass(frozen=True, slots=True)
class ConfigurationChange:
    changed: bool
    path: Path
    backup: Path | None
    before: str
    after: str


def configuration_snippet(repo: Path, executable: str = "llmcut") -> str:
    document = tomlkit.document()
    servers = tomlkit.table()
    server = tomlkit.table()
    server.add("command", executable)
    server.add("args", ["mcp", "serve", "--repo", str(repo.resolve())])
    server.add("required", True)
    servers.add("llmcut", server)
    document.add("mcp_servers", servers)
    return tomlkit.dumps(document)


def configure_codex(
    path: Path,
    repo: Path,
    *,
    remove: bool = False,
    dry_run: bool = False,
    executable: str = "llmcut",
) -> ConfigurationChange:
    before = path.read_text() if path.exists() else ""
    document = tomlkit.parse(before) if before else tomlkit.document()
    servers = document.get("mcp_servers")
    if remove:
        if servers is not None and "llmcut" in servers:
            del servers["llmcut"]
    else:
        if servers is None:
            servers = tomlkit.table()
            document["mcp_servers"] = servers
        server = tomlkit.table()
        server["command"] = executable
        server["args"] = ["mcp", "serve", "--repo", str(repo.resolve())]
        server["required"] = True
        servers["llmcut"] = server
    after = tomlkit.dumps(document)
    tomlkit.parse(after)
    changed = before != after
    if dry_run or not changed:
        return ConfigurationChange(changed, path, None, before, after)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".llmcut.bak")
        shutil.copy2(path, backup)
        os.chmod(backup, 0o600)
    descriptor, temporary = tempfile.mkstemp(prefix=".config.", suffix=".toml", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return ConfigurationChange(True, path, backup, before, after)
