from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from llmcut.index import RepositoryIndex
from llmcut.index.repository import FileRecord

MAX_RESULT_BYTES = 128 * 1024
MAX_RANGE_LINES = 2_000
_SAFE_QUERY = re.compile(r"^[\w\s./:@+*?^$|()[\]{}\\=-]{1,256}$")


class RepositoryContext:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.records = {item.path: item for item in RepositoryIndex(self.root).build()}

    def map(self) -> str:
        return json.dumps(
            [
                {
                    "path": item.path,
                    "language": item.language,
                    "size": item.size,
                    "digest": item.digest,
                    "imports": item.imports,
                    "symbols": item.symbols,
                }
                for item in self.records.values()
            ],
            sort_keys=True,
        )

    def read(self, relative: str) -> tuple[FileRecord, str]:
        record = self.records.get(relative)
        if record is None or record.binary:
            raise ValueError("source is unavailable or excluded")
        target = (self.root / relative).resolve(strict=True)
        if self.root not in target.parents or target.is_symlink():
            raise ValueError("source escapes repository allowlist")
        content = target.read_text(errors="replace")
        from llmcut.model import digest_bytes

        if digest_bytes(content.encode()) != record.digest:
            raise ValueError("repository evidence is stale")
        return record, content

    def bounded(self, value: str) -> str:
        raw = value.encode()
        if len(raw) > MAX_RESULT_BYTES:
            raise ValueError("MCP result exceeds configured bound")
        return value


def create_mcp_server(repo: Path) -> FastMCP:
    context = RepositoryContext(repo)
    server = FastMCP(
        "llmcut",
        instructions="Retrieve exact, digest-verified evidence from the allowlisted repository.",
        json_response=True,
    )

    @server.tool()
    def llmcut_plan(task: str) -> dict[str, Any]:
        """Plan a task-scoped repository working set without executing commands."""
        if not task or len(task) > 8_192:
            raise ValueError("task must be bounded and non-empty")
        words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", task)}
        ranked = sorted(
            context.records.values(),
            key=lambda item: (
                -sum(
                    word in item.path.lower() or word in {s.lower() for s in item.symbols}
                    for word in words
                ),
                item.path,
            ),
        )
        selected = [item.path for item in ranked[: min(20, len(ranked))]]
        return {
            "repository": str(context.root),
            "selected": selected,
            "deferred": len(ranked) - len(selected),
        }

    @server.tool()
    def llmcut_context_get(context_id: str) -> dict[str, Any]:
        """Return exact repository evidence by stable relative identifier."""
        record, content = context.read(context_id)
        return {"id": context_id, "digest": record.digest, "content": context.bounded(content)}

    @server.tool()
    def llmcut_source_range(path: str, start: int, end: int) -> dict[str, Any]:
        """Return a bounded 1-based range from an allowlisted source file."""
        record, content = context.read(path)
        lines = content.splitlines()
        if start < 1 or end < start or end - start + 1 > MAX_RANGE_LINES or end > len(lines):
            raise ValueError("invalid or oversized source range")
        return {
            "path": path,
            "start": start,
            "end": end,
            "digest": record.digest,
            "content": context.bounded("\n".join(lines[start - 1 : end])),
        }

    @server.tool()
    def llmcut_symbol_get(path: str, symbol: str) -> dict[str, Any]:
        """Return exact source around a named indexed symbol."""
        record, content = context.read(path)
        if symbol not in record.symbols:
            raise ValueError("symbol is unavailable")
        lines = content.splitlines()
        match = next(
            (i for i, line in enumerate(lines) if re.search(rf"\b{re.escape(symbol)}\b", line)),
            None,
        )
        if match is None:
            raise ValueError("indexed symbol became stale")
        return {
            "path": path,
            "symbol": symbol,
            "digest": record.digest,
            "content": context.bounded("\n".join(lines[match : match + 200])),
        }

    @server.tool()
    def llmcut_dependencies(path: str) -> dict[str, Any]:
        """Return indexed direct dependencies for an allowlisted source."""
        record, _ = context.read(path)
        return {"path": path, "digest": record.digest, "dependencies": record.imports}

    @server.tool()
    def llmcut_log_search(path: str, pattern: str, limit: int = 20) -> dict[str, Any]:
        """Search bounded plain text in an allowlisted log or output file."""
        record, content = context.read(path)
        if not _SAFE_QUERY.fullmatch(pattern) or not 1 <= limit <= 100:
            raise ValueError("unsafe or oversized search")
        hits = [line for line in content.splitlines() if pattern in line][:limit]
        return {"path": path, "digest": record.digest, "matches": hits}

    @server.tool()
    def llmcut_checkpoint_get(checkpoint_id: str) -> dict[str, Any]:
        """Return a tracked checkpoint document by repository-relative identifier."""
        record, content = context.read(checkpoint_id)
        if "checkpoint" not in checkpoint_id.lower():
            raise ValueError("identifier is not an indexed checkpoint")
        return {"id": checkpoint_id, "digest": record.digest, "content": context.bounded(content)}

    @server.tool()
    def llmcut_tool_discover(category: str) -> dict[str, Any]:
        """Describe llmcut's compact retrieval capabilities by category."""
        catalog = {
            "repository": [
                "llmcut_plan",
                "llmcut_context_get",
                "llmcut_source_range",
                "llmcut_symbol_get",
                "llmcut_dependencies",
            ],
            "logs": ["llmcut_log_search"],
            "checkpoints": ["llmcut_checkpoint_get"],
        }
        if category not in catalog:
            raise ValueError("unknown capability category")
        return {"category": category, "tools": catalog[category]}

    @server.resource("llmcut://repository/map")
    def repository_map() -> str:
        return context.bounded(context.map())

    @server.resource("llmcut://context/{context_id}")
    def context_resource(context_id: str) -> str:
        record, content = context.read(context_id)
        return context.bounded(
            json.dumps({"id": context_id, "digest": record.digest, "content": content})
        )

    return server


def serve(repo: Path) -> None:
    create_mcp_server(repo).run(transport="stdio")
