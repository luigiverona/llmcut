from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llmcut.errors import RetrievalError
from llmcut.managed.planner import ContextPlan
from llmcut.managed.protocol import ManagedRequest, ToolDefinition
from llmcut.model import digest_bytes
from llmcut.store.evidence import EvidenceStore

MAX_RANGE_LINES = 2_000
MAX_PATTERN = 256
_SAFE_PATTERN = re.compile(r"^[\w\s./:@+*?^$|()[\]{}\\=-]+$")
_SECRET_PATH = re.compile(
    r"(?:^|/)(?:\.env(?:\.|$)|id_(?:rsa|ed25519)$|credentials?|secrets?)(?:/|\.|$)", re.I
)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    operation: str
    context_id: str
    content: str
    source: str
    digest: str
    revision: str | None
    cached: bool = False

    def model_content(self) -> str:
        return self.content

    def diagnostic_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "context_id": self.context_id,
            "source": self.source,
            "digest": self.digest,
            "revision": self.revision,
            "bytes": len(self.content.encode()),
            "cached": self.cached,
        }


class RetrievalService:
    def __init__(self, store: EvidenceStore, request: ManagedRequest, plan: ContextPlan) -> None:
        self.store = store
        self.request = request
        self.plan = plan
        self.context = {item.id: item for item in request.context}
        self.tools = {item.name: item for item in request.tools}
        self._cache: dict[str, RetrievalResult] = {}

    def execute(self, operation: str, arguments: dict[str, Any]) -> RetrievalResult:
        if operation not in self.plan.retrieval_operations:
            raise RetrievalError(f"retrieval operation is not available: {operation}")
        key = _cache_key(operation, arguments)
        if key in self._cache:
            previous = self._cache[key]
            return (
                RetrievalResult(*previous.__dict__.values())
                if hasattr(previous, "__dict__")
                else RetrievalResult(
                    previous.operation,
                    previous.context_id,
                    previous.content,
                    previous.source,
                    previous.digest,
                    previous.revision,
                    True,
                )
            )
        result = self._execute(operation, arguments)
        if len(result.content.encode()) > self.request.execution.max_retrieval_bytes:
            raise RetrievalError("retrieval result exceeds configured volume bound")
        self._cache[key] = result
        return result

    def _execute(self, operation: str, arguments: dict[str, Any]) -> RetrievalResult:
        if operation == "tool.discover":
            name = _required_string(arguments, "name")
            tool = self.tools.get(name)
            if tool is None or name not in self.plan.deferred_tools:
                raise RetrievalError(f"tool is unavailable: {name}")
            content = _tool_json(tool)
            return RetrievalResult(
                operation, name, content, "managed:tools", digest_bytes(content.encode()), None
            )
        identifier = _required_string(arguments, "id")
        item = self.context.get(identifier)
        if item is None or identifier not in self.plan.deferred:
            raise RetrievalError(f"context is unavailable or already model-bound: {identifier}")
        if item.extensions.get("secret") is True or _SECRET_PATH.search(item.source_path or ""):
            raise RetrievalError("secret evidence is excluded from managed retrieval")
        digest = self.plan.evidence[identifier]
        content = self.store.get(digest)
        if digest_bytes(content.encode()) != digest:
            raise RetrievalError("retrieved evidence digest mismatch")
        if item.revision and arguments.get("revision") not in {None, item.revision}:
            raise RetrievalError("stale repository evidence revision")
        if operation in {"source.range", "log.range"}:
            content = _range(content, arguments)
        elif operation == "log.search":
            content = _search(content, arguments)
        elif operation == "dependency.get":
            requested = arguments.get("dependency")
            dependencies = item.dependencies if requested is None else (str(requested),)
            if any(dep not in item.dependencies for dep in dependencies):
                raise RetrievalError("requested context is not a declared dependency")
            chunks = []
            for dependency in dependencies:
                dependency_digest = self.plan.evidence[dependency]
                chunks.append(self.store.get(dependency_digest))
            content = "\n".join(chunks)
        elif operation == "symbol.get":
            symbol = _required_string(arguments, "symbol")
            content = _symbol(content, symbol)
        elif operation not in {"evidence.get", "context.expand", "repository.map"}:
            raise RetrievalError(f"unsupported retrieval operation: {operation}")
        return RetrievalResult(
            operation,
            identifier,
            content,
            item.source_path or f"managed:{identifier}",
            digest,
            item.revision,
        )


def _range(content: str, arguments: dict[str, Any]) -> str:
    start, end = int(arguments.get("start", 1)), int(arguments.get("end", 0))
    lines = content.splitlines()
    end = end or len(lines)
    if start < 1 or end < start or end - start + 1 > MAX_RANGE_LINES or end > len(lines):
        raise RetrievalError("invalid or oversized 1-based line range")
    return "\n".join(lines[start - 1 : end])


def _search(content: str, arguments: dict[str, Any]) -> str:
    pattern = _required_string(arguments, "pattern")
    if len(pattern) > MAX_PATTERN or not _SAFE_PATTERN.fullmatch(pattern):
        raise RetrievalError("unsafe or oversized search pattern")
    regex = bool(arguments.get("regex", False))
    if regex and (
        re.search(r"\([^)]*[*+][^)]*\)[*+{]", pattern) or re.search(r"\\[1-9]|\(\?[=!<]", pattern)
    ):
        raise RetrievalError("unsafe or oversized search pattern")
    matcher = re.compile(pattern) if regex else None
    limit = min(max(int(arguments.get("limit", 20)), 1), 100)
    hits = [
        line
        for line in content.splitlines()
        if (matcher.search(line) if matcher else pattern in line)
    ]
    return "\n".join(hits[:limit])


def _symbol(content: str, symbol: str) -> str:
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if re.search(rf"\b(?:class|def|function|const|let|var)\s+{re.escape(symbol)}\b", line):
            start = index
            end = min(len(lines), index + 200)
            return "\n".join(lines[start:end])
    raise RetrievalError(f"symbol not found: {symbol}")


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value or len(value) > 256:
        raise RetrievalError(f"{key} must be a bounded non-empty string")
    return value


def _cache_key(operation: str, arguments: dict[str, Any]) -> str:
    import json

    return operation + ":" + json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _tool_json(tool: ToolDefinition) -> str:
    import json

    return json.dumps(tool.transport_dict(), sort_keys=True, separators=(",", ":"))
