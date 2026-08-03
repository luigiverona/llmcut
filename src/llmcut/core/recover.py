from __future__ import annotations

import re

from llmcut.model import CanonicalRequest
from llmcut.store.evidence import EvidenceStore


class Recovery:
    def __init__(self, store: EvidenceStore) -> None:
        self.store = store

    def evidence(self, digest: str) -> str:
        return self.store.get(digest)

    def source_range(self, digest: str, start: int, end: int) -> str:
        if start < 1 or end < start:
            raise ValueError("invalid 1-based line range")
        return "\n".join(self.evidence(digest).splitlines()[start - 1 : end])

    def matching(self, digest: str, pattern: str, *, regex: bool = False, context: int = 2) -> str:
        lines = self.evidence(digest).splitlines()
        matcher = re.compile(pattern) if regex else None
        hits = [
            index
            for index, line in enumerate(lines)
            if (matcher.search(line) if matcher else pattern in line)
        ]
        selected: set[int] = set()
        for hit in hits:
            selected.update(range(max(0, hit - context), min(len(lines), hit + context + 1)))
        return "\n".join(f"{index + 1}: {lines[index]}" for index in sorted(selected))

    def restore_request(self, optimized: CanonicalRequest) -> CanonicalRequest:
        digest = optimized.passthrough.get("llmcut_original")
        if not isinstance(digest, str):
            # Size-aware fallback returns the untouched canonical request; it is already restored.
            return CanonicalRequest.from_dict(optimized.to_dict())
        import json

        return CanonicalRequest.from_dict(json.loads(self.evidence(digest)))

    def dependencies(self, request: CanonicalRequest, block_id: str) -> list[str]:
        blocks = {block.id: block for block in [*request.blocks, *request.tools]}
        recovered: list[str] = []
        for item in blocks[block_id].dependencies:
            if item in blocks:
                reference = blocks[item].reference
                if reference is not None:
                    recovered.append(self.evidence(reference.digest))
        return recovered
