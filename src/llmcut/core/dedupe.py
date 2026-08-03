from dataclasses import dataclass

from llmcut.model import ContextBlock


@dataclass(frozen=True, slots=True)
class Duplicate:
    removed_id: str
    retained_id: str
    digest: str


def deduplicate(blocks: list[ContextBlock]) -> tuple[list[ContextBlock], list[Duplicate]]:
    """Remove byte-identical blocks only within the same semantic kind."""
    seen: dict[tuple[str, str], ContextBlock] = {}
    kept: list[ContextBlock] = []
    removed: list[Duplicate] = []
    for block in blocks:
        key = (block.kind.value, block.digest)
        canonical = seen.get(key)
        if canonical is None:
            seen[key] = block
            kept.append(block)
        else:
            removed.append(Duplicate(block.id, canonical.id, block.digest))
    return kept, removed
