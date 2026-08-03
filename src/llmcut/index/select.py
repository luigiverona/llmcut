from __future__ import annotations

from pathlib import Path

from llmcut.index.repository import FileRecord
from llmcut.index.score import score
from llmcut.model import BlockKind, ContextBlock
from llmcut.store.evidence import EvidenceStore


def pack_repository(
    repo: Path,
    records: list[FileRecord],
    task: str,
    store: EvidenceStore,
    max_file_bytes: int = 200_000,
) -> list[ContextBlock]:
    ranked = sorted(records, key=lambda item: (-score(item, task).value, item.path))
    blocks: list[ContextBlock] = []
    for record in ranked:
        path = repo / record.path
        if (
            record.binary
            or record.vendored
            or record.generated
            or record.lock_file
            or record.size > max_file_bytes
        ):
            continue
        content = path.read_text(errors="replace")
        relevance = score(record, task)
        reference = store.put(content, f"repo:{record.path}", metadata={"digest": record.digest})
        # Conservative packing: instructions, changed, and term-matched files are in-context.
        include = relevance.value > 0
        if include:
            blocks.append(
                ContextBlock(
                    f"repo:{record.path}",
                    BlockKind.REPOSITORY,
                    content,
                    record.path,
                    priority=min(100, relevance.value),
                    dependencies=record.imports,
                    metadata={
                        "reasons": relevance.reasons,
                        "parser": record.parser,
                        "tests": record.tests,
                    },
                    reference=reference,
                )
            )
    return blocks
