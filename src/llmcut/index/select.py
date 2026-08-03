from __future__ import annotations

import re
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
    by_path = {record.path: record for record in records}
    requested: set[str] = set()
    task_terms = {term.lower() for term in re.findall(r"[A-Za-z_][\w.-]+", task)}
    for record in ranked:
        relevance = score(record, task)
        if relevance.value > 0:
            requested.add(record.path)
            requested.update(record.tests)
            for imported in record.imports:
                for candidate in _import_candidates(record.path, imported):
                    if candidate in by_path:
                        requested.add(candidate)
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
        full_content = path.read_text(errors="replace")
        relevance = score(record, task)
        reference = store.put(
            full_content, f"repo:{record.path}", metadata={"digest": record.digest}
        )
        # Conservative packing: instructions, changed, and term-matched files are in-context.
        include = record.path in requested
        if include:
            content, ranges = _selected_content(record, full_content, task_terms)
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
                        "selected_ranges": ranges,
                        "full_file_digest": reference.digest,
                        "range_selected": bool(ranges),
                    },
                    reference=reference,
                )
            )
    return blocks


def _selected_content(
    record: FileRecord, content: str, task_terms: set[str], context_lines: int = 2
) -> tuple[str, list[list[int]]]:
    matched = [
        item
        for item in record.symbol_ranges
        if any(term in item.name.lower() for term in task_terms)
    ]
    if not matched or Path(record.path).name in {
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
    }:
        return content, []
    lines = content.splitlines()
    import_end = 0
    for index, line in enumerate(lines):
        if line.startswith(("import ", "from ", "const ", "export type ", "interface ")):
            import_end = index + 1
        elif line.strip() and import_end:
            break
    selected: set[int] = set(range(import_end))
    ranges: list[list[int]] = []
    for symbol in matched:
        start = max(1, symbol.start_line - context_lines)
        end = min(len(lines), symbol.end_line + context_lines)
        ranges.append([start, end])
        selected.update(range(start - 1, end))
    rendered: list[str] = []
    previous = -2
    for index in sorted(selected):
        if index > previous + 1:
            rendered.append(f"# … omitted lines; recoverable from {record.path} …")
        rendered.append(lines[index])
        previous = index
    return "\n".join(rendered), ranges


def _import_candidates(source: str, imported: str) -> list[str]:
    parent = Path(source).parent
    module = imported.removeprefix("./").replace(".", "/")
    return [str(parent / f"{module}{suffix}") for suffix in (".py", ".js", ".ts", "/__init__.py")]
